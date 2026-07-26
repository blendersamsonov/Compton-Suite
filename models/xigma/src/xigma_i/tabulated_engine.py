"""GUI-facing engine on the tabulated-energy pipeline (particles.py/
deposition.py/spectrum4d.py/reference.py), covering everything
gui_adapter.py needs: total yield, angle-integrated spectrum, angular
spectrum, angular-range spectrum, temporal envelope, spatial distribution.

`TabulatedEngine` wraps an already-configured `config.Compton` instance
purely for its config-bag properties (k0_las, Wph, N_l, a0, beta_ff,
ellipticity, sigma_ex/sigma_ey, ...) -- particles.bunch_from_macrobunch/
push_and_sample already take a `compton` object as their parameter source,
so this is reuse, not a new dependency. `compton` itself runs no
computation; all of it happens in this class's `.run()`/property methods.

temporal_envelope/spatial_distribution: particles.push_and_sample's
n_time_bins/n_spatial_bins bin the exact same per-timestep `contribution`
array that already gets summed into `weight` (L), during the same
trajectory-integration loop -- not a second pass, and not per-timestep a0
sampling (see push_and_sample's/CLAUDE.md's "a0 is a trajectory average"
warnings, which are about a0 specifically; time and space are orthogonal
axes with no such regime-validity caveat). Needs no post-hoc rescale to
reproduce total_yield: built from the same already-correctly-normalised
`contribution` array L sums over time, so it integrates to total_yield
exactly by construction -- verified numerically, not just asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import particles, deposition, spectrum4d, reference

# The a0 range the weakly-nonlinear approximation this whole codebase is
# built on (a0 <~ 1, see CLAUDE.md's "a0 is a trajectory average" section)
# is meant to be valid over -- a fixed *model* parameter, not derived from
# any particular collision's actual compton.a0. See deposition.retarget_a0
# and CLAUDE.md's "Architecture: new path" for the full rationale.
DEFAULT_A0_MAX = 0.5


@dataclass
class TabulatedEngine:
    """Drives Stage 0/1/2 of the new path for one `compton` config. `table`/
    `gamma`/`weight`/`backend`/`diagnostics` are None until `.run()` is
    called."""

    compton: object
    table: object = field(default=None, repr=False)
    gamma: np.ndarray = field(default=None, repr=False)
    weight: np.ndarray = field(default=None, repr=False)
    backend: str = field(default='numpy', repr=False)
    diagnostics: object = field(default=None, repr=False)

    def run(self, *, n_steps=100, n_bins=(48, 48, 48, 12), scheme='nearest',
            backend='numpy', a0_max=DEFAULT_A0_MAX,
            n_time_bins=None, n_spatial_bins=None, bunch):
        """Stage 0 (particles.push_and_sample) + Stage 1
        (deposition.build_table, a0_kind='shape') + retarget to this
        engine's compton.a0 (deposition.retarget_a0) -- one physical,
        spectrum-ready table.

        backend: 'numpy' or 'cupy', passed to push_and_sample and
        reference.angle_integrated_spectrum (build_table's own `device` is
        left to auto-detect from the Stage 0 output arrays -- same
        'cpu'/'gpu' outcome, different vocabulary, no need to translate
        twice). Table.H always ends up host/numpy regardless (see
        deposition.Table's docstring), so `backend` only matters for the
        raw gamma/weight arrays this class keeps for `.spectrum()`.

        n_time_bins/n_spatial_bins (optional): forwarded to
        push_and_sample -- when given, populates
        `.temporal_envelope`/`.spatial_distribution` (None otherwise). The
        auto-derived bunch-wide time/spatial windows are used (no
        t_edges/spatial_edges override exposed here; construct via
        particles.push_and_sample directly if a specific window is needed).

        bunch: a pre-built particles.Bunch (e.g. from
        particles.bunch_from_macrobunch) to push -- required, keyword-only.
        Electron sampling is the caller's job, not this engine's: there is
        exactly one place electron bunches get drawn from a beam
        description (compton_io.bunch.sample_gaussian_bunch), not one per
        model (mirrors kascade's run_simulation: electrons is mandatory,
        no internal sampling fallback).
        """
        result = particles.push_and_sample(
            self.compton, bunch, n_steps=n_steps, backend=backend,
            n_time_bins=n_time_bins, n_spatial_bins=n_spatial_bins)
        if n_time_bins is not None or n_spatial_bins is not None:
            gamma, tx, ty, a0_shape, w, diagnostics = result
        else:
            gamma, tx, ty, a0_shape, w = result
            diagnostics = None

        table = deposition.build_table(
            gamma, tx, ty, a0_shape, w, n_bins=n_bins, scheme=scheme,
            a0_kind='shape')
        table = deposition.retarget_a0(table, self.compton.a0, a0_max=a0_max)

        self.table = table
        self.gamma = gamma
        self.weight = w
        self.backend = backend
        self.diagnostics = diagnostics
        return table

    @property
    def total_yield(self):
        """table.total_weight -- retarget_a0 preserves it exactly."""
        return float(self.table.total_weight)

    @property
    def temporal_envelope(self):
        """(t_seconds, rate) bin-center arrays -- photon-emission rate vs
        time. None if .run() wasn't called with n_time_bins."""
        if self.diagnostics is None or self.diagnostics.time_envelope is None:
            return None
        edges = self.diagnostics.t_edges
        t_centers = 0.5 * (edges[:-1] + edges[1:])
        return t_centers, self.diagnostics.time_envelope

    @property
    def spatial_distribution(self):
        """(x_centers, y_centers, density) -- transverse areal density
        [photons/cm^2]. None if .run() wasn't called with
        n_spatial_bins."""
        if self.diagnostics is None or self.diagnostics.spatial_envelope is None:
            return None
        x_edges, y_edges = self.diagnostics.spatial_x_edges, self.diagnostics.spatial_y_edges
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        return x_centers, y_centers, self.diagnostics.spatial_envelope

    def spectrum(self, s):
        """dN/ds, angle-integrated over all emission solid angle --
        reference.angle_integrated_spectrum on this run's raw Stage 0/1
        samples (not the table -- this observable doesn't need the
        angular/a0 binning at all, see that function's docstring)."""
        return reference.angle_integrated_spectrum(
            self.gamma, self.weight, s, backend=self.backend)

    def angular_spectrum(self, s, theta_x, theta_y, phi_pol,
                          samples_per_point=32, device=None):
        """d2N/(ds dOmega) grid -- spectrum4d.calculate_angular_spectrum_4d
        on this run's table, auto-selecting the GPU or CPU kernel unless
        `device` is given explicitly.
        `theta_x`/`theta_y`/`s` must already be `cp`/`np` arrays matching
        the chosen device, same convention as calculate_angular_spectrum_4d
        itself (caller converts via `xp.asarray`, not this method)."""
        return spectrum4d.calculate_angular_spectrum_4d(
            self.table, s, theta_x, theta_y, phi_pol,
            samples_per_point=samples_per_point, device=device)
