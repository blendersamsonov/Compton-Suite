"""GUI-facing engine on the new tabulated-energy path (particles.py/
deposition.py/spectrum4d.py/reference.py), covering the "spectrum-shaped"
observables gui_adapter.py needs: total yield, angle-integrated spectrum,
angular spectrum, angular-range spectrum (plan.md Phase 2 Stage B's gap
table -- these are wiring work, the new path already computes all of them).

Deliberately not named Compton, to avoid colliding with/shadowing
core.Compton while both paths coexist during the interim migration (see
CLAUDE.md's GUI integration section, "Relationship to the CPU-fallback PR"
open decision in plan.md): TabulatedEngine wraps an existing, already
`set_electron_parameters`/`set_laser_parameters`/`set_foci_displacement`-
configured core.Compton instance purely for its config-bag properties
(k0_las, Wph, N_l, a0, beta_ff, ellipticity, sigma_ex/sigma_ey, ...) --
particles.sample_bunch/push_and_sample already take a `compton` object as
their parameter source, so this is reuse, not a new dependency. The new
path never calls any of core.Compton's calculate_*/GPU-kernel methods.

Does NOT cover the temporal envelope or spatial distribution (plan.md
Stage C, not implemented in the new path yet -- Stage 0 collapses each
particle's trajectory into one (gamma, theta_x, theta_y, a0, weight) tuple
by design, see CLAUDE.md's "a0 is a trajectory average" section). Callers
needing those two observables still go through core.Compton's
calculate_intersection/time_envelope/spatial_envelope directly, in
parallel -- see gui_adapter.py's run_simulation, which builds one
core.Compton instance and drives both this engine and the legacy
temporal/spatial calls off it during this interim migration period.
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
    `gamma`/`weight`/`backend` are None until `.run()` is called."""

    compton: object
    table: object = field(default=None, repr=False)
    gamma: np.ndarray = field(default=None, repr=False)
    weight: np.ndarray = field(default=None, repr=False)
    backend: str = field(default='numpy', repr=False)

    def run(self, n_particles, gamma_0, sigma_gamma0, *, n_steps=100,
            n_bins=(48, 48, 48, 12), scheme='nearest', backend='numpy',
            a0_max=DEFAULT_A0_MAX, chirp=0.0, angle_energy_corr=0.0, rng=None):
        """Stage 0 (particles.sample_bunch/push_and_sample) + Stage 1
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
        """
        bunch = particles.sample_bunch(
            self.compton, n_particles, gamma_0, sigma_gamma0,
            chirp=chirp, angle_energy_corr=angle_energy_corr, rng=rng)
        gamma, tx, ty, a0_shape, w = particles.push_and_sample(
            self.compton, bunch, n_steps=n_steps, backend=backend)

        table = deposition.build_table(
            gamma, tx, ty, a0_shape, w, n_bins=n_bins, scheme=scheme,
            a0_kind='shape')
        table = deposition.retarget_a0(table, self.compton.a0, a0_max=a0_max)

        self.table = table
        self.gamma = gamma
        self.weight = w
        self.backend = backend
        return table

    @property
    def total_yield(self):
        """table.total_weight -- validated to 1-3% against core.Compton's
        calculate_total() by construction (deposition.py Stage 1, see
        CLAUDE.md "Current state"); retarget_a0 preserves it exactly."""
        return float(self.table.total_weight)

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
        on this run's table, auto-selecting the GPU or CPU kernel per
        plan.md Phase 2 Stage A unless `device` is given explicitly.
        `theta_x`/`theta_y`/`s` must already be `cp`/`np` arrays matching
        the chosen device, same convention as calculate_angular_spectrum_4d
        itself (caller converts via `xp.asarray`, not this method)."""
        return spectrum4d.calculate_angular_spectrum_4d(
            self.table, s, theta_x, theta_y, phi_pol,
            samples_per_point=samples_per_point, device=device)
