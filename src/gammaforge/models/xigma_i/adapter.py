"""ModelAdapter wrappers around this package's tabulated-overlap pipeline.

Two adapters, registered under two names (see ``gammaforge.models.api.
discover_models``):

  * ``XigmaAdapter`` (registered as ``"xigma-i"``) drives the full
    Stage 0/1/2 tabulated pipeline (``tabulated_engine.TabulatedEngine``):
    semi-analytic, GPU/CPU, smooth binned spectral-density output.
  * ``DirectAdapter`` (registered as ``"delta"``) is this package's own
    brute-force per-macroparticle resonance-binning mode -- no table, no
    importance sampling, reusing this package's Stage 0
    (``particles.push_and_sample``) directly. Not a separate model
    package; delta IS xigma_i, just a different Stage-2 evaluation.

Neither adapter has a standalone ``Config`` dataclass: each adapter
instance IS the model-specific parameter state (numeric knobs as plain
``self`` attributes, updated from ``job.extra`` on every ``run()``) --
there's no extra object to shuttle values through. There is also no shared
CGS parameter-bundle type: each call into Stage 0/1/2
(``particles.push_and_sample``, ``TabulatedEngine.run()``) converts exactly
the ``GaussianElectronBeam``/``GaussianParaxialLaser`` fields it needs,
right at that call site, as plain floats -- see ``particles.push_and_sample``'s
docstring. ``XigmaAdapter`` holds its ``TabulatedEngine`` as ``self.engine``
after a run, so ``spectrum_in_angular_range()`` can recompute an on-demand
angular-range query without rerunning the whole simulation; ``DirectAdapter``
has no persistent table, so it instead keeps the raw per-particle arrays
needed for its own on-demand requery. Both cache ``self.device`` (not a
unit conversion, so it isn't part of the "convert at the call site"
principle above -- see ``gammaforge.misc.get_xp``/``.to_host``).
``beta_ff``/``phi_pol``/``ellipticity`` are laser-pulse properties, read
straight off ``job.interaction.laser`` -- not adapter state, not
``Job.extra`` keys.
"""

from __future__ import annotations

import warnings

import numpy as np

from gammaforge.io.interaction import InteractionParameters
from gammaforge.io.photons import AngularRangeSpectrumResult, PhasespaceSlice
from gammaforge.io.units import HBAR_Q, LIGHT_TIME_CONTEXT, M_TO_CM
from gammaforge.misc import get_xp, to_host
from gammaforge.models.api import (
    AXIS_ENERGY,
    AXIS_THETA_X,
    AXIS_THETA_Y,
    AXIS_TIME,
    AXIS_X,
    AXIS_Y,
    Job,
    Results,
    find_slice_request,
)

# The a0 range the weakly-nonlinear approximation this whole codebase is
# built on (a0 <~ 1) is meant to be valid over -- a fixed *model*
# parameter, not derived from any particular collision's actual a0.
# See deposition.retarget_a0.
DEFAULT_A0_MAX = 0.5


def _theta_grid(gamma0: float, n_points: int = 33,
                theta_range: tuple[float, float] | None = None) -> np.ndarray:
    """A generous fixed window around the current collimation angle, wide
    enough that a cheap on-demand re-integration (no re-run) stays valid
    for any collimation angle the user is likely to dial in without
    clicking Calculate again.

    If ``theta_range`` is given (e.g. by ``spectrum_in_angular_range`` for a
    live, user-picked angular window), it's used verbatim instead of the
    generous auto-derived window."""
    if theta_range is not None:
        lo, hi = theta_range
        return np.linspace(lo, hi, n_points, dtype=np.float32)
    half_window = 3.0 / gamma0
    return np.linspace(-half_window, half_window, n_points, dtype=np.float32)


def _resolve_device(device_preference: str) -> str:
    from gammaforge.misc import detect_device

    if device_preference == "auto":
        return detect_device()
    device = device_preference
    if device == "gpu":
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() == 0:
                device = "cpu"
        except Exception:
            device = "cpu"
    if device not in ("gpu", "cpu"):
        raise ValueError(f"device_preference must be 'auto', 'gpu', or 'cpu', got {device_preference!r}")
    return device


# ---------------------------------------------------------------------------
# XigmaAdapter (full tabulated Stage 0/1/2 pipeline)
# ---------------------------------------------------------------------------
class XigmaAdapter:
    def __init__(self):
        # Numeric knobs -- this adapter IS the config, no separate Config
        # dataclass. Defaults mirror what model_params() has always declared.
        self.device_preference = "auto"
        self.n_steps_0 = 64
        self.n_bins_gamma = 48
        self.n_bins_theta_x = 48
        self.n_bins_theta_y = 48
        self.n_bins_a0 = 12
        self.a0_max = DEFAULT_A0_MAX
        self.samples_per_point_2 = 32
        self.chunk = 0  # 0 = auto-size from free VRAM, see particles.estimate_chunk_size

        # Recompute-cache state, populated by run() -- lets
        # spectrum_in_angular_range() reuse the already-built table without
        # rerunning the whole simulation.
        self.interaction: InteractionParameters | None = None
        self.engine = None      # tabulated_engine.TabulatedEngine, once run() has been called
        self.device = "cpu"     # resolved device string for the last run
        self.angular_rescale = 1.0

    def model_params(self) -> list[tuple[str, float | str, str]]:
        return [
            ("Device (auto/gpu/cpu)", self.device_preference, "device_preference"),
            ("Stage 0 trajectory steps", self.n_steps_0, "n_steps_0"),
            ("Grid bins: gamma", self.n_bins_gamma, "n_bins_gamma"),
            ("Grid bins: theta_x", self.n_bins_theta_x, "n_bins_theta_x"),
            ("Grid bins: theta_y", self.n_bins_theta_y, "n_bins_theta_y"),
            ("Grid bins: a0", self.n_bins_a0, "n_bins_a0"),
            ("a0_max (model valid range)", self.a0_max, "a0_max"),
            ("Stage 2 samples/point", self.samples_per_point_2, "samples_per_point_2"),
            ("Particle chunk size (0=auto from VRAM)", self.chunk, "chunk"),
        ]

    def model_choices(self) -> dict[str, list[str]]:
        return {"device_preference": ["auto", "gpu", "cpu"]}

    def run(self, job: Job) -> Results:
        from . import particles
        from .tabulated_engine import TabulatedEngine

        extra = job.extra
        self.device_preference = str(extra.get("device_preference", self.device_preference))
        self.n_steps_0 = int(float(extra.get("n_steps_0", self.n_steps_0)))
        self.n_bins_gamma = int(float(extra.get("n_bins_gamma", self.n_bins_gamma)))
        self.n_bins_theta_x = int(float(extra.get("n_bins_theta_x", self.n_bins_theta_x)))
        self.n_bins_theta_y = int(float(extra.get("n_bins_theta_y", self.n_bins_theta_y)))
        self.n_bins_a0 = int(float(extra.get("n_bins_a0", self.n_bins_a0)))
        self.a0_max = float(extra.get("a0_max", self.a0_max))
        self.samples_per_point_2 = int(float(extra.get("samples_per_point_2", self.samples_per_point_2)))
        self.chunk = int(float(extra.get("chunk", self.chunk)))

        self.interaction = job.interaction
        electrons = job.interaction.electrons
        device = _resolve_device(self.device_preference)
        self.device = device

        if job.interaction.laser.crossing_angle != 0.0:
            warnings.warn(
                "XigmaAdapter (xigma-i) is head-on-only; ignoring nonzero "
                f"crossing_angle ({job.interaction.laser.crossing_angle} rad)."
            )

        beam, laser = self.interaction.electrons.gaussian_fit, self.interaction.laser
        xp = get_xp(device)

        gamma_0 = beam.gamma0

        time_req = find_slice_request(job.output.slices, AXIS_TIME)
        space_req = find_slice_request(job.output.slices, AXIS_X, AXIS_Y)
        energy_req = find_slice_request(job.output.slices, AXIS_ENERGY)
        energy_angle_req = find_slice_request(job.output.slices, AXIS_ENERGY, AXIS_THETA_X, AXIS_THETA_Y)

        n_time_bins = time_req.bins[0] if time_req is not None else 128
        n_spatial_bins = space_req.bins if space_req is not None else (64, 64)

        # Total yield, angle-integrated spectrum, angular spectrum, temporal
        # envelope, and spatial distribution all come from TabulatedEngine,
        # which holds (beam, laser) directly and converts to CGS itself,
        # exactly where push_and_sample needs it -- no shared parameter
        # bundle here.
        engine = TabulatedEngine(beam=beam, laser=laser)
        push_backend = 'cupy' if device == 'gpu' else 'numpy'

        n_particles_new = electrons.n_particles
        n_bins = (self.n_bins_gamma, self.n_bins_theta_x, self.n_bins_theta_y, self.n_bins_a0)
        chunk = self.chunk if self.chunk > 0 else particles.estimate_chunk_size(
            n_particles_new, self.n_steps_0, push_backend)
        engine.run(n_steps=self.n_steps_0,
                   n_bins=n_bins, backend=push_backend, a0_max=self.a0_max,
                   n_time_bins=n_time_bins, n_spatial_bins=n_spatial_bins,
                   bunch=electrons, chunk=chunk)
        total_yield = engine.total_yield
        photon_slices = [PhasespaceSlice(axes={}, distr=np.asarray(total_yield))]

        # engine.temporal_envelope/.spatial_distribution: rate vs seconds /
        # areal density vs cm. Convert cm/photons-per-cm^2 to SI. These stay
        # on-device (cupy) when push_backend='cupy' -- particles.py's
        # _bin_temporal/_bin_spatial never convert to host -- so must be
        # converted here before matplotlib (which cannot implicitly convert a
        # cupy array).
        if time_req is not None:
            t_seconds_raw, rate_raw = engine.temporal_envelope
            t_seconds = to_host(t_seconds_raw, device)
            rate = to_host(rate_raw, device)
            photon_slices.append(PhasespaceSlice(axes={AXIS_TIME: t_seconds}, distr=rate))

        if space_req is not None:
            x_centers_cm, y_centers_cm, density_per_cm2 = engine.spatial_distribution
            x_centers = to_host(x_centers_cm, device) / M_TO_CM
            y_centers = to_host(y_centers_cm, device) / M_TO_CM
            density = to_host(density_per_cm2, device) * (M_TO_CM ** 2)
            photon_slices.append(PhasespaceSlice(axes={AXIS_X: x_centers, AXIS_Y: y_centers}, distr=density))

        Wph_MeV = (HBAR_Q * laser.omega0).to("megaelectron_volt").magnitude
        s_scale_MeV = 4.0 * Wph_MeV

        E_gamma_eV_mean = None
        if energy_req is not None:
            n_e = energy_req.bins[0]
            if energy_req.range is not None and energy_req.range[0] is not None:
                lo_eV, hi_eV = energy_req.range[0]
                s_lo, s_hi = lo_eV / (s_scale_MeV * 1e6), hi_eV / (s_scale_MeV * 1e6)
            else:
                s_lo, s_hi = 0.0, 1.1 * gamma_0 ** 2
            s_tot = xp.linspace(s_lo, s_hi, n_e, dtype=xp.float32)
            dNds_tot = to_host(engine.spectrum(s_tot), device)
            E_eV = (to_host(s_tot, device) * s_scale_MeV) * 1e6
            dNdE_per_eV = dNds_tot / s_scale_MeV / 1e6  # dN/ds -> dN/dE(MeV) -> dN/dE(eV)
            photon_slices.append(PhasespaceSlice(axes={AXIS_ENERGY: E_eV}, distr=dNdE_per_eV))
            E_gamma_eV_mean = float(np.average(E_eV, weights=dNdE_per_eV)) if dNdE_per_eV.sum() else 0.0

        # Angular spectrum, over a generous fixed theta window (or the
        # requested one) and an energy grid (kept smaller for kernel-launch
        # cost: grid size = theta_x.size * theta_y.size * s.size). Always
        # computed (even if the joint slice itself isn't requested) so
        # ``angular_rescale`` stays calibrated for spectrum_in_angular_range's
        # on-demand queries.
        if energy_angle_req is not None:
            n_e_ang, n_tx, n_ty = energy_angle_req.bins
            e_range, tx_range, ty_range = energy_angle_req.range or (None, None, None)
        else:
            n_e_ang, n_tx, n_ty = 96, 33, 33
            e_range = tx_range = ty_range = None

        theta_x = _theta_grid(gamma_0, n_points=n_tx, theta_range=tx_range)
        theta_y = _theta_grid(gamma_0, n_points=n_ty, theta_range=ty_range)
        if e_range is not None:
            lo_eV, hi_eV = e_range
            s_lo, s_hi = lo_eV / (s_scale_MeV * 1e6), hi_eV / (s_scale_MeV * 1e6)
        else:
            s_lo, s_hi = 0.0, 1.1 * gamma_0 ** 2
        s_ang = xp.linspace(s_lo, s_hi, n_e_ang, dtype=xp.float32)

        d2Nds_dOmega, _dt, _debug = engine.angular_spectrum(
            s_ang, xp.asarray(theta_x), xp.asarray(theta_y), laser.phi_pol,
            samples_per_point=self.samples_per_point_2, device=device)
        E_ang_eV = (to_host(s_ang, device) * s_scale_MeV) * 1e6
        d2NdEdOmega = d2Nds_dOmega / s_scale_MeV / 1e6  # -> eV^-1 sr^-1

        _dtx, _dty = np.gradient(theta_x), np.gradient(theta_y)
        _dE_ang = np.gradient(E_ang_eV)
        _full_integral = float(np.einsum("ijk,i,j,k->", d2NdEdOmega, _dtx, _dty, _dE_ang))
        angular_rescale = (total_yield / _full_integral) if _full_integral > 0 else 1.0
        d2NdEdOmega = d2NdEdOmega * angular_rescale

        if energy_angle_req is not None:
            photon_slices.append(PhasespaceSlice(
                axes={AXIS_THETA_X: theta_x, AXIS_THETA_Y: theta_y, AXIS_ENERGY: E_ang_eV},
                distr=d2NdEdOmega))

        model_specific = dict(
            a0=laser.a0_focus,
            # FLAGGED: see the "QUICK FIX" comment above angular_rescale's
            # computation -- 1.0 would mean the kernel's own normalisation
            # already agreed with total_yield; it currently doesn't (~2*pi-ish),
            # so this is visibly != 1.0 until the underlying kernel issue is
            # actually root-caused.
            angular_spectrum_rescale_applied=angular_rescale,
        )
        if E_gamma_eV_mean is not None:
            model_specific["E_gamma_eV_mean"] = E_gamma_eV_mean

        res = Results(photon_slices=photon_slices, model_specific=model_specific)

        self.engine = engine
        self.angular_rescale = angular_rescale
        return res

    def spectrum_in_angular_range(
            self, theta_x_range: tuple[float, float], theta_y_range: tuple[float, float],
            n_points: int = 33, n_energy: int = 96) -> AngularRangeSpectrumResult:
        """Fresh, on-demand spectrum over an arbitrary user-picked angular
        sub-range, using the already-built ``TabulatedEngine``/table cached
        on ``self`` by ``run()``.

        ``calculate_angular_spectrum_4d`` already accepts arbitrary theta_x/
        theta_y arrays -- no new table build needed; this just launches a
        second, purpose-built kernel call over the cached table instead of
        reslicing the coarse generous grid ``run()`` precomputes for the
        collimation-window UI fields.
        """
        if self.engine is None or self.engine.table is None:
            raise RuntimeError("XigmaAdapter.spectrum_in_angular_range: run() must be called first")
        laser = self.interaction.laser
        xp = get_xp(self.device)

        gamma_0 = self.interaction.electrons.gaussian_fit.gamma0
        theta_x = _theta_grid(gamma_0, n_points=n_points, theta_range=theta_x_range)
        theta_y = _theta_grid(gamma_0, n_points=n_points, theta_range=theta_y_range)
        s_ang = (xp.linspace(0.0, 1.1, n_energy, dtype=xp.float32) * gamma_0 ** 2)
        d2Nds_dOmega, _dt, _debug = self.engine.angular_spectrum(
            s_ang, xp.asarray(theta_x), xp.asarray(theta_y), laser.phi_pol,
            samples_per_point=self.samples_per_point_2, device=self.device)
        s_scale_MeV = 4.0 * (HBAR_Q * laser.omega0).to("megaelectron_volt").magnitude
        E_eV = (to_host(s_ang, self.device) * s_scale_MeV) * 1e6
        d2NdEdOmega = d2Nds_dOmega / s_scale_MeV / 1e6
        # Same QUICK-FIX rescale run() applied, cached on self so an
        # on-demand angular-range query stays numerically consistent with
        # the main run's total_yield too.
        d2NdEdOmega = d2NdEdOmega * self.angular_rescale

        dtx = np.gradient(theta_x)
        dty = np.gradient(theta_y)
        dNdE_per_eV = np.einsum("ijk,i,j->k", d2NdEdOmega, dtx, dty)
        n_photons_in_range = float(np.einsum("ijk,i,j,k->", d2NdEdOmega, dtx, dty, np.gradient(E_eV)))

        return AngularRangeSpectrumResult(
            spectrum=PhasespaceSlice(axes={AXIS_ENERGY: E_eV}, distr=dNdE_per_eV),
            theta_x_range=theta_x_range, theta_y_range=theta_y_range,
            n_photons_in_range=n_photons_in_range,
            angle_energy_grid=PhasespaceSlice(
                # d2NdEdOmega is already host numpy here -- calculate_angular_
                # spectrum_4d's own GPU branch already calls .get() before
                # returning (see spectrum4d.py), so no further to_host() needed
                # (and would error: to_host('gpu', already-numpy) calls a
                # nonexistent .get()).
                axes={AXIS_THETA_X: theta_x, AXIS_THETA_Y: theta_y, AXIS_ENERGY: E_eV},
                distr=d2NdEdOmega))


# ---------------------------------------------------------------------------
# DirectAdapter ("delta"): brute-force per-macroparticle resonance binning
# ---------------------------------------------------------------------------
class DirectAdapter:
    def __init__(self):
        # Numeric knobs -- this adapter IS the config, no separate Config
        # dataclass.
        self.device_preference = "auto"
        self.n_steps_0 = 64
        self.n_theta_grid = 9
        self.chunk = 0  # 0 = auto-size from free VRAM, see particles.estimate_chunk_size

        # Recompute-cache state, populated by run() -- the raw per-particle
        # arrays needed for spectrum_in_angular_range()'s on-demand requery
        # (no persistent table to reuse the way XigmaAdapter has one).
        self.interaction: InteractionParameters | None = None
        self.device = "cpu"
        self.gamma = None
        self.theta_x = None
        self.theta_y = None
        self.a0 = None
        self.weight = None
        self.s_scale_MeV = None
        self.angular_rescale = 1.0

    def model_params(self) -> list[tuple[str, float | str, str]]:
        return [
            ("Device (auto/gpu/cpu)", self.device_preference, "device_preference"),
            ("Stage 0 trajectory steps", self.n_steps_0, "n_steps_0"),
            ("Angular-spectrum grid points/axis", self.n_theta_grid, "n_theta_grid"),
            ("Particle chunk size (0=auto from VRAM)", self.chunk, "chunk"),
        ]

    def model_choices(self) -> dict[str, list[str]]:
        return {"device_preference": ["auto", "gpu", "cpu"]}

    def run(self, job: Job) -> Results:
        """``job.interaction.electrons`` is required: electron sampling is
        the caller's job, not this adapter's -- there's exactly one place
        electrons get drawn from a beam description
        (``gammaforge.io.bunch.sample_gaussian_bunch``), not one per
        model."""
        from . import particles, spectrum_from_particles

        extra = job.extra
        self.device_preference = str(extra.get("device_preference", self.device_preference))
        self.n_steps_0 = int(float(extra.get("n_steps_0", self.n_steps_0)))
        self.n_theta_grid = int(float(extra.get("n_theta_grid", self.n_theta_grid)))
        self.chunk = int(float(extra.get("chunk", self.chunk)))

        self.interaction = job.interaction
        electrons = job.interaction.electrons
        device = _resolve_device(self.device_preference)
        self.device = device

        if job.interaction.laser.crossing_angle != 0.0:
            warnings.warn(
                "DirectAdapter (delta) is head-on-only; ignoring nonzero "
                f"crossing_angle ({job.interaction.laser.crossing_angle} rad)."
            )

        beam, laser = self.interaction.electrons.gaussian_fit, self.interaction.laser

        gamma_0 = beam.gamma0

        n_particles_new = electrons.n_particles

        time_req = find_slice_request(job.output.slices, AXIS_TIME)
        space_req = find_slice_request(job.output.slices, AXIS_X, AXIS_Y)
        energy_req = find_slice_request(job.output.slices, AXIS_ENERGY)
        energy_angle_req = find_slice_request(job.output.slices, AXIS_ENERGY, AXIS_THETA_X, AXIS_THETA_Y)

        push_backend = 'cupy' if device == 'gpu' else 'numpy'
        n_time_bins = time_req.bins[0] if time_req is not None else 128
        n_spatial_bins = space_req.bins if space_req is not None else (64, 64)
        chunk = self.chunk if self.chunk > 0 else particles.estimate_chunk_size(
            n_particles_new, self.n_steps_0, push_backend)
        gamma, tx, ty, a0_shape, w, diagnostics = particles.push_and_sample(
            electrons,
            k0_las=(2.0 * np.pi / laser.wavelength_m.quantity).to("1 / centimeter").magnitude,
            beta_ff=laser.beta_ff,
            sigma_lr0=laser.waist_rms_x_m.to_unit("centimeter").magnitude,
            sigma_lz=laser.duration_rms_s.to_unit("centimeter", context=LIGHT_TIME_CONTEXT).magnitude,
            omega_las=laser.omega0.to("1 / second").magnitude,
            N_l=laser.n_photons, ellipticity=laser.ellipticity, N_e=beam.N_e,
            sigma_ex=beam.sigma_x_m.to_unit("centimeter").magnitude,
            sigma_ey=beam.sigma_y_m.to_unit("centimeter").magnitude,
            n_steps=self.n_steps_0, backend=push_backend,
            n_time_bins=n_time_bins, n_spatial_bins=n_spatial_bins, chunk=chunk)
        a0_val = laser.a0_focus
        a0 = a0_shape * a0_val ** 2   # exact per-particle retarget (no table/binning needed)

        total_yield = float(to_host(w, device).sum() if hasattr(w, "get") else np.sum(w))
        photon_slices = [PhasespaceSlice(axes={}, distr=np.asarray(total_yield))]

        s_scale_MeV = 4.0 * (HBAR_Q * laser.omega0).to("megaelectron_volt").magnitude
        gamma_h = to_host(gamma, device) if hasattr(gamma, "get") else np.asarray(gamma)
        w_h = to_host(w, device) if hasattr(w, "get") else np.asarray(w)

        # Total angle-integrated spectrum: angle_integrated_spectrum, no known
        # normalization issue (unlike direct_binning_spectrum's own
        # angle-integrated total) -- needs only gamma/weight.
        E_gamma_eV_mean = None
        if energy_req is not None:
            n_e = energy_req.bins[0]
            if energy_req.range is not None and energy_req.range[0] is not None:
                lo_eV, hi_eV = energy_req.range[0]
                s_lo, s_hi = lo_eV / (s_scale_MeV * 1e6), hi_eV / (s_scale_MeV * 1e6)
            else:
                s_lo, s_hi = 0.0, 1.1 * gamma_0 ** 2
            s_grid = np.linspace(s_lo, s_hi, n_e)
            dNds_tot = to_host(spectrum_from_particles.angle_integrated_spectrum(
                gamma, w, s_grid, backend=push_backend), device)
            E_eV = s_grid * s_scale_MeV * 1e6
            dNdE_per_eV = dNds_tot / s_scale_MeV / 1e6
            photon_slices.append(PhasespaceSlice(axes={AXIS_ENERGY: E_eV}, distr=dNdE_per_eV))
            E_gamma_eV_mean = float(np.average(E_eV, weights=dNdE_per_eV)) if dNdE_per_eV.sum() else 0.0

        # Angular spectrum: the genuinely "brute-force particle binning" part
        # -- direct_binning_spectrum evaluated at each (theta_x, theta_y) grid
        # point, no table, no importance sampling. Always computed (even if
        # the joint slice itself isn't requested) so ``angular_rescale`` stays
        # calibrated for spectrum_in_angular_range's on-demand queries.
        tx_h = to_host(tx, device) if hasattr(tx, "get") else np.asarray(tx)
        ty_h = to_host(ty, device) if hasattr(ty, "get") else np.asarray(ty)
        a0_h = to_host(a0, device) if hasattr(a0, "get") else np.asarray(a0)

        if energy_angle_req is not None:
            n_e_ang, n_tx, n_ty = energy_angle_req.bins
            e_range, tx_range, ty_range = energy_angle_req.range or (None, None, None)
        else:
            n_e_ang, n_tx, n_ty = 65, self.n_theta_grid, self.n_theta_grid
            e_range = tx_range = ty_range = None

        theta_x_grid = _theta_grid(gamma_0, n_points=n_tx, theta_range=tx_range)
        theta_y_grid = _theta_grid(gamma_0, n_points=n_ty, theta_range=ty_range)
        if e_range is not None:
            lo_eV, hi_eV = e_range
            s_lo, s_hi = lo_eV / (s_scale_MeV * 1e6), hi_eV / (s_scale_MeV * 1e6)
        else:
            s_lo, s_hi = 0.0, 1.1 * gamma_0 ** 2
        s_edges = np.linspace(s_lo, s_hi, n_e_ang + 1)
        s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
        d2NdEdOmega = np.empty((n_tx, n_ty, s_centers.size))
        for i, x0 in enumerate(theta_x_grid):
            for j, y0 in enumerate(theta_y_grid):
                hist = spectrum_from_particles.direct_binning_spectrum(
                    gamma, tx, ty, w, a0, x0, y0, s_edges, laser.phi_pol, backend=push_backend)
                d2NdEdOmega[i, j, :] = to_host(hist, device) / s_scale_MeV / 1e6
        E_ang_eV = s_centers * s_scale_MeV * 1e6

        # QUICK FIX, FLAGGED FOR FUTURE INVESTIGATION: direct_binning_spectrum's
        # own angle-integrated total is KNOWN to disagree with the correctly-
        # normalised total_yield/angle_integrated_spectrum by a still-open,
        # deliberately-deferred ~2*pi residual (see spectrum_from_particles.py's
        # docstring). Rather than leave angular_spectrum over-normalised --
        # which can show a collimated flux exceeding total flux for a wide
        # enough window, a real bug users can trip over -- rescale d2NdEdOmega
        # so integrating it over this run's own full theta/energy grid
        # reproduces total_yield exactly, by construction. Does NOT fix the
        # underlying kernel normalisation (still unexplained, still worth
        # chasing) -- only guarantees the GUI's numbers are mutually
        # consistent in the meantime.
        dtx_full, dty_full = np.gradient(theta_x_grid), np.gradient(theta_y_grid)
        dE_full = np.gradient(E_ang_eV)
        full_integral = float(np.einsum("ijk,i,j,k->", d2NdEdOmega, dtx_full, dty_full, dE_full))
        angular_rescale = (total_yield / full_integral) if full_integral > 0 else 1.0
        d2NdEdOmega = d2NdEdOmega * angular_rescale

        if energy_angle_req is not None:
            photon_slices.append(PhasespaceSlice(
                axes={AXIS_THETA_X: theta_x_grid, AXIS_THETA_Y: theta_y_grid, AXIS_ENERGY: E_ang_eV},
                distr=d2NdEdOmega))

        # diagnostics arrays stay on-device (cupy) when push_backend='cupy' --
        # particles.py's _bin_temporal/_bin_spatial never convert to host, so
        # this adapter must, before these reach matplotlib (which cannot
        # implicitly convert a cupy array). to_host() is a no-op on CPU.
        if time_req is not None and diagnostics is not None and diagnostics.time_envelope is not None:
            edges = to_host(diagnostics.t_edges, device)
            t_seconds = 0.5 * (edges[:-1] + edges[1:])
            rate = to_host(diagnostics.time_envelope, device)
            photon_slices.append(PhasespaceSlice(axes={AXIS_TIME: t_seconds}, distr=rate))

        if space_req is not None and diagnostics is not None and diagnostics.spatial_envelope is not None:
            x_edges = to_host(diagnostics.spatial_x_edges, device)
            y_edges = to_host(diagnostics.spatial_y_edges, device)
            density = to_host(diagnostics.spatial_envelope, device)
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:]) / M_TO_CM
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:]) / M_TO_CM
            photon_slices.append(PhasespaceSlice(
                axes={AXIS_X: x_centers, AXIS_Y: y_centers}, distr=density * (M_TO_CM ** 2)))

        model_specific = dict(
            a0=a0_val,
            # FLAGGED: see the "QUICK FIX" comment above -- != 1.0 until the
            # underlying direct_binning_spectrum normalisation is root-caused.
            angular_spectrum_rescale_applied=angular_rescale,
        )
        if E_gamma_eV_mean is not None:
            model_specific["E_gamma_eV_mean"] = E_gamma_eV_mean

        res = Results(photon_slices=photon_slices, model_specific=model_specific)

        self.gamma, self.theta_x, self.theta_y, self.a0, self.weight = gamma_h, tx_h, ty_h, a0_h, w_h
        self.s_scale_MeV = s_scale_MeV
        self.angular_rescale = angular_rescale
        return res

    def spectrum_in_angular_range(
            self, theta_x_range: tuple[float, float], theta_y_range: tuple[float, float],
            n_points: int = 17, n_energy: int = 65) -> AngularRangeSpectrumResult:
        """Evaluate direct_binning_spectrum at a grid of OBSERVATION angles
        spanning the requested range, and integrate -- NOT a mask over which
        electrons happen to have that transverse momentum angle.

        An electron's own momentum angle (theta_x/theta_y) is not the photon's
        emission direction: direct_binning_spectrum's resonance condition
        (s_res = g**2/(1+a0+g**2*r_sq), r_sq = (theta_x-x0)**2+(theta_y-y0)**2)
        is a function of the OBSERVATION direction (x0, y0), not of theta_x/
        theta_y alone -- physically, a single electron radiates into a whole
        cone of observation angles, each Doppler-shifted to a different
        resonant photon energy, not into one direction matching its own
        momentum angle. Masking electrons by their own angle would silently
        mix up "electrons pointing this way" with "photons observed this way".
        Mirrors ``XigmaAdapter.spectrum_in_angular_range`` (xigma-i's own
        version), which already evaluates its kernel at observation points
        for exactly this reason.
        """
        from . import spectrum_from_particles

        if self.gamma is None:
            raise RuntimeError("DirectAdapter.spectrum_in_angular_range: run() must be called first")

        gamma_0 = self.interaction.electrons.gaussian_fit.gamma0
        theta_x_grid = np.linspace(theta_x_range[0], theta_x_range[1], n_points)
        theta_y_grid = np.linspace(theta_y_range[0], theta_y_range[1], n_points)
        s_edges = np.linspace(0.0, 1.1, n_energy + 1) * gamma_0 ** 2
        s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
        d2NdEdOmega = np.empty((n_points, n_points, s_centers.size))
        for i, x0 in enumerate(theta_x_grid):
            for j, y0 in enumerate(theta_y_grid):
                hist = spectrum_from_particles.direct_binning_spectrum(
                    self.gamma, self.theta_x, self.theta_y, self.weight, self.a0,
                    x0, y0, s_edges, self.interaction.laser.phi_pol)
                d2NdEdOmega[i, j, :] = hist / self.s_scale_MeV / 1e6
        # Same QUICK-FIX rescale run() applied to angular_spectrum, cached on
        # self -- keeps an on-demand query numerically consistent with the
        # main run's total_yield too.
        d2NdEdOmega = d2NdEdOmega * self.angular_rescale

        E_eV = s_centers * self.s_scale_MeV * 1e6
        dtx, dty = np.gradient(theta_x_grid), np.gradient(theta_y_grid)
        dNdE_per_eV = np.einsum("ijk,i,j->k", d2NdEdOmega, dtx, dty)
        n_photons_in_range = float(np.einsum("ijk,i,j,k->", d2NdEdOmega, dtx, dty, np.gradient(E_eV)))

        spectrum = PhasespaceSlice(axes={AXIS_ENERGY: E_eV}, distr=dNdE_per_eV)
        return AngularRangeSpectrumResult(
            spectrum=spectrum, theta_x_range=theta_x_range, theta_y_range=theta_y_range,
            n_photons_in_range=n_photons_in_range,
            angle_energy_grid=PhasespaceSlice(
                axes={AXIS_THETA_X: theta_x_grid, AXIS_THETA_Y: theta_y_grid, AXIS_ENERGY: E_eV},
                distr=d2NdEdOmega))
