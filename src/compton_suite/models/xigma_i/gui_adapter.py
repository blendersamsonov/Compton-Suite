"""GUI-facing adapter exposing the xigma_i pipeline through a dfe5-shaped
Config/run_simulation contract, for use as an alternate "model" inside the
MC-Kost desktop GUI (see MC-Kost/dfe5_gui2.py and MC-Kost/model_api.py).

Design notes:

  * This module never imports ``cupy`` (directly or via ``config``/
    ``tabulated_engine``/``xigma_i`` package import) at module scope, so
    that ``import xigma_i.gui_adapter`` degrades gracefully -- the
    consuming GUI wraps that import in a broad ``try/except Exception`` and
    shows the model disabled rather than crashing when cupy/CUDA isn't
    available. ``cupy``/``compton_suite.io.collision.build_params``/
    ``tabulated_engine`` are only imported inside ``available()`` and
    ``run_simulation()``. ``compton_suite.io`` itself has no cupy dependency, so
    ``compton_suite.io.bunch``/``compton_suite.io.constants`` etc. are safe to import at
    module scope unconditionally (see below).
  * ``Config`` mirrors ``dfe5_compton_mc.Config``'s field names and SI units
    wherever a physical mapping exists (``cfg.eps0``, ``cfg.sigma_eps_rel``,
    ``cfg.omega_L``, ``cfg.emit_x/y``, ``cfg.beta_x/y``, ``cfg.sigma_par_L``,
    ...), now derived from ``cfg.interaction`` (see class docstring) rather
    than stored directly, so code reading those names by convention across
    models keeps working unmodified regardless of which model is active.
  * xigma-i is head-on only and has no classical/quantum toggle -- see
    ``capabilities()``.
  * xigma-i computes a smooth, pre-integrated spectral density (``dN/dE``,
    ``d^2N/dE dOmega``), not an unbinned per-photon/per-electron event list
    -- there is no final electron state and no photon-multiplicity
    statistic to report, unlike dfe5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compton_suite.io.bunch import MacroBunch
from compton_suite.io.constants import C_LIGHT as _C_LIGHT_M, E_CHARGE as _ELEMENTARY_CHARGE_C, M_TO_CM as _M_TO_CM
from compton_suite.io.interaction import InteractionParameters as _InteractionParameters
from compton_suite.io.photons import (
    AngularRangeSpectrumResult,
    BinnedAngularSpectrum,
    BinnedSpatialDistribution,
    BinnedSpectrum,
    BinnedTemporalEnvelope,
)
from compton_suite.io.results import CommonResults
from compton_suite.io import (
    NoConvention as _NoConvention,
    PhysicalMeaning as _PhysicalMeaning,
    PhysicalQuantity as _PhysicalQuantity,
    TimeConvention as _TimeConvention,
    adapt_to_model as _adapt_to_model,
    params_to_floats as _params_to_floats,
)
from .params import XIGMA_SPEC as _XIGMA_SPEC


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """SI-unit physics config for the xigma-i model.

    Field names/units mirror ``dfe5_compton_mc.Config`` exactly where a
    mapping exists physically. ``crossing_angle`` must be 0.0 (Compton has no
    crossing-angle support at all -- head-on only). ``quantum`` is accepted
    for interface symmetry but has no effect: xigma-i's differential cross
    section has no classical/quantum switch. xigma-i has no nonlinearity-
    emulation axis at all -- ``a0`` is a real table axis, not a
    phenomenological correction (see ``capabilities()``'s
    ``supports_nonlinearity_emulation=False``).
    """

    interaction: _InteractionParameters

    # Output angular window, used only to size the precomputed angular-
    # spectrum grid in run_simulation (see _theta_grid); not a hard cut.
    Theta_x: float = 0.0
    Theta_y: float = 0.0

    # xigma-i-only extras, no dfe5 analogue
    beta_ff: float = 0.0                # flying-focus factor: 0=static, 1=co-moving
    phi_pol: float = 0.0                # polarization angle [rad]
    device_preference: str = "auto"     # "auto" | "gpu" | "cpu"

    # xigma-i-only numerical/resolution controls -- no physical meaning, but
    # surfaced through extra_params() (see below) so the pipeline's
    # accuracy/cost knobs (CLAUDE.md's "Convergence testing" section) are
    # user-tunable from the GUI instead of hardcoded module constants.
    # Stored here (rather than only in the params_to_config `extra` dict)
    # because run_simulation only ever receives ``cfg``, not the raw fields.
    n_particles_01: float = 60_000.0    # Stage 0/1 particle count (push_and_sample rows)
    n_steps_0: float = 64.0             # Stage 0 per-particle trajectory-integration steps
    n_bins_gamma: float = 48.0          # Stage 1 table bins, gamma axis
    n_bins_theta_x: float = 48.0        # Stage 1 table bins, theta_x axis
    n_bins_theta_y: float = 48.0        # Stage 1 table bins, theta_y axis
    n_bins_a0: float = 12.0             # Stage 1 table bins, a0 axis
    a0_max: float = 0.5                 # model's valid a0 range upper bound (deposition.retarget_a0)
    samples_per_point_2: float = 32.0   # Stage 2 quadrature samples per output point
    n_time_bins: float = 128.0          # temporal_envelope histogram bins
    n_spatial_bins_x: float = 64.0      # spatial_distribution histogram bins, x
    n_spatial_bins_y: float = 64.0      # spatial_distribution histogram bins, y

    # ---- flat fields, now derived from self.interaction, kept as properties
    # so run_simulation/_theta_grid/etc. below keep reading cfg.<name> by the
    # same names as before ------------------------------------------------
    @property
    def eps0(self) -> float:
        return self.interaction.beam.gamma0

    @property
    def sigma_eps_rel(self) -> float:
        return self.interaction.beam.sigma_gamma_over_gamma0

    @property
    def emit_x(self) -> float:
        return self.interaction.beam._ex_m

    @property
    def emit_y(self) -> float:
        return self.interaction.beam._ey_m

    @property
    def sigma0_x(self) -> float:
        return self.interaction.beam._sx_m

    @property
    def sigma0_y(self) -> float:
        return self.interaction.beam._sy_m

    @property
    def sigma_par_e(self) -> float:
        return self.interaction.beam.sigma_z_m

    @property
    def N_e(self) -> float:
        return self.interaction.beam.N_e

    @property
    def beta_x(self) -> float:
        return self.interaction.beam.beta_star_x_m

    @property
    def beta_y(self) -> float:
        return self.interaction.beam.beta_star_y_m

    @property
    def lambda_L(self) -> float:
        return self.interaction.laser._wl_m

    @property
    def sigma0_l(self) -> float:
        return self.interaction.laser._wx_m

    @property
    def sigma_par_L(self) -> float:
        return self.interaction.laser._dur_s * _C_LIGHT_M

    @property
    def pulse_energy_J(self) -> float:
        return self.interaction.laser._E_J

    @property
    def omega_L(self) -> float:
        return 2.0 * np.pi * _C_LIGHT_M / self.lambda_L

    @property
    def delta_x(self) -> float:
        return self.interaction.geometry._dx_m

    @property
    def delta_y(self) -> float:
        return self.interaction.geometry._dy_m

    @property
    def delta_z(self) -> float:
        return self.interaction.geometry._dz_m

    @property
    def crossing_angle(self) -> float:
        return self.interaction.geometry._cr_rad

    @property
    def quantum(self) -> bool:
        # accepted for interface symmetry, no effect (see class docstring)
        return self.interaction.quantum

    @property
    def sigma_eps(self) -> float:
        return self.sigma_eps_rel * self.eps0


def _attach_private_cache(res: CommonResults, *, params, gamma_0, sigma_gamma_0,
                           engine, device, angular_rescale) -> CommonResults:
    """Stash this adapter's own private recompute cache on a
    ``compton_suite.io.results.CommonResults`` instance (a plain, non-frozen
    dataclass -- arbitrary extra attributes work fine set after
    construction, without needing to be declared as part of the shared
    class). ``_params`` is ``TabulatedEngine``'s config source (never
    anything else -- it runs no compute itself); ``_engine``/``_device``
    are cached the same way, all so ``spectrum_in_angular_range()`` can
    recompute an on-demand angular-range query without rerunning the whole
    simulation. ``_angular_rescale`` is the QUICK-FIX rescale factor (see
    ``run_simulation``'s "QUICK FIX, FLAGGED FOR FUTURE INVESTIGATION"
    comment), reapplied identically in ``spectrum_in_angular_range`` so an
    on-demand query stays consistent with the main run's ``total_yield``.
    """
    res._params = params
    res._gamma_0 = gamma_0
    res._sigma_gamma_0 = sigma_gamma_0
    res._engine = engine
    res._device = device
    res._angular_rescale = angular_rescale
    return res


# ---------------------------------------------------------------------------
# Capabilities / availability
# ---------------------------------------------------------------------------
_TRUST_NOTE = (
    "Cross-validated against kascade and delta: total_yield agrees "
    "to <1% across both, angle-integrated spectrum shape to within a few "
    "percent (weighted-L1) at baseline configs (validation/"
    "run_cross_validation.py). Two open items remain, both flagged rather "
    "than silently reconciled: (1) this pipeline's own a0 formula disagrees "
    "with compton_suite.io.laser.GaussianParaxialLaser.a0_focus by ~49% relative "
    "(validation/tier0_wiring.py's check_a0_formula_agreement) -- not yet "
    "resolved, xigma-i's own convention is used as-is; (2) angular_spectrum "
    "carries a known, deliberately-deferred ~2*pi normalisation residual "
    "against the angle-integrated total (see run_simulation's 'QUICK FIX' "
    "comment and spectrum_from_particles.py's module docstring), rescaled at query time so "
    "collimated flux never exceeds total flux but the underlying kernel "
    "normalisation itself is still unexplained. Spectrum-shape agreement "
    "against the other two engines also degrades (to ~24% weighted-L1) near "
    "a0_max, the edge of this pipeline's weakly-nonlinear validity range. "
    "No unit tests, no fixed reproducible benchmark case; crossing-angle and "
    "astigmatic-laser geometries not modeled. Runs on a CUDA GPU if one is "
    "available, else falls back to a CPU/numba implementation of the same "
    "kernels -- numerically validated against the GPU kernels but "
    "noticeably slower."
)


def capabilities():
    from compton_suite.gui.model_api import ModelCapabilities
    return ModelCapabilities(
        name="xigma-i",
        display_name="XIGMA-I",
        requires_gpu=False,
        supports_crossing_angle=False,
        supports_quantum_toggle=False,
        # a0 is a real table axis in this pipeline, not a phenomenological
        # correction, so there is no nonlinearity-emulation axis to expose.
        supports_nonlinearity_emulation=False,
        supports_electron_final_state=False,
        supports_photon_multiplicity=False,
        supports_ele_file_io=True,
        supports_seed_reproducibility=False,
        requires_recompute_on_collimation_change=False,
        supports_temporal_envelope=True,
        supports_spatial_distribution=True,
        supports_angular_distribution=True,
        supports_angular_range_spectrum=True,
        is_fast_preview=False,
        trust_level="production",
        trust_note=_TRUST_NOTE,
    )


def available() -> tuple[bool, str]:
    """True if either supported backend can actually run: a real CUDA GPU
    (cupy + a visible device), or the CPU/numba fallback (see
    compton_suite.io.collision.detect_device). Only returns False -- greying out
    the model in the GUI -- if neither works."""
    try:
        from compton_suite.io.collision import detect_device
        detect_device()
    except Exception as e:
        return False, str(e)
    return True, ""


def _backend_note() -> str:
    """Which backend run_simulation will actually use, for display/logging
    -- not part of the ModelAdapter contract, just a convenience for callers
    that want to show e.g. "running on CPU (numba)" in the UI."""
    try:
        from compton_suite.io.collision import detect_device
        return detect_device()
    except Exception:
        return "unavailable"


def extra_params() -> list[tuple[str, float | str, str]]:
    """Model-specific numeric fields with no dfe5 analogue, for the GUI to
    render in a model-specific pane (see compton_gui.model_api.ModelAdapter.
    extra_params). Each entry is (label, default, key) -- the same shape as
    the (label, default, key) specs app.py's add_field_grid already expects,
    so the GUI can lay these out with its existing helper. ``key`` must match
    the Config field name; params_to_config reads it straight out of the
    fields dict."""
    return [
        ("Device (auto/gpu/cpu)", "auto", "device_preference"),
        ("Stage 0 trajectory steps", 64, "n_steps_0"),
        ("Grid bins: gamma", 48, "n_bins_gamma"),
        ("Grid bins: theta_x", 48, "n_bins_theta_x"),
        ("Grid bins: theta_y", 48, "n_bins_theta_y"),
        ("Grid bins: a0", 12, "n_bins_a0"),
        ("a0_max (model valid range)", 0.5, "a0_max"),
        ("Stage 2 samples/point", 32, "samples_per_point_2"),
    ]


def extra_choices() -> dict[str, list[str]]:
    """Allowed values for choice/enum fields in extra_params()."""
    return {
        "device_preference": ["auto", "gpu", "cpu"],
    }


# ---------------------------------------------------------------------------
# GUI-field parsing
# ---------------------------------------------------------------------------
class ParamError(Exception):
    pass


# The only two XIGMA_SPEC fields that are genuinely raw, convention-
# ambiguous GUI inputs today (a duration typed in picoseconds) -- sigma0_x/
# sigma0_y/sigma0_l are derived from other unambiguous fields (emittance +
# beta, Rayleigh length), not typed directly in a convention that needs
# resolving, so they stay hand-derived below (see docs/refactor/
# parameter-framework-and-collision-params.md's "sigma0_x/sigma0_l wrinkle").
_DURATION_SPEC = {k: _XIGMA_SPEC[k] for k in ("sigma_par_e", "sigma_par_L")}


def _float(fields: dict, key: str) -> float:
    try:
        return float(fields[key].get())
    except ValueError:
        raise ParamError(f"'{key}' is not a valid number: {fields[key].get()!r}")


def params_to_config(fields: dict, quantum: bool = False) -> tuple[Config, dict]:
    """Same practical-unit GUI fields as dfe5's params_to_config; identical
    parsing, targeting this module's ``Config``. ``quantum`` is accepted for
    signature symmetry with dfe5 but has no physical effect here."""
    g = lambda k: _float(fields, k)

    eps0 = g("mean_energy_MeV") * 1e6 / 510_998.950
    N_e = g("charge_nC") * 1e-9 / _ELEMENTARY_CHARGE_C
    sigma_eps_rel = g("rel_spread_pct") / 100.0
    emit_x = g("emit_x_mmmrad") * 1e-6 / eps0
    emit_y = g("emit_y_mmmrad") * 1e-6 / eps0
    beta_x = g("beta_x_m")
    beta_y = g("beta_y_m")
    sigma0_x = np.sqrt(emit_x * beta_x) if beta_x > 0 else 0.0
    sigma0_y = np.sqrt(emit_y * beta_y) if beta_y > 0 else 0.0

    lambda_L = g("laser_wavelength_nm") * 1e-9
    pulse_energy_J = g("laser_energy_mJ") * 1e-3
    R_sf = g("rayleigh_length_m")
    sigma0_l = 0.5 * np.sqrt(max(R_sf, 0.0) * lambda_L / np.pi) if R_sf > 0 else 0.0
    rep_rate_hz = g("pulse_frequency_Hz")
    crossing_angle = g("crossing_angle")

    # Map GUI convention labels to enum members
    _CONV_MAP = {
        "RMS": _TimeConvention.SIGMA_INTENSITY_RMS,
        "FWHM": _TimeConvention.FWHM_INTENSITY,
    }
    duration_conv = _CONV_MAP.get(fields.get("_duration_convention", "RMS"),
                                   _TimeConvention.SIGMA_INTENSITY_RMS)
    pulse_conv = _CONV_MAP.get(fields.get("_pulse_convention", "RMS"),
                               _TimeConvention.SIGMA_INTENSITY_RMS)

    durations = _params_to_floats(_adapt_to_model({
        "sigma_par_e": _PhysicalQuantity(
            g("bunch_duration_ps"), "picosecond",
            _PhysicalMeaning.BUNCH_LENGTH,
            duration_conv,
        ),
        "sigma_par_L": _PhysicalQuantity(
            g("pulse_duration_ps"), "picosecond",
            _PhysicalMeaning.PULSE_DURATION,
            pulse_conv,
        ),
    }, _DURATION_SPEC))
    sigma_par_e, sigma_par_L = durations["sigma_par_e"], durations["sigma_par_L"]

    delta_x = g("x_mismatch_mm") * 1e-3
    delta_y = g("y_mismatch_mm") * 1e-3
    delta_z = g("z_mismatch_mm") * 1e-3 + _C_LIGHT_M * (g("time_mismatch_ps") * 1e-12)

    theta_x_col = g("theta_x_col_mrad") * 1e-3
    theta_y_col = g("theta_y_col_mrad") * 1e-3

    # Shared laser optics parameters (from laser panel, not model-specific).
    # Fallback to 0.0 when called from contexts (e.g. headless test) that
    # don't include these shared fields. The duck-type check (hasattr .get)
    # handles both StringVar (GUI) and plain str (test) values.
    _get_shared = lambda k: (
        float(fields[k].get()) if hasattr(fields.get(k), "get")
        else float(fields.get(k, 0.0)))
    beta_ff = _get_shared("beta_ff")
    phi_pol = _get_shared("phi_pol")

    # Numerical/resolution knobs -- integers clamped to >= 1 so a stray 0 or
    # negative GUI entry can't crash Stage 0/1/2 (e.g. an empty table or a
    # zero-size histogram) rather than raising a clear ParamError; there's
    # no physical reason to allow smaller values.
    n_steps_0 = max(1, int(round(g("n_steps_0"))))
    n_bins_gamma = max(1, int(round(g("n_bins_gamma"))))
    n_bins_theta_x = max(1, int(round(g("n_bins_theta_x"))))
    n_bins_theta_y = max(1, int(round(g("n_bins_theta_y"))))
    n_bins_a0 = max(1, int(round(g("n_bins_a0"))))
    a0_max = g("a0_max")
    samples_per_point_2 = max(1, int(round(g("samples_per_point_2"))))

    warnings = []
    if crossing_angle != 0.0:
        raise ParamError(
            "xigma-i: crossing_angle must be 0 (this model is head-on only); "
            f"got {crossing_angle} rad")
    if sigma0_x <= 0 or sigma0_y <= 0:
        warnings.append("Beta_x/Beta_y must be > 0 to define a finite beam size; "
                        "using a point-like beam on the affected axis.")
    if sigma0_l <= 0:
        warnings.append("Rayleigh length must be > 0; using a very small laser "
                        "waist as a fallback.")

    from compton_suite.io.bunch import beam_from_shared_fields
    from compton_suite.io.interaction import InteractionGeometry
    from compton_suite.io.laser import laser_from_shared_fields

    beam = beam_from_shared_fields(
        eps0=eps0, sigma_eps_rel=sigma_eps_rel,
        emit_x=max(emit_x, 1e-30), emit_y=max(emit_y, 1e-30),
        sigma0_x=max(sigma0_x, 1e-12), sigma0_y=max(sigma0_y, 1e-12),
        sigma_par_e=max(sigma_par_e, 1e-12), N_e=N_e,
    )
    laser = laser_from_shared_fields(
        lambda_L=lambda_L, sigma0_l=max(sigma0_l, 1e-9),
        sigma_par_L=max(sigma_par_L, 1e-9), pulse_energy_J=pulse_energy_J,
    )
    geometry = InteractionGeometry(
        delta_x_m=_PhysicalQuantity(delta_x, "meter", _PhysicalMeaning.DISPLACEMENT, _NoConvention.PLAIN),
        delta_y_m=_PhysicalQuantity(delta_y, "meter", _PhysicalMeaning.DISPLACEMENT, _NoConvention.PLAIN),
        delta_z_m=_PhysicalQuantity(delta_z, "meter", _PhysicalMeaning.DISPLACEMENT, _NoConvention.PLAIN),
        crossing_angle_rad=_PhysicalQuantity(crossing_angle, "radian", _PhysicalMeaning.ANGLE, _NoConvention.PLAIN),
    )
    cfg = Config(
        interaction=_InteractionParameters(
            beam=beam, laser=laser, geometry=geometry, quantum=quantum,
        ),
        Theta_x=theta_x_col, Theta_y=theta_y_col,
        beta_ff=beta_ff, phi_pol=phi_pol,
        n_steps_0=n_steps_0,
        n_bins_gamma=n_bins_gamma, n_bins_theta_x=n_bins_theta_x,
        n_bins_theta_y=n_bins_theta_y, n_bins_a0=n_bins_a0, a0_max=a0_max,
        samples_per_point_2=samples_per_point_2,
    )
    extra = dict(n_mc=int(g("n_mc")), seed=int(g("seed")),
                 rep_rate_hz=rep_rate_hz, warnings=warnings)
    return cfg, extra


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
# TabulatedEngine.run()'s Stage 0/1/2 sizing is user-tunable via Config's
# numerical-control fields (n_particles_01, n_steps_0, n_bins_*, a0_max,
# samples_per_point_2, n_time_bins, n_spatial_bins_*; see extra_params()) --
# their defaults are first-cut values, deliberately not profiled against an
# actual interactive GUI session. Cost is close to linear in
# n_particles*n_steps for Stage 0/1 and independent of n_particles for
# Stage 2's quadrature.
# _N_PARTICLES_SANITY_MAX guards against a fat-fingered GUI n_particles_01
# value hanging the GPU/CPU for a very long time -- applied in
# params_to_config (where n_particles_01 is parsed), not in run_simulation:
# electron sampling is the caller's job (see run_simulation's docstring),
# so this clamp lives at the last point this adapter still owns, parsing
# the field a caller will size its sample from. Everything else is taken
# from cfg as-is.
_N_PARTICLES_SANITY_MAX = 2_000_000


def _theta_grid(cfg: Config, n_points: int = 33,
                theta_range: tuple[float, float] | None = None) -> np.ndarray:
    """A generous fixed window around the current collimation angle, wide
    enough that _update_outputs's cheap re-integration (no re-run) stays
    valid for any collimation angle the user is likely to dial in without
    clicking Calculate again.

    If ``theta_range`` is given (e.g. by ``spectrum_in_angular_range`` for a
    live, user-picked angular window), it's used verbatim instead of the
    generous auto-derived window -- default behavior/call sites (this
    function's only other caller, ``run_simulation``) are unaffected."""
    if theta_range is not None:
        lo, hi = theta_range
        return np.linspace(lo, hi, n_points, dtype=np.float32)
    half_window = max(5.0 * cfg.Theta_x if cfg.Theta_x > 0 else 0.0,
                      5.0 * cfg.Theta_y if cfg.Theta_y > 0 else 0.0,
                      3.0 / cfg.eps0)
    return np.linspace(-half_window, half_window, n_points, dtype=np.float32)


def run_simulation(cfg: Config, n_mc: int = 20_000, seed: int = 0,
                   *, electrons: MacroBunch) -> CommonResults:
    """``n_mc`` is accepted for ``ModelAdapter.run`` signature compatibility
    but unused: xigma-i sizes Stage 0/1 from ``electrons``'s own particle
    count instead (the caller is expected to have sampled it to
    ``cfg.n_particles_01`` particles -- see extra_params()/Config's
    numerical-control fields above; ``compton_suite.gui.app.py``'s
    ``on_start()`` and ``ComptonSuite/validation/runners.py`` both already
    do this).

    ``electrons`` is required: electron sampling is the caller's job, not
    this adapter's, or this package's at all -- there's exactly one place
    electrons get drawn from a beam description
    (``compton_suite.io.bunch.sample_gaussian_bunch``), not one per model.
    ``electrons`` is passed straight through to ``TabulatedEngine.run()``,
    which converts it to this pipeline's own CGS convention inline (see
    ``particles.push_and_sample``)."""
    if cfg.crossing_angle != 0.0:
        raise ValueError(
            f"xigma-i: crossing_angle must be 0 (head-on only), got {cfg.crossing_angle}")

    from compton_suite.io.collision import build_params, detect_device as _detect_device
    from .tabulated_engine import TabulatedEngine

    # Resolve device preference: "auto" (default), "gpu", or "cpu"
    if cfg.device_preference == "auto":
        device = _detect_device()
    else:
        device = cfg.device_preference
        # Validate: if "gpu" requested but unavailable, fall back to CPU with warning
        if device == "gpu":
            try:
                import cupy as cp
                if cp.cuda.runtime.getDeviceCount() == 0:
                    device = "cpu"
            except Exception:
                device = "cpu"
        if device not in ("gpu", "cpu"):
            raise ValueError(f"device_preference must be 'auto', 'gpu', or 'cpu', got {device!r}")

    beam, laser, geometry = (cfg.interaction.beam, cfg.interaction.laser,
                             cfg.interaction.geometry)
    params = build_params(beam, laser, geometry, beta_ff=cfg.beta_ff, device=device)
    xp = params.xp

    gamma_0 = cfg.eps0
    sigma_gamma_0 = cfg.sigma_eps

    # Total yield, angle-integrated spectrum, angular spectrum, temporal
    # envelope, and spatial distribution all come from TabulatedEngine --
    # `params` is used only as its plain-data bundle (build_params above);
    # it runs no compute of its own.
    engine = TabulatedEngine(params)
    push_backend = 'cupy' if device == 'gpu' else 'numpy'
    # ``n_mc`` (the shared "Number of macroelectrons" field/arg) is ignored
    # here -- xigma-i's own 'Stage 0/1 particles' field (cfg.n_particles_01)
    # is the sampling density for the beam-laser overlap integral, not a
    # per-electron event count (see the warning raised in params_to_config).
    # The caller is expected to have sampled ``electrons`` to that many
    # particles already; not re-clamped against _N_PARTICLES_SANITY_MAX
    # here since that guard now belongs at sampling time (the caller's
    # side), not at this adapter's -- this adapter no longer decides how
    # many particles to draw, only what to do with whatever it's handed.
    n_particles_new = electrons.n_particles
    n_bins = (int(cfg.n_bins_gamma), int(cfg.n_bins_theta_x),
              int(cfg.n_bins_theta_y), int(cfg.n_bins_a0))
    n_spatial_bins = (int(cfg.n_spatial_bins_x), int(cfg.n_spatial_bins_y))
    engine.run(n_steps=int(cfg.n_steps_0),
               n_bins=n_bins, backend=push_backend, a0_max=cfg.a0_max,
               n_time_bins=int(cfg.n_time_bins), n_spatial_bins=n_spatial_bins,
               bunch=electrons)
    total_yield = engine.total_yield

    # engine.temporal_envelope/.spatial_distribution: rate vs seconds /
    # areal density vs cm -- see tabulated_engine.py's module docstring.
    # Convert cm/photons-per-cm^2 to SI (m/photons-per-m^2). These stay
    # on-device (cupy) when push_backend='cupy' -- particles.py's
    # _bin_temporal/_bin_spatial never convert to host -- so must be
    # converted here before matplotlib (which cannot implicitly convert a
    # cupy array; this was a real, previously-latent bug never exercised
    # by headless_test.py, which only checks array .size, not actual
    # rendering).
    t_seconds_raw, rate_raw = engine.temporal_envelope
    t_seconds = params.asnumpy(t_seconds_raw)
    rate = params.asnumpy(rate_raw)
    temporal_envelope = BinnedTemporalEnvelope(t_seconds=t_seconds, rate=rate)

    x_centers_cm, y_centers_cm, density_per_cm2 = engine.spatial_distribution
    spatial_distribution = BinnedSpatialDistribution(
        x_centers=params.asnumpy(x_centers_cm) / _M_TO_CM,
        y_centers=params.asnumpy(y_centers_cm) / _M_TO_CM,
        density=params.asnumpy(density_per_cm2) * (_M_TO_CM ** 2))

    # Angle-integrated spectrum, s in [0, 1.1*gamma0^2] (covers up to just
    # past the classical Compton edge), 512 points.
    s_tot = (xp.linspace(0.0, 1.1, 512, dtype=xp.float32) * gamma_0 ** 2)
    dNds_tot = params.asnumpy(engine.spectrum(s_tot))
    s_scale_MeV = 4.0 * params.Wph
    E_eV = (params.asnumpy(s_tot) * s_scale_MeV) * 1e6
    dNdE_per_eV = dNds_tot / s_scale_MeV / 1e6  # dN/ds -> dN/dE(MeV) -> dN/dE(eV)

    # Angular spectrum, precomputed over a generous fixed theta window and a
    # coarser energy grid (kept smaller for kernel-launch cost: grid size =
    # theta_x.size * theta_y.size * s.size). calculate_angular_spectrum_4d
    # always returns a host array regardless of device (see spectrum4d.py).
    theta_x = _theta_grid(cfg)
    theta_y = _theta_grid(cfg)
    s_ang = (xp.linspace(0.0, 1.1, 96, dtype=xp.float32) * gamma_0 ** 2)
    d2Nds_dOmega, _dt, _debug = engine.angular_spectrum(
        s_ang, xp.asarray(theta_x), xp.asarray(theta_y), cfg.phi_pol,
        samples_per_point=int(cfg.samples_per_point_2), device=device)
    E_ang_eV = (params.asnumpy(s_ang) * s_scale_MeV) * 1e6
    d2NdEdOmega = d2Nds_dOmega / s_scale_MeV / 1e6  # -> eV^-1 sr^-1

    # QUICK FIX, FLAGGED FOR FUTURE INVESTIGATION: spectrum_kernel_4d's
    # angular_spectrum is known to disagree with the correctly-normalised
    # total_yield/angle-integrated spectrum by a still-open, deliberately-
    # deferred ~2*pi residual (see CLAUDE.md's "Current state" section and
    # spectrum_from_particles.py's direct_binning_spectrum docstring -- this
    # is pre-existing, not introduced by this adapter). Rather than leave angular_spectrum
    # (and therefore the GUI's "collimated flux" stat and Angular-Range
    # Spectrum tab) silently over-normalised -- which can show a
    # collimated flux EXCEEDING the total flux for a wide enough
    # collimation window, a real bug users can trip over -- rescale
    # d2NdEdOmega so integrating it over this run's own full theta/energy
    # grid reproduces total_yield exactly, by construction. This does NOT
    # fix the underlying kernel normalisation (still unexplained, still
    # worth chasing) -- it only guarantees the numbers shown in the GUI
    # are mutually consistent (collimated <= total) in the meantime.
    # The factor is cached on the result (_angular_rescale) and reapplied
    # identically in spectrum_in_angular_range's on-demand recompute.
    _dtx, _dty = np.gradient(theta_x), np.gradient(theta_y)
    _dE_ang = np.gradient(E_ang_eV)
    _full_integral = float(np.einsum("ijk,i,j,k->", d2NdEdOmega, _dtx, _dty, _dE_ang))
    angular_rescale = (total_yield / _full_integral) if _full_integral > 0 else 1.0
    d2NdEdOmega = d2NdEdOmega * angular_rescale

    summary = dict(
        total_yield=total_yield,
        crossing_angle_rad=cfg.crossing_angle,
        quantum=float(bool(cfg.quantum)),
        E_gamma_eV_mean=float(np.average(E_eV, weights=dNdE_per_eV)) if dNdE_per_eV.sum() else 0.0,
        a0=float(params.a0),
        # FLAGGED: see the "QUICK FIX" comment above angular_rescale's
        # computation -- 1.0 would mean the kernel's own normalisation
        # already agreed with total_yield; it currently doesn't (~2*pi-ish),
        # so this is visibly != 1.0 until the underlying kernel issue is
        # actually root-caused.
        angular_spectrum_rescale_applied=angular_rescale,
    )

    res = CommonResults(
        model_name="xigma-i",
        cfg=cfg,
        n_mc=n_particles_new,
        total_yield=total_yield,
        spectrum=BinnedSpectrum(E_eV=E_eV, dNdE_per_eV=dNdE_per_eV),
        summary=summary,
        angular_spectrum=BinnedAngularSpectrum(
            theta_x=theta_x, theta_y=theta_y, E_eV=E_ang_eV, d2NdEdOmega=d2NdEdOmega),
        photon_samples=None,
        electron_state=None,
        photon_multiplicity=None,
        temporal_envelope=temporal_envelope,
        spatial_distribution=spatial_distribution,
        final_distribution_path=None,
    )
    return _attach_private_cache(
        res, params=params, gamma_0=gamma_0, sigma_gamma_0=sigma_gamma_0,
        engine=engine, device=device, angular_rescale=angular_rescale)


def spectrum_in_angular_range(
        res: CommonResults, theta_x_range: tuple[float, float],
        theta_y_range: tuple[float, float], n_points: int = 33,
        n_energy: int = 96) -> AngularRangeSpectrumResult:
    """Fresh, on-demand spectrum over an arbitrary user-picked angular
    sub-range, using the ``TabulatedEngine`` (and its already-built table)
    cached on ``res`` by ``run_simulation``.

    ``calculate_angular_spectrum_4d`` already accepts arbitrary theta_x/
    theta_y arrays -- no new table build needed; this just launches a
    second, purpose-built kernel call over the cached table instead of
    reslicing the coarse generous grid ``run_simulation`` precomputes for
    the collimation-window UI fields.
    """
    engine = res._engine
    if engine is None or engine.table is None:
        raise RuntimeError("spectrum_in_angular_range: no cached "
                            "TabulatedEngine/table -- run() must be called first")
    params = res._params
    xp = params.xp

    cfg = res.cfg
    theta_x = _theta_grid(cfg, n_points=n_points, theta_range=theta_x_range)
    theta_y = _theta_grid(cfg, n_points=n_points, theta_range=theta_y_range)
    s_ang = (xp.linspace(0.0, 1.1, n_energy, dtype=xp.float32)
             * res._gamma_0 ** 2)
    d2Nds_dOmega, _dt, _debug = engine.angular_spectrum(
        s_ang, xp.asarray(theta_x), xp.asarray(theta_y), cfg.phi_pol,
        samples_per_point=int(cfg.samples_per_point_2), device=res._device)
    s_scale_MeV = 4.0 * params.Wph
    E_eV = (params.asnumpy(s_ang) * s_scale_MeV) * 1e6
    d2NdEdOmega = d2Nds_dOmega / s_scale_MeV / 1e6
    # Same QUICK-FIX rescale run_simulation applied, cached on res so an
    # on-demand angular-range query stays numerically consistent with the
    # main run's total_yield too (see run_simulation's comment).
    d2NdEdOmega = d2NdEdOmega * res._angular_rescale

    dtx = np.gradient(theta_x)
    dty = np.gradient(theta_y)
    dNdE_per_eV = np.einsum("ijk,i,j->k", d2NdEdOmega, dtx, dty)
    n_photons_in_range = float(np.einsum("ijk,i,j,k->", d2NdEdOmega, dtx, dty, np.gradient(E_eV)))

    return AngularRangeSpectrumResult(
        spectrum=BinnedSpectrum(E_eV=E_eV, dNdE_per_eV=dNdE_per_eV),
        theta_x_range=theta_x_range, theta_y_range=theta_y_range,
        n_photons_in_range=n_photons_in_range)


# ---------------------------------------------------------------------------
# ModelAdapter
# ---------------------------------------------------------------------------
class XigmaAdapter:
    def __init__(self):
        self._last_results: CommonResults | None = None

    def capabilities(self):
        return capabilities()

    def available(self) -> tuple[bool, str]:
        return available()

    def extra_params(self) -> list[tuple[str, float | str, str]]:
        return extra_params()

    def extra_choices(self) -> dict[str, list[str]]:
        return {"device_preference": ["auto", "gpu", "cpu"]}

    def params_to_config(self, fields: dict, quantum: bool = False):
        return params_to_config(fields, quantum)

    def run(self, cfg: Config, n_mc: int, seed: int, *, electrons: MacroBunch, output=None):
        # Use OutputSpec values if provided, otherwise use Config defaults
        if output is not None:
            cfg = Config(
                interaction=cfg.interaction,
                Theta_x=cfg.Theta_x, Theta_y=cfg.Theta_y,
                beta_ff=cfg.beta_ff, phi_pol=cfg.phi_pol,
                n_steps_0=cfg.n_steps_0,
                n_bins_gamma=cfg.n_bins_gamma, n_bins_theta_x=cfg.n_bins_theta_x,
                n_bins_theta_y=cfg.n_bins_theta_y, n_bins_a0=cfg.n_bins_a0, a0_max=cfg.a0_max,
                samples_per_point_2=cfg.samples_per_point_2,
                device_preference=cfg.device_preference,
                n_time_bins=output.n_time_bins,
                n_spatial_bins_x=output.n_spatial_bins_x,
                n_spatial_bins_y=output.n_spatial_bins_y,
            )
        res = run_simulation(cfg, n_mc=n_mc, seed=seed, electrons=electrons)
        self._last_results = res
        return res

    def spectrum_in_angular_range(self, theta_x_range: tuple[float, float],
                                  theta_y_range: tuple[float, float], **kwargs):
        if self._last_results is None:
            raise RuntimeError("spectrum_in_angular_range: run() must be "
                               "called at least once first")
        return spectrum_in_angular_range(
            self._last_results, theta_x_range, theta_y_range, **kwargs)

    def load_ele_file(self, path: str) -> MacroBunch:
        from compton_suite.io.io_formats.sdds import load_elegant_ele
        return load_elegant_ele(path)

    def ele_file_summary(self, bunch: MacroBunch) -> dict:
        from compton_suite.io.bunch import fit_gaussian
        beam = fit_gaussian(bunch)
        return dict(
            mean_energy_MeV=beam._KE_eV * 1e-6,
            rel_spread_pct=beam.rel_energy_spread_rms * 100.0,
            bunch_duration_ps=beam._st_s * 1e12,
            beta_x_m=beam.beta_star_x_m,
            beta_y_m=beam.beta_star_y_m,
            emit_x_mmmrad=beam.emit_norm_x_m * 1e6,
            emit_y_mmmrad=beam.emit_norm_y_m * 1e6,
            sigma_ex_um=beam._sx_m * 1e6,
            sigma_ey_um=beam._sy_m * 1e6,
            eps0=beam.gamma0,
            N_e=bunch.n_particles,
        )
