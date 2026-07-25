"""GUI-facing adapter exposing the xigma_i pipeline through a dfe5-shaped
Config/run_simulation contract, for use as an alternate "model" inside the
MC-Kost desktop GUI (see MC-Kost/dfe5_gui2.py and MC-Kost/model_api.py).

Design notes:

  * This module never imports ``cupy`` (directly or via ``config``/
    ``tabulated_engine``/``xigma_i`` package import) at module scope, so
    that ``import xigma_i.gui_adapter`` degrades gracefully -- the
    consuming GUI wraps that import in a broad ``try/except Exception`` and
    shows the model disabled rather than crashing when cupy/CUDA isn't
    available. ``cupy``/``config.Compton``/``tabulated_engine`` are only
    imported inside ``available()`` and ``run_simulation()``.
  * ``Config`` mirrors ``dfe5_compton_mc.Config``'s field names and SI units
    wherever a physical mapping exists, so the GUI's model-agnostic
    spread-estimate formula (which reads ``cfg.eps0``, ``cfg.sigma_eps_rel``,
    ``cfg.omega_L``, ``cfg.emit_x/y``, ``cfg.beta_x/y``, ``cfg.sigma_par_L``)
    keeps working unmodified regardless of which model is active.
  * xigma-i is head-on only and has no classical/quantum toggle -- see
    ``capabilities()``.
  * xigma-i computes a smooth, pre-integrated spectral density (``dN/dE``,
    ``d^2N/dE dOmega``), not an unbinned per-photon/per-electron event list
    -- there is no final electron state and no photon-multiplicity
    statistic to report, unlike dfe5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# compton_io is load-bearing (not cupy-optional) for this whole package
# already -- config.py itself calls this same bootstrap at module scope
# (see that module's docstring) -- so importing compton_io.photons here
# doesn't weaken this module's "no cupy at module scope" promise, which is
# specifically about cupy/config/tabulated_engine, not compton_io.
from . import _bootstrap

_bootstrap.setup_paths()

from compton_io.bunch import MacroBunch  # noqa: E402
from compton_io.photons import (  # noqa: E402
    AngularRangeSpectrumResult,
    BinnedAngularSpectrum,
    BinnedSpatialDistribution,
    BinnedSpectrum,
    BinnedTemporalEnvelope,
)

# Matches xigma_i.config.elC exactly; duplicated here (rather than
# imported) purely to keep this module's own promise of no xigma_i imports
# at module scope (see module docstring) -- config.py itself is actually
# cupy-optional too, this is just consistent with the rest of the file.
_ELEMENTARY_CHARGE_C = 1.602176634e-19   # [C]
_C_LIGHT_M = 2.99792458e8                # [m/s]
_M_TO_CM = 1.0e2


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
    section has no classical/quantum switch. Use ``emulate_nonlinearity``
    for xigma-i's actual (and unrelated) nonlinearity axis.
    """

    eps0: float = 1000.0
    sigma_eps_rel: float = 0.0001
    emit_x: float = 1.0e-14          # geometric emittance, m*rad
    emit_y: float = 1.0e-14
    sigma0_x: float = 10.0e-6         # m
    sigma0_y: float = 10.0e-6
    sigma_par_e: float = 50.0e-6      # m (rms bunch length, = c*duration)
    N_e: float = 1.0e9

    lambda_L: float = 1.0e-6         # m
    sigma0_l: float = 20.0e-6         # m
    sigma_par_L: float = 1.0e-3       # m (rms pulse length, = c*duration)
    pulse_energy_J: float = 28e-3

    delta_x: float = 0.0              # m
    delta_y: float = 0.0
    delta_z: float = 0.0
    crossing_angle: float = 0.0       # rad; must be 0.0 (head-on only)

    quantum: bool = False              # accepted, no effect (see class docstring)

    # Output angular window, used only to size the precomputed angular-
    # spectrum grid in run_simulation (see _theta_grid); not a hard cut.
    Theta_x: float = 0.0
    Theta_y: float = 0.0
    theta_onaxis_gamma: float = 0.2    # kept only so the GUI's spread-estimate
                                        # formula box still has a value to read

    # xigma-i-only extras, no dfe5 analogue
    beta_ff: float = 0.0                # flying-focus factor: 0=static, 1=co-moving
    phi_pol: float = 0.0                # polarization angle [rad]
    emulate_nonlinearity: bool = True   # phenomenological a0 downshift/broadening

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

    # derived in __post_init__, mirroring dfe5_compton_mc.Config so the
    # model-agnostic spread-estimate formula in the GUI works unmodified
    omega_L: float = field(init=False)
    beta_x: float = field(init=False)
    beta_y: float = field(init=False)

    def __post_init__(self) -> None:
        self.omega_L = 2.0 * np.pi * _C_LIGHT_M / self.lambda_L
        self.beta_x = self.sigma0_x ** 2 / self.emit_x
        self.beta_y = self.sigma0_y ** 2 / self.emit_y

    @property
    def sigma_eps(self) -> float:
        return self.sigma_eps_rel * self.eps0


@dataclass
class XigmaResults:
    model_name: str
    cfg: Config
    n_mc: int
    total_yield: float
    spectrum: BinnedSpectrum
    summary: dict
    angular_spectrum: BinnedAngularSpectrum | None = None
    photon_samples: object | None = None
    electron_state: object | None = None
    photon_multiplicity: object | None = None
    temporal_envelope: BinnedTemporalEnvelope | None = None
    spatial_distribution: object | None = None
    final_distribution_path: str | None = None
    warnings: list | None = None
    # Private: the built Compton config instance (+ params it was built
    # with), cached so XigmaAdapter.run() can stash it for
    # spectrum_in_angular_range()'s on-demand recompute without
    # restructuring the module-function/adapter split above. Not part of
    # the model_api.CommonResults contract. _compton is TabulatedEngine's
    # config source (never anything else -- it runs no compute itself);
    # _engine/_device are cached the same way for
    # spectrum_in_angular_range's on-demand angular-spectrum recompute.
    _compton: object | None = None
    _gamma_0: float | None = None
    _sigma_gamma_0: float | None = None
    _engine: object | None = None
    _device: str | None = None
    # QUICK-FIX rescale factor (see run_simulation's "QUICK FIX, FLAGGED
    # FOR FUTURE INVESTIGATION" comment) -- reapplied identically in
    # spectrum_in_angular_range so an on-demand angular-range query stays
    # consistent with the main run's total_yield too.
    _angular_rescale: float = 1.0


# ---------------------------------------------------------------------------
# Capabilities / availability
# ---------------------------------------------------------------------------
_TRUST_NOTE = (
    "passport.md self-rates this engine trust level C: no unit tests, no "
    "cross-code/external/experimental validation, no fixed reproducible "
    "benchmark case, crossing-angle and astigmatic-laser geometries not "
    "modeled. Valid only for a0 <~ a0_max (default 0.5, a weakly-nonlinear "
    "model range, not a phenomenological correction -- a0 is a real "
    "trajectory-averaged table axis). A known, flagged-but-unresolved "
    "caveat: narrow-angle/sparse-table configs show large unstable "
    "variance in the angular-spectrum kernel vs. its own independent "
    "reference paths. All outputs -- total yield, angle-integrated "
    "spectrum, angular spectrum, temporal envelope, spatial distribution "
    "-- come from the tabulated-energy pipeline (particles.py/"
    "deposition.py/spectrum4d.py/reference.py/tabulated_engine.py, see "
    "CLAUDE.md). Runs on a CUDA GPU if one is available, else falls back "
    "to a CPU/numba implementation of the same kernels -- numerically "
    "validated against the GPU kernels but noticeably slower."
)


def capabilities() -> dict:
    return dict(
        name="xigma-i",
        display_name="XIGMA-I (experimental)",
        requires_gpu=False,
        supports_crossing_angle=False,
        supports_quantum_toggle=False,
        # emulate_nonlinearity has no effect: a0 is a real table axis in
        # this pipeline, not a phenomenological correction. Config still
        # accepts and parses the field (harmless, for interface stability);
        # it just doesn't change anything this adapter reports.
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
        trust_level="experimental-C",
        trust_note=_TRUST_NOTE,
    )


def available() -> tuple[bool, str]:
    """True if either supported backend can actually run: a real CUDA GPU
    (cupy + a visible device), or the CPU/numba fallback (see
    config._detect_device). Only returns False -- greying out the model in
    the GUI -- if neither works."""
    try:
        from .config import _detect_device
        _detect_device()
    except Exception as e:
        return False, str(e)
    return True, ""


def _backend_note() -> str:
    """Which backend run_simulation will actually use, for display/logging
    -- not part of the ModelAdapter contract, just a convenience for callers
    that want to show e.g. "running on CPU (numba)" in the UI."""
    try:
        from .config import _detect_device
        return _detect_device()
    except Exception:
        return "unavailable"


def extra_params() -> list[tuple[str, float, str]]:
    """Model-specific numeric fields with no dfe5 analogue, for the GUI to
    render in a model-specific pane (see compton_gui.model_api.ModelAdapter.
    extra_params). Each entry is (label, default, key) -- the same shape as
    the (label, default, key) specs app.py's add_field_grid already expects,
    so the GUI can lay these out with its existing helper. ``key`` must match
    the Config field name; params_to_config reads it straight out of the
    fields dict."""
    return [
        ("Flying-focus factor (0=static, 1=co-moving)", 0.0, "beta_ff"),
        ("Polarization angle [rad]", 0.0, "phi_pol"),
        ("Stage 0/1 particles", 60_000, "n_particles_01"),
        ("Stage 0 trajectory steps", 64, "n_steps_0"),
        ("Grid bins: gamma", 48, "n_bins_gamma"),
        ("Grid bins: theta_x", 48, "n_bins_theta_x"),
        ("Grid bins: theta_y", 48, "n_bins_theta_y"),
        ("Grid bins: a0", 12, "n_bins_a0"),
        ("a0_max (model valid range)", 0.5, "a0_max"),
        ("Stage 2 samples/point", 32, "samples_per_point_2"),
        ("Temporal envelope bins", 128, "n_time_bins"),
        ("Spatial bins: x", 64, "n_spatial_bins_x"),
        ("Spatial bins: y", 64, "n_spatial_bins_y"),
    ]


# ---------------------------------------------------------------------------
# GUI-field parsing
# ---------------------------------------------------------------------------
class ParamError(Exception):
    pass


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
    sigma_par_e = _C_LIGHT_M * (g("bunch_duration_ps") * 1e-12)
    emit_x = g("emit_x_mmmrad") * 1e-6 / eps0
    emit_y = g("emit_y_mmmrad") * 1e-6 / eps0
    beta_x = g("beta_x_m")
    beta_y = g("beta_y_m")
    sigma0_x = np.sqrt(emit_x * beta_x) if beta_x > 0 else 0.0
    sigma0_y = np.sqrt(emit_y * beta_y) if beta_y > 0 else 0.0

    lambda_L = g("laser_wavelength_nm") * 1e-9
    pulse_energy_J = g("laser_energy_mJ") * 1e-3
    sigma_par_L = _C_LIGHT_M * (g("pulse_duration_ps") * 1e-12)
    R_sf = g("rayleigh_length_m")
    sigma0_l = 0.5 * np.sqrt(max(R_sf, 0.0) * lambda_L / np.pi) if R_sf > 0 else 0.0
    rep_rate_hz = g("pulse_frequency_Hz")
    crossing_angle = g("crossing_angle")

    delta_x = g("x_mismatch_mm") * 1e-3
    delta_y = g("y_mismatch_mm") * 1e-3
    delta_z = g("z_mismatch_mm") * 1e-3 + _C_LIGHT_M * (g("time_mismatch_ps") * 1e-12)

    theta_x_col = g("theta_x_col_mrad") * 1e-3
    theta_y_col = g("theta_y_col_mrad") * 1e-3

    beta_ff = g("beta_ff")
    phi_pol = g("phi_pol")

    # Numerical/resolution knobs -- integers clamped to >= 1 so a stray 0 or
    # negative GUI entry can't crash Stage 0/1/2 (e.g. an empty table or a
    # zero-size histogram) rather than raising a clear ParamError; there's
    # no physical reason to allow smaller values.
    n_particles_01 = max(1, int(round(g("n_particles_01"))))
    n_steps_0 = max(1, int(round(g("n_steps_0"))))
    n_bins_gamma = max(1, int(round(g("n_bins_gamma"))))
    n_bins_theta_x = max(1, int(round(g("n_bins_theta_x"))))
    n_bins_theta_y = max(1, int(round(g("n_bins_theta_y"))))
    n_bins_a0 = max(1, int(round(g("n_bins_a0"))))
    a0_max = g("a0_max")
    samples_per_point_2 = max(1, int(round(g("samples_per_point_2"))))
    n_time_bins = max(1, int(round(g("n_time_bins"))))
    n_spatial_bins_x = max(1, int(round(g("n_spatial_bins_x"))))
    n_spatial_bins_y = max(1, int(round(g("n_spatial_bins_y"))))

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
    warnings.append(
        "xigma-i: 'Number of macroelectrons' is ignored -- use this model's "
        "own 'Stage 0/1 particles' field (Model Parameters panel) instead. "
        "That count sets the sampling density of the beam-laser overlap "
        "integral, not a per-electron photon-emission event count -- this "
        "model has no discrete electron/photon event generator (no final "
        "electron state, no photon-multiplicity stats).")

    cfg = Config(
        eps0=eps0, sigma_eps_rel=sigma_eps_rel,
        emit_x=max(emit_x, 1e-30), emit_y=max(emit_y, 1e-30),
        sigma0_x=max(sigma0_x, 1e-12), sigma0_y=max(sigma0_y, 1e-12),
        sigma_par_e=max(sigma_par_e, 1e-12), N_e=N_e,
        lambda_L=lambda_L, sigma0_l=max(sigma0_l, 1e-9),
        sigma_par_L=max(sigma_par_L, 1e-9),
        pulse_energy_J=pulse_energy_J,
        delta_x=delta_x, delta_y=delta_y, delta_z=delta_z,
        crossing_angle=crossing_angle,
        quantum=quantum,
        Theta_x=theta_x_col, Theta_y=theta_y_col,
        beta_ff=beta_ff, phi_pol=phi_pol,
        n_particles_01=n_particles_01, n_steps_0=n_steps_0,
        n_bins_gamma=n_bins_gamma, n_bins_theta_x=n_bins_theta_x,
        n_bins_theta_y=n_bins_theta_y, n_bins_a0=n_bins_a0, a0_max=a0_max,
        samples_per_point_2=samples_per_point_2, n_time_bins=n_time_bins,
        n_spatial_bins_x=n_spatial_bins_x, n_spatial_bins_y=n_spatial_bins_y,
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
# the defaults on those fields are the same first-cut values this module
# used to hardcode here, still deliberately not profiled against an actual
# interactive GUI session. Cost is close to linear in n_particles*n_steps
# for Stage 0/1 and independent of n_particles for Stage 2's quadrature.
# _N_PARTICLES_SANITY_MAX is the one clamp still applied unconditionally
# (in run_simulation), as a guard against a fat-fingered GUI value hanging
# the GPU/CPU for a very long time -- everything else is taken from cfg as-is.
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
                   electrons: MacroBunch | None = None) -> XigmaResults:
    """``n_mc`` is accepted for ``ModelAdapter.run`` signature compatibility
    but unused: xigma-i sizes Stage 0/1 from ``cfg.n_particles_01`` instead
    (see extra_params()/Config's numerical-control fields above), UNLESS
    ``electrons`` (a loaded ``.ele`` bunch) is given, in which case its own
    size determines the particle count instead (mirrors kascade's
    ``run_simulation``: a loaded bunch overrides internal sampling)."""
    if cfg.crossing_angle != 0.0:
        raise ValueError(
            f"xigma-i: crossing_angle must be 0 (head-on only), got {cfg.crossing_angle}")

    from .config import Compton, _detect_device
    from .tabulated_engine import TabulatedEngine

    device = _detect_device()
    compton = Compton(device=device)
    xp = compton.xp

    # Best-effort reproducibility: particles.sample_bunch takes an explicit
    # rng, not a global seed -- not guaranteed bit-exact across GPU/driver/
    # cupy versions or between the GPU and CPU backends, see
    # capabilities()'s supports_seed_reproducibility=False.
    rng = np.random.default_rng(seed)

    compton.set_electron_parameters(
        chargeNC=cfg.N_e * _ELEMENTARY_CHARGE_C * 1e9,
        emit_x=cfg.emit_x * _M_TO_CM, emit_y=cfg.emit_y * _M_TO_CM,
        sigma_ex=cfg.sigma0_x * _M_TO_CM, sigma_ey=cfg.sigma0_y * _M_TO_CM,
        sigma_ez=cfg.sigma_par_e * _M_TO_CM)
    compton.set_laser_parameters(
        WL=cfg.pulse_energy_J, lambda_l=cfg.lambda_L * _M_TO_CM,
        sigma_lr0=cfg.sigma0_l * _M_TO_CM, sigma_lz=cfg.sigma_par_L * _M_TO_CM,
        beta_ff=cfg.beta_ff)
    compton.set_foci_displacement(
        cfg.delta_x * _M_TO_CM, cfg.delta_y * _M_TO_CM, cfg.delta_z * _M_TO_CM)

    gamma_0 = cfg.eps0
    sigma_gamma_0 = cfg.sigma_eps

    # Total yield, angle-integrated spectrum, angular spectrum, temporal
    # envelope, and spatial distribution all come from TabulatedEngine --
    # `compton` is used only as its config-bag (set_electron_parameters/
    # set_laser_parameters/set_foci_displacement above); it runs no compute
    # of its own.
    # cfg.emulate_nonlinearity has no effect here: a0 is a real table axis
    # in this pipeline, not a phenomenological correction (see
    # spectrum4d.py's module docstring). Still read into summary below and
    # still validated/parsed by params_to_config, for interface stability.
    engine = TabulatedEngine(compton)
    push_backend = 'cupy' if device == 'gpu' else 'numpy'
    # ``n_mc`` (the shared "Number of macroelectrons" field/arg) is ignored
    # here -- xigma-i's own 'Stage 0/1 particles' field (cfg.n_particles_01)
    # is the sampling density for the beam-laser overlap integral, not a
    # per-electron event count (see the warning raised in params_to_config).
    # Only clamped against a runaway-cost sanity ceiling, not a floor --
    # params_to_config already enforces >= 1 on every numerical field.
    pushed_bunch = None
    if electrons is not None:
        from .particles import bunch_from_macrobunch
        pushed_bunch = bunch_from_macrobunch(electrons, compton)
        n_particles_new = pushed_bunch.n_particles
    else:
        n_particles_new = min(int(cfg.n_particles_01), _N_PARTICLES_SANITY_MAX)
    n_bins = (int(cfg.n_bins_gamma), int(cfg.n_bins_theta_x),
              int(cfg.n_bins_theta_y), int(cfg.n_bins_a0))
    n_spatial_bins = (int(cfg.n_spatial_bins_x), int(cfg.n_spatial_bins_y))
    engine.run(n_particles_new, gamma_0, sigma_gamma_0, n_steps=int(cfg.n_steps_0),
               n_bins=n_bins, backend=push_backend, rng=rng, a0_max=cfg.a0_max,
               n_time_bins=int(cfg.n_time_bins), n_spatial_bins=n_spatial_bins,
               bunch=pushed_bunch)
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
    t_seconds = compton.asnumpy(t_seconds_raw)
    rate = compton.asnumpy(rate_raw)
    temporal_envelope = BinnedTemporalEnvelope(t_seconds=t_seconds, rate=rate)

    x_centers_cm, y_centers_cm, density_per_cm2 = engine.spatial_distribution
    spatial_distribution = BinnedSpatialDistribution(
        x_centers=compton.asnumpy(x_centers_cm) / _M_TO_CM,
        y_centers=compton.asnumpy(y_centers_cm) / _M_TO_CM,
        density=compton.asnumpy(density_per_cm2) * (_M_TO_CM ** 2))

    # Angle-integrated spectrum, s in [0, 1.1*gamma0^2] (covers up to just
    # past the classical Compton edge), 512 points.
    s_tot = (xp.linspace(0.0, 1.1, 512, dtype=xp.float32) * gamma_0 ** 2)
    dNds_tot = compton.asnumpy(engine.spectrum(s_tot))
    s_scale_MeV = 4.0 * compton.Wph
    E_eV = (compton.asnumpy(s_tot) * s_scale_MeV) * 1e6
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
    E_ang_eV = (compton.asnumpy(s_ang) * s_scale_MeV) * 1e6
    d2NdEdOmega = d2Nds_dOmega / s_scale_MeV / 1e6  # -> eV^-1 sr^-1

    # QUICK FIX, FLAGGED FOR FUTURE INVESTIGATION: spectrum_kernel_4d's
    # angular_spectrum is known to disagree with the correctly-normalised
    # total_yield/angle-integrated spectrum by a still-open, deliberately-
    # deferred ~2*pi residual (see CLAUDE.md's "Current state" section and
    # reference.py's module docstring -- this is pre-existing, not
    # introduced by this adapter). Rather than leave angular_spectrum
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
        emulate_nonlinearity=float(bool(cfg.emulate_nonlinearity)),
        a0=float(compton.a0),
        # FLAGGED: see the "QUICK FIX" comment above angular_rescale's
        # computation -- 1.0 would mean the kernel's own normalisation
        # already agreed with total_yield; it currently doesn't (~2*pi-ish),
        # so this is visibly != 1.0 until the underlying kernel issue is
        # actually root-caused.
        angular_spectrum_rescale_applied=angular_rescale,
    )

    return XigmaResults(
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
        _compton=compton, _gamma_0=gamma_0, _sigma_gamma_0=sigma_gamma_0,
        _engine=engine, _device=device, _angular_rescale=angular_rescale,
    )


def spectrum_in_angular_range(
        res: XigmaResults, theta_x_range: tuple[float, float],
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
    compton = res._compton
    xp = compton.xp

    cfg = res.cfg
    theta_x = _theta_grid(cfg, n_points=n_points, theta_range=theta_x_range)
    theta_y = _theta_grid(cfg, n_points=n_points, theta_range=theta_y_range)
    s_ang = (xp.linspace(0.0, 1.1, n_energy, dtype=xp.float32)
             * res._gamma_0 ** 2)
    d2Nds_dOmega, _dt, _debug = engine.angular_spectrum(
        s_ang, xp.asarray(theta_x), xp.asarray(theta_y), cfg.phi_pol,
        samples_per_point=int(cfg.samples_per_point_2), device=res._device)
    s_scale_MeV = 4.0 * compton.Wph
    E_eV = (compton.asnumpy(s_ang) * s_scale_MeV) * 1e6
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
class _Capabilities:
    """Thin attribute-access wrapper so callers can use either
    ``capabilities()["x"]`` (dict, used internally) or ``caps.x`` (attribute,
    matching model_api.ModelCapabilities) without importing model_api here."""

    def __init__(self, d: dict):
        self.__dict__.update(d)


class XigmaAdapter:
    def __init__(self):
        self._last_results: XigmaResults | None = None

    def capabilities(self):
        return _Capabilities(capabilities())

    def available(self) -> tuple[bool, str]:
        return available()

    def extra_params(self) -> list[tuple[str, float, str]]:
        return extra_params()

    def params_to_config(self, fields: dict, quantum: bool = False):
        return params_to_config(fields, quantum)

    def run(self, cfg: Config, n_mc: int, seed: int, electrons: MacroBunch | None = None):
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
        from compton_io.io_formats.sdds import load_elegant_ele
        return load_elegant_ele(path)

    def ele_file_summary(self, bunch: MacroBunch) -> dict:
        from compton_io.bunch import fit_gaussian
        beam = fit_gaussian(bunch)
        return dict(
            mean_energy_MeV=beam.kinetic_energy_eV * 1e-6,
            rel_spread_pct=beam.rel_energy_spread_rms * 100.0,
            bunch_duration_ps=beam.sigma_t_s * 1e12,
            beta_x_m=beam.beta_star_x_m,
            beta_y_m=beam.beta_star_y_m,
            emit_x_mmmrad=beam.emit_norm_x_m * 1e6,
            emit_y_mmmrad=beam.emit_norm_y_m * 1e6,
            sigma_ex_um=beam.sigma_x_m * 1e6,
            sigma_ey_um=beam.sigma_y_m * 1e6,
            eps0=beam.gamma0,
            N_e=bunch.n_particles,
        )
