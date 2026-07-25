"""GUI-facing adapter exposing xigma_i.core.Compton through a dfe5-shaped
Config/run_simulation contract, for use as an alternate "model" inside the
MC-Kost desktop GUI (see MC-Kost/dfe5_gui2.py and MC-Kost/model_api.py).

Design notes (see the repo's integration plan for the full rationale):

  * This module never imports ``cupy`` (directly or via ``core``/``xigma_i``
    package import) at module scope, so that ``import xigma_i.gui_adapter``
    degrades gracefully -- the consuming GUI wraps that import in a broad
    ``try/except Exception`` and shows the model disabled rather than
    crashing when cupy/CUDA isn't available. ``cupy``/``core.Compton`` are
    only imported inside ``available()`` and ``run_simulation()``.
  * ``Config`` mirrors ``dfe5_compton_mc.Config``'s field names and SI units
    wherever a physical mapping exists, so the GUI's model-agnostic
    spread-estimate formula (which reads ``cfg.eps0``, ``cfg.sigma_eps_rel``,
    ``cfg.omega_L``, ``cfg.emit_x/y``, ``cfg.beta_x/y``, ``cfg.sigma_par_L``)
    keeps working unmodified regardless of which model is active.
  * ``core.Compton`` is head-on only and has no classical/quantum toggle (its
    only physics-affecting switch is the phenomenological
    ``emulate_nonlinearity`` a0-downshift emulation, a different axis from
    dfe5's Thomson/Klein-Nishina choice) -- see ``capabilities()``.
  * ``core.Compton`` computes a smooth, pre-integrated spectral density
    (``dN/dE``, ``d^2N/dE dOmega``), not an unbinned per-photon/per-electron
    event list -- there is no final electron state and no photon-multiplicity
    statistic to report, unlike dfe5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Matches xigma_i.core.elC exactly; duplicated here (rather than imported)
# so this module stays importable without cupy -- core.py does an
# unconditional `import cupy as cp` at module scope.
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


# ---------------------------------------------------------------------------
# Local result dataclasses (shape-compatible with MC-Kost/model_api.py, but
# not imported from there -- this package stays MC-Kost-agnostic)
# ---------------------------------------------------------------------------
@dataclass
class BinnedSpectrum:
    E_eV: np.ndarray
    dNdE_per_eV: np.ndarray


@dataclass
class BinnedAngularSpectrum:
    theta_x: np.ndarray
    theta_y: np.ndarray
    E_eV: np.ndarray
    d2NdEdOmega: np.ndarray


@dataclass
class BinnedTemporalEnvelope:
    """Photon-emission rate vs. time, read straight off ``Compton``'s own
    ``time_envelope``/``env_ts`` attributes (core.py's calculate_intersection
    already computes this internally; no new kernel work needed)."""

    t_seconds: np.ndarray
    rate: np.ndarray


@dataclass
class BinnedSpatialDistribution:
    """Transverse (x, y) areal density of photon emission, read off
    ``Compton``'s ``spatial_envelope``/``spatial_x_edges``/``spatial_y_edges``
    (a dedicated deposition kernel added alongside ``particle_kernel``'s
    existing time_envelope mechanism -- see core.py's ``particle_kernel``
    and ``calculate_intersection``)."""

    x_centers: np.ndarray
    y_centers: np.ndarray
    density: np.ndarray


@dataclass
class AngularRangeSpectrumResult:
    """Result of an on-demand spectrum computed over a user-picked angular
    sub-range (see ``spectrum_in_angular_range``)."""

    spectrum: BinnedSpectrum
    theta_x_range: tuple[float, float]
    theta_y_range: tuple[float, float]


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
    # Private: the built Compton instance (+ params it was built with), cached
    # so XigmaAdapter.run() can stash it for spectrum_in_angular_range()'s
    # on-demand recompute without restructuring the module-function/adapter
    # split above. Not part of the model_api.CommonResults contract.
    # _compton is still used for temporal_envelope/spatial_distribution
    # (Stage C of plan.md Phase 2 isn't implemented in the new path yet) and
    # as the config-bag TabulatedEngine wraps; _engine/_device are the new
    # tabulated-path counterpart, cached the same way for
    # spectrum_in_angular_range's on-demand angular-spectrum recompute.
    _compton: object | None = None
    _gamma_0: float | None = None
    _sigma_gamma_0: float | None = None
    _engine: object | None = None
    _device: str | None = None


# ---------------------------------------------------------------------------
# Capabilities / availability
# ---------------------------------------------------------------------------
_TRUST_NOTE = (
    "passport.md self-rates this engine trust level C (linear/classical "
    "regime) / D (nonlinear-emulation regime): no unit tests, no "
    "cross-code validation, no guaranteed run-to-run reproducibility, "
    "crossing-angle and astigmatic-laser geometries not modeled. All "
    "outputs -- total yield, angle-integrated spectrum, angular spectrum, "
    "temporal envelope, spatial distribution -- are now computed by the "
    "newer tabulated-energy path (see tabulated_engine.py / CLAUDE.md "
    "plan.md Phase 2 Stages B and C); core.Compton is used only as a "
    "config-bag (parameter setters), none of its GPU kernels are called "
    "anymore. Runs on a CUDA GPU if one is available, else falls back to a "
    "CPU/numba implementation of the same kernels -- numerically validated "
    "against the GPU kernels but noticeably slower. Temporal envelope / "
    "spatial distribution were cross-checked against the original "
    "core.Compton kernels' output on the same bin edges (shape correlation "
    "0.9986 / 0.93 respectively); the spatial correlation is imperfect, "
    "plausibly a real difference between the two paths' Stage-0 sampling "
    "schemes rather than a bug, not investigated further -- see "
    "tabulated_engine.py's module docstring."
)


def capabilities() -> dict:
    return dict(
        name="xigma-i",
        display_name="XIGMA-I (experimental)",
        requires_gpu=False,
        supports_crossing_angle=False,
        supports_quantum_toggle=False,
        # False since plan.md Phase 2 Stage B: emulate_nonlinearity only ever
        # affected calculate_spectrum/calculate_angular_spectrum, both now
        # replaced by the new tabulated path's total_yield/spectrum/
        # angular_spectrum (which have no equivalent switch -- a0 is a real
        # table axis there, not a phenomenological correction). Config still
        # accepts and parses the field (harmless, for interface stability);
        # it just no longer changes anything this adapter reports.
        supports_nonlinearity_emulation=False,
        supports_electron_final_state=False,
        supports_photon_multiplicity=False,
        supports_ele_file_io=False,
        supports_seed_reproducibility=False,
        requires_recompute_on_collimation_change=False,
        supports_temporal_envelope=True,
        supports_spatial_distribution=True,
        supports_angular_distribution=True,
        supports_angular_range_spectrum=True,
        trust_level="experimental-C/D",
        trust_note=_TRUST_NOTE,
    )


def available() -> tuple[bool, str]:
    """True if either backend core.Compton supports can actually run: a real
    CUDA GPU (cupy + a visible device), or the CPU/numba fallback (see
    core.py's device auto-detection, core_cpu.py's kernels). Only returns
    False -- greying out the model in the GUI -- if neither works."""
    try:
        from .core import _detect_device
        _detect_device()
    except Exception as e:
        return False, str(e)
    return True, ""


def _backend_note() -> str:
    """Which backend run_simulation will actually use, for display/logging
    -- not part of the ModelAdapter contract, just a convenience for callers
    that want to show e.g. "running on CPU (numba)" in the UI."""
    try:
        from .core import _detect_device
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
        "xigma-i: 'Number of macroelectrons' controls the sampling density of "
        "the beam-laser overlap integral, not a per-electron photon-emission "
        "event count -- this model has no discrete electron/photon event "
        "generator (no final electron state, no photon-multiplicity stats).")

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
    )
    extra = dict(n_mc=int(g("n_mc")), seed=int(g("seed")),
                 rep_rate_hz=rep_rate_hz, warnings=warnings)
    return cfg, extra


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
# TabulatedEngine.run()'s Stage 0/1 sizing for GUI use. Unlike the legacy
# path's particles_amount (an importance-sampled grid-cell weight, clamped
# below against calculate_intersection's O(particles_amount * theta_num^2)
# cost model), the new path's cost is close to linear in n_particles*n_steps
# for Stage 0/1 and independent of n_particles for Stage 2's quadrature --
# a different cost model, so the legacy clamp range doesn't transfer
# directly. These are a first-cut, deliberately not profiled against an
# actual interactive GUI session (plan.md Phase 2 Stage B flags this
# explicitly: "profile rather than guess") -- revisit with real timings
# before trusting them at the high end of the n_mc range.
_N_PARTICLES_NEW_MIN = 20_000
_N_PARTICLES_NEW_MAX = 150_000
_N_STEPS_NEW = 64
_N_BINS_NEW = (48, 48, 48, 12)
_SAMPLES_PER_POINT_NEW = 32
# Stage C (temporal_envelope/spatial_distribution): matches core.py's own
# N_STEPS=128/N_SPATIAL=64 resolution for a roughly comparable-fidelity
# display, not independently profiled.
_N_TIME_BINS_NEW = 128
_N_SPATIAL_BINS_NEW = (64, 64)


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
                   electrons: dict | None = None) -> XigmaResults:
    if electrons is not None:
        raise NotImplementedError(
            "xigma-i: loaded .ele electron bunches are not supported "
            "(capabilities().supports_ele_file_io is False)")
    if cfg.crossing_angle != 0.0:
        raise ValueError(
            f"xigma-i: crossing_angle must be 0 (head-on only), got {cfg.crossing_angle}")

    from .core import Compton, _detect_device
    from .tabulated_engine import TabulatedEngine

    device = _detect_device()
    compton = Compton(device=device)
    xp = compton.xp

    # Best-effort reseed of core.py's global xp.random calls (cupy on GPU,
    # numpy on the CPU/numba fallback). Not guaranteed bit-exact across
    # GPU/driver/cupy versions, or between the GPU and CPU backends -- see
    # capabilities()'s supports_seed_reproducibility=False.
    xp.random.seed(seed)
    # The new path's Stage 0 (particles.sample_bunch) takes an explicit rng
    # instead of reading a global seed -- same best-effort reproducibility
    # caveat as above, not a stronger guarantee.
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
    # envelope, and spatial distribution all now come from the new
    # tabulated-energy path (plan.md Phase 2 Stages B and C) -- core.Compton
    # is no longer used for anything beyond the config-bag TabulatedEngine
    # wraps (set_electron_parameters/set_laser_parameters/
    # set_foci_displacement above); none of its calculate_*/GPU-kernel
    # methods are called in this function anymore.
    # cfg.emulate_nonlinearity is now inert: it only ever affected the
    # legacy calculate_spectrum/calculate_angular_spectrum, and the new
    # path has no equivalent switch at all, since a0 is a real table axis
    # there, not a phenomenological correction (see spectrum4d.py's module
    # docstring). Still read into summary below and still validated/parsed
    # by params_to_config -- just doesn't change any output anymore.
    engine = TabulatedEngine(compton)
    push_backend = 'cupy' if device == 'gpu' else 'numpy'
    # ``n_mc`` here is a quadrature-sampling density for the beam-laser
    # overlap integral, not a per-electron event count (see the warning
    # raised in params_to_config) -- clamped to _N_PARTICLES_NEW_MIN/MAX
    # regardless of what the GUI field says (see that constant's comment
    # for the new path's different cost model / why the clamp range differs
    # from the legacy path's).
    n_particles_new = int(np.clip(int(n_mc), _N_PARTICLES_NEW_MIN, _N_PARTICLES_NEW_MAX))
    engine.run(n_particles_new, gamma_0, sigma_gamma_0, n_steps=_N_STEPS_NEW,
               n_bins=_N_BINS_NEW, backend=push_backend, rng=rng,
               n_time_bins=_N_TIME_BINS_NEW, n_spatial_bins=_N_SPATIAL_BINS_NEW)
    total_yield = engine.total_yield

    # Temporal envelope / spatial distribution (plan.md Phase 2 Stage C) --
    # engine.temporal_envelope/.spatial_distribution are already the same
    # physical convention (rate vs seconds; areal density vs cm) as
    # core.Compton's former env_ts/time_envelope/spatial_x_edges/
    # spatial_y_edges/spatial_envelope, see tabulated_engine.py's module
    # docstring for the cross-validation this was checked against. Convert
    # cm/photons-per-cm^2 to SI (m/photons-per-m^2) same as before.
    t_seconds, rate = engine.temporal_envelope
    temporal_envelope = BinnedTemporalEnvelope(t_seconds=t_seconds, rate=rate)

    x_centers_cm, y_centers_cm, density_per_cm2 = engine.spatial_distribution
    spatial_distribution = BinnedSpatialDistribution(
        x_centers=x_centers_cm / _M_TO_CM, y_centers=y_centers_cm / _M_TO_CM,
        density=density_per_cm2 * (_M_TO_CM ** 2))

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
        samples_per_point=_SAMPLES_PER_POINT_NEW, device=device)
    E_ang_eV = (compton.asnumpy(s_ang) * s_scale_MeV) * 1e6
    d2NdEdOmega = d2Nds_dOmega / s_scale_MeV / 1e6  # -> eV^-1 sr^-1

    summary = dict(
        total_yield=total_yield,
        crossing_angle_rad=cfg.crossing_angle,
        quantum=float(bool(cfg.quantum)),
        E_gamma_eV_mean=float(np.average(E_eV, weights=dNdE_per_eV)) if dNdE_per_eV.sum() else 0.0,
        emulate_nonlinearity=float(bool(cfg.emulate_nonlinearity)),
        a0=float(compton.a0),
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
        _engine=engine, _device=device,
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
        samples_per_point=_SAMPLES_PER_POINT_NEW, device=res._device)
    s_scale_MeV = 4.0 * compton.Wph
    E_eV = (compton.asnumpy(s_ang) * s_scale_MeV) * 1e6
    d2NdEdOmega = d2Nds_dOmega / s_scale_MeV / 1e6

    dtx = np.gradient(theta_x)
    dty = np.gradient(theta_y)
    dNdE_per_eV = np.einsum("ijk,i,j->k", d2NdEdOmega, dtx, dty)

    return AngularRangeSpectrumResult(
        spectrum=BinnedSpectrum(E_eV=E_eV, dNdE_per_eV=dNdE_per_eV),
        theta_x_range=theta_x_range, theta_y_range=theta_y_range)


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

    def run(self, cfg: Config, n_mc: int, seed: int, electrons: dict | None = None):
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

    def load_ele_file(self, path: str):
        raise NotImplementedError(
            "xigma-i: .ele bunch loading is not supported "
            "(capabilities().supports_ele_file_io is False)")

    def ele_file_summary(self, bunch: dict):
        raise NotImplementedError(
            "xigma-i: .ele bunch loading is not supported "
            "(capabilities().supports_ele_file_io is False)")
