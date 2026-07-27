"""ModelAdapter for the brute-force particle-binning model.

Reuses ``xigma_i.particles.push_and_sample`` (Stage 0 -- the same ballistic
trajectory-through-laser-pulse push the tabulated ``xigma-i`` model uses)
and ``xigma_i.spectrum_from_particles.direct_binning_spectrum``/
``angle_integrated_spectrum`` directly, with NO Stage 1 deposition and NO
Stage 2 kernel -- "no table, no importance sampling -- assumption-free on
both the deposition and the lookup" (direct_binning_spectrum's own
docstring). Those two functions are load-bearing production code, not the
package's validation-only ``reference.py`` (brute-force table quadrature,
no production caller at all -- see that module's own docstring).

Depends on both ``compton_io`` (bunch/laser representations, collision
parameters via ``compton_io.collision.build_params``, and electron
sampling via ``compton_io.bunch.sample_gaussian_bunch``) and ``xigma_i``
(Stage 0 push physics, spectrum_from_particles) -- not embedded in either.
Duck-typed against ``compton_guide.model_api``, same decoupling discipline
every other adapter in this suite uses.

Never imports ``cupy`` unconditionally -- ``compton_io.collision``/
``xigma_i.particles``/``spectrum_from_particles`` are themselves
cupy-optional at import time (cupy only matters when actually running on
``device='gpu'``), so importing them here doesn't force a cupy dependency
either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compton_suite.io.bunch import MacroBunch, fit_gaussian
from compton_suite.io.constants import C_LIGHT as _C_LIGHT_M, E_CHARGE as _ELEMENTARY_CHARGE_C, M_TO_CM as _M_TO_CM
from compton_suite.io.interaction import InteractionParameters as _InteractionParameters
from compton_suite.io.io_formats.sdds import load_elegant_ele
from compton_suite.io.photons import (
    AngularRangeSpectrumResult,
    BinnedAngularSpectrum,
    BinnedSpatialDistribution,
    BinnedSpectrum,
    BinnedTemporalEnvelope,
)
from compton_suite.io.results import CommonResults
from compton_suite.io import (
    PhysicalMeaning as _PhysicalMeaning,
    PhysicalQuantity as _PhysicalQuantity,
    TimeConvention as _TimeConvention,
    adapt_to_model as _adapt_to_model,
    params_to_floats as _params_to_floats,
)
from compton_suite.models.xigma_i.params import XIGMA_SPEC as _XIGMA_SPEC

_TRUST_NOTE = (
    "Cross-validated against kascade and xigma-i: total_yield agrees to "
    "<1% against both (tightest of the three: ~0.17% vs xigma-i), "
    "angle-integrated spectrum shape near-identical to xigma-i's own "
    "(weighted-L1 ~0.2%) since both share Stage 0 physics (validation/"
    "run_cross_validation.py). Direct per-macroparticle resonance binning "
    "(xigma_i.spectrum_from_particles.direct_binning_spectrum), no table/"
    "kernel/importance-sampling -- but that function's own angle-integrated "
    "total is KNOWN to disagree with angle_integrated_spectrum by a "
    "consistent, unexplained ~2*pi (~6.3x) factor (see that module's "
    "docstring). total_yield/spectrum here come from "
    "angle_integrated_spectrum instead (no known issue), but "
    "angular_spectrum inherits that residual -- do not expect "
    "angular_spectrum integrated over solid angle to reproduce total_yield "
    "exactly. Same head-on-only, no-crossing-angle limitation as xigma-i "
    "(shares its Stage 0 physics), and inherits xigma-i's own unresolved "
    "~49% a0-formula discrepancy against compton_io.laser.GaussianParaxial"
    "Laser.a0_focus (validation/tier0_wiring.py) since both build "
    "CollisionParams via compton_io.collision.build_params."
)


@dataclass
class DirectConfig:
    """SI-unit physics config, trimmed to what Stage 0 (particles.py)
    needs -- no Stage 1/2 table/kernel sizing knobs (no n_bins_*, no
    a0_max: a0 is retargeted exactly per-particle, not via a table, so
    there's no a0-range/binning choice to make)."""

    interaction: _InteractionParameters

    Theta_x: float = 0.0            # angular window for angular_spectrum, mrad-derived
    Theta_y: float = 0.0

    beta_ff: float = 0.0
    phi_pol: float = 0.0
    device_preference: str = "auto"     # "auto" | "gpu" | "cpu"

    n_particles_01: float = 20_000.0
    n_steps_0: float = 64.0
    n_theta_grid: float = 9.0       # angular_spectrum grid resolution per axis

    # ---- flat fields, now derived from self.interaction, kept as
    # properties so run_simulation/_theta_grid/etc. below keep reading
    # cfg.<name> by the same names as before ----------------------------
    @property
    def eps0(self) -> float:
        return self.interaction.beam.gamma0

    @property
    def sigma_eps_rel(self) -> float:
        return self.interaction.beam.sigma_gamma_over_gamma0

    @property
    def emit_x(self) -> float:
        return self.interaction.beam.emit_geom_x_m

    @property
    def emit_y(self) -> float:
        return self.interaction.beam.emit_geom_y_m

    @property
    def sigma0_x(self) -> float:
        return self.interaction.beam.sigma_x_m

    @property
    def sigma0_y(self) -> float:
        return self.interaction.beam.sigma_y_m

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
        return self.interaction.laser.wavelength_m

    @property
    def sigma0_l(self) -> float:
        return self.interaction.laser.waist_rms_x_m

    @property
    def sigma_par_L(self) -> float:
        return self.interaction.laser.duration_rms_s * _C_LIGHT_M

    @property
    def pulse_energy_J(self) -> float:
        return self.interaction.laser.pulse_energy_J

    @property
    def omega_L(self) -> float:
        return 2.0 * np.pi * _C_LIGHT_M / self.lambda_L

    @property
    def delta_x(self) -> float:
        return self.interaction.geometry.delta_x_m

    @property
    def delta_y(self) -> float:
        return self.interaction.geometry.delta_y_m

    @property
    def delta_z(self) -> float:
        return self.interaction.geometry.delta_z_m

    @property
    def crossing_angle(self) -> float:
        # must be 0.0 -- head-on only, shares xigma_i's Stage 0
        return self.interaction.geometry.crossing_angle_rad

    @property
    def quantum(self) -> bool:
        # accepted, no effect (Stage 0 has no quantum/classical switch)
        return self.interaction.quantum

    @property
    def sigma_eps(self) -> float:
        return self.sigma_eps_rel * self.eps0


def capabilities() -> dict:
    return dict(
        name="xigma-i-direct",
        display_name="XIGMA-I Direct (brute-force binning)",
        requires_gpu=False,
        supports_crossing_angle=False,
        supports_quantum_toggle=False,
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
        trust_level="production",
        trust_note=_TRUST_NOTE,
    )


def available() -> tuple[bool, str]:
    try:
        from compton_suite.io.collision import detect_device
        detect_device()
    except Exception as e:
        return False, str(e)
    return True, ""


def extra_params() -> list[tuple[str, float | str, str]]:
    return [
        ("Flying-focus factor (0=static, 1=co-moving)", 0.0, "beta_ff"),
        ("Polarization angle [rad]", 0.0, "phi_pol"),
        ("Device (auto/gpu/cpu)", "auto", "device_preference"),
        ("Stage 0 particles", 20_000, "n_particles_01"),
        ("Stage 0 trajectory steps", 64, "n_steps_0"),
        ("Angular-spectrum grid points/axis", 9, "n_theta_grid"),
    ]


def extra_choices() -> dict[str, list[str]]:
    """Allowed values for choice/enum fields in extra_params()."""
    return {
        "device_preference": ["auto", "gpu", "cpu"],
    }


class ParamError(Exception):
    pass


# The only two XIGMA_SPEC fields that are genuinely raw, convention-
# ambiguous GUI inputs today (a duration typed in picoseconds) -- sigma0_x/
# sigma0_y/sigma0_l are derived from other unambiguous fields (emittance +
# beta, Rayleigh length), not typed directly in a convention that needs
# resolving, so they stay hand-derived below (see docs/refactor/
# parameter-framework-and-collision-params.md's "sigma0_x/sigma0_l wrinkle").
# Reuses xigma-i's own XIGMA_SPEC (not a separate copy) since this model
# shares xigma-i's Stage 0 conventions exactly (see module docstring).
_DURATION_SPEC = {k: _XIGMA_SPEC[k] for k in ("sigma_par_e", "sigma_par_L")}


def _float(fields: dict, key: str) -> float:
    try:
        return float(fields[key].get())
    except ValueError:
        raise ParamError(f"'{key}' is not a valid number: {fields[key].get()!r}")


def params_to_config(fields: dict, quantum: bool = False) -> tuple[DirectConfig, dict]:
    """Same shared GUI fields as xigma-i's params_to_config -- identical
    parsing, targeting this module's trimmed Config."""
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

    durations = _params_to_floats(_adapt_to_model({
        "sigma_par_e": _PhysicalQuantity(
            g("bunch_duration_ps"), "picosecond",
            _PhysicalMeaning.BUNCH_LENGTH, _TimeConvention.SIGMA_INTENSITY_RMS,
        ),
        "sigma_par_L": _PhysicalQuantity(
            g("pulse_duration_ps"), "picosecond",
            _PhysicalMeaning.PULSE_DURATION, _TimeConvention.SIGMA_INTENSITY_RMS,
        ),
    }, _DURATION_SPEC))
    sigma_par_e, sigma_par_L = durations["sigma_par_e"], durations["sigma_par_L"]

    delta_x = g("x_mismatch_mm") * 1e-3
    delta_y = g("y_mismatch_mm") * 1e-3
    delta_z = g("z_mismatch_mm") * 1e-3 + 2.99792458e8 * (g("time_mismatch_ps") * 1e-12)

    theta_x_col = g("theta_x_col_mrad") * 1e-3
    theta_y_col = g("theta_y_col_mrad") * 1e-3

    warnings = []
    if crossing_angle != 0.0:
        raise ParamError(
            "xigma-i-direct: crossing_angle must be 0 (this model is head-on "
            f"only, shares xigma-i's Stage 0); got {crossing_angle} rad")
    if sigma0_x <= 0 or sigma0_y <= 0:
        warnings.append("Beta_x/Beta_y must be > 0 to define a finite beam size; "
                        "using a point-like beam on the affected axis.")
    if sigma0_l <= 0:
        warnings.append("Rayleigh length must be > 0; using a very small laser "
                        "waist as a fallback.")
    warnings.append(
        "xigma-i-direct: 'Number of macroelectrons' is ignored -- use this "
        "model's own 'Stage 0 particles' field (Model Parameters panel) instead.")

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
        delta_x_m=delta_x, delta_y_m=delta_y, delta_z_m=delta_z,
        crossing_angle_rad=crossing_angle,
    )
    cfg = DirectConfig(
        interaction=_InteractionParameters(
            beam=beam, laser=laser, geometry=geometry, quantum=quantum,
        ),
        Theta_x=theta_x_col, Theta_y=theta_y_col,
        beta_ff=g("beta_ff"), phi_pol=g("phi_pol"),
        n_particles_01=max(1, int(round(g("n_particles_01")))),
        n_steps_0=max(1, int(round(g("n_steps_0")))),
        n_theta_grid=max(3, int(round(g("n_theta_grid")))),
    )
    extra = dict(n_mc=int(g("n_mc")), seed=int(g("seed")),
                 rep_rate_hz=rep_rate_hz, warnings=warnings)
    return cfg, extra


def _theta_grid(cfg: DirectConfig, n_points: int, theta_range: tuple[float, float] | None = None) -> np.ndarray:
    if theta_range is not None:
        lo, hi = theta_range
        return np.linspace(lo, hi, n_points)
    half_window = max(5.0 * cfg.Theta_x if cfg.Theta_x > 0 else 0.0,
                      5.0 * cfg.Theta_y if cfg.Theta_y > 0 else 0.0,
                      3.0 / cfg.eps0)
    return np.linspace(-half_window, half_window, n_points)


def _attach_private_cache(res: CommonResults, *, gamma, theta_x, theta_y, a0, weight,
                           s_edges, s_scale_MeV, phi_pol, angular_rescale) -> CommonResults:
    """Stash this adapter's own private recompute cache on a
    ``compton_io.results.CommonResults`` instance (a plain, non-frozen
    dataclass -- arbitrary extra attributes work fine set after
    construction, without needing to be declared as part of the shared
    class), for ``spectrum_in_angular_range()``'s on-demand recompute.
    ``_angular_rescale`` is the QUICK-FIX rescale factor (see
    ``run_simulation``'s "QUICK FIX, FLAGGED FOR FUTURE INVESTIGATION"
    comment), reapplied identically in ``spectrum_in_angular_range``.
    """
    res._gamma = gamma
    res._theta_x = theta_x
    res._theta_y = theta_y
    res._a0 = a0
    res._weight = weight
    res._s_edges = s_edges
    res._s_scale_MeV = s_scale_MeV
    res._phi_pol = phi_pol
    res._angular_rescale = angular_rescale
    return res


def run_simulation(cfg: DirectConfig, n_mc: int = 20_000, seed: int = 0,
                   *, electrons: MacroBunch) -> CommonResults:
    """``electrons`` is required: electron sampling is the caller's job,
    not this adapter's -- there's exactly one place electrons get drawn
    from a beam description (``compton_io.bunch.sample_gaussian_bunch``),
    not one per model. The caller is expected to size its sample to
    ``cfg.n_particles_01`` particles, mirroring xigma-i's own convention
    (``compton_guide.app.py``'s ``on_start()`` and
    ``ComptonSuite/validation/runners.py`` both already do this)."""
    if cfg.crossing_angle != 0.0:
        raise ValueError(
            f"xigma-i-direct: crossing_angle must be 0 (head-on only), got {cfg.crossing_angle}")

    from compton_suite.io.collision import build_params, detect_device
    from compton_suite.models.xigma_i import particles, spectrum_from_particles

    # Resolve device preference: "auto" (default), "gpu", or "cpu"
    if cfg.device_preference == "auto":
        device = detect_device()
    else:
        device = cfg.device_preference
        if device == "gpu":
            try:
                import cupy as cp
                if cp.cuda.runtime.getDeviceCount() == 0:
                    device = "cpu"
            except Exception:
                device = "cpu"
        if device not in ("gpu", "cpu"):
            raise ValueError(f"device_preference must be 'auto', 'gpu', or 'cpu', got {device!r}")

    # ``seed`` is unused here: electron sampling is the caller's job (see
    # this function's docstring), so there's no RNG left for this function
    # to own. Kept in the signature for ModelAdapter interface compatibility.
    beam, laser, geometry = (cfg.interaction.beam, cfg.interaction.laser,
                             cfg.interaction.geometry)
    params = build_params(beam, laser, geometry, beta_ff=cfg.beta_ff, device=device)

    gamma_0, sigma_gamma_0 = cfg.eps0, cfg.sigma_eps

    n_particles_new = electrons.n_particles

    push_backend = 'cupy' if device == 'gpu' else 'numpy'
    n_time_bins, n_spatial_bins = 128, (64, 64)
    gamma, tx, ty, a0_shape, w, diagnostics = particles.push_and_sample(
        params, electrons, n_steps=int(cfg.n_steps_0), backend=push_backend,
        n_time_bins=n_time_bins, n_spatial_bins=n_spatial_bins)
    a0 = a0_shape * params.a0 ** 2   # exact per-particle retarget (no table/binning needed)

    total_yield = float(params.asnumpy(w).sum() if hasattr(w, "get") else np.sum(w))

    # Total angle-integrated spectrum:
    # spectrum_from_particles.angle_integrated_spectrum, no known
    # normalization issue (unlike direct_binning_spectrum's own
    # angle-integrated total, see _TRUST_NOTE) -- needs only gamma/weight.
    s_scale_MeV = 4.0 * params.Wph
    s_grid = np.linspace(0.0, 1.1, 512) * gamma_0 ** 2
    gamma_h = params.asnumpy(gamma) if hasattr(gamma, "get") else np.asarray(gamma)
    w_h = params.asnumpy(w) if hasattr(w, "get") else np.asarray(w)
    dNds_tot = spectrum_from_particles.angle_integrated_spectrum(gamma_h, w_h, s_grid)
    E_eV = s_grid * s_scale_MeV * 1e6
    dNdE_per_eV = dNds_tot / s_scale_MeV / 1e6

    # Angular spectrum: the genuinely "brute-force particle binning" part
    # -- direct_binning_spectrum evaluated at each (theta_x, theta_y) grid
    # point, no table, no importance sampling.
    tx_h = params.asnumpy(tx) if hasattr(tx, "get") else np.asarray(tx)
    ty_h = params.asnumpy(ty) if hasattr(ty, "get") else np.asarray(ty)
    a0_h = params.asnumpy(a0) if hasattr(a0, "get") else np.asarray(a0)
    n_grid = int(cfg.n_theta_grid)
    theta_x_grid = _theta_grid(cfg, n_grid)
    theta_y_grid = _theta_grid(cfg, n_grid)
    s_edges = np.linspace(0.0, 1.1, 65) * gamma_0 ** 2
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    d2NdEdOmega = np.empty((n_grid, n_grid, s_centers.size))
    for i, x0 in enumerate(theta_x_grid):
        for j, y0 in enumerate(theta_y_grid):
            hist = spectrum_from_particles.direct_binning_spectrum(
                gamma_h, tx_h, ty_h, w_h, a0_h, x0, y0, s_edges, cfg.phi_pol)
            d2NdEdOmega[i, j, :] = hist / s_scale_MeV / 1e6
    E_ang_eV = s_centers * s_scale_MeV * 1e6

    # QUICK FIX, FLAGGED FOR FUTURE INVESTIGATION: direct_binning_spectrum's
    # own angle-integrated total is KNOWN to disagree with the correctly-
    # normalised total_yield/angle_integrated_spectrum by a still-open,
    # deliberately-deferred ~2*pi residual (see this module's _TRUST_NOTE
    # and xigma_i/spectrum_from_particles.py's docstring -- pre-existing,
    # not introduced here). Rather than leave angular_spectrum over-normalised -- which can
    # show a collimated flux exceeding total flux for a wide enough window,
    # a real bug users can trip over -- rescale d2NdEdOmega so integrating
    # it over this run's own full theta/energy grid reproduces total_yield
    # exactly, by construction. Does NOT fix the underlying kernel
    # normalisation (still unexplained, still worth chasing) -- only
    # guarantees the GUI's numbers are mutually consistent in the meantime.
    dtx_full, dty_full = np.gradient(theta_x_grid), np.gradient(theta_y_grid)
    dE_full = np.gradient(E_ang_eV)
    full_integral = float(np.einsum("ijk,i,j,k->", d2NdEdOmega, dtx_full, dty_full, dE_full))
    angular_rescale = (total_yield / full_integral) if full_integral > 0 else 1.0
    d2NdEdOmega = d2NdEdOmega * angular_rescale

    # diagnostics arrays stay on-device (cupy) when push_backend='cupy' --
    # particles.py's _bin_temporal/_bin_spatial never convert to host, so
    # this adapter must, before these reach matplotlib (which cannot
    # implicitly convert a cupy array). params.asnumpy() is a no-op on CPU.
    t_seconds, rate = None, None
    temporal_envelope = None
    if diagnostics is not None and diagnostics.time_envelope is not None:
        edges = params.asnumpy(diagnostics.t_edges)
        t_seconds = 0.5 * (edges[:-1] + edges[1:])
        rate = params.asnumpy(diagnostics.time_envelope)
        temporal_envelope = BinnedTemporalEnvelope(t_seconds=t_seconds, rate=rate)

    spatial_distribution = None
    if diagnostics is not None and diagnostics.spatial_envelope is not None:
        x_edges = params.asnumpy(diagnostics.spatial_x_edges)
        y_edges = params.asnumpy(diagnostics.spatial_y_edges)
        density = params.asnumpy(diagnostics.spatial_envelope)
        spatial_distribution = BinnedSpatialDistribution(
            x_centers=0.5 * (x_edges[:-1] + x_edges[1:]) / _M_TO_CM,
            y_centers=0.5 * (y_edges[:-1] + y_edges[1:]) / _M_TO_CM,
            density=density * (_M_TO_CM ** 2))

    summary = dict(
        total_yield=total_yield,
        crossing_angle_rad=cfg.crossing_angle,
        quantum=float(bool(cfg.quantum)),
        a0=float(params.a0),
        # FLAGGED: see the "QUICK FIX" comment above -- != 1.0 until the
        # underlying direct_binning_spectrum normalisation is root-caused.
        angular_spectrum_rescale_applied=angular_rescale,
    )

    res = CommonResults(
        model_name="xigma-i-direct",
        cfg=cfg,
        n_mc=n_particles_new,
        total_yield=total_yield,
        spectrum=BinnedSpectrum(E_eV=E_eV, dNdE_per_eV=dNdE_per_eV),
        summary=summary,
        angular_spectrum=BinnedAngularSpectrum(
            theta_x=theta_x_grid, theta_y=theta_y_grid, E_eV=E_ang_eV, d2NdEdOmega=d2NdEdOmega),
        temporal_envelope=temporal_envelope,
        spatial_distribution=spatial_distribution,
    )
    return _attach_private_cache(
        res, gamma=gamma_h, theta_x=tx_h, theta_y=ty_h, a0=a0_h, weight=w_h,
        s_edges=s_edges, s_scale_MeV=s_scale_MeV, phi_pol=cfg.phi_pol,
        angular_rescale=angular_rescale)


def spectrum_in_angular_range(res: CommonResults, theta_x_range: tuple[float, float],
                              theta_y_range: tuple[float, float],
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
    mix up "electrons pointing this way" with "photons observed this way"
    -- a real, previously-made mistake in this function, not a
    hypothetical one. This mirrors xigma_i.gui_adapter's own
    spectrum_in_angular_range, which already evaluates its kernel at
    observation points for exactly this reason.
    """
    from compton_suite.models.xigma_i import spectrum_from_particles

    if res._gamma is None:
        raise RuntimeError("spectrum_in_angular_range: run() must be called first")

    gamma_0 = res.cfg.eps0
    theta_x_grid = np.linspace(theta_x_range[0], theta_x_range[1], n_points)
    theta_y_grid = np.linspace(theta_y_range[0], theta_y_range[1], n_points)
    s_edges = np.linspace(0.0, 1.1, n_energy + 1) * gamma_0 ** 2
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    d2NdEdOmega = np.empty((n_points, n_points, s_centers.size))
    for i, x0 in enumerate(theta_x_grid):
        for j, y0 in enumerate(theta_y_grid):
            hist = spectrum_from_particles.direct_binning_spectrum(
                res._gamma, res._theta_x, res._theta_y, res._weight, res._a0,
                x0, y0, s_edges, res._phi_pol)
            d2NdEdOmega[i, j, :] = hist / res._s_scale_MeV / 1e6
    # Same QUICK-FIX rescale run_simulation applied to angular_spectrum,
    # cached on res -- keeps an on-demand query numerically consistent
    # with the main run's total_yield too (see that comment for why).
    d2NdEdOmega = d2NdEdOmega * res._angular_rescale

    E_eV = s_centers * res._s_scale_MeV * 1e6
    dtx, dty = np.gradient(theta_x_grid), np.gradient(theta_y_grid)
    dNdE_per_eV = np.einsum("ijk,i,j->k", d2NdEdOmega, dtx, dty)
    n_photons_in_range = float(np.einsum("ijk,i,j,k->", d2NdEdOmega, dtx, dty, np.gradient(E_eV)))

    spectrum = BinnedSpectrum(E_eV=E_eV, dNdE_per_eV=dNdE_per_eV)
    return AngularRangeSpectrumResult(
        spectrum=spectrum, theta_x_range=theta_x_range, theta_y_range=theta_y_range,
        n_photons_in_range=n_photons_in_range)


class XigmaDirectAdapter:
    def __init__(self):
        self._last_results: CommonResults | None = None

    def capabilities(self):
        from compton_suite.gui.model_api import ModelCapabilities
        return ModelCapabilities(**capabilities())

    def available(self) -> tuple[bool, str]:
        return available()

    def extra_params(self) -> list[tuple[str, float, str]]:
        return extra_params()

    def params_to_config(self, fields: dict, quantum: bool = False):
        return params_to_config(fields, quantum)

    def run(self, cfg: DirectConfig, n_mc: int, seed: int, *, electrons: MacroBunch):
        res = run_simulation(cfg, n_mc=n_mc, seed=seed, electrons=electrons)
        self._last_results = res
        return res

    def spectrum_in_angular_range(self, theta_x_range: tuple[float, float],
                                  theta_y_range: tuple[float, float], **kwargs):
        if self._last_results is None:
            raise RuntimeError("spectrum_in_angular_range: run() must be "
                               "called at least once first")
        return spectrum_in_angular_range(self._last_results, theta_x_range, theta_y_range)

    def load_ele_file(self, path: str) -> MacroBunch:
        return load_elegant_ele(path)

    def ele_file_summary(self, bunch: MacroBunch) -> dict:
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
