"""ModelAdapter wrappers around this package's tabulated-overlap pipeline.

Two adapters, registered under two names (see ``compton_suite.models.api.
discover_models``):

  * ``XigmaAdapter`` (registered as ``"xigma-i"``) drives the full
    Stage 0/1/2 tabulated pipeline (``tabulated_engine.TabulatedEngine``):
    semi-analytic, GPU/CPU, smooth binned spectral-density output.
  * ``DirectAdapter`` (registered as ``"delta"``) is this package's own
    brute-force per-macroparticle resonance-binning mode -- no table, no
    importance sampling, reusing this package's Stage 0
    (``particles.push_and_sample``) directly. Not a separate model
    package; delta IS xigma_i, just a different Stage-2 evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compton_suite.io.interaction import InteractionParameters
from compton_suite.io.photons import (
    AngularRangeSpectrumResult,
    BinnedAngularSpectrum,
    BinnedSpatialDistribution,
    BinnedSpectrum,
    BinnedTemporalEnvelope,
    Photons,
)
from compton_suite.io.units import M_TO_CM
from compton_suite.models.api import Job, ModelCapabilities

# The a0 range the weakly-nonlinear approximation this whole codebase is
# built on (a0 <~ 1) is meant to be valid over -- a fixed *model*
# parameter, not derived from any particular collision's actual params.a0.
# See deposition.retarget_a0.
DEFAULT_A0_MAX = 0.5


def _theta_grid(cfg, n_points: int = 33,
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
    half_window = max(5.0 * cfg.Theta_x if cfg.Theta_x > 0 else 0.0,
                      5.0 * cfg.Theta_y if cfg.Theta_y > 0 else 0.0,
                      3.0 / cfg.interaction.beam.gamma0)
    return np.linspace(-half_window, half_window, n_points, dtype=np.float32)


def _resolve_device(device_preference: str) -> str:
    from .collision import detect_device

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
@dataclass
class Config:
    """Numerics-only config for the full tabulated pipeline: the shared
    (beam, laser) bundle plus xigma_i-specific numeric/grid knobs.
    ``beta_ff``/``phi_pol`` are laser extras this convention supports that
    the shared ``GaussianParaxialLaser`` deliberately doesn't (see that
    module's docstring); ``crossing_angle``/``quantum`` are always-off
    placeholders (this pipeline is head-on/classical-only today) kept so
    ``summary`` can report them consistently with kascade's."""

    interaction: InteractionParameters
    beta_ff: float = 0.0
    phi_pol: float = 0.0
    device_preference: str = "auto"
    n_steps_0: int = 64
    n_bins_gamma: int = 48
    n_bins_theta_x: int = 48
    n_bins_theta_y: int = 48
    n_bins_a0: int = 12
    a0_max: float = DEFAULT_A0_MAX
    samples_per_point_2: int = 32
    n_time_bins: int = 128
    n_spatial_bins_x: int = 64
    n_spatial_bins_y: int = 64
    Theta_x: float = 0.0
    Theta_y: float = 0.0
    crossing_angle: float = 0.0
    quantum: bool = False


def _attach_xigma_cache(res: Photons, *, params, gamma_0, sigma_gamma_0,
                         engine, device, angular_rescale) -> Photons:
    """Stash this adapter's own private recompute cache on a ``Photons``
    instance (a plain, non-frozen dataclass -- arbitrary extra attributes
    work fine set after construction). ``_params``/``_engine``/``_device``
    let ``spectrum_in_angular_range()`` recompute an on-demand angular-range
    query without rerunning the whole simulation. ``_angular_rescale`` is
    the QUICK-FIX rescale factor (see ``run_simulation``'s "QUICK FIX,
    FLAGGED FOR FUTURE INVESTIGATION" comment), reapplied identically in
    ``spectrum_in_angular_range`` so an on-demand query stays consistent
    with the main run's ``total_yield``."""
    res._params = params
    res._gamma_0 = gamma_0
    res._sigma_gamma_0 = sigma_gamma_0
    res._engine = engine
    res._device = device
    res._angular_rescale = angular_rescale
    return res


def run_simulation(job: Job) -> Photons:
    from .collision import build_params
    from .tabulated_engine import TabulatedEngine

    extra = job.extra
    cfg = Config(
        interaction=job.interaction,
        beta_ff=float(extra.get("beta_ff", 0.0)),
        phi_pol=float(extra.get("phi_pol", 0.0)),
        device_preference=str(extra.get("device_preference", "auto")),
        n_steps_0=int(float(extra.get("n_steps_0", 64))),
        n_bins_gamma=int(float(extra.get("n_bins_gamma", 48))),
        n_bins_theta_x=int(float(extra.get("n_bins_theta_x", 48))),
        n_bins_theta_y=int(float(extra.get("n_bins_theta_y", 48))),
        n_bins_a0=int(float(extra.get("n_bins_a0", 12))),
        a0_max=float(extra.get("a0_max", DEFAULT_A0_MAX)),
        samples_per_point_2=int(float(extra.get("samples_per_point_2", 32))),
        n_time_bins=int(job.output.n_time_bins),
        n_spatial_bins_x=int(job.output.n_spatial_bins_x),
        n_spatial_bins_y=int(job.output.n_spatial_bins_y),
    )
    electrons = job.electrons
    device = _resolve_device(cfg.device_preference)

    beam, laser = cfg.interaction.beam, cfg.interaction.laser
    params = build_params(beam, laser, beta_ff=cfg.beta_ff, device=device)
    xp = params.xp

    gamma_0 = cfg.interaction.beam.gamma0
    sigma_gamma_0 = cfg.interaction.beam.sigma_gamma

    # Total yield, angle-integrated spectrum, angular spectrum, temporal
    # envelope, and spatial distribution all come from TabulatedEngine --
    # `params` is used only as its plain-data bundle (build_params above);
    # it runs no compute of its own.
    engine = TabulatedEngine(params)
    push_backend = 'cupy' if device == 'gpu' else 'numpy'

    n_particles_new = electrons.n_particles
    n_bins = (cfg.n_bins_gamma, cfg.n_bins_theta_x, cfg.n_bins_theta_y, cfg.n_bins_a0)
    n_spatial_bins = (cfg.n_spatial_bins_x, cfg.n_spatial_bins_y)
    engine.run(n_steps=cfg.n_steps_0,
               n_bins=n_bins, backend=push_backend, a0_max=cfg.a0_max,
               n_time_bins=cfg.n_time_bins, n_spatial_bins=n_spatial_bins,
               bunch=electrons)
    total_yield = engine.total_yield

    # engine.temporal_envelope/.spatial_distribution: rate vs seconds /
    # areal density vs cm. Convert cm/photons-per-cm^2 to SI. These stay
    # on-device (cupy) when push_backend='cupy' -- particles.py's
    # _bin_temporal/_bin_spatial never convert to host -- so must be
    # converted here before matplotlib (which cannot implicitly convert a
    # cupy array).
    t_seconds_raw, rate_raw = engine.temporal_envelope
    t_seconds = params.asnumpy(t_seconds_raw)
    rate = params.asnumpy(rate_raw)
    temporal_envelope = BinnedTemporalEnvelope(t_seconds=t_seconds, rate=rate)

    x_centers_cm, y_centers_cm, density_per_cm2 = engine.spatial_distribution
    spatial_distribution = BinnedSpatialDistribution(
        x_centers=params.asnumpy(x_centers_cm) / M_TO_CM,
        y_centers=params.asnumpy(y_centers_cm) / M_TO_CM,
        density=params.asnumpy(density_per_cm2) * (M_TO_CM ** 2))

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
    # always returns a host array regardless of device.
    theta_x = _theta_grid(cfg)
    theta_y = _theta_grid(cfg)
    s_ang = (xp.linspace(0.0, 1.1, 96, dtype=xp.float32) * gamma_0 ** 2)
    d2Nds_dOmega, _dt, _debug = engine.angular_spectrum(
        s_ang, xp.asarray(theta_x), xp.asarray(theta_y), cfg.phi_pol,
        samples_per_point=cfg.samples_per_point_2, device=device)
    E_ang_eV = (params.asnumpy(s_ang) * s_scale_MeV) * 1e6
    d2NdEdOmega = d2Nds_dOmega / s_scale_MeV / 1e6  # -> eV^-1 sr^-1

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

    res = Photons(
        model_name="xigma-i",
        cfg=cfg,
        n_mc=n_particles_new,
        total_yield=total_yield,
        spectrum=BinnedSpectrum(E_eV=E_eV, dNdE_per_eV=dNdE_per_eV),
        summary=summary,
        angular_spectrum=BinnedAngularSpectrum(
            theta_x=theta_x, theta_y=theta_y, E_eV=E_ang_eV, d2NdEdOmega=d2NdEdOmega),
        temporal_envelope=temporal_envelope,
        spatial_distribution=spatial_distribution,
    )
    return _attach_xigma_cache(
        res, params=params, gamma_0=gamma_0, sigma_gamma_0=sigma_gamma_0,
        engine=engine, device=device, angular_rescale=angular_rescale)


def spectrum_in_angular_range(
        res: Photons, theta_x_range: tuple[float, float],
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
        samples_per_point=cfg.samples_per_point_2, device=res._device)
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


class XigmaAdapter:
    def __init__(self):
        self._last_res: Photons | None = None

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(display_name="XIGMA", uses_shared_sample_count=True)

    def model_params(self) -> list[tuple[str, float | str, str]]:
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

    def model_choices(self) -> dict[str, list[str]]:
        return {"device_preference": ["auto", "gpu", "cpu"]}

    def run(self, job: Job) -> Photons:
        res = run_simulation(job)
        self._last_res = res
        return res

    def spectrum_in_angular_range(self, theta_x_range, theta_y_range, **kwargs) -> AngularRangeSpectrumResult:
        if self._last_res is None:
            raise RuntimeError("XigmaAdapter.spectrum_in_angular_range: run() must be called first")
        return spectrum_in_angular_range(self._last_res, theta_x_range, theta_y_range, **kwargs)


# ---------------------------------------------------------------------------
# DirectAdapter ("delta"): brute-force per-macroparticle resonance binning
# ---------------------------------------------------------------------------
@dataclass
class DirectConfig:
    """Numerics-only config for the brute-force per-macroparticle mode.
    Same field shape as ``Config`` minus the Stage-1/2 table-grid knobs,
    plus its own angular-grid resolution."""

    interaction: InteractionParameters
    beta_ff: float = 0.0
    phi_pol: float = 0.0
    device_preference: str = "auto"
    n_steps_0: int = 64
    n_theta_grid: int = 9
    Theta_x: float = 0.0
    Theta_y: float = 0.0
    crossing_angle: float = 0.0
    quantum: bool = False


def _attach_delta_cache(res: Photons, *, gamma, theta_x, theta_y, a0, weight,
                         s_edges, s_scale_MeV, phi_pol, angular_rescale) -> Photons:
    """Stash the raw per-particle arrays this mode needs to answer an
    on-demand ``spectrum_in_angular_range_direct`` query -- no table to
    requery (unlike xigma-i's ``_attach_xigma_cache``), just re-evaluate
    ``direct_binning_spectrum`` at a new (x0, y0) grid over the same
    cached particles."""
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


def run_simulation_direct(job: Job) -> Photons:
    """``job.electrons`` is required: electron sampling is the caller's
    job, not this adapter's -- there's exactly one place electrons get
    drawn from a beam description (``compton_suite.io.bunch.
    sample_gaussian_bunch``), not one per model."""
    from . import particles, spectrum_from_particles
    from .collision import build_params

    extra = job.extra
    cfg = DirectConfig(
        interaction=job.interaction,
        beta_ff=float(extra.get("beta_ff", 0.0)),
        phi_pol=float(extra.get("phi_pol", 0.0)),
        device_preference=str(extra.get("device_preference", "auto")),
        n_steps_0=int(float(extra.get("n_steps_0", 64))),
        n_theta_grid=int(float(extra.get("n_theta_grid", 9))),
    )
    if cfg.crossing_angle != 0.0:
        raise ValueError(
            f"delta: crossing_angle must be 0 (head-on only), got {cfg.crossing_angle}")

    electrons = job.electrons
    device = _resolve_device(cfg.device_preference)

    beam, laser = cfg.interaction.beam, cfg.interaction.laser
    params = build_params(beam, laser, beta_ff=cfg.beta_ff, device=device)

    gamma_0, sigma_gamma_0 = cfg.interaction.beam.gamma0, cfg.interaction.beam.sigma_gamma

    n_particles_new = electrons.n_particles

    push_backend = 'cupy' if device == 'gpu' else 'numpy'
    n_time_bins, n_spatial_bins = 128, (64, 64)
    gamma, tx, ty, a0_shape, w, diagnostics = particles.push_and_sample(
        params, electrons, n_steps=cfg.n_steps_0, backend=push_backend,
        n_time_bins=n_time_bins, n_spatial_bins=n_spatial_bins)
    a0 = a0_shape * params.a0 ** 2   # exact per-particle retarget (no table/binning needed)

    total_yield = float(params.asnumpy(w).sum() if hasattr(w, "get") else np.sum(w))

    # Total angle-integrated spectrum: angle_integrated_spectrum, no known
    # normalization issue (unlike direct_binning_spectrum's own
    # angle-integrated total) -- needs only gamma/weight.
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
    n_grid = cfg.n_theta_grid
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

    # diagnostics arrays stay on-device (cupy) when push_backend='cupy' --
    # particles.py's _bin_temporal/_bin_spatial never convert to host, so
    # this adapter must, before these reach matplotlib (which cannot
    # implicitly convert a cupy array). params.asnumpy() is a no-op on CPU.
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
            x_centers=0.5 * (x_edges[:-1] + x_edges[1:]) / M_TO_CM,
            y_centers=0.5 * (y_edges[:-1] + y_edges[1:]) / M_TO_CM,
            density=density * (M_TO_CM ** 2))

    summary = dict(
        total_yield=total_yield,
        crossing_angle_rad=cfg.crossing_angle,
        quantum=float(bool(cfg.quantum)),
        a0=float(params.a0),
        # FLAGGED: see the "QUICK FIX" comment above -- != 1.0 until the
        # underlying direct_binning_spectrum normalisation is root-caused.
        angular_spectrum_rescale_applied=angular_rescale,
    )

    res = Photons(
        model_name="delta",
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
    return _attach_delta_cache(
        res, gamma=gamma_h, theta_x=tx_h, theta_y=ty_h, a0=a0_h, weight=w_h,
        s_edges=s_edges, s_scale_MeV=s_scale_MeV, phi_pol=cfg.phi_pol,
        angular_rescale=angular_rescale)


def spectrum_in_angular_range_direct(res: Photons, theta_x_range: tuple[float, float],
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
    mix up "electrons pointing this way" with "photons observed this way".
    Mirrors ``spectrum_in_angular_range`` (xigma-i's own version), which
    already evaluates its kernel at observation points for exactly this
    reason.
    """
    from . import spectrum_from_particles

    if res._gamma is None:
        raise RuntimeError("spectrum_in_angular_range_direct: run() must be called first")

    gamma_0 = res.cfg.interaction.beam.gamma0
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
    # Same QUICK-FIX rescale run_simulation_direct applied to
    # angular_spectrum, cached on res -- keeps an on-demand query
    # numerically consistent with the main run's total_yield too.
    d2NdEdOmega = d2NdEdOmega * res._angular_rescale

    E_eV = s_centers * res._s_scale_MeV * 1e6
    dtx, dty = np.gradient(theta_x_grid), np.gradient(theta_y_grid)
    dNdE_per_eV = np.einsum("ijk,i,j->k", d2NdEdOmega, dtx, dty)
    n_photons_in_range = float(np.einsum("ijk,i,j,k->", d2NdEdOmega, dtx, dty, np.gradient(E_eV)))

    spectrum = BinnedSpectrum(E_eV=E_eV, dNdE_per_eV=dNdE_per_eV)
    return AngularRangeSpectrumResult(
        spectrum=spectrum, theta_x_range=theta_x_range, theta_y_range=theta_y_range,
        n_photons_in_range=n_photons_in_range)


class DirectAdapter:
    def __init__(self):
        self._last_res: Photons | None = None

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(display_name="delta", uses_shared_sample_count=True)

    def model_params(self) -> list[tuple[str, float | str, str]]:
        return [
            ("Device (auto/gpu/cpu)", "auto", "device_preference"),
            ("Stage 0 trajectory steps", 64, "n_steps_0"),
            ("Angular-spectrum grid points/axis", 9, "n_theta_grid"),
        ]

    def model_choices(self) -> dict[str, list[str]]:
        return {"device_preference": ["auto", "gpu", "cpu"]}

    def run(self, job: Job) -> Photons:
        res = run_simulation_direct(job)
        self._last_res = res
        return res

    def spectrum_in_angular_range(self, theta_x_range, theta_y_range, **kwargs) -> AngularRangeSpectrumResult:
        if self._last_res is None:
            raise RuntimeError("DirectAdapter.spectrum_in_angular_range: run() must be called first")
        return spectrum_in_angular_range_direct(self._last_res, theta_x_range, theta_y_range, **kwargs)
