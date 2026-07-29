"""Electron-bunch representations.

Two distinct types, matching the distinction the ``gaussian_6d_waist``
v0.1 I/O spec (``specs/electron_beam_io_v0.1_full.md``) draws between raw
data and its analytic description:

* :class:`MacroBunch` -- raw, engine-agnostic macroparticle arrays, e.g. as
  loaded from an elegant ``.ele`` file, at whatever slice the tracking code
  produced them at (not necessarily the beam waist).
* :class:`GaussianElectronBeam` -- the ``gaussian_6d_waist`` v0.1 analytic
  contract: a 6D factorized Gaussian defined AT the waist slice
  (``alpha_x = alpha_y = 0`` there by construction), with the waist located
  at the interaction point ``z=0``. Every downstream consumer (sampling,
  the analytical model) treats this as the single source of truth for a
  Gaussian beam, relying on Liouville's theorem -- geometric emittance is
  invariant under linear/ballistic drift -- rather than re-deriving beam
  parameters at each z independently. See :func:`fit_gaussian` for the
  concrete mechanism.

Every physical parameter travels as a :class:`PhysicalQuantity` (never a bare
float) -- models extract the value in whatever unit they need internally.

Key functions:

* :func:`sample_gaussian_bunch` / :func:`sample_gaussian_canonical` -- sample
  macroparticles from a beam description using canonical variables with
  mass-shell enforcement for physically consistent particles.
* :func:`drift` -- propagate beam in vacuum over distance L, naturally
  producing Twiss tilt from waist sampling.
* :func:`fit_beam_full` -- fit structured Gaussian model with physical
  correlations (Twiss, chirp, dispersion) and quality metrics.
* :func:`evaluate_fit_quality` -- evaluate fit quality using Mahalanobis
  distance, KS statistics, and log-likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import C_LIGHT, E_CHARGE, MEC2_EV
from .enums import NoConvention, PhysicalMeaning, TimeConvention, WidthConvention
from .quantities import PhysicalQuantity

__all__ = ["MacroBunch", "GaussianElectronBeam", "BeamFittedParams", "validate",
           "sample_gaussian_bunch", "sample_gaussian_canonical", "drift",
           "fit_gaussian", "fit_beam_full", "evaluate_fit_quality",
           "beam_from_shared_fields", "beta_star_from_sigma_emit",
           "divergence_from_sigma_emit", "sigma_from_emittance"]

# ---------------------------------------------------------------------------
# Internal helper: build a PhysicalQuantity shorthand.
# ---------------------------------------------------------------------------
_BEAM_WIDTH_CONV = WidthConvention.SIGMA_INTENSITY_RMS
_BUNCH_LEN_CONV = TimeConvention.SIGMA_INTENSITY_RMS


def _pq(value: float, unit: str, meaning: PhysicalMeaning, convention=None) -> PhysicalQuantity:
    """Shortcut to build a PhysicalQuantity."""
    return PhysicalQuantity(value, unit, meaning, convention)


def _to_si_float(v: float | PhysicalQuantity, unit: str) -> float:
    """Extract a raw float in the given unit from a value that may be a
    PhysicalQuantity or a plain float (interpreted as already in *unit*)."""
    if isinstance(v, PhysicalQuantity):
        return v.to_unit(unit).magnitude
    return float(v)


def beta_star_from_sigma_emit(sigma_m: float, emit_geom_m: float) -> float:
    """Beta function at the waist, ``sigma**2 / emit_geom`` -- same units in
    as out squared over rad (SI in/out here; callers in other unit systems,
    e.g. xigma_i's CGS, convert at their own boundary)."""
    return sigma_m**2 / emit_geom_m


def divergence_from_sigma_emit(sigma_m: float, emit_geom_m: float) -> float:
    """RMS divergence at the waist, ``emit_geom / sigma`` -- a dimensionless
    angle, unit-system-invariant (no conversion needed between SI and CGS
    callers)."""
    return emit_geom_m / sigma_m


def sigma_from_emittance(emit_norm_m: float, beta_m: float, gamma: float) -> float:
    """Transverse rms beam size [m] from normalised emittance, beta function,
    and Lorentz gamma.

    ``sigma = sqrt(emit_geom * beta)`` where
    ``emit_geom = emit_norm / gamma``.

    All inputs and the output are SI.  Returns 0.0 for non-physical
    inputs (negative emittance, non-positive beta/gamma) rather than
    raising, so the GUI can display "--" gracefully.
    """
    if emit_norm_m is None or beta_m is None or gamma is None:
        return 0.0
    if emit_norm_m < 0 or beta_m <= 0 or gamma <= 0:
        return 0.0
    emit_geom = emit_norm_m / gamma
    return float(np.sqrt(emit_geom * beta_m))


@dataclass
class MacroBunch:
    """Raw, engine-agnostic macroparticle representation. SI units.

    ``x``/``y``/``z`` are the transverse/longitudinal position at whatever
    slice the data was produced at (head-tail sign for ``z``); ``thx``/
    ``thy`` are momentum angles ``px/pz``, ``py/pz`` -- not positions.
    ``weight`` is electrons-per-macroparticle, uniform across the bunch
    (matches what both kascade and xigma_i already assume throughout their
    physics, not just at I/O -- see the plan's Open Questions if that ever
    needs to change to support non-uniform/lossy bunches).
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    thx: np.ndarray
    thy: np.ndarray
    gamma: np.ndarray
    weight: float
    meta: dict = field(default_factory=dict)

    @property
    def n_particles(self) -> int:
        return int(np.asarray(self.x).shape[0])

    @property
    def N_e(self) -> float:
        return self.weight * self.n_particles


@dataclass(frozen=True)
class GaussianElectronBeam:
    """The ``gaussian_6d_waist`` v0.1 I/O contract.

    A 6D factorized Gaussian defined at the beam waist (``alpha_x =
    alpha_y = 0`` there), waist located at the interaction point ``z=0``.
    Every physical field is a :class:`PhysicalQuantity`; ``rel_energy_spread_rms``
    and ``sigma_pz`` are plain floats because they are dimensionless relative
    values (unitless), not physical quantities.
    """

    bunch_charge_C: PhysicalQuantity
    kinetic_energy_eV: PhysicalQuantity
    rel_energy_spread_rms: float
    sigma_x_m: PhysicalQuantity
    sigma_y_m: PhysicalQuantity
    emit_geom_x_m: PhysicalQuantity
    emit_geom_y_m: PhysicalQuantity
    sigma_t_s: PhysicalQuantity
    sigma_pz: float  # Relative RMS dispersion for longitudinal momentum (dimensionless)

    # -- SI convenience helpers (used by every derived property) ------------
    @property
    def _q_C(self) -> float:
        return self.bunch_charge_C.to_unit("coulomb").magnitude

    @property
    def _KE_eV(self) -> float:
        return self.kinetic_energy_eV.to_unit("electron_volt").magnitude

    @property
    def _sx_m(self) -> float:
        return self.sigma_x_m.to_unit("meter").magnitude

    @property
    def _sy_m(self) -> float:
        return self.sigma_y_m.to_unit("meter").magnitude

    @property
    def _ex_m(self) -> float:
        return self.emit_geom_x_m.to_unit("meter").magnitude

    @property
    def _ey_m(self) -> float:
        return self.emit_geom_y_m.to_unit("meter").magnitude

    @property
    def _st_s(self) -> float:
        return self.sigma_t_s.to_unit("second").magnitude

    # -- derived properties ------------------------------------------------
    @property
    def N_e(self) -> float:
        return self._q_C / E_CHARGE

    @property
    def total_energy_eV(self) -> float:
        return self._KE_eV + MEC2_EV

    @property
    def gamma0(self) -> float:
        return self.total_energy_eV / MEC2_EV

    @property
    def beta0(self) -> float:
        g = self.gamma0
        return (1.0 - 1.0 / g**2) ** 0.5

    @property
    def sigma_E_kin_eV(self) -> float:
        return self.rel_energy_spread_rms * self._KE_eV

    @property
    def sigma_gamma(self) -> float:
        return self.sigma_E_kin_eV / MEC2_EV

    @property
    def sigma_gamma_over_gamma0(self) -> float:
        return self.sigma_gamma / self.gamma0

    @property
    def sigma_z_m(self) -> float:
        return self.beta0 * C_LIGHT * self._st_s

    @property
    def sigma_pz_abs(self) -> float:
        """Absolute RMS dispersion for longitudinal momentum (normalized to mc)."""
        return self.sigma_pz * self.gamma0 * self.beta0

    @property
    def divergence_x_rad(self) -> float:
        return divergence_from_sigma_emit(self._sx_m, self._ex_m)

    @property
    def divergence_y_rad(self) -> float:
        return divergence_from_sigma_emit(self._sy_m, self._ey_m)

    @property
    def beta_star_x_m(self) -> float:
        return beta_star_from_sigma_emit(self._sx_m, self._ex_m)

    @property
    def beta_star_y_m(self) -> float:
        return beta_star_from_sigma_emit(self._sy_m, self._ey_m)

    @property
    def emit_norm_x_m(self) -> float:
        return self.beta0 * self.gamma0 * self._ex_m

    @property
    def emit_norm_y_m(self) -> float:
        return self.beta0 * self.gamma0 * self._ey_m

    @property
    def peak_current_A(self) -> float:
        return self._q_C / ((2.0 * np.pi) ** 0.5 * self._st_s)

    @property
    def peak_density_m3(self) -> float:
        return self.N_e / (
            (2.0 * np.pi) ** 1.5 * self._sx_m * self._sy_m * self.sigma_z_m
        )


def validate(beam: GaussianElectronBeam) -> list[str]:
    """Validate a :class:`GaussianElectronBeam` per spec Sec. 16.

    Raises ``ValueError`` on the spec's hard requirements (non-positive
    physical quantities); returns a list of warning strings for the
    spec's soft checks (large spread/divergence, suspicious unit mix-ups).
    """
    _q = beam._q_C
    _ke = beam._KE_eV
    _sx = beam._sx_m
    _sy = beam._sy_m
    _ex = beam._ex_m
    _ey = beam._ey_m
    _st = beam._st_s

    if _q <= 0:
        raise ValueError("GaussianElectronBeam: bunch_charge must be > 0")
    if _ke <= 0:
        raise ValueError("GaussianElectronBeam: kinetic_energy must be > 0")
    if beam.rel_energy_spread_rms < 0:
        raise ValueError("GaussianElectronBeam: rel_energy_spread_rms must be >= 0")
    if _sx <= 0 or _sy <= 0:
        raise ValueError("GaussianElectronBeam: sigma_x/sigma_y must be > 0")
    if _ex <= 0 or _ey <= 0:
        raise ValueError("GaussianElectronBeam: emit_geom_x/emit_geom_y must be > 0")
    if _st <= 0:
        raise ValueError("GaussianElectronBeam: sigma_t must be > 0")

    warnings: list[str] = []
    if beam.rel_energy_spread_rms > 0.1:
        warnings.append("Large relative energy spread; check whether this is intended.")
    if beam.divergence_x_rad * 1e3 > 100 or beam.divergence_y_rad * 1e3 > 100:
        warnings.append("Large angular divergence; paraxial approximation may be questionable.")
    if _ex > _sx or _ey > _sy:
        warnings.append(
            "Check units: geometric emittance is larger than the beam size -- "
            "emittance is a length*angle, not a length; this usually indicates "
            "a units mix-up (e.g. mm*mrad entered as m*rad)."
        )
    if _q > 1.0e-8:
        warnings.append("Very large bunch charge (> 10 nC); check pC/nC conversion is correct.")
    return warnings


def sample_gaussian_bunch(beam: GaussianElectronBeam, n_particles: int, *,
                           rng=None) -> MacroBunch:
    """Draw macroparticles from a :class:`GaussianElectronBeam`.

    Uses canonical sampling with mass-shell enforcement for physically
    consistent particles. This is the single entry point for electron
    bunch sampling -- no model has its own internal bunch sampler.

    Delegates to :func:`sample_gaussian_canonical` for the actual sampling.
    """
    return sample_gaussian_canonical(beam, n_particles, rng=rng)


def sample_gaussian_canonical(
    beam: GaussianElectronBeam,
    n_particles: int,
    *,
    rng=None,
) -> MacroBunch:
    """Draw macroparticles from a :class:`GaussianElectronBeam` using canonical
    variables with mass-shell enforcement.

    Sampling is done in canonical variables (x, p_x, y, p_y, z, p_z) and
    then mapped to beam variables (x, x', y, y', z, gamma) with the
    relativistic mass-shell constraint gamma^2 = 1 + p_x^2 + p_y^2 + p_z^2
    automatically satisfied by construction.

    Algorithm:
    1. Sample x, y, z from Gaussians (independent)
    2. Sample p_x, p_y from Gaussians (independent, sigma = gamma0 * divergence)
    3. Sample p_z from Gaussian (independent, sigma = beam.sigma_pz * pz_mean)
    4. Calculate gamma = sqrt(1 + p_x^2 + p_y^2 + p_z^2)
    5. Convert to angles: x' = p_x/p_z, y' = p_y/p_z

    This ensures:
    - Mass-shell constraint is automatically satisfied
    - No rejection sampling needed for mass-shell violations
    - Physically consistent particles
    """
    rng = np.random.default_rng() if rng is None else rng

    # Extract SI values from the beam (PhysicalQuantity -> raw float).
    sx = beam._sx_m
    sy = beam._sy_m
    sz = beam.sigma_z_m
    g0 = beam.gamma0
    dx = beam.divergence_x_rad
    dy = beam.divergence_y_rad
    b0 = beam.beta0

    # Sample positions (independent)
    x = rng.normal(0.0, sx, n_particles)
    y = rng.normal(0.0, sy, n_particles)
    z = rng.normal(0.0, sz, n_particles)

    # Sample transverse momenta (independent)
    px = rng.normal(0.0, g0 * dx, n_particles)
    py = rng.normal(0.0, g0 * dy, n_particles)

    # Sample longitudinal momentum (independent)
    # pz_mean = gamma0 * beta0 for ultra-relativistic beams
    pz_mean = g0 * b0
    sigma_pz_abs = beam.sigma_pz * pz_mean  # Absolute dispersion
    pz = rng.normal(pz_mean, sigma_pz_abs, n_particles)

    # Enforce pz > 1 (physical constraint for relativistic particles)
    mask = pz > 1.0
    while not np.all(mask):
        n_bad = np.sum(~mask)
        pz[~mask] = rng.normal(pz_mean, sigma_pz_abs, n_bad)
        mask = pz > 1.0

    # Calculate gamma from momenta (mass-shell: gamma^2 = 1 + p^2)
    gamma = np.sqrt(1.0 + px**2 + py**2 + pz**2)

    # Convert to angles
    thx = px / pz
    thy = py / pz

    weight = beam.N_e / n_particles
    return MacroBunch(x=x, y=y, z=z, thx=thx, thy=thy, gamma=gamma, weight=weight,
                       meta={"source": "sample_gaussian_canonical", "beam": beam})


def drift(bunch: MacroBunch, L: float) -> MacroBunch:
    """Propagate beam in vacuum over distance L.

    Ballistic propagation:
    - x -> x + x' * L
    - y -> y + y' * L
    - z, thx, thy, gamma unchanged

    This naturally produces Twiss tilt (alpha != 0) from waist sampling.
    When sampling at the waist (alpha_x = alpha_y = 0), the beam will
    develop non-zero alpha after drifting, which is the physically correct
    behavior.
    """
    return MacroBunch(
        x=bunch.x + bunch.thx * L,
        y=bunch.y + bunch.thy * L,
        z=bunch.z,
        thx=bunch.thx,
        thy=bunch.thy,
        gamma=bunch.gamma,
        weight=bunch.weight,
    )


def beam_from_shared_fields(*, eps0: float, sigma_eps_rel: float,
                             emit_x: float | PhysicalQuantity,
                             emit_y: float | PhysicalQuantity,
                             sigma0_x: float | PhysicalQuantity,
                             sigma0_y: float | PhysicalQuantity,
                             sigma_par_e: float | PhysicalQuantity,
                             N_e: float,
                             sigma_pz: float = 0.0) -> GaussianElectronBeam:
    """Build a :class:`GaussianElectronBeam` from the flat SI field set every
    model's own ``Config`` already derives and agrees on (``eps0``,
    ``sigma_eps_rel``, ``emit_x/y``, ``sigma0_x/y``, ``sigma_par_e``, ``N_e``
    -- see ``ComptonSuite/validation/scenarios.py``'s ``scenario_to_shared_
    fields``, this function's exact inverse).

    Accepts either plain floats (interpreted as SI) or ``PhysicalQuantity``
    objects for the length/emittance parameters. ``eps0`` (Lorentz gamma),
    ``sigma_eps_rel`` (relative energy spread wrt gamma), ``N_e`` (electron
    count) and ``sigma_pz`` are dimensionless and always raw floats.

    Exists so a GUI (or any other caller) that already has one model's
    ``Config`` on hand can hand this beam to :func:`sample_gaussian_bunch`
    and pass the SAME sampled :class:`MacroBunch` into every model's
    ``run()`` -- one canonical electron sample per Calculate click, drawn
    here by the IO layer, rather than each model silently falling back to
    its own independent internal sampler whenever no bunch is supplied.

    ``sigma_eps_rel`` here is relative to gamma/total energy (every
    ``Config``'s own convention -- confirmed by ``KASCADE_SPEC``/
    ``XIGMA_SPEC`` "machine-checking" that all these engines already agree
    on it), NOT :class:`GaussianElectronBeam`'s own ``rel_energy_spread_
    rms`` (relative to *kinetic* energy, spec Sec. 10) -- converted
    explicitly below, same distinction that caused a real bug in
    ``AnalyticalConfig`` earlier (see that class's ``sigma_eps_rel``
    docstring).

    ``sigma_pz`` is the relative RMS dispersion for longitudinal momentum
    (dimensionless), normalized to the mean momentum ``pz_mean = gamma0 * beta0``.
    If not provided, defaults to 0.0 (delta function in longitudinal momentum).
    """
    gamma0 = eps0
    kinetic_energy_eV = (gamma0 - 1.0) * MEC2_EV
    sigma_gamma = sigma_eps_rel * gamma0
    rel_energy_spread_rms = (sigma_gamma * MEC2_EV) / kinetic_energy_eV
    beta0 = (1.0 - 1.0 / gamma0**2) ** 0.5

    # Unpack PhysicalQuantity (or float) to SI float.
    emit_x_si = _to_si_float(emit_x, "meter")
    emit_y_si = _to_si_float(emit_y, "meter")
    sx_si = _to_si_float(sigma0_x, "meter")
    sy_si = _to_si_float(sigma0_y, "meter")
    spe_si = _to_si_float(sigma_par_e, "meter")

    return GaussianElectronBeam(
        bunch_charge_C=_pq(N_e * E_CHARGE, "coulomb", PhysicalMeaning.BUNCH_CHARGE),
        kinetic_energy_eV=_pq(kinetic_energy_eV, "electron_volt", PhysicalMeaning.BEAM_ENERGY),
        rel_energy_spread_rms=rel_energy_spread_rms,
        sigma_x_m=_pq(sx_si, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE, _BEAM_WIDTH_CONV),
        sigma_y_m=_pq(sy_si, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE, _BEAM_WIDTH_CONV),
        emit_geom_x_m=_pq(emit_x_si, "meter", PhysicalMeaning.EMITTANCE),
        emit_geom_y_m=_pq(emit_y_si, "meter", PhysicalMeaning.EMITTANCE),
        sigma_t_s=_pq(spe_si / (beta0 * C_LIGHT), "second", PhysicalMeaning.BUNCH_LENGTH, _BUNCH_LEN_CONV),
        sigma_pz=sigma_pz,
    )


@dataclass
class BeamFittedParams:
    """Physically meaningful beam parameters from fitting.

    Contains all parameters extracted from a Gaussian fit to macroparticle
    data, including transverse Twiss parameters, longitudinal dispersion,
    and fit quality metrics.
    """
    # Transverse (at waist)
    sigma_x_waist: float
    sigma_y_waist: float
    emit_geom_x: float
    emit_geom_y: float
    beta_x: float
    beta_y: float
    alpha_x: float  # Should be ~0 at waist
    alpha_y: float

    # Longitudinal
    sigma_z: float
    sigma_pz: float      # Input parameter (relative)
    sigma_gamma: float   # Computed from fit
    chirp: float         # Slope dγ/dz (1/m units)

    # Dispersion
    D_x: float
    D_y: float

    # Fit quality
    fit_quality: dict


def evaluate_fit_quality(
    bunch: MacroBunch,
    mu: np.ndarray,
    Sigma: np.ndarray,
    n_synthetic: int = 3,
) -> dict:
    """Evaluate Gaussian fit quality with sampling-noise baseline.

    Compares the real data against synthetic Gaussian samples generated from
    the fitted parameters to distinguish between sampling noise and true
    model mismatch.

    Returns a dictionary with:
    - ks_real: KS statistic for real data
    - ks_synthetic: Mean KS for synthetic Gaussian samples
    - ks_excess: ks_real - ks_synthetic (model mismatch indicator)
    - mean_d2_real: Mean Mahalanobis distance for real data
    - mean_d2_synthetic: Mean for synthetic
    - log_likelihood_real: Log-likelihood of real data
    - log_likelihood_synthetic: Mean for synthetic

    Interpretation:
    - real ~ synthetic: fit is noise-limited (good)
    - real > synthetic: model mismatch
    - large deviation: non-Gaussian structure
    """
    from scipy.stats import chi2, kstest

    # Build data matrix
    X = np.stack([bunch.x, bunch.thx, bunch.y, bunch.thy, bunch.z, bunch.gamma], axis=1)
    Xc = X - mu

    # Inverse covariance (with regularization for stability)
    try:
        inv = np.linalg.inv(Sigma)
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            raise np.linalg.LinAlgError("Singular covariance")
    except np.linalg.LinAlgError:
        # Regularize if singular
        Sigma_reg = Sigma + 1e-10 * np.eye(Sigma.shape[0])
        inv = np.linalg.inv(Sigma_reg)
        sign, logdet = np.linalg.slogdet(Sigma_reg)

    # Mahalanobis distance for real data
    d2_real = np.einsum("ni,ij,nj->n", Xc, inv, Xc)

    # KS test against chi2(6)
    ks_real, _ = kstest(d2_real, chi2(df=6).cdf)
    mean_d2_real = np.mean(d2_real)

    # Log-likelihood for real data
    k = X.shape[1]
    loglik_real = -0.5 * (k * np.log(2 * np.pi) + logdet + np.mean(d2_real))

    # Synthetic baseline
    rng = np.random.default_rng()
    ks_syn = []
    mean_d2_syn = []
    loglik_syn = []

    for _ in range(n_synthetic):
        Xs = rng.multivariate_normal(mu, Sigma, size=len(X))
        Xsc = Xs - mu

        d2s = np.einsum("ni,ij,nj->n", Xsc, inv, Xsc)
        ks, _ = kstest(d2s, chi2(df=6).cdf)

        ks_syn.append(ks)
        mean_d2_syn.append(np.mean(d2s))
        loglik_syn.append(-0.5 * (k * np.log(2 * np.pi) + logdet + np.mean(d2s)))

    return {
        "ks_real": ks_real,
        "ks_synthetic": np.mean(ks_syn),
        "ks_excess": ks_real - np.mean(ks_syn),
        "mean_d2_real": mean_d2_real,
        "mean_d2_synthetic": np.mean(mean_d2_syn),
        "log_likelihood_real": loglik_real,
        "log_likelihood_synthetic": np.mean(loglik_syn),
    }


def fit_beam_full(bunch: MacroBunch) -> BeamFittedParams:
    """Fit structured Gaussian model with physical correlations.

    Extracts physically meaningful parameters from macroparticle data:
    - Transverse: Twiss alpha, beta, emittance (geometric)
    - Longitudinal: sigma_z, sigma_pz, chirp (slope dγ/dz)
    - Dispersion: D_x, D_y (x-γ, y-γ correlations)
    - Fit quality: Mahalanobis, KS, log-likelihood metrics

    Uses biased covariance for physics (population parameters).
    """
    # Build data matrix: [x, thx, y, thy, z, gamma]
    X = np.stack([bunch.x, bunch.thx, bunch.y, bunch.thy, bunch.z, bunch.gamma], axis=1)

    # Center data
    mu = np.mean(X, axis=0)
    Xc = X - mu

    # Compute covariance (biased for population parameters)
    Sigma = np.cov(Xc, rowvar=False, bias=True)

    # Extract parameters
    ix, ixp, iy, iyp, iz, ig = range(6)

    # Transverse emittance
    emit_x = np.sqrt(Sigma[ix, ix] * Sigma[ixp, ixp] - Sigma[ix, ixp]**2)
    emit_y = np.sqrt(Sigma[iy, iy] * Sigma[iyp, iyp] - Sigma[iy, iyp]**2)

    # Twiss parameters
    beta_x = Sigma[ix, ix] / emit_x
    alpha_x = -Sigma[ix, ixp] / emit_x
    beta_y = Sigma[iy, iy] / emit_y
    alpha_y = -Sigma[iy, iyp] / emit_y

    # Waist sizes (from emittance and divergence)
    sigma_x_waist = emit_x / np.sqrt(Sigma[ixp, ixp])
    sigma_y_waist = emit_y / np.sqrt(Sigma[iyp, iyp])

    # Longitudinal
    sigma_z = np.sqrt(Sigma[iz, iz])
    sigma_pz = np.sqrt(Sigma[ig, ig])  # This is sigma_gamma from fit
    sigma_gamma = sigma_pz  # For Gaussian, they're related

    # Chirp (slope dγ/dz)
    chirp = Sigma[iz, ig] / Sigma[iz, iz] if Sigma[iz, iz] > 0 else 0.0

    # Dispersion
    D_x = Sigma[ix, ig] / Sigma[ig, ig] if Sigma[ig, ig] > 0 else 0.0
    D_y = Sigma[iy, ig] / Sigma[ig, ig] if Sigma[ig, ig] > 0 else 0.0

    # Fit quality
    fit_quality = evaluate_fit_quality(bunch, mu, Sigma)

    return BeamFittedParams(
        sigma_x_waist=sigma_x_waist,
        sigma_y_waist=sigma_y_waist,
        emit_geom_x=emit_x,
        emit_geom_y=emit_y,
        beta_x=beta_x,
        beta_y=beta_y,
        alpha_x=alpha_x,
        alpha_y=alpha_y,
        sigma_z=sigma_z,
        sigma_pz=sigma_pz,
        sigma_gamma=sigma_gamma,
        chirp=chirp,
        D_x=D_x,
        D_y=D_y,
        fit_quality=fit_quality,
    )


def fit_gaussian(bunch: MacroBunch) -> GaussianElectronBeam:
    """Fit a :class:`GaussianElectronBeam` (at its waist) from raw macroparticles.

    Not a plain moment calculation: ``alpha=0`` (needed for
    ``beta* = sigma**2/emit_geom``) only holds at the waist, and a loaded
    bunch is generally *not* sampled exactly there. The closed-form fix
    exploits that the divergence ``sigma_x'`` is invariant under a pure
    ballistic drift (only position changes, not angle) -- so the Twiss
    "gamma" function ``gamma_twiss = sigma_x'^2 / epsilon`` is itself
    drift-invariant, and at the waist ``beta* = 1/gamma_twiss =
    epsilon/sigma_x'^2`` exactly, with no need to explicitly solve for or
    propagate to the waist location. Emittance ``epsilon`` is itself a
    Liouville invariant, unchanged by the drift. This is the concrete
    mechanism behind "ballistic propagation + Liouville makes this
    trivial": a real drift-to-waist transform, done algebraically instead
    of numerically re-simulating the drift.
    """
    x, thx = np.asarray(bunch.x, dtype=float), np.asarray(bunch.thx, dtype=float)
    y, thy = np.asarray(bunch.y, dtype=float), np.asarray(bunch.thy, dtype=float)
    gamma = np.asarray(bunch.gamma, dtype=float)
    z = np.asarray(bunch.z, dtype=float)

    def _waist_sigma_and_emit(pos: np.ndarray, ang: np.ndarray) -> tuple[float, float]:
        sig_pos2 = float(np.var(pos))
        sig_ang2 = float(np.var(ang))
        sig_cross = float(np.mean((pos - pos.mean()) * (ang - ang.mean())))
        epsilon = (sig_pos2 * sig_ang2 - sig_cross**2) ** 0.5
        sigma_ang = sig_ang2**0.5
        sigma_waist = epsilon / sigma_ang  # = sqrt(epsilon * beta_star), beta_star = epsilon/sig_ang2
        return sigma_waist, epsilon

    sigma_x_m, emit_geom_x_m = _waist_sigma_and_emit(x, thx)
    sigma_y_m, emit_geom_y_m = _waist_sigma_and_emit(y, thy)

    gamma0 = float(np.mean(gamma))
    sigma_gamma = float(np.std(gamma))
    kinetic_energy_eV = (gamma0 - 1.0) * MEC2_EV
    rel_energy_spread_rms = (sigma_gamma * MEC2_EV) / kinetic_energy_eV

    beta0 = (1.0 - 1.0 / gamma0**2) ** 0.5
    sigma_t_s = float(np.std(z)) / (beta0 * C_LIGHT)

    bunch_charge = bunch.N_e * E_CHARGE

    # Estimate sigma_pz from gamma spread (good approximation for ultra-relativistic beams)
    # For ultra-relativistic: pz ≈ gamma, so sigma_pz / pz_mean ≈ sigma_gamma / gamma0
    sigma_pz = sigma_gamma / gamma0 if gamma0 > 0 else 0.0

    return GaussianElectronBeam(
        bunch_charge_C=_pq(bunch_charge, "coulomb", PhysicalMeaning.BUNCH_CHARGE),
        kinetic_energy_eV=_pq(kinetic_energy_eV, "electron_volt", PhysicalMeaning.BEAM_ENERGY),
        rel_energy_spread_rms=rel_energy_spread_rms,
        sigma_x_m=_pq(sigma_x_m, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE, _BEAM_WIDTH_CONV),
        sigma_y_m=_pq(sigma_y_m, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE, _BEAM_WIDTH_CONV),
        emit_geom_x_m=_pq(emit_geom_x_m, "meter", PhysicalMeaning.EMITTANCE),
        emit_geom_y_m=_pq(emit_geom_y_m, "meter", PhysicalMeaning.EMITTANCE),
        sigma_t_s=_pq(sigma_t_s, "second", PhysicalMeaning.BUNCH_LENGTH, _BUNCH_LEN_CONV),
        sigma_pz=sigma_pz,
    )
