"""Electron-bunch representations.

Two distinct types, matching the distinction between raw data and its
analytic description:

* :class:`Bunch` -- raw, engine-agnostic macroparticle arrays (flat
  ``x``/``y``/``z``/``thx``/``thy``/``gamma``/``weight``), e.g. as loaded
  from an elegant ``.ele`` file, at whatever slice the tracking code
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

A third type, :class:`BeamFittedParams`, is the *output* of fitting a
:class:`Bunch` back to a structured Gaussian model (:func:`fit_beam_full`):
Twiss parameters, chirp, dispersion, and fit-quality diagnostics -- kept
separate from :class:`GaussianElectronBeam` since it carries fit-derived
information (beta/alpha, fit quality) that an *input* beam description
never needs.

Every physical parameter on :class:`GaussianElectronBeam` travels as a
:class:`PhysicalQuantity` (never a bare float) -- models extract the value
in whatever unit they need internally. :class:`Bunch` and
:class:`BeamFittedParams` are plain SI floats/arrays throughout: they are
either raw simulation data or fit diagnostics, neither of which carries the
RMS-vs-FWHM-vs-1/e^2 convention ambiguity a GUI-facing input does.

Key functions:

* :func:`sample_gaussian_canonical` / :func:`sample_gaussian_bunch` --
  sample macroparticles from a beam description using canonical variables
  (x, y, z, thx, thy, gamma) with mass-shell enforcement for physically
  consistent particles.
* :func:`drift` -- propagate a bunch in vacuum over a distance L, naturally
  producing Twiss tilt from waist sampling.
* :func:`fit_beam_full` -- fit a structured Gaussian model with physical
  correlations (Twiss, chirp, dispersion) and quality metrics.
* :func:`evaluate_fit_quality` -- evaluate fit quality using Mahalanobis
  distance, KS statistics, and log-likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterator

import numpy as np

from .units import (
    C_LIGHT,
    E_CHARGE,
    MEC2_EV,
    NoConvention,
    PhysicalMeaning,
    PhysicalQuantity,
    TimeConvention,
    WidthConvention,
)

__all__ = [
    "Bunch", "GaussianElectronBeam", "BeamFittedParams", "validate",
    "sample_gaussian_bunch", "sample_gaussian_canonical", "drift", "propagate", "stream",
    "fit_gaussian", "fit_beam_full", "evaluate_fit_quality",
    "beam_from_shared_fields", "beta_star_from_sigma_emit",
    "divergence_from_sigma_emit", "sigma_from_emittance",
    "ballistic_position_simultaneous", "ballistic_position_z0_reference",
    "laser_overlap_time_window",
]

# ---------------------------------------------------------------------------
# Internal helper: build a PhysicalQuantity shorthand.
# ---------------------------------------------------------------------------
_BEAM_WIDTH_CONV = WidthConvention.SIGMA_INTENSITY_RMS
_BUNCH_LEN_CONV = TimeConvention.SIGMA_INTENSITY_RMS


def _pq(value: float, unit: str, meaning: PhysicalMeaning, convention=None) -> PhysicalQuantity:
    """Shortcut to build a PhysicalQuantity."""
    return PhysicalQuantity(value, unit, meaning, convention)


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

    All inputs and the output are SI. Returns 0.0 for non-physical inputs
    (negative emittance, non-positive beta/gamma) rather than raising, so
    the GUI can display "--" gracefully.
    """
    if emit_norm_m is None or beta_m is None or gamma is None:
        return 0.0
    if emit_norm_m < 0 or beta_m <= 0 or gamma <= 0:
        return 0.0
    emit_geom = emit_norm_m / gamma
    return float(np.sqrt(emit_geom * beta_m))


# ---------------------------------------------------------------------------
# Raw macroparticle bunch
# ---------------------------------------------------------------------------
@dataclass
class Bunch:
    """Macroparticle electron bunch. SI units, flat arrays.

    ``x``/``y``/``z`` are the transverse/longitudinal position at whatever
    slice the data was produced at (head-tail sign for ``z``); ``thx``/
    ``thy`` are momentum angles ``px/pz``, ``py/pz`` -- not positions.
    ``gamma`` is the per-particle Lorentz factor. ``weight`` is
    electrons-per-macroparticle, uniform across the bunch.
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
    def n_electrons(self) -> float:
        """Total number of physical electrons."""
        return self.weight * self.n_particles


# ---------------------------------------------------------------------------
# Analytic Gaussian beam description (the gaussian_6d_waist v0.1 contract)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GaussianElectronBeam:
    """The ``gaussian_6d_waist`` v0.1 I/O contract.

    A 6D factorized Gaussian defined at the beam waist (``alpha_x =
    alpha_y = 0`` there), waist located at the interaction point ``z=0``.
    Every physical field is a :class:`PhysicalQuantity`; ``rel_energy_spread_rms``
    and ``sigma_pz`` are plain floats because they are dimensionless relative
    values (unitless), not physical quantities.

    Optional correlations, consumed by :func:`sample_gaussian_canonical`:
    - ``chirp_h``: longitudinal chirp dγ/dz (1/m). Controls how energy
      varies along the bunch. Default 0 (no chirp).
    - ``dispersion_x``, ``dispersion_y``: x-γ / y-γ dispersion
      (dimensionless). Controls correlation between transverse position
      and energy. Default 0 (no dispersion).
    """

    bunch_charge_C: PhysicalQuantity
    kinetic_energy_eV: PhysicalQuantity
    rel_energy_spread_rms: float
    sigma_x_m: PhysicalQuantity
    sigma_y_m: PhysicalQuantity
    emit_geom_x_m: PhysicalQuantity
    emit_geom_y_m: PhysicalQuantity
    sigma_t_s: PhysicalQuantity
    sigma_pz: float
    chirp_h: float = 0.0
    dispersion_x: float = 0.0
    dispersion_y: float = 0.0

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

    @property
    def chirp_correlation(self) -> float:
        """Correlation coefficient ρ_zγ = chirp_h · σ_z / σ_γ."""
        sg = self.sigma_gamma
        sz = self.sigma_z_m
        if sg <= 0 or sz <= 0:
            return 0.0
        return self.chirp_h * sz / sg

    @property
    def dispersion_x_correlation(self) -> float:
        """Correlation coefficient ρ_xγ = D_x · σ_γ / σ_x."""
        sg = self.sigma_gamma
        sx = self._sx_m
        if sg <= 0 or sx <= 0:
            return 0.0
        return self.dispersion_x * sg / sx

    @property
    def dispersion_y_correlation(self) -> float:
        """Correlation coefficient ρ_yγ = D_y · σ_γ / σ_y."""
        sg = self.sigma_gamma
        sy = self._sy_m
        if sg <= 0 or sy <= 0:
            return 0.0
        return self.dispersion_y * sg / sy


def validate(beam: GaussianElectronBeam) -> list[str]:
    """Validate a :class:`GaussianElectronBeam`.

    Raises ``ValueError`` on hard requirements (non-positive physical
    quantities); returns a list of warning strings for soft checks (large
    spread/divergence, suspicious unit mix-ups).
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

    if beam.chirp_h != 0.0:
        rho_z = beam.chirp_correlation
        if abs(rho_z) > 0.9:
            warnings.append(
                f"Very strong z-γ chirp correlation (ρ_zγ = {rho_z:.2g}); "
                f"conditional energy variance will be very small."
            )
    if beam.dispersion_x != 0.0 or beam.dispersion_y != 0.0:
        rho_x = beam.dispersion_x_correlation
        rho_y = beam.dispersion_y_correlation
        if abs(rho_x) > 0.9:
            warnings.append(
                f"Very strong x-γ dispersion correlation (ρ_xγ = {rho_x:.2g})."
            )
        if abs(rho_y) > 0.9:
            warnings.append(
                f"Very strong y-γ dispersion correlation (ρ_yγ = {rho_y:.2g})."
            )
    total_corr = beam.chirp_correlation ** 2 + beam.dispersion_x_correlation ** 2 + beam.dispersion_y_correlation ** 2
    if total_corr >= 0.95:
        warnings.append(
            f"Combined correlation strength ρ²_total = {total_corr:.2g} is close to 1; "
            f"sampling may produce very small conditional energy variance."
        )

    return warnings


# ---------------------------------------------------------------------------
# Fit output: BeamFittedParams
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BeamFittedParams:
    """The output of :func:`fit_beam_full`: a structured Gaussian fit to a
    macroparticle :class:`Bunch`, with physical correlations. Plain SI
    floats throughout -- this is fit-derived diagnostic output, not a
    GUI-facing input, so it carries no width/duration convention ambiguity."""

    sigma_x_waist: float
    sigma_y_waist: float
    emit_geom_x: float
    emit_geom_y: float
    beta_x: float
    beta_y: float
    alpha_x: float  # Should be ~0 at waist
    alpha_y: float
    sigma_z: float
    sigma_pz: float
    sigma_gamma: float
    chirp: float          # Slope dγ/dz (1/m)
    D_x: float             # x-γ dispersion (dimensionless)
    D_y: float             # y-γ dispersion (dimensionless)
    fit_quality: dict


def evaluate_fit_quality(
    bunch: Bunch,
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

    X = np.stack([bunch.x, bunch.thx, bunch.y, bunch.thy, bunch.z, bunch.gamma], axis=1)
    Xc = X - mu

    try:
        inv = np.linalg.inv(Sigma)
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            raise np.linalg.LinAlgError("Singular covariance")
    except np.linalg.LinAlgError:
        Sigma_reg = Sigma + 1e-10 * np.eye(Sigma.shape[0])
        inv = np.linalg.inv(Sigma_reg)
        sign, logdet = np.linalg.slogdet(Sigma_reg)

    d2_real = np.einsum("ni,ij,nj->n", Xc, inv, Xc)

    ks_real, _ = kstest(d2_real, chi2(df=6).cdf)
    mean_d2_real = np.mean(d2_real)

    k = X.shape[1]
    loglik_real = -0.5 * (k * np.log(2 * np.pi) + logdet + np.mean(d2_real))

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


def fit_beam_full(bunch: Bunch) -> BeamFittedParams:
    """Fit a structured Gaussian model with physical correlations.

    Extracts physically meaningful parameters from macroparticle data:
    - Transverse: Twiss alpha, beta, emittance (geometric)
    - Longitudinal: sigma_z, sigma_pz, chirp (slope dγ/dz)
    - Dispersion: D_x, D_y (x-γ, y-γ correlations)
    - Fit quality: Mahalanobis, KS, log-likelihood metrics

    Uses biased covariance for physics (population parameters).
    """
    X = np.stack([bunch.x, bunch.thx, bunch.y, bunch.thy, bunch.z, bunch.gamma], axis=1)

    mu = np.mean(X, axis=0)
    Xc = X - mu

    Sigma = np.cov(Xc, rowvar=False, bias=True)

    ix, ixp, iy, iyp, iz, ig = range(6)

    emit_x = np.sqrt(Sigma[ix, ix] * Sigma[ixp, ixp] - Sigma[ix, ixp]**2)
    emit_y = np.sqrt(Sigma[iy, iy] * Sigma[iyp, iyp] - Sigma[iy, iyp]**2)

    beta_x = Sigma[ix, ix] / emit_x
    alpha_x = -Sigma[ix, ixp] / emit_x
    beta_y = Sigma[iy, iy] / emit_y
    alpha_y = -Sigma[iy, iyp] / emit_y

    sigma_x_waist = emit_x / np.sqrt(Sigma[ixp, ixp])
    sigma_y_waist = emit_y / np.sqrt(Sigma[iyp, iyp])

    sigma_z = np.sqrt(Sigma[iz, iz])
    sigma_pz = np.sqrt(Sigma[ig, ig])  # This is sigma_gamma from fit
    sigma_gamma = sigma_pz

    chirp = Sigma[iz, ig] / Sigma[iz, iz] if Sigma[iz, iz] > 0 else 0.0

    D_x = Sigma[ix, ig] / Sigma[ig, ig] if Sigma[ig, ig] > 0 else 0.0
    D_y = Sigma[iy, ig] / Sigma[ig, ig] if Sigma[ig, ig] > 0 else 0.0

    fit_quality = evaluate_fit_quality(bunch, mu, Sigma)

    return BeamFittedParams(
        sigma_x_waist=float(sigma_x_waist),
        sigma_y_waist=float(sigma_y_waist),
        emit_geom_x=float(emit_x),
        emit_geom_y=float(emit_y),
        beta_x=float(beta_x),
        beta_y=float(beta_y),
        alpha_x=float(alpha_x),
        alpha_y=float(alpha_y),
        sigma_z=float(sigma_z),
        sigma_pz=float(sigma_pz),
        sigma_gamma=float(sigma_gamma),
        chirp=float(chirp),
        D_x=float(D_x),
        D_y=float(D_y),
        fit_quality=fit_quality,
    )


def fit_gaussian(bunch: Bunch) -> GaussianElectronBeam:
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
    Liouville invariant, unchanged by the drift.
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

    bunch_charge = bunch.n_electrons * E_CHARGE

    return GaussianElectronBeam(
        bunch_charge_C=_pq(bunch_charge, "coulomb", PhysicalMeaning.BUNCH_CHARGE, NoConvention.PLAIN),
        kinetic_energy_eV=_pq(kinetic_energy_eV, "electron_volt", PhysicalMeaning.BEAM_ENERGY, NoConvention.PLAIN),
        rel_energy_spread_rms=rel_energy_spread_rms,
        sigma_x_m=_pq(sigma_x_m, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE, _BEAM_WIDTH_CONV),
        sigma_y_m=_pq(sigma_y_m, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE, _BEAM_WIDTH_CONV),
        emit_geom_x_m=_pq(emit_geom_x_m, "meter", PhysicalMeaning.EMITTANCE, NoConvention.PLAIN),
        emit_geom_y_m=_pq(emit_geom_y_m, "meter", PhysicalMeaning.EMITTANCE, NoConvention.PLAIN),
        sigma_t_s=_pq(sigma_t_s, "second", PhysicalMeaning.BUNCH_LENGTH, _BUNCH_LEN_CONV),
        sigma_pz=sigma_gamma / gamma0 if gamma0 > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_gaussian_canonical(beam: GaussianElectronBeam, n_particles: int, *, rng=None) -> Bunch:
    """Draw macroparticles from a :class:`GaussianElectronBeam` using
    canonical variables (x, y, z, thx, thy, gamma) with mass-shell
    enforcement.

    x, y, z, thx, thy are independent Gaussians (sigma from the beam's
    transverse/longitudinal sizes and divergences); gamma is sampled
    conditionally on (x, y, z) via the beam's chirp/dispersion
    correlations. Downstream consumers that need canonical momenta derive
    them as ``pz = sqrt((gamma**2-1)/(1+thx**2+thy**2))``, ``px = thx*pz``,
    ``py = thy*pz`` -- since pz is *only* ever obtained this way (never
    independently sampled), ``gamma**2 = 1 + px**2 + py**2 + pz**2`` holds
    automatically by construction for every particle; there is no separate
    "mass-shell enforcement" step.

    Correlations (optional, from ``beam.chirp_h``, ``beam.dispersion_x``,
    ``beam.dispersion_y``):
    - Chirp h = dγ/dz (1/m): Cov[z, γ] = h · σ_z²
    - Dispersion D_x = Cov[x, γ] / σ_γ² (dimensionless)
    - Dispersion D_y = Cov[y, γ] / σ_γ²

    The conditional variance σ²_γ|xyz equals the unconditional variance
    σ²_γ minus the variance explained by the correlations, ensuring the
    total marginal variance of γ is always σ²_γ regardless of chirp/
    dispersion strength.

    Raises:
        ValueError: If chirp/dispersion are too strong (total correlation
            ρ_xγ² + ρ_yγ² + ρ_zγ² >= 1), which would make the conditional
            variance non-positive.
    """
    rng = np.random.default_rng() if rng is None else rng

    sx = beam._sx_m
    sy = beam._sy_m
    sz = beam.sigma_z_m
    g0 = beam.gamma0
    dx = beam.divergence_x_rad
    dy = beam.divergence_y_rad

    sigma_gamma = beam.sigma_gamma

    Dx = beam.dispersion_x
    Dy = beam.dispersion_y
    chirp = beam.chirp_h

    cov_xg = Dx * sigma_gamma ** 2
    cov_yg = Dy * sigma_gamma ** 2
    cov_zg = chirp * sz ** 2

    beta_x = cov_xg / sx ** 2 if sx > 0 else 0.0
    beta_y = cov_yg / sy ** 2 if sy > 0 else 0.0
    beta_z = cov_zg / sz ** 2 if sz > 0 else 0.0

    var_gamma_cond = (
        sigma_gamma ** 2
        - cov_xg ** 2 / sx ** 2
        - cov_yg ** 2 / sy ** 2
        - cov_zg ** 2 / sz ** 2
    )
    if var_gamma_cond < 0:
        raise ValueError(
            f"Chirp/dispersion too strong: conditional variance "
            f"sigma^2_gamma|xyz = {var_gamma_cond:.3g} < 0. "
            f"rho_zg = {beam.chirp_correlation:.3g}, "
            f"rho_xg = {beam.dispersion_x_correlation:.3g}, "
            f"rho_yg = {beam.dispersion_y_correlation:.3g}. "
            f"Need rho_xg^2 + rho_yg^2 + rho_zg^2 < 1."
        )
    sigma_gamma_cond = np.sqrt(var_gamma_cond)

    x = rng.normal(0.0, sx, n_particles)
    y = rng.normal(0.0, sy, n_particles)
    z = rng.normal(0.0, sz, n_particles)
    thx = rng.normal(0.0, dx, n_particles)
    thy = rng.normal(0.0, dy, n_particles)

    gamma_mean = g0 + beta_x * x + beta_y * y + beta_z * z
    gamma = rng.normal(gamma_mean, sigma_gamma_cond, n_particles)

    weight = beam.N_e / n_particles
    return Bunch(x=x, y=y, z=z, thx=thx, thy=thy, gamma=gamma, weight=weight)


def sample_gaussian_bunch(beam: GaussianElectronBeam, n_particles: int, *, rng=None) -> Bunch:
    """Draw macroparticles from ``beam``. Delegates to
    :func:`sample_gaussian_canonical` -- the canonical-variable sampler is
    the only sampler; kept as a separate name for call-site readability
    (``sample_gaussian_bunch`` at GUI/model boundaries,
    ``sample_gaussian_canonical`` when the canonical-variable framing
    matters)."""
    return sample_gaussian_canonical(beam, n_particles, rng=rng)


def beam_from_shared_fields(*, bunch_charge_C: float, kinetic_energy_eV: float,
                             rel_energy_spread_rms: float, sigma_x_m: float, sigma_y_m: float,
                             emit_geom_x_m: float, emit_geom_y_m: float, sigma_t_s: float,
                             sigma_pz: float, chirp_h: float = 0.0, dispersion_x: float = 0.0,
                             dispersion_y: float = 0.0) -> GaussianElectronBeam:
    """Build a :class:`GaussianElectronBeam` from flat SI fields a model's
    own ``Config``/``params_to_config`` already computes -- plain floats,
    not yet wrapped as :class:`PhysicalQuantity`."""
    return GaussianElectronBeam(
        bunch_charge_C=_pq(bunch_charge_C, "coulomb", PhysicalMeaning.BUNCH_CHARGE, NoConvention.PLAIN),
        kinetic_energy_eV=_pq(kinetic_energy_eV, "electron_volt", PhysicalMeaning.BEAM_ENERGY, NoConvention.PLAIN),
        rel_energy_spread_rms=rel_energy_spread_rms,
        sigma_x_m=_pq(sigma_x_m, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE, _BEAM_WIDTH_CONV),
        sigma_y_m=_pq(sigma_y_m, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE, _BEAM_WIDTH_CONV),
        emit_geom_x_m=_pq(emit_geom_x_m, "meter", PhysicalMeaning.EMITTANCE, NoConvention.PLAIN),
        emit_geom_y_m=_pq(emit_geom_y_m, "meter", PhysicalMeaning.EMITTANCE, NoConvention.PLAIN),
        sigma_t_s=_pq(sigma_t_s, "second", PhysicalMeaning.BUNCH_LENGTH, _BUNCH_LEN_CONV),
        sigma_pz=sigma_pz,
        chirp_h=chirp_h,
        dispersion_x=dispersion_x,
        dispersion_y=dispersion_y,
    )


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------
def drift(bunch: Bunch, L: float) -> Bunch:
    """Ballistically propagate a bunch by a longitudinal distance ``L`` (SI
    metres): ``x -> x + thx*L``, ``y -> y + thy*L``. ``z``/``thx``/``thy``/
    ``gamma`` unchanged (straight-line, no acceleration, no light-travel-time
    bookkeeping -- see :func:`propagate` for that). This is the mechanism
    that produces Twiss tilt (α ≠ 0) from waist sampling."""
    return Bunch(
        x=bunch.x + bunch.thx * L,
        y=bunch.y + bunch.thy * L,
        z=bunch.z,
        thx=bunch.thx,
        thy=bunch.thy,
        gamma=bunch.gamma,
        weight=bunch.weight,
    )


def ballistic_position_simultaneous(x0, y0, z0, thx, thy, dt):
    """Straight-line position at time offset ``dt`` later, given a
    per-particle reference ``(x0, y0, z0, thx, thy)`` that describes each
    particle's REAL, simultaneous position at ``dt=0`` (the
    :class:`~compton_suite.io.bunch.Bunch` convention). ``thx``/``thy`` are
    (small-angle) transverse velocity fractions, ``vz = sqrt(1 - thx**2 -
    thy**2)``. Returns ``(x, y, z)``, each the same shape as the broadcast
    of the inputs.
    """
    vz = (1.0 - thx**2 - thy**2) ** 0.5
    x = x0 + thx * vz * dt
    y = y0 + thy * vz * dt
    z = z0 + vz * dt
    return x, y, z


def ballistic_position_z0_reference(x0, y0, z0, thx, thy, t):
    """Straight-line position at time offset ``t``, given a per-particle
    reference where ``x0``/``y0`` are each particle's transverse position
    extrapolated to ``z=0`` (not necessarily its position at its own
    ``z0``) and ``z0`` is that particle's real longitudinal offset -- the
    ``xigma_i.particles`` convention. ``dt0 = z0 / vz`` is the time needed
    to travel from ``z=0`` to ``z0`` at this particle's own ``vz``, folded
    into the ``x``/``y`` evolution (``z`` itself already starts at ``z0``
    directly) so ``x(t)``/``y(t)`` continue that same line correctly.
    """
    vz = (1.0 - thx**2 - thy**2) ** 0.5
    dt0 = z0 / vz
    x = x0 + thx * (t + dt0)
    y = y0 + thy * (t + dt0)
    z = z0 + vz * t
    return x, y, z


def propagate(bunch: Bunch, dt) -> Bunch:
    """Ballistically drift every macroparticle in ``bunch`` by a time
    offset ``dt`` (SI seconds; scalar, or one value per particle).
    ``gamma``/``thx``/``thy`` are unchanged (straight-line, no
    acceleration) -- only ``x``/``y``/``z`` move. Thin :class:`Bunch`
    wrapper over :func:`ballistic_position_simultaneous`, for external SI
    callers (a GUI, a table-building script, tests) that want "this
    bunch's state at a different time" without hand-rolling the drift
    themselves.
    """
    x, y, z = ballistic_position_simultaneous(
        np.asarray(bunch.x, dtype=float), np.asarray(bunch.y, dtype=float),
        np.asarray(bunch.z, dtype=float), np.asarray(bunch.thx, dtype=float),
        np.asarray(bunch.thy, dtype=float), C_LIGHT * np.asarray(dt, dtype=float),
    )
    return replace(bunch, x=x, y=y, z=z)


def laser_overlap_time_window(z0, *, k0_las, sigma_lz, sigma_lr0,
                               beta_ff=0.0, gauss_width=3.0, lorentz_width=8.0, xp=np):
    """Per-particle time window ``[t0, t1]`` (normalised length units,
    e.g. ``k0_las*c*t``) bounding where a ballistic particle at
    longitudinal offset ``z0`` (same normalised units, e.g. ``k0_las*z``)
    is within ``lorentz_width`` Rayleigh ranges transversely and
    ``gauss_width`` pulse-duration Gaussian widths temporally of a
    Gaussian laser pulse centred at the origin -- i.e. the window worth
    integrating a beam-laser overlap over, outside of which the pulse
    envelope is negligible. Model-agnostic: any model doing a ballistic
    push through a Gaussian laser pulse needs the same bound.

    z0: per-particle longitudinal offset, normalised (e.g. ``k0_las * z_cm``
        for xigma_i's CGS/k0-normalised convention).
    k0_las: laser wavenumber (``2*pi/wavelength``) -- multiplies the raw
        ``sigma_lz``/``sigma_lr0`` below to put them in the same normalised
        unit as ``z0``.
    sigma_lz, sigma_lr0: the laser pulse's RMS duration and RMS focal
        radius, as raw lengths in whatever unit ``1/k0_las`` is (e.g. cm
        for xigma_i) -- *not* pre-normalised; this function does that.
    beta_ff: flying-focus factor (0 = static focus, 1 = co-moving) -- an
        engine-specific laser extra, passed through as a plain scalar.
    gauss_width, lorentz_width: how many pulse-duration Gaussian widths /
        Rayleigh-range Lorentzian widths out a trajectory is still
        considered "possibly inside the pulse" -- xigma_i's own defaults
        (3, 8) are passed explicitly by that caller; other models can pick
        their own.
    xp: array module ``z0`` belongs to (``numpy`` or ``cupy``) -- accepts
        an explicit module since this needs ``maximum``/``minimum``, which
        aren't expressible as a bare operator.
    """
    zT = k0_las * sigma_lz
    zR = (k0_las * sigma_lr0) ** 2 * (1.0 + beta_ff) * 2.0

    sigma_tau = gauss_width * zT
    sigma_rayleigh = lorentz_width * zR

    t0 = (xp.maximum(-sigma_tau, (-z0 * (1 + beta_ff) - 2 * sigma_rayleigh) / (1 - beta_ff)) - z0) / 2
    t1 = (xp.minimum(sigma_tau, (-z0 * (1 + beta_ff) + 2 * sigma_rayleigh) / (1 - beta_ff)) - z0) / 2
    return t0, t1


def stream(bunch: Bunch, t_grid) -> Iterator[Bunch]:
    """Yield a propagated :class:`Bunch` snapshot at each time in
    ``t_grid`` (SI seconds, monotonic) -- for a caller that wants distinct
    bunch-state samples at a sequence of time instances (e.g. building a
    time-resolved table one slice at a time) without hand-rolling the
    drift at each step itself. Each snapshot is independently computed
    from ``bunch`` (the original, not the previous snapshot), so ``t_grid``
    need not be evenly spaced and yields are exact, not accumulated.
    """
    for t in t_grid:
        yield propagate(bunch, t)
