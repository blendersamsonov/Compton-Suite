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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import C_LIGHT, E_CHARGE, MEC2_EV

__all__ = ["MacroBunch", "GaussianElectronBeam", "validate", "sample_gaussian_bunch", "fit_gaussian",
           "beam_from_shared_fields"]


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
    Field units are SI; ``rel_energy_spread_rms`` is relative to *kinetic*
    energy, not gamma (spec Sec. 10) -- ``sigma_gamma`` below converts.
    """

    bunch_charge_C: float
    kinetic_energy_eV: float
    rel_energy_spread_rms: float
    sigma_x_m: float
    sigma_y_m: float
    emit_geom_x_m: float
    emit_geom_y_m: float
    sigma_t_s: float

    @property
    def N_e(self) -> float:
        return self.bunch_charge_C / E_CHARGE

    @property
    def total_energy_eV(self) -> float:
        return self.kinetic_energy_eV + MEC2_EV

    @property
    def gamma0(self) -> float:
        return self.total_energy_eV / MEC2_EV

    @property
    def beta0(self) -> float:
        g = self.gamma0
        return (1.0 - 1.0 / g**2) ** 0.5

    @property
    def sigma_E_kin_eV(self) -> float:
        return self.rel_energy_spread_rms * self.kinetic_energy_eV

    @property
    def sigma_gamma(self) -> float:
        return self.sigma_E_kin_eV / MEC2_EV

    @property
    def sigma_gamma_over_gamma0(self) -> float:
        return self.sigma_gamma / self.gamma0

    @property
    def sigma_z_m(self) -> float:
        return self.beta0 * C_LIGHT * self.sigma_t_s

    @property
    def divergence_x_rad(self) -> float:
        return self.emit_geom_x_m / self.sigma_x_m

    @property
    def divergence_y_rad(self) -> float:
        return self.emit_geom_y_m / self.sigma_y_m

    @property
    def beta_star_x_m(self) -> float:
        return self.sigma_x_m**2 / self.emit_geom_x_m

    @property
    def beta_star_y_m(self) -> float:
        return self.sigma_y_m**2 / self.emit_geom_y_m

    @property
    def emit_norm_x_m(self) -> float:
        return self.beta0 * self.gamma0 * self.emit_geom_x_m

    @property
    def emit_norm_y_m(self) -> float:
        return self.beta0 * self.gamma0 * self.emit_geom_y_m

    @property
    def peak_current_A(self) -> float:
        return self.bunch_charge_C / ((2.0 * np.pi) ** 0.5 * self.sigma_t_s)

    @property
    def peak_density_m3(self) -> float:
        return self.N_e / (
            (2.0 * np.pi) ** 1.5 * self.sigma_x_m * self.sigma_y_m * self.sigma_z_m
        )


def validate(beam: GaussianElectronBeam) -> list[str]:
    """Validate a :class:`GaussianElectronBeam` per spec Sec. 16.

    Raises ``ValueError`` on the spec's hard requirements (non-positive
    physical quantities); returns a list of warning strings for the
    spec's soft checks (large spread/divergence, suspicious unit mix-ups).
    """
    if beam.bunch_charge_C <= 0:
        raise ValueError("GaussianElectronBeam: bunch_charge_C must be > 0")
    if beam.kinetic_energy_eV <= 0:
        raise ValueError("GaussianElectronBeam: kinetic_energy_eV must be > 0")
    if beam.rel_energy_spread_rms < 0:
        raise ValueError("GaussianElectronBeam: rel_energy_spread_rms must be >= 0")
    if beam.sigma_x_m <= 0 or beam.sigma_y_m <= 0:
        raise ValueError("GaussianElectronBeam: sigma_x_m/sigma_y_m must be > 0")
    if beam.emit_geom_x_m <= 0 or beam.emit_geom_y_m <= 0:
        raise ValueError("GaussianElectronBeam: emit_geom_x_m/emit_geom_y_m must be > 0")
    if beam.sigma_t_s <= 0:
        raise ValueError("GaussianElectronBeam: sigma_t_s must be > 0")

    warnings: list[str] = []
    if beam.rel_energy_spread_rms > 0.1:
        warnings.append("Large relative energy spread; check whether this is intended.")
    if beam.divergence_x_rad * 1e3 > 100 or beam.divergence_y_rad * 1e3 > 100:
        warnings.append("Large angular divergence; paraxial approximation may be questionable.")
    if beam.emit_geom_x_m > beam.sigma_x_m or beam.emit_geom_y_m > beam.sigma_y_m:
        warnings.append(
            "Check units: geometric emittance is larger than the beam size -- "
            "emittance is a length*angle, not a length; this usually indicates "
            "a units mix-up (e.g. mm*mrad entered as m*rad)."
        )
    if beam.bunch_charge_C > 1.0e-8:
        warnings.append("Very large bunch charge (> 10 nC); check pC/nC conversion is correct.")
    return warnings


def sample_gaussian_bunch(beam: GaussianElectronBeam, n_particles: int, *, rng=None) -> MacroBunch:
    """Draw macroparticles from a :class:`GaussianElectronBeam`.

    Independent factorized Gaussians per spec Sec. 13, defined at the beam
    waist (``z=0``): ``x``/``thx`` and ``y``/``thy`` uncorrelated
    (``alpha=0`` at the waist), no chirp (``<z, delta_E> = 0``).
    """
    rng = np.random.default_rng() if rng is None else rng

    x = rng.normal(0.0, beam.sigma_x_m, n_particles)
    thx = rng.normal(0.0, beam.divergence_x_rad, n_particles)
    y = rng.normal(0.0, beam.sigma_y_m, n_particles)
    thy = rng.normal(0.0, beam.divergence_y_rad, n_particles)

    t = rng.normal(0.0, beam.sigma_t_s, n_particles)
    z = beam.beta0 * C_LIGHT * t
    gamma = beam.gamma0 + rng.normal(0.0, beam.sigma_gamma, n_particles)

    weight = beam.N_e / n_particles
    return MacroBunch(x=x, y=y, z=z, thx=thx, thy=thy, gamma=gamma, weight=weight,
                       meta={"source": "sample_gaussian_bunch", "beam": beam})


def beam_from_shared_fields(*, eps0: float, sigma_eps_rel: float, emit_x: float, emit_y: float,
                             sigma0_x: float, sigma0_y: float, sigma_par_e: float,
                             N_e: float) -> GaussianElectronBeam:
    """Build a :class:`GaussianElectronBeam` from the flat SI field set every
    model's own ``Config`` already derives and agrees on (``eps0``,
    ``sigma_eps_rel``, ``emit_x/y``, ``sigma0_x/y``, ``sigma_par_e``, ``N_e``
    -- see ``ComptonSuite/validation/scenarios.py``'s ``scenario_to_shared_
    fields``, this function's exact inverse).

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
    """
    gamma0 = eps0
    kinetic_energy_eV = (gamma0 - 1.0) * MEC2_EV
    sigma_gamma = sigma_eps_rel * gamma0
    rel_energy_spread_rms = (sigma_gamma * MEC2_EV) / kinetic_energy_eV
    beta0 = (1.0 - 1.0 / gamma0**2) ** 0.5

    return GaussianElectronBeam(
        bunch_charge_C=N_e * E_CHARGE,
        kinetic_energy_eV=kinetic_energy_eV,
        rel_energy_spread_rms=rel_energy_spread_rms,
        sigma_x_m=sigma0_x,
        sigma_y_m=sigma0_y,
        emit_geom_x_m=emit_x,
        emit_geom_y_m=emit_y,
        sigma_t_s=sigma_par_e / (beta0 * C_LIGHT),
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

    bunch_charge_C = bunch.N_e * E_CHARGE

    return GaussianElectronBeam(
        bunch_charge_C=bunch_charge_C,
        kinetic_energy_eV=kinetic_energy_eV,
        rel_energy_spread_rms=rel_energy_spread_rms,
        sigma_x_m=sigma_x_m,
        sigma_y_m=sigma_y_m,
        emit_geom_x_m=emit_geom_x_m,
        emit_geom_y_m=emit_geom_y_m,
        sigma_t_s=sigma_t_s,
    )
