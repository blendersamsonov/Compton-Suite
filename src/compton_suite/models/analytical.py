"""Fast, closed-form Compton-source physics: total yield, angle-integrated
spectrum, and an estimated collimated spectrum width -- no per-particle
Monte Carlo, no GPU. Feeds ``analytical_adapter.py``'s always-on GUI
preview, and is directly importable by any other model for its own quick
sanity check.

``estimate_yield``/``estimate_spectrum_width`` were originally ported
(re-parametrized off ``compton_suite.io.bunch.GaussianElectronBeam``/
``compton_suite.io.laser.GaussianParaxialLaser`` instead of a CGS
``CollisionParams`` instance -- SI throughout, otherwise unchanged) from
``xigma_i.config.CollisionParams.estimate_yield``/``estimate_spectrum_width``,
already documented there as "cheap analytic estimate, for sanity-checking
... not used by the real computation" -- i.e. already exactly this role.
That xigma_i pair has since been deleted outright (dead code, no other
caller) rather than kept as a re-export wrapper -- this module is now the
only implementation. ``angle_integrated_spectrum`` is an independent,
numpy-only copy of the same formula as ``xigma_i.spectrum_from_particles.
angle_integrated_spectrum`` (needs only per-particle ``gamma``/``weight``
arrays and an ``s`` grid, no table, no collision config) -- kept
independent rather than imported, since that xigma_i module is
production code for a different model with its own (optional cupy)
dispatch this fast-preview model doesn't need.

LINEAR POLARIZATION ONLY, same caveat as ``compton_suite.io.laser.GaussianParaxialLaser
.a0_at()`` -- these formulas use ``pulse.a0_interaction``, so they inherit
that limitation unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfcx

from compton_suite.io.bunch import Bunch, GaussianElectronBeam
from compton_suite.io.interaction import InteractionParameters
from compton_suite.io.laser import GaussianParaxialLaser
from compton_suite.io.photons import BinnedSpectrum, Photons
from compton_suite.io.units import C_LIGHT, E_CHARGE, HBAR, SIGMA_T_M2
from compton_suite.models.api import Job, ModelCapabilities

__all__ = ["estimate_yield", "estimate_spectrum_width", "angle_integrated_spectrum",
           "AnalyticalConfig", "Adapter"]


def estimate_yield(beam: GaussianElectronBeam, pulse: GaussianParaxialLaser) -> float:
    """Cheap analytic total-photon-yield estimate from an overlap integral
    between two Gaussian bunches, for sanity-checking against a real
    per-particle computation -- not a replacement for one.

    The laser's transverse profile is treated as round (radius
    ``sqrt(waist_rms_x_m * waist_rms_y_m)``, a geometric-mean effective
    size) since the underlying formula assumes a round beam; an
    elliptical laser is only approximated by this estimate, not modeled
    exactly.
    """
    sigma_ex, sigma_ey = beam._sx_m, beam._sy_m
    beta_x, beta_y = beam.beta_star_x_m, beam.beta_star_y_m
    sigma_ez = beam.sigma_z_m
    sigma_lr0 = (pulse._wx_m * pulse._wy_m) ** 0.5
    sigma_lz = pulse._dur_s * C_LIGHT
    lambda_l = pulse._wl_m

    sb_av = np.sqrt(sigma_ex * sigma_ey / beta_x / beta_y)
    sigma0 = np.sqrt(sigma_ex**2 + sigma_lr0**2)
    nu = (
        np.sqrt(2) * sigma0 / np.sqrt(sigma_ez**2 + sigma_lz**2)
        / np.sqrt(sb_av**2 + lambda_l**2 / np.pi**2 / sigma_lr0**2)
    )
    return beam.N_e * pulse.n_photons * SIGMA_T_M2 / 2 / np.sqrt(np.pi) / sigma0**2 * nu * erfcx(nu)


def estimate_spectrum_width(beam: GaussianElectronBeam, pulse: GaussianParaxialLaser,
                             theta_col: float) -> float:
    """Cheap analytic estimate of the collimated-spectrum FWHM (in units of
    the Compton edge, dimensionless), combining angular-collimation,
    angular-divergence, energy-spread, and ponderomotive-broadening terms
    in quadrature. For sanity-checking, not a replacement for a real
    per-particle computation.

    ``theta_col``: collimation half-angle (rad).
    """
    gamma0, sigma_gamma = beam.gamma0, beam.sigma_gamma
    emit_width = np.sqrt(beam.divergence_x_rad * beam.divergence_y_rad)
    a0 = pulse.a0_interaction
    return 0.5 * 2.355 * np.sqrt(
        (gamma0 * theta_col) ** 4 + (gamma0 * emit_width) ** 4
        + (sigma_gamma / gamma0) ** 2 + (0.5 * a0**2) ** 2
    )


def angle_integrated_spectrum(gamma: np.ndarray, particle_weight: np.ndarray, s):
    """dN/ds integrated over all emission solid angle, from the standard
    angle-independent Compton edge shape alone -- no table, no quadrature.

    ``gamma``, ``particle_weight``: 1D arrays, one entry per macroparticle
    (e.g. from ``compton_suite.io.bunch.sample_gaussian_bunch``). ``s``: scalar
    or 1D array of normalized photon energies (``s = E / E_max``, ``E_max
    = 4*gamma**2*E_laser`` at the Compton edge). Returns an array matching
    ``s``'s shape (or a scalar if ``s`` was scalar).
    """
    gamma = np.asarray(gamma)
    particle_weight = np.asarray(particle_weight)
    s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
    gamma_col = gamma[:, None]
    y = s_arr[None, :] / gamma_col**2
    shape = 1.5 * (1.0 - 2.0 * y * (1.0 - y))
    shape = np.where((y < 0) | (y > 1), 0.0, shape)
    out = np.sum(particle_weight[:, None] * shape / gamma_col**2, axis=0)
    return out if np.ndim(s) else out[0]

@dataclass
class AnalyticalConfig:
    """Numerics-only config for the analytical model: the shared (beam,
    laser) bundle plus the one analytical-specific numeric knob (the
    collimation half-angle used for the spectrum-width estimate)."""

    interaction: InteractionParameters
    theta_col_rad: float = 0.0


class Adapter:
    """Fast closed-form model: total yield, angle-integrated spectrum, and
    an estimated collimated-spectrum width -- no per-particle Monte Carlo.
    Meant to run alongside whichever model is actually selected, as an
    always-available real-time preview and base sanity check (see
    ``ModelCapabilities.is_fast_preview``)."""

    def __init__(self):
        self._last_beam: GaussianElectronBeam | None = None

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(display_name="Analytical", is_fast_preview=True,
                                  uses_shared_sample_count=False)

    def model_params(self) -> list[tuple[str, float, str]]:
        return [("Collimation half-angle (rad)", 0.0, "theta_col_rad")]

    def model_choices(self) -> dict[str, list[str]]:
        return {}

    def run(self, job: Job) -> Photons:
        cfg = AnalyticalConfig(
            interaction=job.interaction,
            theta_col_rad=float(job.extra.get("theta_col_rad", 0.0)),
        )
        electrons: Bunch = job.electrons
        # electrons is already a macroparticle sample of cfg.interaction.beam
        # (electron sampling is the caller's job, not this adapter's -- see
        # module docstring), so the exact analytic beam description is used
        # directly rather than re-fitting a noisier copy from electrons.
        beam = cfg.interaction.beam
        self._last_beam = beam
        pulse = cfg.interaction.laser

        total_yield = float(estimate_yield(beam, pulse))
        width = float(estimate_spectrum_width(beam, pulse, cfg.theta_col_rad))

        gamma_arr = np.asarray(electrons.gamma, dtype=float)
        weight_arr = np.full(electrons.n_particles, electrons.weight)

        omega0 = 2.0 * np.pi * C_LIGHT / pulse._wl_m
        Wph_eV = HBAR * omega0 / E_CHARGE
        n_bins = job.output.n_energy_bins
        s_grid = np.linspace(1e-3, 1.0 - 1e-3, n_bins)
        dNds = angle_integrated_spectrum(gamma_arr, weight_arr, s_grid)
        E_eV = 4.0 * beam.gamma0**2 * Wph_eV * s_grid
        dNdE_per_eV = dNds / (4.0 * Wph_eV)

        # QUICK FIX, FLAGGED FOR FUTURE INVESTIGATION: angle_integrated_spectrum
        # returns a per-electron kinematic SHAPE only -- it has no dependence
        # on the laser pulse (a0/n_photons) at all, so its raw absolute scale
        # has nothing to do with the pulse-energy-dependent total_yield
        # estimate_yield() actually computes (confirmed: this raw integral is
        # bit-identical across scenarios that only change pulse_energy_J).
        # Same self-consistent-rescale pattern applied to xigma-i/delta's
        # angular_spectrum vs total_yield mismatch: force the spectrum shape
        # to integrate to the trusted total_yield, rather than trust its own
        # absolute normalization.
        _raw_integral = float(np.trapezoid(dNdE_per_eV, E_eV))
        if _raw_integral > 0:
            dNdE_per_eV = dNdE_per_eV * (total_yield / _raw_integral)

        return Photons(
            model_name="analytical",
            cfg=cfg,
            n_mc=electrons.n_particles,
            total_yield=total_yield,
            spectrum=BinnedSpectrum(E_eV=E_eV, dNdE_per_eV=dNdE_per_eV),
            summary={"estimated_spectrum_width_fwhm": width, "gamma0": beam.gamma0,
                     "N_e": beam.N_e, "n_photons": pulse.n_photons, "a0_interaction": pulse.a0_interaction},
        )
