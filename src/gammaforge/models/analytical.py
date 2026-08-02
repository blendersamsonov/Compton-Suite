"""Fast, closed-form Compton-source physics: total yield, angle-integrated
spectrum, and an estimated collimated spectrum width -- no per-particle
Monte Carlo, no GPU. Feeds ``analytical_adapter.py``'s always-on GUI
preview, and is directly importable by any other model for its own quick
sanity check.

``estimate_yield``/``estimate_spectrum_width`` were originally ported
(re-parametrized off ``gammaforge.io.bunch.GaussianElectronBeam``/
``gammaforge.io.laser.GaussianParaxialLaser`` instead of a CGS
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

LINEAR POLARIZATION ONLY, same caveat as ``gammaforge.io.laser.GaussianParaxialLaser
.a0_at()`` -- these formulas use ``pulse.a0_interaction``, so they inherit
that limitation unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfcx

from gammaforge.io.bunch import Bunch, GaussianElectronBeam
from gammaforge.io.laser import GaussianParaxialLaser
from gammaforge.io.photons import BinnedSpectrum, Photons
from gammaforge.io.units import C_LIGHT, E_CHARGE, HBAR, SIGMA_T_M2
from gammaforge.models.api import Job

__all__ = ["estimate_yield", "estimate_spectrum_width", "angle_integrated_spectrum", "Adapter"]


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
    sigma_ex = beam.sigma_x_m.to_unit("meter").magnitude
    sigma_ey = beam.sigma_y_m.to_unit("meter").magnitude
    beta_x = beam.beta_star_x.to("meter").magnitude
    beta_y = beam.beta_star_y.to("meter").magnitude
    sigma_ez = beam.sigma_z.to("meter").magnitude
    sigma_lr0 = (pulse.waist_rms_x_m.to_unit("meter").magnitude
                 * pulse.waist_rms_y_m.to_unit("meter").magnitude) ** 0.5
    sigma_lz = pulse.duration_rms_s.to_unit("second").magnitude * C_LIGHT
    lambda_l = pulse.wavelength_m.to_unit("meter").magnitude

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
    (e.g. from ``gammaforge.io.bunch.sample_gaussian_bunch``). ``s``: scalar
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

class Adapter:
    """Fast closed-form model: total yield, angle-integrated spectrum, and
    an estimated collimated-spectrum width -- no per-particle Monte Carlo.
    Meant to run alongside whichever model is actually selected, as an
    always-available real-time preview and base sanity check -- the GUI
    hardcodes this (``self.analytical_adapter``), not a metadata flag. No
    standalone Config class -- ``theta_col_rad`` (its one numeric knob)
    lives directly on this adapter."""

    def __init__(self):
        self.theta_col_rad: float = 0.0
        self._last_beam: GaussianElectronBeam | None = None

    def model_params(self) -> list[tuple[str, float, str]]:
        return [("Collimation half-angle (rad)", self.theta_col_rad, "theta_col_rad")]

    def model_choices(self) -> dict[str, list[str]]:
        return {}

    def run(self, job: Job) -> Photons:
        self.theta_col_rad = float(job.extra.get("theta_col_rad", self.theta_col_rad))

        electrons: Bunch = job.interaction.electrons
        # electrons.gaussian_fit is the exact analytic beam description
        # electrons was sampled from (electron sampling is the caller's job,
        # not this adapter's -- see module docstring), used directly rather
        # than re-fitting a noisier copy from the macroparticles.
        beam = job.interaction.electrons.gaussian_fit
        self._last_beam = beam
        pulse = job.interaction.laser

        total_yield = float(estimate_yield(beam, pulse))
        width = float(estimate_spectrum_width(beam, pulse, self.theta_col_rad))

        gamma_arr = np.asarray(electrons.gamma, dtype=float)
        weight_arr = np.full(electrons.n_particles, electrons.weight)

        omega0 = pulse.omega0.to("1 / second").magnitude
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
            n_mc=electrons.n_particles,
            total_yield=total_yield,
            spectrum=BinnedSpectrum(E_eV=E_eV, dNdE_per_eV=dNdE_per_eV),
            summary={"estimated_spectrum_width_fwhm": width, "gamma0": beam.gamma0,
                     "N_e": beam.N_e, "n_photons": pulse.n_photons, "a0_interaction": pulse.a0_interaction},
        )
