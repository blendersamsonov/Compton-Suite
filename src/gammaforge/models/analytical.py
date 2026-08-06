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
only implementation.

``angle_integrated_spectrum`` used to be a macroparticle-based Monte
Carlo sum (an independent copy of ``xigma_i.spectrum_from_particles.
angle_integrated_spectrum``'s formula, summed over real per-particle
``gamma``/``weight`` samples) -- the ONLY place this supposedly "closed
form, no per-particle Monte Carlo" model actually touched macroparticles
at all, and the reason its cost scaled with ``n_particles`` (root cause
of a real user-reported 76.3 GiB allocation: 5,000,000 particles x 2048
energy bins in the always-on preview that runs alongside whichever main
model is selected on every Calculate click). Replaced with a fixed-size
numerical quadrature over the beam's own (assumed Gaussian) energy
distribution (``gamma0``, ``sigma_gamma``) -- the exact same per-particle
Compton-edge kinematic formula, integrated against the *known* analytic
distribution instead of Monte-Carlo-summed over samples of it. Now
``O(n_quad * len(s))`` with ``n_quad`` a small fixed constant,
independent of ``n_particles`` by construction -- matching
``estimate_yield``/``estimate_spectrum_width``, which already never
touched a macroparticle. No chunking machinery needed any more (removed
along with the macroparticle dependency): the largest array this module
ever builds is a few hundred quadrature points x a few thousand energy
bins, trivially small regardless of how many electrons the beam has.

LINEAR POLARIZATION ONLY, same caveat as ``gammaforge.io.laser.GaussianParaxialLaser
.a0_at()`` -- these formulas use ``pulse.a0_interaction``, so they inherit
that limitation unchanged.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.special import erfcx

from gammaforge.io.bunch import GaussianElectronBeam
from gammaforge.io.laser import GaussianParaxialLaser
from gammaforge.io.photons import PhasespaceSlice
from gammaforge.io.units import C_LIGHT, E_CHARGE, HBAR, SIGMA_T_M2
from gammaforge.models.api import AXIS_ENERGY, Job, Results, find_slice_request

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


def angle_integrated_spectrum(gamma0: float, sigma_gamma: float, N_e: float, s, n_quad: int = 401):
    """dN/ds integrated over all emission solid angle AND over the beam's
    own (assumed Gaussian) energy distribution -- a genuinely closed-form
    computation, no macroparticles: the standard angle-independent
    Compton-edge kinematic shape, evaluated on a fixed quadrature grid over
    gamma and weighted by the beam's Gaussian energy density, replacing
    what used to be a Monte Carlo sum over real per-particle gamma samples
    (see module docstring for why that made this "closed form" model's
    cost scale with n_particles -- root cause of a real user-reported OOM).

    ``gamma0``, ``sigma_gamma``: the beam's mean/rms Lorentz factor
    (``GaussianElectronBeam.gamma0``/``.sigma_gamma``). ``N_e``: total
    electron count (``GaussianElectronBeam.N_e``), scaling the output to
    an absolute photon count -- the quadrature weights integrate to 1
    (to quadrature precision) over the sampled +-6 sigma_gamma window, so
    multiplying by N_e reproduces the same total-weight convention the
    previous per-particle-weighted sum used (weights summing to N_e).
    ``s``: scalar or 1D array of normalized photon energies (``s = E /
    E_max``, ``E_max = 4*gamma**2*E_laser`` at the Compton edge). Returns
    an array matching ``s``'s shape (or a scalar if ``s`` was scalar).

    ``n_quad``: number of quadrature points spanning +-6 sigma_gamma
    around gamma0 -- independent of n_particles by construction, so a
    generous default costs nothing: the resulting (n_quad, len(s)) array
    is at most a few hundred x a few thousand elements, orders of
    magnitude smaller than the (n_particles, len(s)) broadcast this
    replaces (n_particles can be millions).
    """
    s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))

    span = 6.0 * sigma_gamma
    gamma_grid = np.linspace(max(gamma0 - span, 1.0 + 1e-9), gamma0 + span, n_quad)
    pdf = np.exp(-0.5 * ((gamma_grid - gamma0) / sigma_gamma) ** 2) / (sigma_gamma * np.sqrt(2.0 * np.pi))
    quad_weight = pdf * np.gradient(gamma_grid) * N_e

    gamma2 = (gamma_grid ** 2)[:, None]
    y = s_arr[None, :] / gamma2
    shape = 1.5 * (1.0 - 2.0 * y * (1.0 - y))
    shape = np.where((y < 0) | (y > 1), 0.0, shape)
    out = np.sum(quad_weight[:, None] * shape / gamma2, axis=0)

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

    def run(self, job: Job) -> Results:
        self.theta_col_rad = float(job.extra.get("theta_col_rad", self.theta_col_rad))

        # beam is the exact analytic beam description the (unused here)
        # macroparticles were sampled from -- electron sampling is the
        # caller's job, not this adapter's (see module docstring); this
        # model needs only beam's Gaussian parameters (gamma0, sigma_gamma,
        # N_e, ...), never the macroparticle array itself.
        beam = job.interaction.electrons.gaussian_fit
        self._last_beam = beam
        pulse = job.interaction.laser

        if pulse.crossing_angle != 0.0:
            warnings.warn(
                "analytical.Adapter is head-on-only; ignoring nonzero "
                f"crossing_angle ({pulse.crossing_angle} rad)."
            )

        total_yield = float(estimate_yield(beam, pulse))
        width = float(estimate_spectrum_width(beam, pulse, self.theta_col_rad))

        photon_slices = [PhasespaceSlice(axes={}, distr=np.asarray(total_yield))]

        energy_req = find_slice_request(job.output.slices, AXIS_ENERGY)
        if energy_req is not None:
            omega0 = pulse.omega0.to("1 / second").magnitude
            Wph_eV = HBAR * omega0 / E_CHARGE
            n_bins = energy_req.bins[0]
            s_grid = np.linspace(1e-3, 1.0 - 1e-3, n_bins)
            dNds = angle_integrated_spectrum(beam.gamma0, beam.sigma_gamma, beam.N_e, s_grid)
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

            photon_slices.append(PhasespaceSlice(axes={AXIS_ENERGY: E_eV}, distr=dNdE_per_eV))

        return Results(
            photon_slices=photon_slices,
            model_specific={"estimated_spectrum_width_fwhm": width, "gamma0": beam.gamma0,
                            "N_e": beam.N_e, "n_photons": pulse.n_photons, "a0_interaction": pulse.a0_interaction},
        )
