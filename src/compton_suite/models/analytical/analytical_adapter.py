"""ModelAdapter for the fast, always-on analytical model.

Duck-typed against ``compton_suite.gui.model_api`` (never imports it) --
same decoupling discipline ``xigma_i.gui_adapter`` already uses, now made
easy because the actual result types (``compton_suite.io.photons.*``) are a
shared dependency both this module and ``compton_suite.gui`` import directly,
not something either side has to define a lookalike of.

No physics lives in ``compton_suite.gui`` -- this adapter is pure GUI-field
parsing glue around ``compton_suite.analytical``'s plain functions and
``compton_suite.io.bunch``/``compton_suite.io.laser``'s representations, all of which
are independently importable and usable outside the GUI entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compton_suite.io.bunch import GaussianElectronBeam, MacroBunch, fit_gaussian
from compton_suite.io.constants import C_LIGHT, E_CHARGE, HBAR
from compton_suite.io.io_formats.sdds import load_elegant_ele
from compton_suite.io.photons import BinnedSpectrum
from compton_suite.io import (
    PhysicalMeaning as _PhysicalMeaning,
    PhysicalQuantity as _PhysicalQuantity,
    TimeConvention as _TimeConvention,
    adapt_to_model as _adapt_to_model,
    params_to_floats as _params_to_floats,
)
from compton_suite.io.bunch import beam_from_shared_fields as _beam_from_shared_fields
from compton_suite.io.laser import laser_from_shared_fields as _laser_from_shared_fields

from . import analytical

# Analytical shares kascade's duration conventions (same raw GUI fields,
# same sigma_par_e/sigma_par_L meaning) -- reuse its ModelSpec rather than
# duplicating a spec that would be byte-identical.
from compton_suite.models.kascade.params.spec import KASCADE_SPEC as _KASCADE_SPEC
_DURATION_SPEC = {k: _KASCADE_SPEC[k] for k in ("sigma_par_e", "sigma_par_L")}


class ParamError(Exception):
    pass


def _float(fields: dict, key: str) -> float:
    try:
        return float(fields[key].get())
    except ValueError:
        raise ParamError(f"'{key}' is not a valid number: {fields[key].get()!r}")


@dataclass
class AnalyticalConfig:
    """Wraps beam/pulse, but ALSO exposes a handful of flat SI fields every
    other model's Config carries (eps0, beta_x/y, lambda_L) -- these are
    read by validation/tier0_wiring.py's cross-model wiring/units sanity
    checks (gamma0/beta_x/beta_y/lambda_L must agree exactly across all
    four models), not by this model's own run() (which reads .beam/.pulse
    directly)."""

    beam: GaussianElectronBeam
    pulse: GaussianParaxialLaser
    theta_col_rad: float
    crossing_angle: float = 0.0
    quantum: bool = False

    @property
    def eps0(self) -> float:
        return self.beam.gamma0

    @property
    def beta_x(self) -> float:
        return self.beam.beta_star_x_m

    @property
    def beta_y(self) -> float:
        return self.beam.beta_star_y_m

    @property
    def lambda_L(self) -> float:
        return self.pulse.wavelength_m


class AnalyticalAdapter:
    """Fast closed-form model: total yield, angle-integrated spectrum, and
    an estimated collimated-spectrum width -- no per-particle Monte Carlo.
    Meant to run alongside whichever model is actually selected, as an
    always-available real-time preview and base sanity check (see
    ``ModelCapabilities.is_fast_preview``)."""

    def __init__(self):
        self._last_beam: GaussianElectronBeam | None = None

    def capabilities(self):
        from compton_suite.gui.model_api import ModelCapabilities
        return ModelCapabilities(
            name="analytical",
            display_name="Analytical (fast estimate)",
            requires_gpu=False,
            supports_crossing_angle=False,
            supports_quantum_toggle=False,
            supports_nonlinearity_emulation=False,
            supports_electron_final_state=False,
            supports_photon_multiplicity=False,
            supports_ele_file_io=True,
            supports_seed_reproducibility=False,
            requires_recompute_on_collimation_change=True,
            trust_level="experimental-analytical",
            trust_note="Closed-form Gaussian-overlap estimate (head-on, classical only -- "
                       "no crossing angle, no quantum recoil). Fast preview / base "
                       "validation, not a substitute for a real per-particle model.",
            is_fast_preview=True,
            uses_shared_sample_count=False,  # analytical model doesn't run electrons
        )

    def available(self) -> tuple[bool, str]:
        return True, ""

    def extra_params(self) -> list[tuple[str, float, str]]:
        return [("Collimation half-angle (mrad)", 1.0, "theta_col_mrad")]

    def extra_choices(self) -> dict[str, list[str]]:
        return {}

    def params_to_config(self, fields: dict, quantum: bool = False):
        g = lambda k: _float(fields, k)

        # gamma0 from mean_energy_MeV, matching the existing shared
        # Electrons-panel convention every other adapter already uses
        # (kascade_adapter.py: eps0 = mean_energy_MeV*1e6/MEC2_EV, i.e.
        # this field is used directly as total-energy/mec2 -- not a
        # kinetic-energy field at the GUI-field level).
        from compton_suite.io.constants import MEC2_EV
        gamma0 = g("mean_energy_MeV") * 1e6 / MEC2_EV
        sigma_eps_rel = g("rel_spread_pct") / 100.0

        emit_x = g("emit_x_mmmrad") * 1e-6 / gamma0
        emit_y = g("emit_y_mmmrad") * 1e-6 / gamma0
        beta_x_m, beta_y_m = g("beta_x_m"), g("beta_y_m")
        sigma_x_m = np.sqrt(emit_x * beta_x_m) if beta_x_m > 0 else 0.0
        sigma_y_m = np.sqrt(emit_y * beta_y_m) if beta_y_m > 0 else 0.0

        # --- duration conversion through the canonical framework ---
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

        # --- laser ---
        lambda_L = g("laser_wavelength_nm") * 1e-9
        pulse_energy_J = g("laser_energy_mJ") * 1e-3
        R_sf = g("rayleigh_length_m")
        sigma0_l = 0.5 * np.sqrt(max(R_sf, 0.0) * lambda_L / np.pi) if R_sf > 0 else 1e-9

        # --- build shared representations via io/ ---
        beam = _beam_from_shared_fields(
            eps0=gamma0, sigma_eps_rel=sigma_eps_rel,
            emit_x=max(emit_x, 1e-30), emit_y=max(emit_y, 1e-30),
            sigma0_x=max(sigma_x_m, 1e-12), sigma0_y=max(sigma_y_m, 1e-12),
            sigma_par_e=max(sigma_par_e, 1e-12),
            N_e=g("charge_nC") * 1e-9 / E_CHARGE,
        )
        pulse = _laser_from_shared_fields(
            lambda_L=lambda_L, sigma0_l=max(sigma0_l, 1e-9),
            sigma_par_L=max(sigma_par_L, 1e-9), pulse_energy_J=pulse_energy_J,
        )

        cfg = AnalyticalConfig(
            beam=beam, pulse=pulse,
            theta_col_rad=g("theta_col_mrad") * 1e-3,
            quantum=quantum,
        )
        warnings: list[str] = []
        extra = dict(n_mc=5000, seed=0, rep_rate_hz=g("pulse_frequency_Hz"), warnings=warnings)
        return cfg, extra

    def run(self, cfg: AnalyticalConfig, n_mc: int, seed: int, *, electrons: MacroBunch):
        beam = fit_gaussian(electrons)
        self._last_beam = beam
        pulse = cfg.pulse

        total_yield = float(analytical.estimate_yield(beam, pulse))
        width = float(analytical.estimate_spectrum_width(beam, pulse, cfg.theta_col_rad))

        # Electron sampling is the caller's job now, not this adapter's
        # (see module docstring / the cross-repo "macrobunching" change
        # this is part of) -- this used to draw its OWN separate preview
        # bunch here (max(n_mc, 2000) particles via sample_gaussian_bunch)
        # for the per-electron kinematic shape calculation below, even
        # though ``electrons`` had already been fitted into ``beam`` two
        # lines above. That was strictly more self-sampling than
        # necessary: ``electrons`` already IS a macroparticle sample of
        # this same beam, so its own gamma/weight are used directly below
        # instead of drawing a second, independent one.
        #
        # Particle-count check: every real caller sizes ``electrons`` far
        # above the old max(n_mc, 2000) floor -- compton_suite.gui.app.py's
        # on_start() samples it to whichever *other* active model's own
        # convention (kascade's shared n_mc field, default 200_000; xigma-
        # i/-direct's n_particles_01, default 60_000/20_000), and
        # ComptonSuite/validation/runners.py samples DEFAULT_N_MC=100_000
        # particles for this adapter specifically -- so in practice this
        # is always a strictly larger, not noisier, sample than the old
        # internal draw. The one case that could hand this a small
        # ``electrons`` is a hand-loaded .ele file with few macroparticles
        # (supports_ele_file_io=True); if that ever proves too noisy for
        # this shape estimate, the fix is a caller-side minimum bunch
        # size, not resurrecting an internal resample here -- reintroducing
        # sampling in this adapter is exactly what this change removes.
        gamma_arr = np.asarray(electrons.gamma, dtype=float)
        weight_arr = np.full(electrons.n_particles, electrons.weight)

        omega0 = 2.0 * np.pi * C_LIGHT / pulse.wavelength_m
        Wph_eV = HBAR * omega0 / E_CHARGE
        s_grid = np.linspace(1e-3, 1.0 - 1e-3, 256)
        dNds = analytical.angle_integrated_spectrum(gamma_arr, weight_arr, s_grid)
        E_eV = 4.0 * beam.gamma0**2 * Wph_eV * s_grid
        dNdE_per_eV = dNds / (4.0 * Wph_eV)

        # QUICK FIX, FLAGGED FOR FUTURE INVESTIGATION: angle_integrated_spectrum
        # returns a per-electron kinematic SHAPE only -- it has no dependence
        # on the laser pulse (a0/n_photons) at all, so its raw absolute scale
        # has nothing to do with the pulse-energy-dependent total_yield
        # estimate_yield() actually computes (confirmed: this raw integral is
        # bit-identical across scenarios that only change pulse_energy_J).
        # Same self-consistent-rescale pattern already applied this session to
        # xigma-i/delta's angular_spectrum vs total_yield mismatch:
        # force the spectrum shape to integrate to the trusted total_yield,
        # rather than trust its own absolute normalization.
        _raw_integral = float(np.trapezoid(dNdE_per_eV, E_eV))
        if _raw_integral > 0:
            dNdE_per_eV = dNdE_per_eV * (total_yield / _raw_integral)

        from compton_suite.io.results import CommonResults
        return CommonResults(
            model_name="analytical",
            cfg=cfg,
            n_mc=electrons.n_particles,
            total_yield=total_yield,
            spectrum=BinnedSpectrum(E_eV=E_eV, dNdE_per_eV=dNdE_per_eV),
            summary={"estimated_spectrum_width_fwhm": width, "gamma0": beam.gamma0,
                     "N_e": beam.N_e, "n_photons": pulse.n_photons, "a0_interaction": pulse.a0_interaction},
        )

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

    def spectrum_in_angular_range(self, theta_x_range, theta_y_range, **kwargs):
        raise NotImplementedError("analytical: no angular data (supports_angular_range_spectrum is False)")
