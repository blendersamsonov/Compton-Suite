"""Shared physical scenarios for the cross-model validation suite, and the
per-model Config builders that construct each model's own native Config
directly from one canonical (beam, pulse) pair -- bypassing the string-
parsing params_to_config/GUI-field layer entirely (that layer is a UI
convenience, not physically meaningful; skipping it removes a source of
avoidable float-formatting noise from cross-model comparisons).

kascade.Config, xigma_i.gui_adapter.Config, and xigma_direct.DirectConfig
each hold a single ``interaction: compton_io.interaction.InteractionParameters``
field (see that module's docstring) instead of their own flat physics
fields, so ``Scenario.interaction_parameters`` feeds all three directly.
AnalyticalConfig wraps (beam, pulse) natively, no conversion needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from compton_suite.io.bunch import GaussianElectronBeam
from compton_suite.io.constants import MEC2_EV
from compton_suite.io.interaction import InteractionGeometry, InteractionParameters
from compton_suite.io.laser import GaussianParaxialLaser

# Default random seed for every scenario run -- deterministic comparisons,
# same convention as Xigma/validation/params.py's DEFAULT_SEED.
DEFAULT_SEED = 20260721

# Default macro-particle/electron count for models whose n_mc argument is
# actually used (kascade -- xigma_i/xigma_direct ignore it, sizing Stage 0
# from their own Config.n_particles_01 instead, already defaulted on their
# Config dataclasses).
DEFAULT_N_MC = 100_000

__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_N_MC",
    "Scenario",
    "BASELINE",
    "LOW_A0",
    "NEAR_A0_MAX",
    "build_kascade_config",
    "build_xigma_config",
    "build_xigma_direct_config",
    "build_analytical_config",
]


@dataclass(frozen=True)
class Scenario:
    """One canonical physical scenario. crossing_angle_rad/quantum stay
    outside (beam, pulse) -- deliberately excluded from compton_io's
    shared representation since only kascade supports non-zero/quantum;
    0.0/False here means "cross-model comparable", non-zero/True means
    "kascade-only, no cross-check partner"."""

    name: str
    beam: GaussianElectronBeam
    pulse: GaussianParaxialLaser
    crossing_angle_rad: float = 0.0
    quantum: bool = False
    beta_ff: float = 0.0          # xigma-i/xigma-i-direct-only extra
    phi_pol: float = 0.0          # xigma-i/xigma-i-direct-only extra
    a0_max: float = 0.5           # xigma-i-only Stage 1/2 table sizing knob
    theta_col_rad: float = 1.0e-3  # collimation half-angle, for analytical's estimate_spectrum_width

    @property
    def interaction_parameters(self) -> InteractionParameters:
        """This scenario's (beam, pulse, geometry) as one shared
        compton_io.interaction.InteractionParameters bundle -- the
        physics-parameter subset every model's Config should ultimately be
        built from (see that module's docstring)."""
        return InteractionParameters(
            beam=self.beam, laser=self.pulse,
            geometry=InteractionGeometry(crossing_angle_rad=self.crossing_angle_rad),
            quantum=self.quantum,
        )


def _baseline() -> Scenario:
    # Ported from Xigma/validation/params.py's ELECTRON/LASER dicts (CGS,
    # cm/s) -- gamma0=2000, 10nC, 20J 1030nm laser -- already the
    # representative, already-exercised operating point that repo's own
    # figures/benchmarks use, not a new arbitrary choice. Converted here to
    # SI GaussianElectronBeam/GaussianParaxialLaser via code (not
    # hand-computed literals) to avoid transcription error.
    charge_nC = 10.0
    gamma0 = 2000.0
    sigma_gamma_rel = 0.005          # sigma_gamma0 / gamma0 (relative to gamma/total energy)
    sigma_x_cm = 1e-3
    sigma_y_cm = 1e-3
    duration_s = 10e-12              # -> sigma_ez = duration * c
    norm_emit_x_cm_rad = 1e-4
    norm_emit_y_cm_rad = 0.01e-4

    kinetic_energy_eV = (gamma0 - 1.0) * MEC2_EV
    sigma_gamma = sigma_gamma_rel * gamma0
    # rel_energy_spread_rms is relative to KINETIC energy (compton_io's
    # convention, spec Sec.10) -- converted exactly from params.py's
    # relative-to-gamma sigma_gamma_rel, not assumed equal (they agree to
    # ~1/gamma0 here, but computed correctly rather than approximated).
    rel_energy_spread_rms = (sigma_gamma * MEC2_EV) / kinetic_energy_eV

    beam = GaussianElectronBeam(
        bunch_charge_C=charge_nC * 1e-9,
        kinetic_energy_eV=kinetic_energy_eV,
        rel_energy_spread_rms=rel_energy_spread_rms,
        sigma_x_m=sigma_x_cm * 1e-2,
        sigma_y_m=sigma_y_cm * 1e-2,
        emit_geom_x_m=(norm_emit_x_cm_rad / gamma0) * 1e-2,
        emit_geom_y_m=(norm_emit_y_cm_rad / gamma0) * 1e-2,
        sigma_t_s=duration_s,
    )

    pulse_energy_J = 20.0
    lambda_l_cm = 1030e-7
    sigma_r_cm = 10e-4
    laser_duration_s = 30e-12

    pulse = GaussianParaxialLaser(
        pulse_energy_J=pulse_energy_J,
        wavelength_m=lambda_l_cm * 1e-2,
        waist_rms_x_m=sigma_r_cm * 1e-2,
        waist_rms_y_m=sigma_r_cm * 1e-2,
        duration_rms_s=laser_duration_s,
    )

    return Scenario(name="baseline", beam=beam, pulse=pulse)


BASELINE = _baseline()

# Same beam, scaled laser pulse energy -- a0 scales ~linearly with
# pulse_energy_J (N_l), so this is the simplest lever to move the
# scenario's regime while holding the electron bunch fixed. Calibrated
# empirically against compton_io.collision.build_params's own a0 (the value the
# engines actually use, not compton_io's independent formula -- see
# tier0_wiring.py's flagged a0-formula discrepancy): baseline itself sits
# at a0~0.093; LOW_A0 at ~0.009 (deep in the classical linear regime,
# a0 << a0_max=0.5); NEAR_A0_MAX at ~0.46 (close to xigma's documented
# valid-range boundary).
def _scaled(name: str, pulse_energy_J: float) -> Scenario:
    from dataclasses import replace
    pulse = replace(BASELINE.pulse, pulse_energy_J=pulse_energy_J)
    return replace(BASELINE, name=name, pulse=pulse)


LOW_A0 = _scaled("low_a0", pulse_energy_J=2.0)          # a0 ~ 0.009
NEAR_A0_MAX = _scaled("near_a0_max", pulse_energy_J=100.0)  # a0 ~ 0.46


def build_kascade_config(scenario: Scenario):
    from compton_suite.models.kascade import kascade
    return kascade.Config(interaction=scenario.interaction_parameters)


def build_xigma_config(scenario: Scenario):
    from compton_suite.models.xigma_i import gui_adapter
    return gui_adapter.Config(
        interaction=scenario.interaction_parameters,
        beta_ff=scenario.beta_ff, phi_pol=scenario.phi_pol,
        a0_max=scenario.a0_max,
    )


def build_xigma_direct_config(scenario: Scenario):
    from compton_suite.models.xigma_direct import gui_adapter
    return gui_adapter.DirectConfig(
        interaction=scenario.interaction_parameters,
        beta_ff=scenario.beta_ff, phi_pol=scenario.phi_pol,
    )


def build_analytical_config(scenario: Scenario):
    from compton_suite.models.analytical import analytical_adapter
    return analytical_adapter.AnalyticalConfig(
        beam=scenario.beam, pulse=scenario.pulse,
        theta_col_rad=scenario.theta_col_rad,
        crossing_angle=scenario.crossing_angle_rad, quantum=scenario.quantum,
    )


def build_params_for_xigma(cfg, device: str | None = None):
    """A compton_io.collision.CollisionParams instance from an xigma_i/
    xigma_direct Config -- exactly the beam/laser/geometry ->
    build_params() call xigma_i.gui_adapter.run_simulation itself makes
    (SI -> CGS at this boundary), so Tier 0/1 can read params.a0/.N_l/.N_e
    the same way the real run does, not a reimplementation that could
    silently drift out of sync."""
    from compton_suite.io.collision import build_params, detect_device

    device = device or detect_device()
    return build_params(cfg.interaction.beam, cfg.interaction.laser,
                         cfg.interaction.geometry, beta_ff=cfg.beta_ff, device=device)
