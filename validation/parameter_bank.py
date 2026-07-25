"""User-curated bank of additional cross-validation parameter sets.

Deliberately empty on creation -- meant to be populated by hand with
whatever (GaussianElectronBeam, GaussianParaxialLaser) combinations are
worth regularly cross-checking (specific experimental operating points,
previously-buggy regimes worth guarding against a regression, edge cases
away from scenarios.py's own fixed reference scenarios). Not yet wired
into run_cross_validation.py or visualize.py -- both iterate over
scenarios.BASELINE/LOW_A0/NEAR_A0_MAX today; once this bank has real
entries, iterate over PARAMETER_BANK there the same way.

Each entry is a scenarios.Scenario -- see that module for the full field
list (name, beam, pulse, crossing_angle_rad, quantum, beta_ff, phi_pol,
a0_max, theta_col_rad) and scenarios.BASELINE for a fully-worked example
of constructing one from raw electron/laser numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenarios import Scenario  # noqa: E402

# Example (delete once real entries are added):
#
# from compton_io.bunch import GaussianElectronBeam
# from compton_io.laser import GaussianParaxialLaser
#
# MY_SCENARIO = Scenario(
#     name="my_scenario",
#     beam=GaussianElectronBeam(
#         bunch_charge_C=1e-9, kinetic_energy_eV=1e9, rel_energy_spread_rms=1e-3,
#         sigma_x_m=1e-5, sigma_y_m=1e-5, emit_geom_x_m=1e-10, emit_geom_y_m=1e-10,
#         sigma_t_s=1e-12,
#     ),
#     pulse=GaussianParaxialLaser(
#         pulse_energy_J=0.3, wavelength_m=1e-6,
#         waist_rms_x_m=1e-5, waist_rms_y_m=1e-5, duration_rms_s=3e-12,
#     ),
# )
# PARAMETER_BANK = [MY_SCENARIO]

PARAMETER_BANK: list[Scenario] = []
