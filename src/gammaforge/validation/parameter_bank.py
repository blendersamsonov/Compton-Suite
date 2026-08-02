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

from gammaforge.validation.scenarios import Scenario

PARAMETER_BANK: list[Scenario] = []
