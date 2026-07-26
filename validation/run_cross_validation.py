"""Cross-model validation suite -- entry point.

Runs kascade/xigma-i/xigma-i-direct/analytical once each per scenario (via
runners.py) and shares the results across Tiers 0-3, instead of every tier
re-running an expensive GPU/MC computation independently.

Tiers 0-2 are pass/fail gates (non-zero exit code on failure). Tiers 3-4
are report-only canaries by design (see their own module docstrings for
why a hard assertion there would currently be wrong, not just strict) --
they always print findings but never affect the exit code.

Usage: conda run -n core --no-capture-output python3 run_cross_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tier0_wiring  # noqa: E402
import tier1_yield  # noqa: E402
import tier2_spectrum  # noqa: E402
import tier3_angular_shape  # noqa: E402
import tier4_regime_boundary  # noqa: E402
from scenarios import BASELINE, LOW_A0  # noqa: E402
from runners import run_analytical, run_kascade, run_xigma, run_xigma_direct  # noqa: E402


def _run_gated_scenario(scenario) -> bool:
    """Tier 0 (config-only, no run needed) + Tier 1/2 (share one run of
    each model) + Tier 3 (canary, reported not gated) for one scenario."""
    ok = tier0_wiring.run(scenario)

    print()
    results = dict(
        kascade=run_kascade(scenario),
        xigma_i=run_xigma(scenario),
        xigma_direct=run_xigma_direct(scenario),
        analytical=run_analytical(scenario),
    )

    print()
    ok &= tier1_yield.run(scenario, results=results)

    print()
    ok &= tier2_spectrum.run(scenario, results=results)

    print()
    tier3_angular_shape.run(scenario, results=results)

    return ok


def run() -> bool:
    print("############################################")
    print("# ComptonSuite cross-model validation suite #")
    print("############################################")
    print()

    ok = True

    print(f"########## Scenario: {BASELINE.name} ##########")
    ok &= _run_gated_scenario(BASELINE)

    print()
    print(f"########## Scenario: {LOW_A0.name} ##########")
    ok &= _run_gated_scenario(LOW_A0)

    print()
    print("########## Tier 4: regime boundary (report-only) ##########")
    tier4_regime_boundary.run()

    print()
    print("=" * 60)
    print(f"CROSS-VALIDATION SUITE: {'ALL GATED TIERS PASS' if ok else 'FAILURES ABOVE'}")
    print("(Tiers 3-4 are report-only canaries and never affect this result "
          "-- see their printed findings above.)")
    print("=" * 60)
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
