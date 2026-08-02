"""Tier 0: wiring/units sanity.

Only kascade has a standalone ``Config`` object. xigma_i/delta/analytical
read ``job.interaction`` directly with no re-derivation, so there is no
separate Config copy that could drift from the Scenario's InteractionParameters.

xigma_i reads laser properties (a0_focus, n_photons) directly from
``job.interaction.laser`` (see ``particles.push_and_sample``'s docstring),
so there is no separate CGS copy that could silently diverge.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gammaforge.validation.scenarios import (  # noqa: E402
    BASELINE,
    Scenario,
    build_interaction,
    build_kascade_config,
)


def check_interaction_identity(scenario: Scenario = BASELINE) -> bool:
    """kascade's Config.interaction must be the exact same
    InteractionParameters object build_interaction() produced -- not an
    independent re-derivation. xigma_i/delta/analytical have no Config left
    to check this way (see module docstring). Identity (``is``), not
    structural equality (``==``/``!=``): InteractionParameters now holds a
    Bunch with numpy-array fields, and the dataclass's auto-generated
    ``__eq__`` comparing those pairwise would raise (truth value of an
    array is ambiguous)."""
    ref = build_interaction(scenario, n_mc=100, seed=0)
    kcfg = build_kascade_config(scenario, ref)
    if kcfg.interaction is not ref:
        print("  [FAIL] kascade.Config.interaction is not the scenario's own InteractionParameters")
        return False
    print("  [PASS] kascade.Config.interaction is the scenario's own InteractionParameters")
    return True


def run(scenario: Scenario = BASELINE) -> bool:
    print(f"=== Tier 0: wiring/units sanity ({scenario.name}) ===")
    checks = [check_interaction_identity]
    results = [check(scenario) for check in checks]
    ok = all(results)
    print(f"-> Tier 0: {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
