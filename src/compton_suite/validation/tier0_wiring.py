"""Tier 0: wiring/units sanity.

kascade is the only model left with a real, standalone ``Config`` object
(its own physics engine's parameter type) -- xigma_i/delta/analytical read
``job.interaction`` directly with nothing to re-derive, so the
"Config drifted from the Scenario's own InteractionParameters" failure mode
this tier used to guard against for all four models structurally can't
happen anymore for those three (there's no Config to hold a divergent
copy).

This tier used to also check that xigma_i's CGS ``CollisionParams.a0``/
``.N_l`` stayed an exact pass-through of ``GaussianParaxialLaser.a0_focus``/
``.n_photons`` -- that check no longer applies: ``CollisionParams`` is
gone, and every xigma_i call site now reads ``laser.a0_focus``/
``.n_photons`` directly (see ``particles.push_and_sample``'s docstring), so
there is no separate CGS copy left that could silently drift out of sync.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compton_suite.validation.scenarios import (  # noqa: E402
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
