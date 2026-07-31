"""Tier 0: wiring/units sanity.

Every model's Config is built directly from the same Scenario's
``interaction_parameters`` (the shared ``compton_suite.io.interaction.
InteractionParameters`` bundle) -- not re-derived independently per model
-- so ``gamma0``/``N_e``/``wavelength``/etc. agreeing across all four
models is now guaranteed by construction (each Config just holds a
reference to the same ``beam``/``laser`` objects), not something that can
drift and needs runtime checking. What *can* still drift is the boundary
conversion into xigma_i's own CGS ``CollisionParams`` bundle
(``models.xigma_i.collision.build_params``): its ``a0``/``N_l`` fields are
meant to be a direct pass-through of ``GaussianParaxialLaser.a0_focus``/
``.n_photons`` (the shared, model-agnostic SI formula -- see that module's
docstring), so this tier checks that pass-through stays exact, catching a
real regression if someone reintroduces an independent CGS a0/N_l formula.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compton_suite.validation.scenarios import (  # noqa: E402
    BASELINE,
    Scenario,
    build_analytical_config,
    build_params_for_xigma,
    build_kascade_config,
    build_xigma_config,
    build_delta_config,
)

EXACT_TOL = 1e-9        # same-source floats -- should agree to float precision


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def check_interaction_identity(scenario: Scenario = BASELINE) -> bool:
    """Every model's Config.interaction must be the exact same
    InteractionParameters object the Scenario built -- not an independent
    re-derivation. This is what actually guarantees gamma0/N_e/wavelength/
    etc. agree across models now (see module docstring); a model
    re-deriving its own copy instead of holding this reference is the
    wiring bug this tier exists to catch."""
    ref = scenario.interaction_parameters
    configs = dict(
        kascade=build_kascade_config(scenario), xigma_i=build_xigma_config(scenario),
        delta=build_delta_config(scenario), analytical=build_analytical_config(scenario),
    )
    bad = {name: cfg.interaction for name, cfg in configs.items() if cfg.interaction != ref}
    if bad:
        print(f"  [FAIL] Config.interaction is not the scenario's own InteractionParameters: {list(bad)}")
        return False
    print("  [PASS] every model's Config.interaction is the scenario's own InteractionParameters")
    return True


def check_a0_N_l_passthrough(scenario: Scenario = BASELINE) -> bool:
    """xigma_i's CollisionParams.a0/.N_l must exactly match
    GaussianParaxialLaser.a0_focus/.n_photons -- a straight pass-through
    (see collision.build_params's docstring), not an independently
    re-derived CGS formula."""
    xcfg = build_xigma_config(scenario)
    params = build_params_for_xigma(xcfg, device="cpu")
    ok = True
    for label, engine_val, io_val in (
            ("a0", float(params.a0), scenario.pulse.a0_focus),
            ("N_l", float(params.N_l), scenario.pulse.n_photons)):
        rel = _rel(engine_val, io_val)
        if rel > EXACT_TOL:
            print(f"  [FAIL] {label} pass-through broken: CollisionParams.{label}={engine_val:.6g} vs "
                  f"GaussianParaxialLaser value={io_val:.6g} (rel diff {rel:.3g}, tol {EXACT_TOL:.3g})")
            ok = False
        else:
            print(f"  [PASS] {label} pass-through exact: {engine_val:.6g} (rel diff {rel:.3g})")
    return ok


def run(scenario: Scenario = BASELINE) -> bool:
    print(f"=== Tier 0: wiring/units sanity ({scenario.name}) ===")
    checks = [check_interaction_identity, check_a0_N_l_passthrough]
    results = [check(scenario) for check in checks]
    ok = all(results)
    print(f"-> Tier 0: {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
