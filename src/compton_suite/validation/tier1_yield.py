"""Tier 1: cross-model total_yield agreement.

kascade (event-generator MC), xigma-i (tabulated kernel), and
xigma-i-direct (brute-force per-particle binning) are genuinely
independent computations of the same physical total photon yield --
xigma-i and xigma-i-direct share the exact same Stage 0 (particles.
sample_bunch/push_and_sample), so should agree tightly; kascade uses
entirely different physics (a sequential multi-photon emission chain, not
a luminosity-functional overlap integral), so a looser tolerance applies.
analytical is a closed-form "cheap estimate" its own original docstring
says is "not used by the real computation" -- compared with an
order-of-magnitude tolerance, not a precision one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compton_suite.validation.scenarios import BASELINE, Scenario  # noqa: E402
from compton_suite.validation.runners import run_analytical, run_kascade, run_xigma, run_xigma_direct  # noqa: E402

# Tiered tolerances (relative). Calibrated against the baseline scenario's
# actual observed spread (kascade/xigma-i/xigma-i-direct agreed to <0.5%
# there -- these leave real margin above that, not tuned to just barely
# pass) -- not validated across every scenario/regime this suite will
# eventually cover, see Tier 4 for where tighter agreement is explicitly
# NOT expected (near a0_max).
TIGHT_TOL = 0.05    # xigma-i vs xigma-i-direct (shared Stage 0 physics)
LOOSE_TOL = 0.15    # kascade vs {xigma-i, xigma-i-direct} -- matches the
                     # ~15% spread Xigma/CLAUDE.md already documents
                     # between its own three internal methods
ANALYTICAL_TOL = 3.0  # order-of-magnitude bound (observed ~70% deviation
                       # at baseline, consistent underestimate direction
                       # -- see this module's own printed output), not a
                       # precision one


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def _total_yield_kascade(res) -> float:
    return float(res.summary["N_gamma_total"])


def run(scenario: Scenario = BASELINE, results: dict | None = None) -> bool:
    """results (optional): a dict with any of "kascade"/"xigma_i"/
    "xigma_direct"/"analytical" keys already populated with each model's
    result object (see runners.py) -- reused instead of re-running, so
    run_cross_validation.py can run each model once and share results
    across tiers."""
    print(f"=== Tier 1: total_yield cross-check ({scenario.name}) ===")
    results = results or {}
    res_kascade = results.get("kascade") or run_kascade(scenario)
    res_xigma = results.get("xigma_i") or run_xigma(scenario)
    res_xigma_direct = results.get("xigma_direct") or run_xigma_direct(scenario)
    res_analytical = results.get("analytical") or run_analytical(scenario)

    y_kascade = _total_yield_kascade(res_kascade)
    y_xigma = float(res_xigma.total_yield)
    y_xigma_direct = float(res_xigma_direct.total_yield)
    y_analytical = float(res_analytical.total_yield)

    print(f"  kascade        total_yield = {y_kascade:.6e}")
    print(f"  xigma-i        total_yield = {y_xigma:.6e}")
    print(f"  xigma-i-direct total_yield = {y_xigma_direct:.6e}")
    print(f"  analytical     total_yield = {y_analytical:.6e}")

    ok = True

    rel_xx = _rel(y_xigma_direct, y_xigma)
    pass_xx = rel_xx <= TIGHT_TOL
    print(f"  [{'PASS' if pass_xx else 'FAIL'}] xigma-i-direct/xigma-i rel diff = {rel_xx:.3g} (tol {TIGHT_TOL:.2g})")
    ok &= pass_xx

    rel_kx = _rel(y_kascade, y_xigma)
    pass_kx = rel_kx <= LOOSE_TOL
    print(f"  [{'PASS' if pass_kx else 'FAIL'}] kascade/xigma-i rel diff = {rel_kx:.3g} (tol {LOOSE_TOL:.2g})")
    ok &= pass_kx

    rel_kxd = _rel(y_kascade, y_xigma_direct)
    pass_kxd = rel_kxd <= LOOSE_TOL
    print(f"  [{'PASS' if pass_kxd else 'FAIL'}] kascade/xigma-i-direct rel diff = {rel_kxd:.3g} (tol {LOOSE_TOL:.2g})")
    ok &= pass_kxd

    rel_ax = _rel(y_analytical, y_xigma)
    pass_ax = rel_ax <= ANALYTICAL_TOL
    print(f"  [{'PASS' if pass_ax else 'FAIL'}] analytical/xigma-i rel diff = {rel_ax:.3g} "
          f"(tol {ANALYTICAL_TOL:.2g}, order-of-magnitude bound)")
    ok &= pass_ax

    print(f"-> Tier 1: {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
