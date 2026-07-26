"""Tier 4: regime boundary (low-a0 vs near-a0_max).

Originally planned as a hard assertion that kascade-vs-xigma disagreement
GROWS approaching xigma's documented a0_max validity boundary (a0~0.5).
Running Tier 1/2 at NEAR_A0_MAX (a0~0.46) first, empirically: total_yield
agreement stays just as tight (<0.5%) as at LOW_A0 (a0~0.009) -- the
disagreement does NOT visibly grow in total_yield alone. Rather than
hard-assert something the actual data doesn't support (which would make
this tier flaky/wrong, not useful), this tier REPORTS the low-a0 vs
near-a0_max comparison -- both total_yield ratios and spectrum-shape
errors -- as a finding to monitor, same "report, don't gate" treatment as
Tier 3. If the a0_max approximation's breakdown shows up at all in these
scenarios, it may be more visible in spectral/angular SHAPE than in
total_yield -- this tier's spectrum-shape numbers are the place to look
for that, not a pass/fail signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from .scenarios import LOW_A0, NEAR_A0_MAX  # noqa: E402
from .runners import run_analytical, run_kascade, run_xigma, run_xigma_direct  # noqa: E402
from .tier1_yield import _rel, _total_yield_kascade  # noqa: E402
from .tier2_spectrum import compton_edge_eV, to_dNdE_on_grid  # noqa: E402
from .metrics import window_integrated_relative_error  # noqa: E402
import numpy as np  # noqa: E402


def _yield_ratios(scenario) -> dict:
    res_kascade = run_kascade(scenario)
    res_xigma = run_xigma(scenario)
    res_xigma_direct = run_xigma_direct(scenario)
    y_k = _total_yield_kascade(res_kascade)
    y_x = float(res_xigma.total_yield)
    y_xd = float(res_xigma_direct.total_yield)
    return dict(
        kascade_vs_xigma=_rel(y_k, y_x),
        kascade_vs_xigma_direct=_rel(y_k, y_xd),
        xigma_direct_vs_xigma=_rel(y_xd, y_x),
        _results=(res_kascade, res_xigma, res_xigma_direct),
    )


def _spectrum_shape_error(scenario, res_kascade, res_xigma) -> tuple[float, float]:
    e_max = compton_edge_eV(scenario)
    e_centers = np.linspace(1e-3, 1.05, 256) * e_max
    mu = 0.02 * e_max
    dNdE_kascade = to_dNdE_on_grid(res_kascade, e_centers)
    dNdE_xigma = to_dNdE_on_grid(res_xigma, e_centers)
    l1, mx, _ = window_integrated_relative_error(e_centers, dNdE_kascade, dNdE_xigma, mu)
    return l1, mx


def run() -> bool:
    """Always returns True -- this tier reports findings, it does not gate
    the suite (see module docstring for why a hard assertion here would be
    wrong given the actual observed behavior)."""
    print("=== Tier 4: regime boundary (low-a0 vs near-a0_max), reported not gated ===")

    low = _yield_ratios(LOW_A0)
    high = _yield_ratios(NEAR_A0_MAX)

    print(f"  kascade/xigma-i yield rel diff:        low-a0={low['kascade_vs_xigma']:.3g}  "
          f"near-a0_max={high['kascade_vs_xigma']:.3g}")
    print(f"  kascade/xigma-i-direct yield rel diff:  low-a0={low['kascade_vs_xigma_direct']:.3g}  "
          f"near-a0_max={high['kascade_vs_xigma_direct']:.3g}")
    print(f"  xigma-i-direct/xigma-i yield rel diff:  low-a0={low['xigma_direct_vs_xigma']:.3g}  "
          f"near-a0_max={high['xigma_direct_vs_xigma']:.3g}")

    res_k_low, res_x_low, _ = low["_results"]
    res_k_high, res_x_high, _ = high["_results"]
    l1_low, mx_low = _spectrum_shape_error(LOW_A0, res_k_low, res_x_low)
    l1_high, mx_high = _spectrum_shape_error(NEAR_A0_MAX, res_k_high, res_x_high)
    print(f"  kascade/xigma-i spectrum-shape weighted-L1: low-a0={l1_low:.3g}  near-a0_max={l1_high:.3g}")
    print(f"  kascade/xigma-i spectrum-shape max-rel:     low-a0={mx_low:.3g}  near-a0_max={mx_high:.3g}")

    grew = l1_high > l1_low
    print(f"  observation: spectrum-shape disagreement {'grows' if grew else 'does NOT grow'} "
          f"from low-a0 to near-a0_max ({l1_low:.3g} -> {l1_high:.3g})")
    print("-> Tier 4: reported (see observation above; not a pass/fail gate)")
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
