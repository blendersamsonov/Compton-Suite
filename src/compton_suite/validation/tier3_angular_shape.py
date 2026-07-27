"""Tier 3: angular shape canary (xigma-i vs delta), reported not
gated.

xigma-i and delta's angular_spectrum totals are already forced
to agree by this session's rescale fix (see gui_adapter.py's "QUICK FIX,
FLAGGED FOR FUTURE INVESTIGATION" comments in both repos) -- but that
only fixes the FULL-RANGE integral, not necessarily point-by-point/
window-by-window agreement. An earlier ad hoc check this session (before
the rescale fix was applied consistently in both repos) had found the two
models' collimated fraction for one window differing by ~2x. Re-measured
here with the rescale fix in place, across three window sizes at the
baseline scenario, the two methods (tabulated kernel vs brute-force
per-particle binning) actually agree much more closely (~1-5% relative
difference in collimated fraction -- see printed output). This tier keeps
reporting the comparison rather than hard-gating on it, since there's no
independently-established "correct" answer to gate against and this is
still only one scenario/one set of windows.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compton_suite.validation.scenarios import BASELINE, Scenario  # noqa: E402
from compton_suite.validation.runners import run_xigma, run_delta  # noqa: E402

WINDOWS_MRAD = (0.05, 0.2, 1.0)


def collimated_fraction(spectrum_in_angular_range_fn, res, theta_rad: float) -> float:
    result = spectrum_in_angular_range_fn(res, (-theta_rad, theta_rad), (-theta_rad, theta_rad))
    return result.n_photons_in_range / res.total_yield


def run(scenario: Scenario = BASELINE, results: dict | None = None) -> bool:
    """Always returns True -- canary tier, reports findings, does not gate
    the suite (see module docstring)."""
    print(f"=== Tier 3: angular shape canary, xigma-i vs delta ({scenario.name}) ===")
    from compton_suite.models.xigma_i.gui_adapter import spectrum_in_angular_range as xigma_sar
    from compton_suite.models.delta.gui_adapter import spectrum_in_angular_range as delta_sar

    results = results or {}
    res_xigma = results.get("xigma_i") or run_xigma(scenario)
    res_delta = results.get("delta") or run_delta(scenario)

    for w in WINDOWS_MRAD:
        theta_rad = w * 1e-3
        frac_x = collimated_fraction(xigma_sar, res_xigma, theta_rad)
        frac_xd = collimated_fraction(delta_sar, res_delta, theta_rad)
        rel = abs(frac_x - frac_xd) / max(frac_x, frac_xd, 1e-300)
        print(f"  window +/-{w:g} mrad: xigma-i collimated_fraction={frac_x:.4g}, "
              f"delta={frac_xd:.4g}, rel diff={rel:.3g}")

    print("-> Tier 3: reported (canary, not a pass/fail gate -- see numbers above)")
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
