"""Tier 3: angular shape canary (xigma-i vs delta), reported not gated.

xigma-i and delta's angular_spectrum totals are normalized to agree exactly.
This tier validates point-by-point/window-by-window agreement: the two methods
(tabulated kernel vs per-particle binning) agree within ~1-5% relative
difference in collimated fraction across multiple window sizes at baseline
scenarios. This tier reports the comparison rather than hard-gating on it,
since there's no
independently-established "correct" answer to gate against and this is
still only one scenario/one set of windows.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gammaforge.models.api import total_yield  # noqa: E402
from gammaforge.validation.scenarios import BASELINE, Scenario  # noqa: E402
from gammaforge.validation.runners import run_xigma_live, run_delta_live  # noqa: E402

WINDOWS_MRAD = (0.05, 0.2, 1.0)


def collimated_fraction(adapter, res, theta_rad: float) -> float:
    """``adapter`` is the live XigmaAdapter/DirectAdapter instance that
    produced ``res`` -- its own spectrum_in_angular_range() reuses the
    cached TabulatedEngine/per-particle arrays for the on-demand recompute,
    which a cached (pickled) Photons result alone can't provide (see
    runners.py's module docstring)."""
    result = adapter.spectrum_in_angular_range((-theta_rad, theta_rad), (-theta_rad, theta_rad))
    return result.n_photons_in_range / total_yield(res)


def run(scenario: Scenario = BASELINE, results: dict | None = None) -> bool:
    """Always returns True -- canary tier, reports findings, does not gate
    the suite (see module docstring). ``results`` (a cache-shared dict of
    plain Photons from other tiers) isn't usable here -- this tier needs
    the live adapter for its on-demand angular recompute, so it always
    computes its own xigma-i/delta run fresh via run_xigma_live/
    run_delta_live rather than reusing a cached Photons-only result."""
    print(f"=== Tier 3: angular shape canary, xigma-i vs delta ({scenario.name}) ===")

    res_xigma, xigma_adapter = run_xigma_live(scenario)
    res_delta, delta_adapter = run_delta_live(scenario)

    for w in WINDOWS_MRAD:
        theta_rad = w * 1e-3
        frac_x = collimated_fraction(xigma_adapter, res_xigma, theta_rad)
        frac_xd = collimated_fraction(delta_adapter, res_delta, theta_rad)
        rel = abs(frac_x - frac_xd) / max(frac_x, frac_xd, 1e-300)
        print(f"  window +/-{w:g} mrad: xigma-i collimated_fraction={frac_x:.4g}, "
              f"delta={frac_xd:.4g}, rel diff={rel:.3g}")

    print("-> Tier 3: reported (canary, not a pass/fail gate -- see numbers above)")
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
