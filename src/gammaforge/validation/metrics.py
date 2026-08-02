"""Shared comparison metric for the cross-model validation suite.

Promoted (moved conceptually, copied verbatim) from
``Xigma/validation/metrics.py`` -- that module is pure numpy with nothing
xigma_i-specific about it, so it's reused here as the canonical
implementation for cross-*model* comparisons too, not just xigma_i's own
internal cross-*method* ones. ``Xigma/validation/metrics.py`` itself is
intentionally left untouched by this move (avoids unrelated churn in a
repo not otherwise part of this suite); unifying the two copies so
Xigma's own scripts import this one instead is a reasonable future
follow-up, not required for this suite to work.

Convention: bin the spectrum into windows of width `mu` (the reporting
resolution), integrate flux within each window (trapezoidal in the
frequency-like variable), and report two numbers relative to a reference
spectrum computed on the same grid:

  - the reference-flux-weighted L1 relative deviation (the headline
    number, representative of the whole spectrum), and
  - the max relative deviation over windows (sensitive to localized
    failures, e.g. near the Compton edge).

Windows with negligible reference flux are down-weighted rather than
excluded, so a handful of empty-reference bins at the grid edges cannot
produce spurious inf/nan.
"""

from __future__ import annotations

import numpy as np

__all__ = ["window_integrated_relative_error", "resample_to"]


def window_integrated_relative_error(s, spec, spec_ref, mu, floor_fraction=1e-6):
    """s, spec, spec_ref: 1D arrays of equal shape (same grid for both
    spectra -- resample one onto the other's grid via resample_to() first
    if they differ). mu: window width in the same units as s.

    Returns (weighted_l1, max_rel, n_windows).
    """
    s = np.asarray(s, dtype=np.float64)
    spec = np.asarray(spec, dtype=np.float64)
    spec_ref = np.asarray(spec_ref, dtype=np.float64)
    order = np.argsort(s)
    s, spec, spec_ref = s[order], spec[order], spec_ref[order]

    lo, hi = s[0], s[-1]
    n_windows = max(1, int(round((hi - lo) / mu)))
    edges = np.linspace(lo, hi, n_windows + 1)
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, n_windows - 1)

    ds = np.gradient(s)
    flux = spec * ds
    flux_ref = spec_ref * ds

    win_flux = np.bincount(idx, weights=flux, minlength=n_windows)
    win_flux_ref = np.bincount(idx, weights=flux_ref, minlength=n_windows)

    total_ref = win_flux_ref.sum()
    floor = floor_fraction * total_ref / n_windows if total_ref > 0 else 0.0
    denom = np.maximum(np.abs(win_flux_ref), max(floor, 1e-300))
    rel = np.abs(win_flux - win_flux_ref) / denom

    significant = win_flux_ref > floor
    if not np.any(significant):
        return 0.0, 0.0, n_windows

    weights = np.where(significant, win_flux_ref, 0.0)
    weights = weights / weights.sum()

    l1 = float(np.sum(weights * rel))
    mx = float(rel[significant].max())
    return l1, mx, n_windows


def resample_to(s_ref, s_src, spec_src):
    """Linear-interpolate spec_src(s_src) onto s_ref, zero outside s_src's range."""
    return np.interp(s_ref, s_src, spec_src, left=0.0, right=0.0)
