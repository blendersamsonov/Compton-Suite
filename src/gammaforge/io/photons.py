"""Output-side observable representations: the phase-space-slice contract
every model reports results through, and the GUI renders from directly.

Every observable -- energy spectrum, temporal envelope, spatial
distribution, angular distribution, or a joint combination of these -- is a
single :class:`PhasespaceSlice`: a density array over a named set of axes.
There is no separate "sampled" vs "binned" shape: a model that only has
per-macroparticle samples (kascade) histograms them into the same density-
array shape every other model produces directly (see :func:`make_slice`).
A 0D slice (``axes={}``, ``distr`` a 0-d array) is the fully
phase-space-integrated total yield -- every adapter includes exactly one of
these in :attr:`Results.photon_slices`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gammaforge.io.bunch import Bunch

__all__ = [
    "AXIS_ENERGY",
    "AXIS_TIME",
    "AXIS_X",
    "AXIS_Y",
    "AXIS_THETA_X",
    "AXIS_THETA_Y",
    "ALLOWED_SLICE_AXES",
    "PhasespaceSlice",
    "Results",
    "AngularRangeSpectrumResult",
    "find_slice",
    "total_yield",
    "make_slice",
    "validate_results",
]

AXIS_ENERGY = "E_eV"
AXIS_TIME = "t_seconds"
AXIS_X = "x"
AXIS_Y = "y"
AXIS_THETA_X = "theta_x"
AXIS_THETA_Y = "theta_y"

# The closed set of axis-groupings a PhasespaceSlice may carry. No
# combination beyond these nine is valid -- in particular energy never
# combines with space or time, only with angle.
ALLOWED_SLICE_AXES: frozenset[frozenset[str]] = frozenset({
    frozenset(),  # 0D: total yield
    frozenset({AXIS_ENERGY}),
    frozenset({AXIS_TIME}),
    frozenset({AXIS_X, AXIS_Y}),
    frozenset({AXIS_THETA_X, AXIS_THETA_Y}),
    frozenset({AXIS_X, AXIS_Y, AXIS_TIME}),
    frozenset({AXIS_THETA_X, AXIS_THETA_Y, AXIS_TIME}),
    frozenset({AXIS_X, AXIS_Y, AXIS_THETA_X, AXIS_THETA_Y}),
    frozenset({AXIS_ENERGY, AXIS_THETA_X, AXIS_THETA_Y}),
})


@dataclass
class PhasespaceSlice:
    """Density over <=4 axes of the 6D photon phase space (time, space,
    angle, energy).

    ``axes`` maps axis name -> bin-centers array, in the same order as
    ``distr``'s dimensions. An empty dict means a 0D slice (``distr`` is a
    0-d array) -- the fully phase-space-integrated total yield.
    """

    axes: dict[str, np.ndarray]
    distr: np.ndarray


@dataclass
class AngularRangeSpectrumResult:
    """On-demand angle-integrated spectrum restricted to an arbitrary
    ``(theta_x_range, theta_y_range)`` window, returned by
    ``ModelAdapter.spectrum_in_angular_range()`` (xigma-i/delta only). This
    is a live recompute against cached engine/table state, not part of the
    static :class:`Results` a ``run()`` call returns.

    ``angle_energy_grid``: the joint (theta_x, theta_y, energy) grid
    ``spectrum`` is itself collapsed from (``einsum``'d over the two angle
    axes) -- both adapters already materialise this internally before
    collapsing it, so exposing it is free (no extra compute), and lets a
    caller that wants a genuine 2D angle-resolved view (not just the
    angle-integrated 1D spectrum) reuse the SAME collimation-window-sized
    query instead of falling back to a generic, uncollimated grid. ``None``
    only if a caller explicitly opts out (no current caller does).
    """

    spectrum: PhasespaceSlice
    theta_x_range: tuple[float, float]
    theta_y_range: tuple[float, float]
    n_photons_in_range: float | None = None
    angle_energy_grid: PhasespaceSlice | None = None


@dataclass
class Results:
    """What every model's ``run()`` returns.

    ``photon_slices`` always includes a 0D slice (``axes={}``) holding the
    total photon yield; everything else is whatever slices ``job.output``
    requested that this model is able to produce -- a model simply omits a
    requested slice it can't compute, there is no all-or-nothing
    requirement. ``macrophoton``/``electrons`` are the raw final-state
    per-macroparticle bunches when a model has them (kascade only today),
    else ``None``. ``model_specific`` is a free-form dict for anything that
    doesn't fit the slice/bunch shapes (e.g. kascade's photon-multiplicity
    fractions, xigma-i's mean photon energy).
    """

    photon_slices: list[PhasespaceSlice]
    macrophoton: Bunch | None = None
    electrons: Bunch | None = None
    model_specific: dict = field(default_factory=dict)


def find_slice(slices: list[PhasespaceSlice], *axis_names: str) -> PhasespaceSlice | None:
    """Look up a slice by its exact axis-name set (order-independent).

    ``find_slice(res.photon_slices)`` (no axis names) returns the required
    0D total-yield slice.
    """
    wanted = frozenset(axis_names)
    for s in slices:
        if frozenset(s.axes) == wanted:
            return s
    return None


def total_yield(res: Results) -> float:
    """Read the 0D slice every adapter is required to emit."""
    s = find_slice(res.photon_slices)
    return float(s.distr) if s is not None else 0.0


def make_slice(
    samples: dict[str, np.ndarray],
    weight: float,
    bins: dict[str, int],
    ranges: dict[str, tuple[float, float]] | None = None,
) -> PhasespaceSlice:
    """Histogram per-macroparticle samples (with a uniform weight) into a
    :class:`PhasespaceSlice`.

    ``samples`` maps axis name -> per-particle array; all arrays must be
    the same length. ``bins``/``ranges`` are keyed the same way. This is
    the boundary helper an event-generator model (kascade) uses to turn its
    raw per-photon arrays into the same density-array shape every other
    model produces directly. Passing an empty ``samples``/``bins`` dict
    produces the 0D total-yield slice.
    """
    ranges = ranges or {}
    axis_names = list(samples.keys())

    if not axis_names:
        n = 0
        if samples:
            n = next(iter(samples.values())).shape[0]
        return PhasespaceSlice(axes={}, distr=np.asarray(float(n) * weight))

    data = np.stack([np.asarray(samples[name]) for name in axis_names], axis=-1)
    n_bins = [bins[name] for name in axis_names]
    hist_range = [ranges.get(name) for name in axis_names]
    counts, edges = np.histogramdd(data, bins=n_bins, range=hist_range)

    # Divide by the per-cell bin "volume" (product of each axis's bin width)
    # so ``distr`` is a density -- e.g. photons/eV for a 1D energy slice --
    # matching the convention every other (already-binned) model uses
    # directly (``dNdE_per_eV`` etc), not a raw weighted histogram count.
    bin_widths = [np.diff(edge) for edge in edges]
    volume = bin_widths[0]
    for w_arr in bin_widths[1:]:
        volume = np.multiply.outer(volume, w_arr)
    distr = counts * weight / volume

    axes = {
        name: 0.5 * (edge[:-1] + edge[1:])
        for name, edge in zip(axis_names, edges)
    }
    return PhasespaceSlice(axes=axes, distr=distr)


def validate_results(res: Any) -> list[str]:
    """Defensive sanity check, run right after a model's ``run()``
    returns.

    Returns a list of problem descriptions; empty means OK. Never raises --
    the caller decides whether a non-empty list is fatal.
    """
    problems: list[str] = []
    slices = getattr(res, "photon_slices", None)
    if not slices:
        problems.append("missing or empty required field: 'photon_slices'")
        return problems

    if find_slice(slices) is None:
        problems.append("photon_slices is missing the required 0D total-yield slice")

    for s in slices:
        axis_set = frozenset(s.axes)
        if axis_set not in ALLOWED_SLICE_AXES:
            problems.append(f"slice has disallowed axis combination: {sorted(axis_set)!r}")
            continue
        expected_shape = tuple(np.asarray(v).shape[0] for v in s.axes.values())
        if s.distr.shape != expected_shape:
            problems.append(
                f"slice {sorted(axis_set)!r} distr.shape {s.distr.shape} "
                f"doesn't match axes shape {expected_shape}"
            )
    return problems
