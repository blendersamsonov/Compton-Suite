"""Ballistic (straight-line, no-acceleration) electron-beam propagation,
and a time-streaming helper built on it.

``bunch.py``'s module docstring already describes this codebase's implicit
reliance on ballistic drift (Liouville's theorem / the Twiss-invariant
trick behind ``fit_gaussian``); this module makes the forward operation --
"where is this bunch's electrons at a different time?" -- an explicit,
reusable primitive, so a model that needs bunch state at several time
instances (e.g. to build a time-resolved table) calls into this rather
than re-deriving the drift itself.

Two related but genuinely different position conventions show up across
this codebase, and this module is deliberately explicit about which is
which rather than conflating them into one "generic" formula that would
silently be wrong for one side:

* :class:`~compton_io.bunch.MacroBunch`'s ``x``/``y``/``z`` are each
  particle's REAL, simultaneous position at one common reference time
  (``t=0``) -- see that class's own docstring. :func:`propagate`/
  :func:`stream` below, built on :func:`ballistic_position_simultaneous`,
  serve this convention: no correction term is needed, since ``x``/``y``/
  ``z`` already describe one consistent instant.
* ``xigma_i.particles.Bunch``'s ``x0``/``y0`` are instead each particle's
  transverse position extrapolated to ``z=0`` along its own straight-line
  trajectory (``sample_bunch`` draws them independently of ``z0``, i.e. as
  the beam's transverse profile AT the waist/interaction plane) -- while
  ``z0`` is that same particle's real, independently-drawn longitudinal
  offset. ``(x0, y0, z0)`` together are therefore NOT a simultaneous
  position the way ``MacroBunch``'s fields are; using them as such would
  silently drop a real ``thx * z0``-scale correction. :func:`ballistic_position_z0_reference`
  serves this second convention -- extracted verbatim from
  ``xigma_i.particles``' existing formula (not re-derived), so
  ``xigma_i.particles`` can call it directly with zero behavior change
  (see that module's integration).

Both are backend-agnostic by construction: only ``+``, ``-``, ``*``,
``**0.5`` (no ``numpy``/``cupy``-specific calls), so they run unmodified
on numpy arrays, cupy arrays, or plain floats -- the same trick
``bunch.fit_gaussian`` already relies on to stay cupy-safe without
importing it.

Reconciling ``sample_gaussian_bunch``'s own longitudinal draw (``z =
beta0 * C_LIGHT * t``, a single beam-mean ``beta0``) with either
per-particle ``vz`` convention above is a deliberate non-goal here:
that function constructs a bunch at the waist from scratch, not
propagating an existing one, and is left as-is (already tested).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterator

import numpy as np

from .bunch import MacroBunch

__all__ = [
    "ballistic_position_simultaneous",
    "ballistic_position_z0_reference",
    "propagate",
    "stream",
]


def ballistic_position_simultaneous(x0, y0, z0, thx, thy, dt):
    """Straight-line position at time offset ``dt`` later, given a
    per-particle reference ``(x0, y0, z0, thx, thy)`` that describes each
    particle's REAL, simultaneous position at ``dt=0`` (the
    :class:`~compton_io.bunch.MacroBunch` convention -- see module
    docstring). ``thx``/``thy`` are (small-angle) transverse velocity
    fractions, ``vz = sqrt(1 - thx**2 - thy**2)``. Returns ``(x, y, z)``,
    each the same shape as the broadcast of the inputs.
    """
    vz = (1.0 - thx**2 - thy**2) ** 0.5
    x = x0 + thx * vz * dt
    y = y0 + thy * vz * dt
    z = z0 + vz * dt
    return x, y, z


def ballistic_position_z0_reference(x0, y0, z0, thx, thy, t):
    """Straight-line position at time offset ``t``, given a per-particle
    reference where ``x0``/``y0`` are each particle's transverse position
    extrapolated to ``z=0`` (not necessarily its position at its own
    ``z0``) and ``z0`` is that particle's real longitudinal offset -- the
    ``xigma_i.particles.Bunch`` convention (see module docstring).
    ``dt0 = z0 / vz`` is the time needed to travel from ``z=0`` to ``z0``
    at this particle's own ``vz``, folded into the ``x``/``y`` evolution
    (``z`` itself already starts at ``z0`` directly) so ``x(t)``/``y(t)``
    continue that same line correctly. Extracted verbatim from
    ``xigma_i.particles``' existing formula, not re-derived -- see that
    module's integration for why it calls this instead of
    :func:`ballistic_position_simultaneous`.
    """
    vz = (1.0 - thx**2 - thy**2) ** 0.5
    dt0 = z0 / vz
    x = x0 + thx * (t + dt0)
    y = y0 + thy * (t + dt0)
    z = z0 + vz * t
    return x, y, z


def propagate(bunch: MacroBunch, dt) -> MacroBunch:
    """Ballistically drift every macroparticle in ``bunch`` by a time
    offset ``dt`` (SI seconds; scalar, or one value per particle).
    ``gamma``/``thx``/``thy`` are unchanged (straight-line, no
    acceleration) -- only ``x``/``y``/``z`` move. Thin :class:`MacroBunch`
    wrapper over :func:`ballistic_position_simultaneous`, for external SI
    callers (a GUI, a table-building script, tests) that want "this
    bunch's state at a different time" without hand-rolling the drift
    themselves.
    """
    from .constants import C_LIGHT

    x, y, z = ballistic_position_simultaneous(
        np.asarray(bunch.x, dtype=float), np.asarray(bunch.y, dtype=float),
        np.asarray(bunch.z, dtype=float), np.asarray(bunch.thx, dtype=float),
        np.asarray(bunch.thy, dtype=float), C_LIGHT * np.asarray(dt, dtype=float),
    )
    return replace(bunch, x=x, y=y, z=z)


def stream(bunch: MacroBunch, t_grid) -> Iterator[MacroBunch]:
    """Yield a propagated :class:`MacroBunch` snapshot at each time in
    ``t_grid`` (SI seconds, monotonic) -- for a caller that wants distinct
    bunch-state samples at a sequence of time instances (e.g. building a
    time-resolved table one slice at a time) without hand-rolling the
    drift at each step itself. Each snapshot is independently computed
    from ``bunch`` (the original, not the previous snapshot), so ``t_grid``
    need not be evenly spaced and yields are exact, not accumulated.
    """
    for t in t_grid:
        yield propagate(bunch, t)
