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
* ``xigma_i.particles``' internal ``x0``/``y0`` (computed inline in
  ``push_and_sample``, no longer a separate class -- see that module's
  ``_normalise_bunch``) are instead each particle's transverse position
  extrapolated to ``z=0`` along its own straight-line trajectory
  (``compton_io.bunch.sample_gaussian_bunch`` draws ``x``/``y``
  independently of ``z``, i.e. as the beam's transverse profile AT the
  waist/interaction plane, carried straight through unchanged into
  ``x0``/``y0``) -- while ``z0`` is that same particle's real,
  independently-drawn longitudinal offset. ``(x0, y0, z0)`` together are
  therefore NOT a simultaneous position the way ``MacroBunch``'s fields
  are; using them as such would
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
    "laser_overlap_time_window",
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
    ``xigma_i.particles`` convention (see module docstring).
    ``dt0 = z0 / vz`` is the time needed to travel from ``z=0`` to ``z0``
    at this particle's own ``vz``, folded into the ``x``/``y`` evolution
    (``z`` itself already starts at ``z0`` directly) so ``x(t)``/``y(t)``
    continue that same line correctly. Extracted verbatim from
    ``xigma_i.particles``' own formula, not re-derived --
    ``xigma_i.particles.push_and_sample`` calls this directly (see that
    module for why it needs this convention instead of
    :func:`ballistic_position_simultaneous`).
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


def laser_overlap_time_window(z0, *, k0_las, sigma_lz, sigma_lr0,
                               beta_ff=0.0, gauss_width=3.0, lorentz_width=8.0, xp=np):
    """Per-particle time window ``[t0, t1]`` (normalised length units,
    e.g. ``k0_las*c*t``) bounding where a ballistic particle at
    longitudinal offset ``z0`` (same normalised units, e.g. ``k0_las*z``)
    is within ``lorentz_width`` Rayleigh ranges transversely and
    ``gauss_width`` pulse-duration Gaussian widths temporally of a
    Gaussian laser pulse centred at the origin -- i.e. the window worth
    integrating a beam-laser overlap over, outside of which the pulse
    envelope is negligible. Model-agnostic: any model doing a ballistic
    push through a Gaussian laser pulse needs the same bound (originally
    ``xigma_i.particles._time_window``, moved here since nothing about it
    is xigma-specific).

    z0: per-particle longitudinal offset, normalised (e.g. ``k0_las * z_cm``
        for xigma_i's CGS/k0-normalised convention).
    k0_las: laser wavenumber (``2*pi/wavelength``) -- multiplies the raw
        ``sigma_lz``/``sigma_lr0`` below to put them in the same normalised
        unit as ``z0``.
    sigma_lz, sigma_lr0: the laser pulse's RMS duration and RMS focal
        radius, as raw lengths in whatever unit ``1/k0_las`` is (e.g. cm
        for xigma_i) -- *not* pre-normalised; this function does that.
    beta_ff: flying-focus factor (0 = static focus, 1 = co-moving) -- an
        engine-specific laser extra (see ``compton_io.laser``'s module
        docstring on why this isn't part of the shared laser
        representation), passed through as a plain scalar.
    gauss_width, lorentz_width: how many pulse-duration Gaussian widths /
        Rayleigh-range Lorentzian widths out a trajectory is still
        considered "possibly inside the pulse" -- xigma_i's own defaults
        (3, 8) are passed explicitly by that caller; other models can pick
        their own.
    xp: array module ``z0`` belongs to (``numpy`` or ``cupy``) -- accepts
        an explicit module (rather than the ``**0.5``-only trick the other
        functions in this file use) since this needs ``maximum``/``minimum``,
        which aren't expressible as a bare operator.
    """
    zT = k0_las * sigma_lz
    zR = (k0_las * sigma_lr0) ** 2 * (1.0 + beta_ff) * 2.0

    sigma_tau = gauss_width * zT
    sigma_rayleigh = lorentz_width * zR

    t0 = (xp.maximum(-sigma_tau, (-z0 * (1 + beta_ff) - 2 * sigma_rayleigh) / (1 - beta_ff)) - z0) / 2
    t1 = (xp.minimum(sigma_tau, (-z0 * (1 + beta_ff) + 2 * sigma_rayleigh) / (1 - beta_ff)) - z0) / 2
    return t0, t1


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
