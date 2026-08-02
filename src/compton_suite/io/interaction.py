"""The shared "interaction parameters" bundle: one canonical (laser,
electrons) pair every model's ``run(job)`` reads from directly
(``job.interaction``), rather than each model re-deriving/duplicating the
physics parameters itself.

``electrons`` is the full sampled :class:`Bunch` (not just its
``gaussian_fit``) -- electron sampling is the caller's job (see
``compton_suite.io.bunch.sample_gaussian_bunch``), so by the time an
``InteractionParameters`` exists, sampling has already happened. A model
that needs beam-level scalars (charge, emittance, ...) reads
``electrons.gaussian_fit.<attr>``; every ``Bunch`` reaching a ``Job`` is
expected to have ``gaussian_fit`` populated (only a raw ``.ele``-file load
that hasn't been fit yet would leave it ``None`` -- see ``gui/app.py``'s
``on_start()``).

Deliberately minimal otherwise: geometry/crossing-angle and any classical/
quantum toggle are NOT part of this shared bundle -- not every model
supports a nonzero/True value (kascade has a real ``crossing_angle``/
``quantum`` toggle; xigma_i/delta are head-on-only with no quantum toggle),
so those stay model-owned fields, read straight from ``Job.extra`` at each
adapter's own boundary (see each adapter's ``run()``).

Every physical parameter travels as a :class:`PhysicalQuantity` (never a
bare float).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bunch import Bunch
from .units import C_LIGHT, E_CHARGE, HBAR, MEC2_EV, PhysicalQuantity
from .laser import GaussianParaxialLaser

__all__ = ["InteractionParameters", "recoil_parameter"]

@dataclass(frozen=True)
class InteractionParameters:
    """One canonical (laser, electrons) pair -- the physics-parameter
    bundle every model's ``run(job)`` reads from, instead of each model
    owning its own copy of these fields."""

    laser: GaussianParaxialLaser
    electrons: Bunch


def recoil_parameter(gamma: float, wavelength_m: float) -> float:
    """Quantum recoil parameter ``q = 4 * gamma * hbar * omega / (m_e * c^2)``.

    Dimensionless; ``q << 1`` is the classical (Thomson) regime,
    ``q ~ 1`` is the quantum (Klein-Nishina) regime.  Used by the GUI's
    status bar and by kascade's classical/quantum cross-section toggle.
    """
    omega = 2.0 * np.pi * C_LIGHT / wavelength_m
    return 4.0 * gamma * HBAR * omega / (MEC2_EV * E_CHARGE)
