"""The shared "interaction parameters" bundle: an electron beam, a laser
pulse, and their collision geometry, considered together.

Every model's own ``Config`` used to carry beam+laser+geometry fields as a
flat, independently-duplicated bag (see ``bunch.beam_from_shared_fields``/
``laser.laser_from_shared_fields`` for the per-piece migration this
generalizes). This module gives that bundle one shared shape so a caller
holding one canonical (beam, pulse, geometry) triple can build every
model's own numerics-only ``Config`` from it, rather than each model
re-deriving/duplicating the physics parameters themselves.

Deliberately excludes anything with no cross-model meaning (grid/step/bin
counts, per-model numerical-control knobs) -- those stay model-owned, see
each model's own ``Config``. ``crossing_angle_rad``/``quantum`` are real
physics, not numerics, but not every model supports a nonzero/True value
(``xigma_i``/``delta`` are head-on-only with no quantum toggle) --
included here anyway since they're still physically meaningful collision
parameters, just sometimes unsupported by a particular model (that model's
own ``params_to_config`` is responsible for rejecting what it can't do,
same as today).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bunch import GaussianElectronBeam
from .constants import C_LIGHT, E_CHARGE, HBAR, MEC2_EV
from .laser import GaussianParaxialLaser

__all__ = ["InteractionGeometry", "InteractionParameters", "recoil_parameter"]


@dataclass(frozen=True)
class InteractionGeometry:
    """Collision geometry: foci displacement and crossing angle. SI units."""

    delta_x_m: float = 0.0
    delta_y_m: float = 0.0
    delta_z_m: float = 0.0
    crossing_angle_rad: float = 0.0


@dataclass(frozen=True)
class InteractionParameters:
    """One canonical (beam, pulse, geometry) triple -- the physics-parameter
    subset every model's ``Config`` should be built from, instead of each
    model owning its own copy of these fields."""

    beam: GaussianElectronBeam
    laser: GaussianParaxialLaser
    geometry: InteractionGeometry = InteractionGeometry()
    quantum: bool = False


def recoil_parameter(gamma: float, wavelength_m: float) -> float:
    """Quantum recoil parameter ``q = 4 * gamma * hbar * omega / (m_e * c^2)``.

    Dimensionless; ``q << 1`` is the classical (Thomson) regime,
    ``q ~ 1`` is the quantum (Klein-Nishina) regime.  Used by the GUI's
    status bar and by kascade's classical/quantum cross-section toggle.
    """
    omega = 2.0 * np.pi * C_LIGHT / wavelength_m
    return 4.0 * gamma * HBAR * omega / (MEC2_EV * E_CHARGE)
