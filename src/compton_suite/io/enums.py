"""Shared vocabulary for parameter semantics.

A ``PhysicalMeaning`` names *what* a number is (a laser transverse size, a
pulse length, ...). A convention enum (``WidthConvention``/``TimeConvention``/
``AmplitudeConvention``) names *which* mathematical definition was used to
turn a fuzzy physical width into one number (RMS of the intensity profile?
FWHM? the 1/e^2 radius?). The two are orthogonal on purpose: two codes can
agree on meaning (both call it "laser width") while disagreeing on
convention (one means sigma, the other FWHM), which is exactly the silent
mismatch this package exists to catch.
"""

from __future__ import annotations

from enum import Enum, auto


class PhysicalMeaning(Enum):
    LASER_WIDTH = auto()          # transverse laser (photon-density) size
    PULSE_DURATION = auto()       # laser longitudinal/temporal extent
    LASER_AMPLITUDE = auto()      # normalised vector potential a0
    ELECTRON_BEAM_SIZE = auto()   # transverse electron-bunch size
    BUNCH_LENGTH = auto()         # electron-bunch longitudinal/temporal extent

    # New meanings for 2026-07 quantity-aware refactor: parameters without
    # convention ambiguity (see the schema's "PhysicalQuantity everywhere"
    # design).  Every one of these carries NoConvention.PLAIN as its
    # convention -- the value means what the meaning says, no further
    # interpretation needed.
    PULSE_ENERGY = auto()         # laser pulse energy (J) -- no convention
    WAVELENGTH = auto()           # laser central wavelength (m) -- no convention
    BEAM_ENERGY = auto()          # electron kinetic energy (eV) -- no convention
    EMITTANCE = auto()           # geometric emittance (m*rad) -- no convention
    BUNCH_CHARGE = auto()        # bunch charge (C) -- no convention
    DISPLACEMENT = auto()        # spatial offset (m) -- no convention
    ANGLE = auto()               # crossing/polarisation angle (rad) -- no convention


class NoConvention(Enum):
    """Sentinel for parameters whose meaning is unambiguous -- the value
    means exactly what ``PhysicalMeaning`` says, with no RMS-vs-FWHM-vs-1/e^2
    style ambiguity to resolve.  Used instead of ``None`` so the convention
    slot on every ``PhysicalQuantity`` is a real ``Enum`` member, never
    ``None``, keeping ``validate_quantity``'s check simple."""
    PLAIN = auto()


class WidthConvention(Enum):
    """Definitions of a transverse Gaussian width. Reused verbatim for
    ``ELECTRON_BEAM_SIZE`` -- the same Gaussian-width algebra applies to a
    particle-density profile as to a photon-density (intensity) profile."""

    SIGMA_INTENSITY_RMS = auto()   # I(r) ~ exp(-r^2 / (2 sigma^2))
    SIGMA_FIELD_RMS = auto()       # E(r) ~ exp(-r^2 / (2 sigma_field^2)); sigma_field = sigma_intensity*sqrt(2)
    FWHM_INTENSITY = auto()        # full width at half of peak intensity
    W0_1E2 = auto()                # radius at which intensity falls to 1/e^2 of peak (laser "waist")


class TimeConvention(Enum):
    """Definitions of a longitudinal/temporal Gaussian width. Same algebra
    as ``WidthConvention`` minus ``W0_1E2`` (no standard "waist" analogue in
    time); kept as a separate enum rather than reusing ``WidthConvention``
    so a width value can never be silently accepted where a duration was
    expected, or vice versa."""

    SIGMA_INTENSITY_RMS = auto()
    SIGMA_FIELD_RMS = auto()
    FWHM_INTENSITY = auto()


class AmplitudeConvention(Enum):
    A0_PEAK = auto()   # amplitude at the trajectory/profile peak
    A0_RMS = auto()     # cycle-averaged sqrt(<a^2>); for linear polarisation, a0_rms = a0_peak / sqrt(2)
