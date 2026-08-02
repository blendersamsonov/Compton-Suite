"""Shared pint unit registry, physical constants, and the parameter-
semantics/convention vocabulary every ``PhysicalQuantity`` carries.

**One registry for the whole suite** -- every consumer (``compton_suite.gui``,
``xigma_i``, ``kascade``) imports this exact ``ureg``/``Quantity``, not one
of its own, because pint quantities built from different registries can't
interoperate.

Also defines a custom ``"light_time"`` context that lets a length convert
to/from a duration via ``length = c * time``. Both ``xigma_i`` and
``kascade`` store pulse/bunch "length" natively as a position-domain sigma
(``sigma_par_L = c * pulse_duration``, see each engine's
``params_to_config``) rather than as a time, so ``PULSE_DURATION``/
``BUNCH_LENGTH`` quantities need this context to move between the unit a
GUI would naturally show (seconds, picoseconds) and the unit the physics
code actually stores (metres, centimetres).

**SI values are canonical.** ``compton_suite.gui`` and ``kascade`` are SI
throughout and use the constants below directly (plain floats, SI base
units -- the registry itself is a default/SI-system ``pint.UnitRegistry``,
*not* constructed with ``system="cgs"``, precisely so ``to_base_units()``
below lands in SI, not CGS). ``xigma_i`` is CGS-Gaussian throughout (see
its own docs) and converts at its own boundary -- CGS-Gaussian isn't a pure
unit re-scaling of SI for electromagnetic quantities (charge isn't
dimensionally the same in the two systems), so pint's registry can't
convert electromagnetic quantities between them directly; ``xigma_i``
applies the textbook exact conversion (``1 C = c[cm/s] / 10 statC``) by
hand instead.

Also hosts the parameter-semantics/convention vocabulary: a
``PhysicalMeaning`` names *what* a number is (a laser transverse size, a
pulse length, ...); a convention enum (``WidthConvention``/
``TimeConvention``/``AmplitudeConvention``) names *which* mathematical
definition was used to turn a fuzzy physical width into one number (RMS of
the intensity profile? FWHM? the 1/e^2 radius?). The two are orthogonal on
purpose: two values can agree on meaning (both "a laser width") while
disagreeing on convention (one means sigma, the other FWHM) -- exactly the
silent mismatch ``PhysicalQuantity`` exists to catch. Every value that
crosses a model/GUI boundary should travel as a ``PhysicalQuantity`` (value
+ unit + meaning + convention), never a bare float; each model converts a
``PhysicalQuantity`` to whatever convention/unit it needs directly via
``convert_width``/``convert_time``/``convert_amplitude`` at its own
boundary (there is no generic "spec"/"adapt to model" indirection layer).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import pint

ureg = pint.UnitRegistry()
Quantity = ureg.Quantity


def _const(name: str) -> float:
    """A CODATA constant from pint's own table, as a plain float in SI base
    units -- self-verifying (no risk of a hand-typed digit drifting from the
    real value) and usable directly in numpy/numba/cupy arithmetic, which a
    bare pint ``Quantity`` is not."""
    return float(Quantity(1, name).to_base_units().magnitude)


C_LIGHT = _const("speed_of_light")
E_CHARGE = _const("elementary_charge")
HBAR = _const("hbar")
ME = _const("electron_mass")
MEC2 = ME * C_LIGHT ** 2
MEC2_EV = float(
    (Quantity(1, "electron_mass") * Quantity(1, "speed_of_light") ** 2).to("eV").magnitude
)
EPS0 = _const("vacuum_permittivity")
ALPHA = _const("fine_structure_constant")
R_E_M = _const("classical_electron_radius")
SIGMA_T_M2 = _const("thomson_cross_section")

# Quantity companions of the bare floats above -- for formulas that combine
# one of these constants with a PhysicalQuantity field's own `.quantity`
# (a bare pint Quantity) directly, so the arithmetic is genuine pint
# dimensional analysis start to finish rather than a bare-float rescale of
# whatever unit the field happens to be stored in. The bare floats above are
# untouched and stay the only constants kascade.py/xigma_i's CGS code use.
C_LIGHT_Q = Quantity(C_LIGHT, "meter / second")
E_CHARGE_Q = Quantity(E_CHARGE, "coulomb")
HBAR_Q = Quantity(HBAR, "joule * second")
ME_Q = Quantity(ME, "kilogram")
MEC2_EV_Q = Quantity(MEC2_EV, "electron_volt")
EPS0_Q = Quantity(EPS0, "farad / meter")

# CGS views (erg, gram, cm, statcoulomb) of the constants above -- xigma_i
# is CGS-Gaussian throughout and needs these directly. EL_STATC uses the
# exact textbook conversion (1 C = c[cm/s] / 10 statC) since pint's registry
# can't convert electromagnetic quantities between SI and Gaussian directly
# (see module docstring) -- applied by hand instead of via pint.
HBAR_ERG_S = HBAR * 1e7
ME_G = ME * 1e3
C_CM_S = C_LIGHT * 1e2
EL_STATC = E_CHARGE * C_LIGHT * 10.0
M_TO_CM = 100.0
R_E_CM = R_E_M * 1e2

LIGHT_TIME_CONTEXT = "light_time"

# Defined via the DSL (`Context.from_lines`, the same mechanism pint's own
# built-in contexts like "spectroscopy" use) rather than the plain-Python
# `Context.add_transformation` API: the latter registers its dimensionality
# keys as `ParserHelper` instances, which don't compare equal to the
# `UnitsContainer` instances `Quantity.to()` looks them up with in this pint
# version (0.25) -- a silent `DimensionalityError` despite the context being
# "registered". The DSL form doesn't have this mismatch.
_ctx = pint.Context.from_lines([
    f"@context {LIGHT_TIME_CONTEXT}",
    "    [length] -> [time]: value / speed_of_light",
    "    [time] -> [length]: value * speed_of_light",
])
ureg.add_context(_ctx)


# ---------------------------------------------------------------------------
# Parameter semantics: meaning + convention vocabulary
# ---------------------------------------------------------------------------
class PhysicalMeaning(Enum):
    LASER_WIDTH = auto()          # transverse laser (photon-density) size
    PULSE_DURATION = auto()       # laser longitudinal/temporal extent
    LASER_AMPLITUDE = auto()      # normalised vector potential a0
    ELECTRON_BEAM_SIZE = auto()   # transverse electron-bunch size
    BUNCH_LENGTH = auto()         # electron-bunch longitudinal/temporal extent

    # Meanings without convention ambiguity -- every one of these carries
    # NoConvention.PLAIN: the value means what the meaning says, no further
    # interpretation needed.
    PULSE_ENERGY = auto()         # laser pulse energy (J) -- no convention
    WAVELENGTH = auto()           # laser central wavelength (m) -- no convention
    BEAM_ENERGY = auto()          # electron kinetic energy (eV) -- no convention
    EMITTANCE = auto()            # geometric emittance (m*rad) -- no convention
    BUNCH_CHARGE = auto()         # bunch charge (C) -- no convention
    DISPLACEMENT = auto()         # spatial offset (m) -- no convention
    ANGLE = auto()                # crossing/polarisation angle (rad) -- no convention


class NoConvention(Enum):
    """Sentinel for parameters whose meaning is unambiguous -- the value
    means exactly what ``PhysicalMeaning`` says, with no RMS-vs-FWHM-vs-1/e^2
    style ambiguity to resolve. Used instead of ``None`` so the convention
    slot on every ``PhysicalQuantity`` is a real ``Enum`` member, never
    ``None``."""
    PLAIN = auto()


class WidthConvention(Enum):
    """Definitions of a transverse Gaussian width (laser or electron beam)."""

    SIGMA_INTENSITY_RMS = auto()   # I(r) ~ exp(-r^2 / (2 sigma^2))
    SIGMA_FIELD_RMS = auto()       # E(r) ~ exp(-r^2 / (2 sigma_field^2)); sigma_field = sigma_intensity*sqrt(2)
    FWHM_INTENSITY = auto()        # full width at half of peak intensity
    W0_1E2 = auto()                # 1/e^2 intensity radius ("waist" convention)


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
    """Definitions of a laser normalised-vector-potential amplitude
    (linear polarisation: ``<a^2> = a0_peak^2 / 2``)."""

    A0_PEAK = auto()
    A0_RMS = auto()


_FWHM_OVER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


# ---------------------------------------------------------------------------
# Gaussian width/time algebra
# ---------------------------------------------------------------------------
def fwhm_to_sigma_intensity(fwhm: float) -> float:
    return fwhm / _FWHM_OVER_SIGMA


def sigma_intensity_to_fwhm(sigma: float) -> float:
    return sigma * _FWHM_OVER_SIGMA


def sigma_field_to_sigma_intensity(sigma_field: float) -> float:
    return sigma_field / np.sqrt(2.0)


def sigma_intensity_to_sigma_field(sigma_intensity: float) -> float:
    return sigma_intensity * np.sqrt(2.0)


def w0_to_sigma_intensity(w0: float) -> float:
    return w0 / 2.0


def sigma_intensity_to_w0(sigma_intensity: float) -> float:
    return sigma_intensity * 2.0


# ---------------------------------------------------------------------------
# Amplitude algebra (linear polarisation: <a^2> = a0_peak^2 / 2)
# ---------------------------------------------------------------------------
def a0_peak_to_a0_rms(a0_peak: float) -> float:
    return a0_peak / np.sqrt(2.0)


def a0_rms_to_a0_peak(a0_rms: float) -> float:
    return a0_rms * np.sqrt(2.0)


# ---------------------------------------------------------------------------
# Unified per-family engines: convert(value, from_convention, to_convention)
# ---------------------------------------------------------------------------
_WIDTH_TO_CANONICAL = {
    WidthConvention.SIGMA_INTENSITY_RMS: lambda v: v,
    WidthConvention.SIGMA_FIELD_RMS: sigma_field_to_sigma_intensity,
    WidthConvention.FWHM_INTENSITY: fwhm_to_sigma_intensity,
    WidthConvention.W0_1E2: w0_to_sigma_intensity,
}
_WIDTH_FROM_CANONICAL = {
    WidthConvention.SIGMA_INTENSITY_RMS: lambda v: v,
    WidthConvention.SIGMA_FIELD_RMS: sigma_intensity_to_sigma_field,
    WidthConvention.FWHM_INTENSITY: sigma_intensity_to_fwhm,
    WidthConvention.W0_1E2: sigma_intensity_to_w0,
}

_TIME_TO_CANONICAL = {
    TimeConvention.SIGMA_INTENSITY_RMS: lambda v: v,
    TimeConvention.SIGMA_FIELD_RMS: sigma_field_to_sigma_intensity,
    TimeConvention.FWHM_INTENSITY: fwhm_to_sigma_intensity,
}
_TIME_FROM_CANONICAL = {
    TimeConvention.SIGMA_INTENSITY_RMS: lambda v: v,
    TimeConvention.SIGMA_FIELD_RMS: sigma_intensity_to_sigma_field,
    TimeConvention.FWHM_INTENSITY: sigma_intensity_to_fwhm,
}

_AMPLITUDE_TO_CANONICAL = {
    AmplitudeConvention.A0_PEAK: lambda v: v,
    AmplitudeConvention.A0_RMS: a0_rms_to_a0_peak,
}
_AMPLITUDE_FROM_CANONICAL = {
    AmplitudeConvention.A0_PEAK: lambda v: v,
    AmplitudeConvention.A0_RMS: a0_peak_to_a0_rms,
}


class UnknownConversionError(ValueError):
    """Raised when a convention has no registered conversion in its family."""


def _convert(value: float, from_conv, to_conv, to_canonical: dict, from_canonical: dict, family: str) -> float:
    try:
        to_c = to_canonical[from_conv]
    except KeyError as exc:
        raise UnknownConversionError(f"no {family} conversion registered for {from_conv!r}") from exc
    try:
        from_c = from_canonical[to_conv]
    except KeyError as exc:
        raise UnknownConversionError(f"no {family} conversion registered for {to_conv!r}") from exc
    return from_c(to_c(value))


def convert_width(value: float, from_conv: WidthConvention, to_conv: WidthConvention) -> float:
    return _convert(value, from_conv, to_conv, _WIDTH_TO_CANONICAL, _WIDTH_FROM_CANONICAL, "width")


def convert_time(value: float, from_conv: TimeConvention, to_conv: TimeConvention) -> float:
    return _convert(value, from_conv, to_conv, _TIME_TO_CANONICAL, _TIME_FROM_CANONICAL, "time")


def convert_amplitude(value: float, from_conv: AmplitudeConvention, to_conv: AmplitudeConvention) -> float:
    return _convert(value, from_conv, to_conv, _AMPLITUDE_TO_CANONICAL, _AMPLITUDE_FROM_CANONICAL, "amplitude")


# ---------------------------------------------------------------------------
# The value + unit + meaning + convention wrapper
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PhysicalQuantity:
    magnitude: float
    unit: str
    meaning: PhysicalMeaning
    convention: Enum | None = None

    @property
    def quantity(self) -> Quantity:
        return ureg.Quantity(self.magnitude, self.unit)

    def to_unit(self, unit: str, *, context: str | None = None) -> "PhysicalQuantity":
        """Pure unit conversion (metres <-> centimetres, ps <-> s, ...).
        Never changes ``convention`` -- use ``convert_width``/``convert_time``/
        ``convert_amplitude`` for that. ``context`` is a pint context name
        (see ``LIGHT_TIME_CONTEXT`` above) for conversions pint can't do
        unaided, such as length <-> time via c."""
        q = self.quantity
        converted = q.to(unit, context) if context else q.to(unit)
        return PhysicalQuantity(float(converted.magnitude), unit, self.meaning, self.convention)

    @property
    def si_magnitude(self) -> float:
        """Convenience: magnitude in the SI base unit for this quantity's
        dimension. Raises if the pint dimension can't be expressed in a
        single SI base unit (dimensionless values already are; tagged units
        like ``"hour"`` work fine)."""
        return float(self.quantity.to_base_units().magnitude)
