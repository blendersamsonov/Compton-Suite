"""Fail-fast checks. Every error below is meant to be raised, not logged --
a convention/unit mismatch must never be silently "fixed" by guessing (see
compton_suite.gui's ``Conventions-and-units.md``, section 10, the design doc this
package implements)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .quantities import PhysicalQuantity
    from .schema import ParameterSpec


class PhysicsParamsError(Exception):
    """Base class for every error this package raises."""


class MissingConventionError(PhysicsParamsError):
    """A ``PhysicalQuantity`` was constructed/used without a convention."""


class UnknownConversionError(PhysicsParamsError):
    """No conversion is registered between two conventions of the same
    meaning (e.g. a convention enum member with no entry in the converter
    table)."""


class UnitMismatchError(PhysicsParamsError):
    """A unit conversion pint itself refused (incompatible dimensions and
    no context bridges them)."""


class MeaningMismatchError(PhysicsParamsError):
    """A quantity's ``meaning`` doesn't match what the caller/spec expected
    (e.g. a ``LASER_WIDTH`` value handed to a ``BUNCH_LENGTH`` slot)."""


def validate_quantity(q: "PhysicalQuantity") -> None:
    if q.meaning is None:
        raise MissingConventionError(f"{q!r} has no PhysicalMeaning")
    # Meaning without convention ambiguity may carry None convention;
    # meanings with ambiguity must have one set.
    from .enums import NoConvention, PhysicalMeaning as _PM

    _NO_CONV = {_PM.PULSE_ENERGY, _PM.WAVELENGTH, _PM.BEAM_ENERGY,
                 _PM.EMITTANCE, _PM.BUNCH_CHARGE, _PM.DISPLACEMENT, _PM.ANGLE}
    if q.convention is None and q.meaning not in _NO_CONV:
        raise MissingConventionError(
            f"{q!r} has no convention, but {q.meaning!r} requires one "
            f"(not a NoConvention meaning)"
        )


def validate_against_spec(params: dict[str, "PhysicalQuantity"], spec: dict[str, "ParameterSpec"]) -> None:
    """Check that ``params`` supplies exactly what ``spec`` requires, with
    matching physical meaning. Does not check convention/unit -- those are
    reconciled by ``canonical.from_canonical``/``adapter.adapt_to_model``,
    which raise their own errors if a conversion is unknown."""
    for key, pspec in spec.items():
        if key not in params:
            raise MissingConventionError(f"missing required parameter {key!r} (spec: {pspec.description})")
        q = params[key]
        validate_quantity(q)
        if q.meaning != pspec.meaning:
            raise MeaningMismatchError(
                f"parameter {key!r}: expected meaning {pspec.meaning}, got {q.meaning}"
            )
