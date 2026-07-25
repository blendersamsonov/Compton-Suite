"""Parameter semantics & unit normalisation layer for this model's own
parameter contract.

Solves two independent problems for values crossing the compton-gui <->
xigma-i boundary:

1. **Units** -- handled by ``pint`` (see ``units.py``).
2. **Semantics/convention** -- is a "width" the RMS of the intensity
   profile, the FWHM, or the 1/e^2 radius? Is a "duration" a sigma or a
   FWHM? Is an amplitude peak or RMS? Handled by this module: every value
   is a ``PhysicalQuantity`` (value + unit + meaning + convention), never a
   bare float, converted through one canonical representation per meaning
   (``canonical.py``) and out to this model's own convention/unit,
   declared in ``spec.py``'s ``XIGMA_SPEC``.

Originally implemented inside compton-gui (see that repo's
``Conventions-and-units.md``) as a GUI-side framework with one
``ModelSpec`` per pluggable physics engine (``schemas/xigma.py``,
``schemas/kascade.py``); moved here so this model declares its own
parameter contract instead of the GUI declaring it on the model's behalf.
compton-gui's copy of the generic framework (``enums``/``units``/
``quantities``/``canonical``/``converters``/``validation``/``schema``/
``adapter``, unchanged) stays in place there for ``kascade``, which hasn't
had the same schema-ownership move yet.

Typical use (see ``spec.py`` for the concrete spec, and compton-gui's
``scripts/physics_params_demo.py`` for a full example):

    from xigma_i.params import (
        PhysicalQuantity, PhysicalMeaning, WidthConvention, adapt_to_model,
    )
    from xigma_i.params import XIGMA_SPEC

    laser_width = PhysicalQuantity(
        magnitude=5.0, unit="micrometer",
        meaning=PhysicalMeaning.LASER_WIDTH,
        convention=WidthConvention.FWHM_INTENSITY,
    )
    adapted = adapt_to_model({"sigma0_l": laser_width, ...}, XIGMA_SPEC)
    # adapted["sigma0_l"] is now in XIGMA_SPEC's own convention/unit

**Not yet wired into ``gui_adapter.params_to_config``** -- that still does
the FWHM/waist/duration arithmetic by hand.
"""

from .adapter import adapt_to_model, params_to_floats
from .canonical import CANONICAL_CONVENTIONS, CANONICAL_UNIT, from_canonical, to_canonical
from .enums import AmplitudeConvention, PhysicalMeaning, TimeConvention, WidthConvention
from .quantities import PhysicalQuantity
from .schema import ModelSpec, ParameterSpec
from .spec import XIGMA_DIAGNOSTIC_SPEC, XIGMA_SPEC
from .validation import (
    MeaningMismatchError,
    MissingConventionError,
    PhysicsParamsError,
    UnitMismatchError,
    UnknownConversionError,
    validate_against_spec,
    validate_quantity,
)

__all__ = [
    "PhysicalQuantity",
    "PhysicalMeaning",
    "WidthConvention",
    "TimeConvention",
    "AmplitudeConvention",
    "ParameterSpec",
    "ModelSpec",
    "XIGMA_SPEC",
    "XIGMA_DIAGNOSTIC_SPEC",
    "CANONICAL_CONVENTIONS",
    "CANONICAL_UNIT",
    "to_canonical",
    "from_canonical",
    "adapt_to_model",
    "params_to_floats",
    "validate_quantity",
    "validate_against_spec",
    "PhysicsParamsError",
    "MissingConventionError",
    "UnknownConversionError",
    "UnitMismatchError",
    "MeaningMismatchError",
]
