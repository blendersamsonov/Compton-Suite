"""Thin re-export of ``compton_suite.io``'s parameter-semantics/unit
normalisation framework, per ``Conventions-and-units.md``.

The framework itself (``PhysicalQuantity``, the enums, canonical
conversion, ``ParameterSpec``/``ModelSpec``, ``adapt_to_model``) lives in
``compton_suite.io`` -- **not defined here**, so
``compton_suite.gui.physics_params.PhysicalQuantity`` is the literal same
class as e.g. ``compton_suite.models.xigma_i.params.PhysicalQuantity``, not
an independently-defined look-alike (an earlier version of this move gave
each repo its own copy, which broke exactly that; see git history if
curious). This module exists so existing `from compton_suite.gui.
physics_params import ...` call sites don't need to change.

**Every model's own ``ModelSpec`` now lives with that model**, not here:
``compton_suite.models.xigma_i.params.XIGMA_SPEC`` and
``compton_suite.models.kascade.params.KASCADE_SPEC`` (the `schemas/`
sub-package that used to hold `KASCADE_SPEC` on the GUI's behalf has been
removed -- the model declares its own parameter contract instead of the
GUI declaring it for it, matching xigma-i's already-moved pattern).

Typical use (see `compton_suite.models.xigma_i.params`/
`compton_suite.models.kascade.params` for the concrete specs, and
`scripts/physics_params_demo.py` for a full example):

    from compton_suite.gui.physics_params import (
        PhysicalQuantity, PhysicalMeaning, WidthConvention, adapt_to_model,
    )
    from compton_suite.models.xigma_i.params import XIGMA_SPEC

    laser_width = PhysicalQuantity(
        magnitude=5.0, unit="micrometer",
        meaning=PhysicalMeaning.LASER_WIDTH,
        convention=WidthConvention.FWHM_INTENSITY,
    )
    adapted = adapt_to_model({"sigma0_l": laser_width, ...}, XIGMA_SPEC)
    # adapted["sigma0_l"] is now in XIGMA_SPEC's own convention/unit
"""

from compton_suite.io import (
    CANONICAL_CONVENTIONS,
    CANONICAL_UNIT,
    AmplitudeConvention,
    MeaningMismatchError,
    MissingConventionError,
    ModelSpec,
    ParameterSpec,
    PhysicalMeaning,
    PhysicalQuantity,
    PhysicsParamsError,
    TimeConvention,
    UnitMismatchError,
    UnknownConversionError,
    WidthConvention,
    adapt_to_model,
    from_canonical,
    params_to_floats,
    to_canonical,
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
