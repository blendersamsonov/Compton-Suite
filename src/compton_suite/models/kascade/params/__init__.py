"""This model's own parameter contract, built on the shared
``compton_suite.io`` framework.

The parameter-semantics/units framework itself (``PhysicalQuantity``, the
``PhysicalMeaning``/``WidthConvention``/``TimeConvention``/
``AmplitudeConvention`` enums, canonical conversion, ``ParameterSpec``/
``ModelSpec``, ``adapt_to_model``) lives in ``compton_suite.io`` and is
re-exported here unchanged, so ``models.kascade.params.PhysicalQuantity``
and (say) ``models.xigma_i.params.PhysicalQuantity`` are the *same* class,
not two independently-defined look-alikes. Only ``spec.py``'s
``KASCADE_SPEC``/``KASCADE_DIAGNOSTIC_SPEC`` -- this model's own parameter
contract instance -- is defined here.

Typical use (see ``spec.py`` for the concrete spec, and
``scripts/physics_params_demo.py`` for a full example):

    from compton_suite.models.kascade.params import (
        PhysicalQuantity, PhysicalMeaning, TimeConvention, adapt_to_model,
    )
    from compton_suite.models.kascade.params import KASCADE_SPEC

    bunch_length = PhysicalQuantity(
        magnitude=50.0, unit="picosecond",
        meaning=PhysicalMeaning.BUNCH_LENGTH,
        convention=TimeConvention.SIGMA_INTENSITY_RMS,
    )
    adapted = adapt_to_model({"sigma_par_e": bunch_length, ...}, KASCADE_SPEC)
    # adapted["sigma_par_e"] is now in KASCADE_SPEC's own convention/unit
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

from .spec import KASCADE_DIAGNOSTIC_SPEC, KASCADE_SPEC

__all__ = [
    "PhysicalQuantity",
    "PhysicalMeaning",
    "WidthConvention",
    "TimeConvention",
    "AmplitudeConvention",
    "ParameterSpec",
    "ModelSpec",
    "KASCADE_SPEC",
    "KASCADE_DIAGNOSTIC_SPEC",
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
