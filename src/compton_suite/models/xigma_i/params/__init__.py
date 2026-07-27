"""This model's own parameter contract, built on the shared
``compton_io`` framework.

The parameter-semantics/units framework itself (``PhysicalQuantity``, the
``PhysicalMeaning``/``WidthConvention``/``TimeConvention``/
``AmplitudeConvention`` enums, canonical conversion, ``ParameterSpec``/
``ModelSpec``, ``adapt_to_model``) lives in ``compton_io`` and is
re-exported here unchanged, so ``xigma_i.params.PhysicalQuantity`` and (say)
``compton_guide.physics_params.PhysicalQuantity`` are the *same* class, not
two independently-defined look-alikes. Only ``spec.py``'s ``XIGMA_SPEC``/
``XIGMA_DIAGNOSTIC_SPEC`` -- this model's own parameter contract instance --
is defined here.

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

**Partially wired into ``gui_adapter.params_to_config``**: ``sigma_par_e``/
``sigma_par_L`` (raw, convention-ambiguous duration inputs) go through
``adapt_to_model``/``params_to_floats`` against this package's own
``XIGMA_SPEC``; ``sigma0_x``/``sigma0_y``/``sigma0_l`` still do the
emittance/beta/Rayleigh-length arithmetic by hand, since they aren't raw
convention-ambiguous inputs today -- see ``spec.py``'s module docstring.
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

from .spec import XIGMA_DIAGNOSTIC_SPEC, XIGMA_SPEC

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
