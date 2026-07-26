"""Shared physical constants, pint unit registry, parameter-semantics/
convention framework, and electron-bunch/laser-pulse representations for
the ComptonSuite toolkit (``compton_suite``, ``compton_guide``, ``xigma_i``,
``xigma_direct``, ``kascade``).

Sibling repos plug physics engines into a shared GUI through a duck-typed
``ModelAdapter`` contract, and previously each hand-maintained its own copy
of physical constants and (where a convention layer existed at all) its own
copy of the enums/dataclasses used to describe parameter semantics -- with
a real, ~1.6e-8 relative numeric disagreement between two of the constant
copies (see ``constants.py``'s docstring), and, for the convention layer,
structurally-identical-but-not-the-same-class duplication once more than
one repo had a copy. This package exists so there is exactly one copy of
each, that every consumer imports directly (never vendors or re-derives):

1. **Physical constants** -- ``constants.py``, derived from pint's own
   built-in CODATA values rather than hand-typed literals.
2. **A shared pint unit registry** -- ``units.py``, including a custom
   ``"light_time"`` context (length <-> ``c * duration``).
3. **Parameter semantics/convention** -- is a "width" the RMS of the
   intensity profile, the FWHM, or the 1/e^2 radius? Is a "duration" a
   sigma or a FWHM? Is an amplitude peak or RMS? Every value that crosses a
   model/GUI boundary should travel as a ``PhysicalQuantity`` (value + unit
   + ``PhysicalMeaning`` + a convention enum), never a bare float, converted
   through one canonical representation per meaning (``canonical.py``) and
   out to whatever convention/unit a specific model declares in its own
   ``ModelSpec`` (owned by that model's own repo -- e.g.
   ``xigma_i.params.spec.XIGMA_SPEC`` -- not by this package).
4. **Electron-bunch representation** -- ``bunch.py``: ``MacroBunch`` (raw
   macroparticle arrays, e.g. loaded from an elegant ``.ele`` file) and
   ``GaussianElectronBeam`` (the ``gaussian_6d_waist`` v0.1 analytic
   contract, defined at the beam waist -- see ``specs/
   electron_beam_io_v0.1_full.md``), plus ``sample_gaussian_bunch``/
   ``fit_gaussian`` to convert between the two.
5. **Laser-pulse representation** -- ``laser.py``: ``GaussianParaxialLaser``,
   the ``gaussian_paraxial`` v0.1 analytic contract (see ``specs/
   gaussian_paraxial_laser_io_v0.1.md``).
6. **Photon/observable representations** -- ``photons.py``: the
   ``Sampled*``/``Binned*`` spectrum, angular-spectrum, temporal-envelope,
   and spatial-distribution dataclasses every model reports results
   through, and the GUI renders from -- the single shared type every model
   constructs directly, instead of each defining its own structurally-
   identical-but-separate lookalikes.
7. **External-format I/O** -- ``io_formats/``: ``sdds.py`` (elegant
   ``.ele``), ``yaml_spec.py`` (this package's own ``gaussian_6d_waist``/
   ``gaussian_paraxial`` YAML formats).

Typical use:

    from compton_io import (
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

"""

from . import bunch, constants, interaction, io_formats, laser, photons, propagation, results
from .adapter import adapt_to_model, params_to_floats
from .canonical import CANONICAL_CONVENTIONS, CANONICAL_UNIT, from_canonical, to_canonical
from .enums import AmplitudeConvention, PhysicalMeaning, TimeConvention, WidthConvention
from .quantities import PhysicalQuantity
from .schema import ModelSpec, ParameterSpec
from .units import LIGHT_TIME_CONTEXT, Quantity, ureg
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
    "constants",
    "ureg",
    "Quantity",
    "LIGHT_TIME_CONTEXT",
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
    "bunch",
    "laser",
    "photons",
    "interaction",
    "propagation",
    "results",
    "io_formats",
]
