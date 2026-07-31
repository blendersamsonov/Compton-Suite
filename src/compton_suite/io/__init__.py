"""Shared physical constants, pint unit registry, parameter-semantics
vocabulary, and electron-bunch/laser-pulse representations for the
ComptonSuite toolkit (``compton_suite.gui``, ``xigma_i``, ``kascade``,
``analytical``).

Physics engines plug into a shared GUI through a duck-typed ``ModelAdapter``
contract (``compton_suite.models.api``); this package is the one shared
layer under every one of them and under the GUI itself, so there is exactly
one copy of the physical constants, unit registry, and beam/laser/photon
representations every consumer needs -- never vendored or re-derived
per-model.

1. **Physical constants + shared pint registry + parameter-semantics
   vocabulary** -- ``units.py``: CODATA constants derived from pint's own
   built-in values, the shared ``ureg``/``Quantity``, a custom
   ``"light_time"`` context (length <-> ``c * duration``), and the
   ``PhysicalMeaning``/``WidthConvention``/``TimeConvention``/
   ``AmplitudeConvention``/``NoConvention`` enums plus ``PhysicalQuantity``
   (value + unit + meaning + convention). Every value that crosses a
   model/GUI boundary should travel as a ``PhysicalQuantity``, never a bare
   float. There is no generic "spec"/"adapt to model" framework -- each
   model converts a ``PhysicalQuantity`` to whatever convention/unit it
   needs directly via ``convert_width``/``convert_time``/
   ``convert_amplitude`` at its own boundary.
2. **Electron-bunch representation** -- ``bunch.py``: ``Bunch`` (raw
   macroparticle arrays), ``GaussianElectronBeam`` (the
   ``gaussian_6d_waist`` v0.1 analytic contract), ``BeamFittedParams``
   (structured-fit output), plus ``sample_gaussian_bunch``/
   ``fit_gaussian``/``fit_beam_full``/``drift`` to move between them.
3. **Laser-pulse representation** -- ``laser.py``: ``GaussianParaxialLaser``,
   the ``gaussian_paraxial`` v0.1 analytic contract, plus
   ``gaussian_pulse_envelope`` (the full (x, y, z, t) evaluator).
4. **The shared interaction bundle** -- ``interaction.py``:
   ``InteractionParameters(beam, laser)``, the (beam, pulse) pair every
   model's own ``Config`` builds from.
5. **Photon/observable representations** -- ``photons.py``: the
   spectrum/angular-spectrum/temporal-envelope/spatial-distribution
   dataclasses every model reports results through, and the results
   contract every model's ``run()`` returns (``Photons``).
6. **External-format I/O** -- ``io_formats/``: ``sdds.py`` (elegant
   ``.ele``), ``yaml_spec.py`` (this package's own ``gaussian_6d_waist``/
   ``gaussian_paraxial`` YAML formats).

Typical use:

    from compton_suite.io import PhysicalQuantity, PhysicalMeaning, WidthConvention
    from compton_suite.io.units import convert_width

    laser_width = PhysicalQuantity(
        magnitude=5.0, unit="micrometer",
        meaning=PhysicalMeaning.LASER_WIDTH,
        convention=WidthConvention.FWHM_INTENSITY,
    )
    sigma_m = convert_width(laser_width.to_unit("meter").magnitude,
                             WidthConvention.FWHM_INTENSITY, WidthConvention.SIGMA_INTENSITY_RMS)
"""

from __future__ import annotations

from . import bunch, interaction, io_formats, laser, photons, units
from .interaction import InteractionParameters, recoil_parameter
from .photons import Photons, validate_results
from .units import (
    ALPHA,
    C_LIGHT,
    E_CHARGE,
    EPS0,
    HBAR,
    LIGHT_TIME_CONTEXT,
    ME,
    MEC2,
    MEC2_EV,
    R_E_M,
    SIGMA_T_M2,
    AmplitudeConvention,
    NoConvention,
    PhysicalMeaning,
    PhysicalQuantity,
    Quantity,
    TimeConvention,
    WidthConvention,
    convert_amplitude,
    convert_time,
    convert_width,
    ureg,
)

__all__ = [
    "ureg",
    "Quantity",
    "LIGHT_TIME_CONTEXT",
    "C_LIGHT", "E_CHARGE", "HBAR", "ME", "MEC2", "MEC2_EV", "EPS0", "ALPHA", "R_E_M", "SIGMA_T_M2",
    "PhysicalQuantity",
    "PhysicalMeaning",
    "NoConvention",
    "WidthConvention",
    "TimeConvention",
    "AmplitudeConvention",
    "convert_width",
    "convert_time",
    "convert_amplitude",
    "InteractionParameters",
    "recoil_parameter",
    "Photons",
    "validate_results",
    "bunch",
    "laser",
    "interaction",
    "photons",
    "units",
    "io_formats",
]
