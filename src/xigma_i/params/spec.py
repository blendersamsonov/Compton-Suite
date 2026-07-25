"""``ModelSpec`` for this package's own GUI-facing ``Config``
(``gui_adapter.Config``), whose fields ``config.Compton.set_electron_parameters``/
``set_laser_parameters`` consume directly after only a ``units.py``-style
CGS conversion (metre->centimetre) -- no convention change happens between
``Config`` and the CUDA/CPU kernels (verified against ``config.py``:
``set_laser_parameters``'s ``sigma_lr0`` comment explicitly says "this is
the RMS radius of the *photon density* distribution").

``Config`` itself is populated from raw GUI fields (picoseconds, a
Rayleigh length in metres, ...) by ``gui_adapter.params_to_config`` today,
by hand -- e.g. ``sigma_par_e = _C_LIGHT_M * (bunch_duration_ps * 1e-12)``
and ``sigma0_l = 0.5 * sqrt(rayleigh_length_m * lambda_L / pi)`` (the
latter is exactly ``converters.w0_to_sigma_intensity`` fed a waist derived
from the Rayleigh length). ``XIGMA_SPEC`` describes the *result* of that
hand-rolled conversion, i.e. what ``Config`` itself expects; it does not
(yet) replace ``params_to_config``'s own arithmetic -- see compton-gui's
``scripts/physics_params_demo.py`` for what doing so would look like.

Originally lived in compton-gui as ``schemas/xigma.py`` (one of several
per-model schemas in a GUI-side, multi-model framework); moved here so
this model declares its own parameter contract directly, on top of the
shared ``compton_io`` framework -- see this package's ``__init__.py``
docstring.
"""

from __future__ import annotations

from compton_io import AmplitudeConvention, ParameterSpec, PhysicalMeaning, TimeConvention, WidthConvention

XIGMA_SPEC: dict[str, ParameterSpec] = {
    "sigma0_x": ParameterSpec(
        name="sigma0_x",
        meaning=PhysicalMeaning.ELECTRON_BEAM_SIZE,
        convention=WidthConvention.SIGMA_INTENSITY_RMS,
        unit="meter",
        description=(
            "Horizontal electron-bunch RMS size at the waist: "
            "n_e(x) ~ exp(-x^2 / (2 sigma0_x^2))."
        ),
    ),
    "sigma0_y": ParameterSpec(
        name="sigma0_y",
        meaning=PhysicalMeaning.ELECTRON_BEAM_SIZE,
        convention=WidthConvention.SIGMA_INTENSITY_RMS,
        unit="meter",
        description="Vertical electron-bunch RMS size at the waist (see sigma0_x).",
    ),
    "sigma_par_e": ParameterSpec(
        name="sigma_par_e",
        meaning=PhysicalMeaning.BUNCH_LENGTH,
        convention=TimeConvention.SIGMA_INTENSITY_RMS,
        unit="meter",
        description=(
            "Electron-bunch RMS length, stored as c * (RMS duration): "
            "n_e(z) ~ exp(-z^2 / (2 sigma_par_e^2))."
        ),
    ),
    "sigma0_l": ParameterSpec(
        name="sigma0_l",
        meaning=PhysicalMeaning.LASER_WIDTH,
        convention=WidthConvention.SIGMA_INTENSITY_RMS,
        unit="meter",
        description=(
            "Laser transverse RMS size of the *photon density* profile at "
            "focus (config.Compton.set_laser_parameters's own comment for "
            "sigma_lr0): I(r) ~ exp(-r^2 / (2 sigma0_l^2))."
        ),
    ),
    "sigma_par_L": ParameterSpec(
        name="sigma_par_L",
        meaning=PhysicalMeaning.PULSE_DURATION,
        convention=TimeConvention.SIGMA_INTENSITY_RMS,
        unit="meter",
        description=(
            "Laser pulse RMS length, stored as c * (RMS duration): "
            "I(z) ~ exp(-z^2 / (2 sigma_par_L^2))."
        ),
    ),
}
"""Config-level input parameters with a width/duration convention."""

XIGMA_DIAGNOSTIC_SPEC: dict[str, ParameterSpec] = {
    "a0": ParameterSpec(
        name="a0",
        meaning=PhysicalMeaning.LASER_AMPLITUDE,
        convention=AmplitudeConvention.A0_PEAK,
        unit="dimensionless",
        description=(
            "Peak normalised vector potential, derived from pulse energy + "
            "geometry inside config.Compton.set_laser_parameters -- not a "
            "direct GUI input, so not part of XIGMA_SPEC. Confirmed peak "
            "(not RMS) by CLAUDE.md's 'a0 is a trajectory average' section: "
            "'a0_local(t) = compton.a0 * sqrt(local intensity / peak "
            "intensity)'."
        ),
    ),
}
"""Read-only/derived quantities with a documented convention, kept separate
from XIGMA_SPEC because they aren't parameters a GUI or adapter ever
constructs and feeds in -- see this module's docstring."""
