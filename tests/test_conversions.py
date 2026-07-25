"""Cross-checks for the parameter-semantics/convention framework
(enums/quantities/canonical/converters/schema/adapter), self-contained --
does not need xigma_i or compton_guide's own ModelSpecs, just an inline
example one, so this suite is runnable on its own.

No cupy/GPU/tkinter needed. Run with `python3 -m pytest tests/` or
`python3 tests/test_conversions.py` directly (plain asserts).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compton_io import (  # noqa: E402
    PhysicalMeaning,
    PhysicalQuantity,
    TimeConvention,
    WidthConvention,
    adapt_to_model,
    params_to_floats,
)
from compton_io.converters import fwhm_to_sigma_intensity  # noqa: E402
from compton_io.schema import ParameterSpec  # noqa: E402
from compton_io.units import ureg  # noqa: E402

_EXAMPLE_SPEC = {
    "sigma0_l": ParameterSpec(
        name="sigma0_l",
        meaning=PhysicalMeaning.LASER_WIDTH,
        convention=WidthConvention.SIGMA_INTENSITY_RMS,
        unit="meter",
        description="Example laser-width parameter for self-test purposes.",
    ),
    "sigma_par_L": ParameterSpec(
        name="sigma_par_L",
        meaning=PhysicalMeaning.PULSE_DURATION,
        convention=TimeConvention.SIGMA_INTENSITY_RMS,
        unit="meter",
        description="Example pulse-duration-as-length parameter (c*duration).",
    ),
}


def test_fwhm_to_sigma_conversion_applies():
    raw = {
        "sigma0_l": PhysicalQuantity(12.0, "micrometer", PhysicalMeaning.LASER_WIDTH, WidthConvention.FWHM_INTENSITY),
        "sigma_par_L": PhysicalQuantity(2.0, "picosecond", PhysicalMeaning.PULSE_DURATION, TimeConvention.SIGMA_INTENSITY_RMS),
    }
    out = params_to_floats(adapt_to_model(raw, _EXAMPLE_SPEC))

    expected_sigma0_l = fwhm_to_sigma_intensity(12.0e-6)
    assert abs(out["sigma0_l"] - expected_sigma0_l) < 1e-12


def test_duration_to_length_context_applies():
    raw = {
        "sigma0_l": PhysicalQuantity(12.0, "micrometer", PhysicalMeaning.LASER_WIDTH, WidthConvention.FWHM_INTENSITY),
        "sigma_par_L": PhysicalQuantity(2.0, "picosecond", PhysicalMeaning.PULSE_DURATION, TimeConvention.SIGMA_INTENSITY_RMS),
    }
    out = params_to_floats(adapt_to_model(raw, _EXAMPLE_SPEC))

    c = ureg.Quantity(1.0, "speed_of_light").to("meter/second").magnitude
    expected_sigma_par_L = 2.0e-12 * c
    assert abs(out["sigma_par_L"] - expected_sigma_par_L) < 1e-12 * expected_sigma_par_L


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
    print("\nAll conversion tests passed.")
