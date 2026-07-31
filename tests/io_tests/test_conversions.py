"""Cross-checks for the width/time/amplitude convention conversions and the
light-time pint context in ``units.py``, self-contained.

No cupy/GPU/tkinter needed. Run with `python3 -m pytest tests/` or
`python3 tests/test_conversions.py` directly (plain asserts).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compton_suite.io.units import (  # noqa: E402
    LIGHT_TIME_CONTEXT,
    AmplitudeConvention,
    PhysicalMeaning,
    PhysicalQuantity,
    TimeConvention,
    WidthConvention,
    convert_amplitude,
    convert_time,
    convert_width,
    fwhm_to_sigma_intensity,
    ureg,
)


def test_fwhm_to_sigma_conversion_applies():
    raw = PhysicalQuantity(12.0, "micrometer", PhysicalMeaning.LASER_WIDTH, WidthConvention.FWHM_INTENSITY)
    sigma_m = convert_width(
        raw.to_unit("meter").magnitude, WidthConvention.FWHM_INTENSITY, WidthConvention.SIGMA_INTENSITY_RMS,
    )
    expected_sigma0_l = fwhm_to_sigma_intensity(12.0e-6)
    assert abs(sigma_m - expected_sigma0_l) < 1e-12


def test_duration_to_length_context_applies():
    raw = PhysicalQuantity(2.0, "picosecond", PhysicalMeaning.PULSE_DURATION, TimeConvention.SIGMA_INTENSITY_RMS)
    as_length_m = raw.to_unit("meter", context=LIGHT_TIME_CONTEXT).magnitude

    c = ureg.Quantity(1.0, "speed_of_light").to("meter/second").magnitude
    expected_sigma_par_L = 2.0e-12 * c
    assert abs(as_length_m - expected_sigma_par_L) < 1e-12 * expected_sigma_par_L


def test_width_round_trip():
    for conv in (WidthConvention.SIGMA_FIELD_RMS, WidthConvention.FWHM_INTENSITY, WidthConvention.W0_1E2):
        sigma = convert_width(1.0, WidthConvention.SIGMA_INTENSITY_RMS, conv)
        back = convert_width(sigma, conv, WidthConvention.SIGMA_INTENSITY_RMS)
        assert abs(back - 1.0) < 1e-12


def test_time_round_trip():
    for conv in (TimeConvention.SIGMA_FIELD_RMS, TimeConvention.FWHM_INTENSITY):
        sigma = convert_time(1.0, TimeConvention.SIGMA_INTENSITY_RMS, conv)
        back = convert_time(sigma, conv, TimeConvention.SIGMA_INTENSITY_RMS)
        assert abs(back - 1.0) < 1e-12


def test_amplitude_round_trip():
    rms = convert_amplitude(1.0, AmplitudeConvention.A0_PEAK, AmplitudeConvention.A0_RMS)
    back = convert_amplitude(rms, AmplitudeConvention.A0_RMS, AmplitudeConvention.A0_PEAK)
    assert abs(back - 1.0) < 1e-12
    assert abs(rms - 1.0 / 2.0**0.5) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
    print("\nAll conversion tests passed.")
