"""Cross-checks for laser.py's GaussianParaxialLaser.

No cupy/GPU/tkinter needed. Run with `python3 -m pytest tests/` or
`python3 tests/test_laser.py` directly (plain asserts).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compton_suite.io.laser import GaussianParaxialLaser, laser_from_shared_fields, validate  # noqa: E402
from compton_suite.io.units import (  # noqa: E402
    C_LIGHT,
    NoConvention,
    PhysicalMeaning,
    PhysicalQuantity,
    TimeConvention,
    WidthConvention,
)

_EXAMPLE_PULSE = GaussianParaxialLaser(
    pulse_energy_J=PhysicalQuantity(0.05, "joule", PhysicalMeaning.PULSE_ENERGY, NoConvention.PLAIN),
    wavelength_m=PhysicalQuantity(0.8e-6, "meter", PhysicalMeaning.WAVELENGTH, NoConvention.PLAIN),
    waist_rms_x_m=PhysicalQuantity(2.5e-6, "meter", PhysicalMeaning.LASER_WIDTH, WidthConvention.SIGMA_INTENSITY_RMS),
    waist_rms_y_m=PhysicalQuantity(2.5e-6, "meter", PhysicalMeaning.LASER_WIDTH, WidthConvention.SIGMA_INTENSITY_RMS),
    duration_rms_s=PhysicalQuantity(12.74e-15, "second", PhysicalMeaning.PULSE_DURATION, TimeConvention.SIGMA_INTENSITY_RMS),
    focus_z_m=PhysicalQuantity(0.0, "meter", PhysicalMeaning.DISPLACEMENT, NoConvention.PLAIN),
)


def test_derived_quantities_are_sane():
    assert validate(_EXAMPLE_PULSE) == []
    assert _EXAMPLE_PULSE.rayleigh_x_m > 0
    assert _EXAMPLE_PULSE.peak_intensity_focus_W_m2 > 0
    assert _EXAMPLE_PULSE.peak_power_W > 0
    assert _EXAMPLE_PULSE.a0_focus > 0


def test_a0_focus_equals_a0_interaction_when_focused_at_ip():
    assert abs(_EXAMPLE_PULSE.a0_focus - _EXAMPLE_PULSE.a0_interaction) < 1e-12 * _EXAMPLE_PULSE.a0_focus


def test_a0_matches_practical_approximation():
    # spec Sec.9's practical shortcut: a0 ~= 0.855 * lambda_um * sqrt(I0 / 1e18 W/cm^2).
    # Cross-check the exact SI formula against this approximation (should
    # agree to a few tenths of a percent -- the 0.855 coefficient is itself
    # rounded).
    lambda_um = _EXAMPLE_PULSE._wl_m * 1e6
    I0_W_cm2 = _EXAMPLE_PULSE.peak_intensity_focus_W_m2 * 1e-4
    approx = 0.855 * lambda_um * (I0_W_cm2 / 1e18) ** 0.5
    assert abs(_EXAMPLE_PULSE.a0_focus / approx - 1.0) < 5e-3


def test_a0sq_is_a0_squared():
    assert abs(_EXAMPLE_PULSE.a0sq_focus - _EXAMPLE_PULSE.a0_focus ** 2) < 1e-12 * _EXAMPLE_PULSE.a0sq_focus
    assert abs(_EXAMPLE_PULSE.a0sq_interaction - _EXAMPLE_PULSE.a0_interaction ** 2) < 1e-12 * _EXAMPLE_PULSE.a0sq_interaction
    assert abs(_EXAMPLE_PULSE.a0sq_at(1e-6) - _EXAMPLE_PULSE.a0_at(1e-6) ** 2) < 1e-12 * _EXAMPLE_PULSE.a0sq_at(1e-6)


def test_defocused_interaction_intensity_is_lower():
    defocused = GaussianParaxialLaser(
        pulse_energy_J=PhysicalQuantity(0.05, "joule", PhysicalMeaning.PULSE_ENERGY, NoConvention.PLAIN),
        wavelength_m=PhysicalQuantity(0.8e-6, "meter", PhysicalMeaning.WAVELENGTH, NoConvention.PLAIN),
        waist_rms_x_m=PhysicalQuantity(2.5e-6, "meter", PhysicalMeaning.LASER_WIDTH, WidthConvention.SIGMA_INTENSITY_RMS),
        waist_rms_y_m=PhysicalQuantity(2.5e-6, "meter", PhysicalMeaning.LASER_WIDTH, WidthConvention.SIGMA_INTENSITY_RMS),
        duration_rms_s=PhysicalQuantity(12.74e-15, "second", PhysicalMeaning.PULSE_DURATION, TimeConvention.SIGMA_INTENSITY_RMS),
        focus_z_m=PhysicalQuantity(500e-6, "meter", PhysicalMeaning.DISPLACEMENT, NoConvention.PLAIN),  # well past a few-micron Rayleigh range
    )
    assert defocused.peak_intensity_at(0.0) < defocused.peak_intensity_focus_W_m2
    assert defocused.a0_interaction < defocused.a0_focus
    warnings = validate(defocused)
    assert any("far from the interaction point" in w for w in warnings)


def test_validate_rejects_nonpositive_fields():
    bad = GaussianParaxialLaser(
        pulse_energy_J=PhysicalQuantity(-1.0, "joule", PhysicalMeaning.PULSE_ENERGY, NoConvention.PLAIN),
        wavelength_m=PhysicalQuantity(0.8e-6, "meter", PhysicalMeaning.WAVELENGTH, NoConvention.PLAIN),
        waist_rms_x_m=PhysicalQuantity(2.5e-6, "meter", PhysicalMeaning.LASER_WIDTH, WidthConvention.SIGMA_INTENSITY_RMS),
        waist_rms_y_m=PhysicalQuantity(2.5e-6, "meter", PhysicalMeaning.LASER_WIDTH, WidthConvention.SIGMA_INTENSITY_RMS),
        duration_rms_s=PhysicalQuantity(12.74e-15, "second", PhysicalMeaning.PULSE_DURATION, TimeConvention.SIGMA_INTENSITY_RMS),
    )
    try:
        validate(bad)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_laser_from_shared_fields_round_trips():
    rebuilt = laser_from_shared_fields(
        lambda_L=_EXAMPLE_PULSE._wl_m,
        sigma0_l=_EXAMPLE_PULSE._wx_m,
        sigma_par_L=_EXAMPLE_PULSE._dur_s * C_LIGHT,
        pulse_energy_J=_EXAMPLE_PULSE._E_J,
        focus_z_m=_EXAMPLE_PULSE._fz_m,
    )
    assert abs(rebuilt._wl_m - _EXAMPLE_PULSE._wl_m) < 1e-12 * _EXAMPLE_PULSE._wl_m
    assert abs(rebuilt._wx_m - _EXAMPLE_PULSE._wx_m) < 1e-12 * _EXAMPLE_PULSE._wx_m
    assert abs(rebuilt._wy_m - _EXAMPLE_PULSE._wy_m) < 1e-12 * _EXAMPLE_PULSE._wy_m
    assert abs(rebuilt._dur_s - _EXAMPLE_PULSE._dur_s) < 1e-12 * _EXAMPLE_PULSE._dur_s
    assert abs(rebuilt._E_J - _EXAMPLE_PULSE._E_J) < 1e-12 * _EXAMPLE_PULSE._E_J
    assert abs(rebuilt.a0_focus - _EXAMPLE_PULSE.a0_focus) < 1e-9 * _EXAMPLE_PULSE.a0_focus


def test_validate_warns_on_astigmatism():
    astig = GaussianParaxialLaser(
        pulse_energy_J=PhysicalQuantity(0.05, "joule", PhysicalMeaning.PULSE_ENERGY, NoConvention.PLAIN),
        wavelength_m=PhysicalQuantity(0.8e-6, "meter", PhysicalMeaning.WAVELENGTH, NoConvention.PLAIN),
        waist_rms_x_m=PhysicalQuantity(2.5e-6, "meter", PhysicalMeaning.LASER_WIDTH, WidthConvention.SIGMA_INTENSITY_RMS),
        waist_rms_y_m=PhysicalQuantity(5.0e-6, "meter", PhysicalMeaning.LASER_WIDTH, WidthConvention.SIGMA_INTENSITY_RMS),
        duration_rms_s=PhysicalQuantity(12.74e-15, "second", PhysicalMeaning.PULSE_DURATION, TimeConvention.SIGMA_INTENSITY_RMS),
    )
    warnings = validate(astig)
    assert any("elliptical" in w.lower() for w in warnings)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
    print("\nAll laser tests passed.")
