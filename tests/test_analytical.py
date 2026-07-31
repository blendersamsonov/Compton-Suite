"""Cross-checks for models/analytical.py's estimate_yield/
estimate_spectrum_width/angle_integrated_spectrum.

Needs the dev-install (see this repo's top-level CLAUDE.md) so
compton_suite.io and compton_suite.models.analytical are importable. Run
with `python3 -m pytest tests/` or `python3 tests/test_analytical.py`
directly (plain asserts).
"""

from __future__ import annotations

import numpy as np

from compton_suite.models import analytical
from compton_suite.io.bunch import GaussianElectronBeam
from compton_suite.io.laser import GaussianParaxialLaser
from compton_suite.io.units import NoConvention, PhysicalMeaning, PhysicalQuantity, TimeConvention, WidthConvention

_EXAMPLE_BEAM = GaussianElectronBeam(
    bunch_charge_C=PhysicalQuantity(100.0e-12, "coulomb", PhysicalMeaning.BUNCH_CHARGE),
    kinetic_energy_eV=PhysicalQuantity(200.0e6, "electron_volt", PhysicalMeaning.BEAM_ENERGY),
    rel_energy_spread_rms=0.001,
    sigma_x_m=PhysicalQuantity(10.0e-6, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE, WidthConvention.SIGMA_INTENSITY_RMS),
    sigma_y_m=PhysicalQuantity(10.0e-6, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE, WidthConvention.SIGMA_INTENSITY_RMS),
    emit_geom_x_m=PhysicalQuantity(0.05e-6, "meter", PhysicalMeaning.EMITTANCE),
    emit_geom_y_m=PhysicalQuantity(0.05e-6, "meter", PhysicalMeaning.EMITTANCE),
    sigma_t_s=PhysicalQuantity(1.0e-12, "second", PhysicalMeaning.BUNCH_LENGTH, TimeConvention.SIGMA_INTENSITY_RMS),
    sigma_pz=0.001,
)
_EXAMPLE_PULSE = GaussianParaxialLaser(
    pulse_energy_J=PhysicalQuantity(0.05, "joule", PhysicalMeaning.PULSE_ENERGY, NoConvention.PLAIN),
    wavelength_m=PhysicalQuantity(0.8e-6, "meter", PhysicalMeaning.WAVELENGTH, NoConvention.PLAIN),
    waist_rms_x_m=PhysicalQuantity(2.5e-6, "meter", PhysicalMeaning.LASER_WIDTH, WidthConvention.SIGMA_INTENSITY_RMS),
    waist_rms_y_m=PhysicalQuantity(2.5e-6, "meter", PhysicalMeaning.LASER_WIDTH, WidthConvention.SIGMA_INTENSITY_RMS),
    duration_rms_s=PhysicalQuantity(12.74e-15, "second", PhysicalMeaning.PULSE_DURATION, TimeConvention.SIGMA_INTENSITY_RMS),
)


def test_estimate_yield_is_positive_finite():
    y = analytical.estimate_yield(_EXAMPLE_BEAM, _EXAMPLE_PULSE)
    assert np.isfinite(y) and y > 0


def test_estimate_spectrum_width_is_positive_finite():
    w = analytical.estimate_spectrum_width(_EXAMPLE_BEAM, _EXAMPLE_PULSE, theta_col=1e-3)
    assert np.isfinite(w) and w > 0


def test_estimate_spectrum_width_grows_with_larger_collimation_angle():
    # Larger collimation aperture admits a wider range of Doppler-shifted
    # angles -> broader collected spectrum.
    narrow = analytical.estimate_spectrum_width(_EXAMPLE_BEAM, _EXAMPLE_PULSE, theta_col=1e-4)
    wide = analytical.estimate_spectrum_width(_EXAMPLE_BEAM, _EXAMPLE_PULSE, theta_col=1e-2)
    assert wide > narrow


def test_angle_integrated_spectrum_shape_and_scalar_input():
    rng = np.random.default_rng(0)
    n = 10_000
    gamma = rng.normal(_EXAMPLE_BEAM.gamma0, _EXAMPLE_BEAM.sigma_gamma, n)
    weight = np.full(n, _EXAMPLE_BEAM.N_e / n)

    s_array = np.linspace(0.01, 0.99, 16)
    out_array = analytical.angle_integrated_spectrum(gamma, weight, s_array)
    assert out_array.shape == s_array.shape
    assert np.all(np.isfinite(out_array)) and np.all(out_array >= 0)

    out_scalar = analytical.angle_integrated_spectrum(gamma, weight, 0.5)
    assert np.ndim(out_scalar) == 0 or isinstance(out_scalar, float)


def test_angle_integrated_spectrum_zero_outside_kinematic_range():
    # s = E/E_max(gamma) must vanish for s/gamma^2 outside [0, 1] --
    # a single monoenergetic bunch has a hard Compton edge.
    gamma = np.array([100.0])
    weight = np.array([1.0])
    s_beyond_edge = np.array([gamma[0] ** 2 * 1.5])
    out = analytical.angle_integrated_spectrum(gamma, weight, s_beyond_edge)
    assert out[0] == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
    print("\nAll analytical tests passed.")
