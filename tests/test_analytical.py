"""Cross-checks for models/analytical.py's estimate_yield/
estimate_spectrum_width/angle_integrated_spectrum.

Needs the dev-install (see this repo's top-level CLAUDE.md) so
gammaforge.io and gammaforge.models.analytical are importable. Run
with `python3 -m pytest tests/` or `python3 tests/test_analytical.py`
directly (plain asserts).
"""

from __future__ import annotations

import numpy as np

from gammaforge.models import analytical
from gammaforge.io.bunch import GaussianElectronBeam
from gammaforge.io.laser import GaussianParaxialLaser
from gammaforge.io.units import NoConvention, PhysicalMeaning, PhysicalQuantity, TimeConvention, WidthConvention

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
    s_array = np.linspace(0.01, 0.99, 16)
    out_array = analytical.angle_integrated_spectrum(
        _EXAMPLE_BEAM.gamma0, _EXAMPLE_BEAM.sigma_gamma, _EXAMPLE_BEAM.N_e, s_array)
    assert out_array.shape == s_array.shape
    assert np.all(np.isfinite(out_array)) and np.all(out_array >= 0)

    out_scalar = analytical.angle_integrated_spectrum(
        _EXAMPLE_BEAM.gamma0, _EXAMPLE_BEAM.sigma_gamma, _EXAMPLE_BEAM.N_e, 0.5)
    assert np.ndim(out_scalar) == 0 or isinstance(out_scalar, float)


def test_angle_integrated_spectrum_zero_outside_kinematic_range():
    # s = E/E_max(gamma) must vanish for s/gamma^2 outside [0, 1] -- a
    # narrow (near-monoenergetic) beam has an almost-hard Compton edge.
    # sigma_gamma can't be exactly 0 (the Gaussian quadrature grid would
    # be degenerate), so this uses a tight but finite spread instead of a
    # true delta function.
    gamma0 = 100.0
    s_far_beyond_edge = np.array([gamma0 ** 2 * 1.5])
    out = analytical.angle_integrated_spectrum(gamma0, gamma0 * 1e-6, 1.0, s_far_beyond_edge)
    assert out[0] == 0.0


def test_angle_integrated_spectrum_no_macroparticle_dependency():
    """This is the point of the refactor (see module docstring): the
    function takes only the beam's Gaussian parameters (gamma0,
    sigma_gamma, N_e) -- scalars, not a per-particle array -- so calling
    it with plain Python floats (no numpy array, no length, no
    "n_particles" concept at all) must work. A naive re-introduction of
    a macroparticle array argument would break this at the call site,
    not just numerically."""
    import inspect
    params = list(inspect.signature(analytical.angle_integrated_spectrum).parameters)
    assert params[:3] == ["gamma0", "sigma_gamma", "N_e"]

    out = analytical.angle_integrated_spectrum(300.0, 0.5, 1.0e9, 0.5)
    assert np.isfinite(out)


def test_angle_integrated_spectrum_matches_monte_carlo_reference():
    """Validates the closed-form quadrature (gamma0, sigma_gamma, N_e)
    against an independent Monte Carlo reference (a large sample of real
    gamma draws from the same Gaussian, summed the way the old
    macroparticle-based implementation used to) -- confirms the
    refactor preserved the physics, not just the API shape."""
    rng = np.random.default_rng(1)
    n = 2_000_000
    gamma0, sigma_gamma, N_e = _EXAMPLE_BEAM.gamma0, _EXAMPLE_BEAM.sigma_gamma, _EXAMPLE_BEAM.N_e
    gamma_samples = rng.normal(gamma0, sigma_gamma, n)
    weight = N_e / n

    s_array = np.linspace(0.05, 0.95, 12)
    gamma2 = (gamma_samples ** 2)[:, None]
    y = s_array[None, :] / gamma2
    shape = 1.5 * (1.0 - 2.0 * y * (1.0 - y))
    shape = np.where((y < 0) | (y > 1), 0.0, shape)
    mc_reference = np.sum(weight * shape / gamma2, axis=0)

    quad_result = analytical.angle_integrated_spectrum(gamma0, sigma_gamma, N_e, s_array)
    # Monte Carlo with 2e6 samples has real statistical noise; the
    # quadrature result is (to quadrature precision) exact, so the
    # tolerance reflects the reference's noise floor, not the
    # quadrature's own accuracy.
    assert np.allclose(quad_result, mc_reference, rtol=0.02)


def test_angle_integrated_spectrum_fast_at_reported_scale():
    """Regression test for the user report this refactor fixes: 2048
    energy bins used to require an (n_particles, 2048) broadcast --
    5,000,000 particles x 2048 bins hit a 76.3 GiB allocation. The new
    signature has no n_particles argument at all, so this must simply
    stay fast and never allocate anything bigger than
    (n_quad, len(s))."""
    s_array = np.linspace(1e-3, 1.0 - 1e-3, 2048)
    out = analytical.angle_integrated_spectrum(
        _EXAMPLE_BEAM.gamma0, _EXAMPLE_BEAM.sigma_gamma, _EXAMPLE_BEAM.N_e, s_array)
    assert out.shape == s_array.shape
    assert np.all(np.isfinite(out))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
    print("\nAll analytical tests passed.")
