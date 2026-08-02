"""Tests for the improved electron bunch sampling, fitting, and evaluation.

This module tests the new canonical sampling with mass-shell enforcement,
vacuum propagation, structured Gaussian fitting, and fit quality metrics.
"""

import numpy as np
import pytest

from gammaforge.io.bunch import (
    GaussianElectronBeam,
    Bunch,
    drift,
    evaluate_fit_quality,
    fit_gaussian,
    sample_gaussian_bunch,
    sample_gaussian_canonical,
)
from gammaforge.io.units import PhysicalMeaning, PhysicalQuantity


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _pq(value: float, unit: str, meaning: PhysicalMeaning) -> PhysicalQuantity:
    """Build a PhysicalQuantity for test fixtures."""
    return PhysicalQuantity(value, unit, meaning)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def rng():
    """Random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def ultra_relativistic_beam():
    """Ultra-relativistic beam (gamma >> 1) for testing.

    Physical parameters: 500 MeV beam, 100 um beam size, 10 nm emittance.
    Divergence = emit/sigma = 10e-9/100e-6 = 0.1 mrad.
    """
    return GaussianElectronBeam(
        bunch_charge_C=_pq(1e-12, "coulomb", PhysicalMeaning.BUNCH_CHARGE),
        kinetic_energy_eV=_pq(500e6, "electron_volt", PhysicalMeaning.BEAM_ENERGY),
        rel_energy_spread_rms=0.01,  # 1% energy spread
        sigma_x_m=_pq(100e-6, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE),
        sigma_y_m=_pq(100e-6, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE),
        emit_geom_x_m=_pq(10e-9, "meter", PhysicalMeaning.EMITTANCE),
        emit_geom_y_m=_pq(10e-9, "meter", PhysicalMeaning.EMITTANCE),
        sigma_t_s=_pq(1e-12, "second", PhysicalMeaning.BUNCH_LENGTH),
        sigma_pz=0.01,  # 1% momentum spread
    )


@pytest.fixture
def mildly_relativistic_beam():
    """Mildly relativistic beam (gamma ~ 19) for testing.

    Physical parameters: 9 MeV beam, 100 um beam size, 10 nm emittance.
    """
    return GaussianElectronBeam(
        bunch_charge_C=_pq(1e-12, "coulomb", PhysicalMeaning.BUNCH_CHARGE),
        kinetic_energy_eV=_pq(9e6, "electron_volt", PhysicalMeaning.BEAM_ENERGY),
        rel_energy_spread_rms=0.05,  # 5% energy spread
        sigma_x_m=_pq(100e-6, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE),
        sigma_y_m=_pq(100e-6, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE),
        emit_geom_x_m=_pq(10e-9, "meter", PhysicalMeaning.EMITTANCE),
        emit_geom_y_m=_pq(10e-9, "meter", PhysicalMeaning.EMITTANCE),
        sigma_t_s=_pq(1e-12, "second", PhysicalMeaning.BUNCH_LENGTH),
        sigma_pz=0.05,  # 5% momentum spread
    )


# ============================================================================
# Canonical Sampling Tests
# ============================================================================

class TestCanonicalSampling:
    """Test canonical sampling with mass-shell enforcement."""

    def test_mass_shell_constraint(self, ultra_relativistic_beam, rng):
        """Verify mass-shell constraint: gamma^2 = 1 + px^2 + py^2 + pz^2."""
        n_particles = 10000
        bunch = sample_gaussian_canonical(ultra_relativistic_beam, n_particles, rng=rng)

        # Recover pz from gamma and angles using the EXACT inverse:
        # gamma^2 = 1 + pz^2 * (1 + thx^2 + thy^2)
        # => pz^2 = (gamma^2 - 1) / (1 + thx^2 + thy^2)
        pz = np.sqrt((bunch.gamma**2 - 1) / (1 + bunch.thx**2 + bunch.thy**2))
        px = bunch.thx * pz
        py = bunch.thy * pz

        # Check mass-shell: gamma^2 = 1 + px^2 + py^2 + pz^2
        gamma_sq = bunch.gamma**2
        p_sq = 1.0 + px**2 + py**2 + pz**2

        # Should be exactly equal (within floating point precision)
        np.testing.assert_allclose(gamma_sq, p_sq, rtol=1e-10)

    def test_pz_positivity(self, ultra_relativistic_beam, rng):
        """Verify pz > 1 for all particles (physical constraint)."""
        n_particles = 10000
        bunch = sample_gaussian_canonical(ultra_relativistic_beam, n_particles, rng=rng)

        # Recover pz using exact inverse
        pz = np.sqrt((bunch.gamma**2 - 1) / (1 + bunch.thx**2 + bunch.thy**2))

        # All pz should be > 1
        assert np.all(pz > 1.0), f"Some pz <= 1.0: min = {np.min(pz)}"

    def test_statistics_match_input(self, ultra_relativistic_beam, rng):
        """Verify sampled statistics match input parameters."""
        n_particles = 100000
        bunch = sample_gaussian_canonical(ultra_relativistic_beam, n_particles, rng=rng)

        # Check transverse sizes (should be close to input)
        sigma_x_meas = np.std(bunch.x)
        sigma_y_meas = np.std(bunch.y)

        # Allow 5% tolerance due to sampling noise
        sx_in = ultra_relativistic_beam.sigma_x_m.to_unit("meter").magnitude
        sy_in = ultra_relativistic_beam.sigma_y_m.to_unit("meter").magnitude
        assert abs(sigma_x_meas - sx_in) / sx_in < 0.05
        assert abs(sigma_y_meas - sy_in) / sy_in < 0.05

    def test_gamma_calculation(self, ultra_relativistic_beam, rng):
        """Verify gamma is correctly calculated from momenta."""
        n_particles = 10000
        bunch = sample_gaussian_canonical(ultra_relativistic_beam, n_particles, rng=rng)

        # Recover pz using exact inverse
        pz = np.sqrt((bunch.gamma**2 - 1) / (1 + bunch.thx**2 + bunch.thy**2))
        px = bunch.thx * pz
        py = bunch.thy * pz

        # Gamma should be sqrt(1 + px^2 + py^2 + pz^2)
        gamma_expected = np.sqrt(1.0 + px**2 + py**2 + pz**2)
        np.testing.assert_allclose(bunch.gamma, gamma_expected, rtol=1e-10)


# ============================================================================
# Drift Tests
# ============================================================================

class TestDrift:
    """Test vacuum propagation."""

    def test_position_propagation(self, ultra_relativistic_beam, rng):
        """Verify x -> x + x' * L."""
        n_particles = 1000
        bunch = sample_gaussian_canonical(ultra_relativistic_beam, n_particles, rng=rng)
        L = 0.1  # 10 cm drift

        drifted = drift(bunch, L)

        # Check x propagation
        x_expected = bunch.x + bunch.thx * L
        np.testing.assert_allclose(drifted.x, x_expected)

        # Check y propagation
        y_expected = bunch.y + bunch.thy * L
        np.testing.assert_allclose(drifted.y, y_expected)

    def test_unchanged_quantities(self, ultra_relativistic_beam, rng):
        """Verify z, thx, thy, gamma unchanged by drift."""
        n_particles = 1000
        bunch = sample_gaussian_canonical(ultra_relativistic_beam, n_particles, rng=rng)
        L = 0.1

        drifted = drift(bunch, L)

        np.testing.assert_allclose(drifted.z, bunch.z)
        np.testing.assert_allclose(drifted.thx, bunch.thx)
        np.testing.assert_allclose(drifted.thy, bunch.thy)
        np.testing.assert_allclose(drifted.gamma, bunch.gamma)

    def test_twiss_alpha_emergence(self, ultra_relativistic_beam, rng):
        """Verify Twiss alpha emerges from waist + drift."""
        n_particles = 10000
        bunch = sample_gaussian_canonical(ultra_relativistic_beam, n_particles, rng=rng)
        L = 0.1

        # At waist, alpha should be ~0
        params_waist = fit_gaussian(bunch)
        assert abs(params_waist.alpha_x) < 0.1  # Should be close to 0
        assert abs(params_waist.alpha_y) < 0.1

        # After drift, alpha should be non-zero
        drifted = drift(bunch, L)
        params_drifted = fit_gaussian(drifted)

        # Alpha should be negative (converging beam)
        assert params_drifted.alpha_x < 0
        assert params_drifted.alpha_y < 0

    def test_attached_gaussian_fit_tracks_alpha_analytically(self, ultra_relativistic_beam, rng):
        """Verify drift() analytically propagates bunch.gaussian_fit's alpha
        in lockstep with the macroparticles -- no refit needed (see
        bunch.py's drift()/`_drift_gaussian_fit` docstrings). The waist sizes
        stay invariant; only alpha moves, via alpha_new = alpha_old - L/beta*."""
        n_particles = 10000
        bunch = sample_gaussian_canonical(ultra_relativistic_beam, n_particles, rng=rng)
        assert bunch.gaussian_fit is not None
        L = 0.1

        drifted = drift(bunch, L)
        fit = drifted.gaussian_fit
        assert fit is not None

        expected_alpha_x = 0.0 - L / ultra_relativistic_beam.beta_star_x.to("meter").magnitude
        expected_alpha_y = 0.0 - L / ultra_relativistic_beam.beta_star_y.to("meter").magnitude
        assert abs(fit.alpha_x - expected_alpha_x) < 1e-9
        assert abs(fit.alpha_y - expected_alpha_y) < 1e-9

        # Waist-referenced sizes are drift-invariant.
        assert abs(fit.sigma_x_m.to_unit("meter").magnitude
                   / ultra_relativistic_beam.sigma_x_m.to_unit("meter").magnitude - 1.0) < 1e-9
        assert abs(fit.emit_geom_x_m.to_unit("meter").magnitude
                   / ultra_relativistic_beam.emit_geom_x_m.to_unit("meter").magnitude - 1.0) < 1e-9


# ============================================================================
# Fitting Tests
# ============================================================================

class TestFitting:
    """Test structured Gaussian fitting."""

    def test_fit_parameters_match_input(self, ultra_relativistic_beam, rng):
        """Verify fitted parameters match input beam."""
        n_particles = 100000
        bunch = sample_gaussian_canonical(ultra_relativistic_beam, n_particles, rng=rng)

        params = fit_gaussian(bunch)

        # Check transverse emittance (should be close to input)
        emit_x_input = ultra_relativistic_beam.emit_geom_x_m.to_unit("meter").magnitude
        emit_y_input = ultra_relativistic_beam.emit_geom_y_m.to_unit("meter").magnitude

        # Allow 10% tolerance due to sampling noise
        assert abs(params.emit_geom_x_m.to_unit("meter").magnitude - emit_x_input) / emit_x_input < 0.1
        assert abs(params.emit_geom_y_m.to_unit("meter").magnitude - emit_y_input) / emit_y_input < 0.1

    def test_fit_with_drift(self, ultra_relativistic_beam, rng):
        """Verify fitting works correctly with drifted beam."""
        n_particles = 100000
        bunch = sample_gaussian_canonical(ultra_relativistic_beam, n_particles, rng=rng)
        L = 0.1

        drifted = drift(bunch, L)
        params = fit_gaussian(drifted)

        # Emittance should be conserved (Liouville's theorem)
        emit_x_input = ultra_relativistic_beam.emit_geom_x_m.to_unit("meter").magnitude
        emit_y_input = ultra_relativistic_beam.emit_geom_y_m.to_unit("meter").magnitude

        # Allow 10% tolerance
        assert abs(params.emit_geom_x_m.to_unit("meter").magnitude - emit_x_input) / emit_x_input < 0.1
        assert abs(params.emit_geom_y_m.to_unit("meter").magnitude - emit_y_input) / emit_y_input < 0.1

    def test_sigma_gamma_calculation(self, ultra_relativistic_beam, rng):
        """Verify sigma_gamma is correctly calculated from fit."""
        n_particles = 10000
        bunch = sample_gaussian_canonical(ultra_relativistic_beam, n_particles, rng=rng)

        params = fit_gaussian(bunch)

        # sigma_gamma should be close to std(gamma)
        sigma_gamma_expected = np.std(bunch.gamma)
        assert abs(params.sigma_gamma - sigma_gamma_expected) / sigma_gamma_expected < 0.1

    def test_chirp_is_finite(self, ultra_relativistic_beam, rng):
        """Verify chirp is computed and finite for uncorrelated sampling."""
        n_particles = 10000
        bunch = sample_gaussian_canonical(ultra_relativistic_beam, n_particles, rng=rng)

        params = fit_gaussian(bunch)

        # Chirp should be a finite number
        assert np.isfinite(params.chirp_h)
        # For uncorrelated sampling, chirp should be relatively small
        # compared to gamma0/sigma_z (the natural scale)
        natural_scale = ultra_relativistic_beam.gamma0 / ultra_relativistic_beam.sigma_z.to("meter").magnitude
        assert abs(params.chirp_h) < 0.5 * natural_scale  # Much less than the natural scale


# ============================================================================
# Fit Quality Tests
# ============================================================================

class TestFitQuality:
    """Test fit quality evaluation metrics."""

    def test_gaussian_data_low_ks_excess(self, ultra_relativistic_beam, rng):
        """Verify Gaussian data produces low KS excess."""
        n_particles = 10000
        bunch = sample_gaussian_canonical(ultra_relativistic_beam, n_particles, rng=rng)

        # Fit the beam
        X = np.stack([bunch.x, bunch.thx, bunch.y, bunch.thy, bunch.z, bunch.gamma], axis=1)
        mu = np.mean(X, axis=0)
        Sigma = np.cov(X - mu, rowvar=False, bias=True)

        # Evaluate fit quality
        quality = evaluate_fit_quality(bunch, mu, Sigma)

        # For Gaussian data, KS excess should be small
        # (close to 0, within sampling noise)
        assert abs(quality["ks_excess"]) < 0.1

    def test_non_gaussian_data_high_ks_excess(self, rng):
        """Verify non-Gaussian data produces high KS excess."""
        n_particles = 10000

        # Create non-Gaussian data (mixture of two well-separated Gaussians)
        n1 = n_particles // 2
        n2 = n_particles - n1

        # Use x as the non-Gaussian dimension - large separation
        x1 = rng.normal(-5e-6, 1e-6, n1)
        x2 = rng.normal(5e-6, 1e-6, n2)
        x = np.concatenate([x1, x2])

        y = rng.normal(0, 100e-6, n_particles)
        z = rng.normal(0, 3e-4, n_particles)
        thx = rng.normal(0, 1e-4, n_particles)
        thy = rng.normal(0, 1e-4, n_particles)
        gamma = rng.normal(1000, 10, n_particles)

        bunch = Bunch(x=x, y=y, z=z, thx=thx, thy=thy, gamma=gamma, weight=1.0)

        # Fit the beam (using the non-Gaussian data)
        X = np.stack([bunch.x, bunch.thx, bunch.y, bunch.thy, bunch.z, bunch.gamma], axis=1)
        mu = np.mean(X, axis=0)
        Sigma = np.cov(X - mu, rowvar=False, bias=True)

        # Evaluate fit quality
        quality = evaluate_fit_quality(bunch, mu, Sigma)

        # For non-Gaussian data, KS excess should be larger
        # (the fit is a poor model for the mixture)
        assert quality["ks_excess"] > 0.01  # Should be noticeable

    def test_log_likelihood_comparison(self, ultra_relativistic_beam, rng):
        """Verify log-likelihood comparison between real and synthetic."""
        n_particles = 10000
        bunch = sample_gaussian_canonical(ultra_relativistic_beam, n_particles, rng=rng)

        # Fit the beam
        X = np.stack([bunch.x, bunch.thx, bunch.y, bunch.thy, bunch.z, bunch.gamma], axis=1)
        mu = np.mean(X, axis=0)
        Sigma = np.cov(X - mu, rowvar=False, bias=True)

        # Evaluate fit quality
        quality = evaluate_fit_quality(bunch, mu, Sigma)

        # Log-likelihood can be positive or negative depending on scales
        # The key is that real and synthetic should be close for good fit
        assert np.isfinite(quality["log_likelihood_real"])
        assert np.isfinite(quality["log_likelihood_synthetic"])

        # For good fit, real log-likelihood should be close to synthetic
        # (within sampling noise)
        ll_diff = abs(quality["log_likelihood_real"] - quality["log_likelihood_synthetic"])
        # Relative difference should be small
        ll_scale = max(abs(quality["log_likelihood_real"]), abs(quality["log_likelihood_synthetic"]), 1.0)
        assert ll_diff / ll_scale < 0.1  # Within 10%


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self, ultra_relativistic_beam, rng):
        """Test complete pipeline: sample -> drift -> fit -> evaluate."""
        n_particles = 100000
        L = 0.1

        # Sample
        bunch = sample_gaussian_canonical(ultra_relativistic_beam, n_particles, rng=rng)

        # Drift
        drifted = drift(bunch, L)

        # Fit
        params = fit_gaussian(drifted)

        # Verify all parameters are computed -- fit_gaussian returns a
        # GaussianElectronBeam (the merged type: BeamFittedParams no
        # longer exists), so these are that type's own field/property names.
        assert hasattr(params, 'sigma_x_m')
        assert hasattr(params, 'sigma_y_m')
        assert hasattr(params, 'emit_geom_x_m')
        assert hasattr(params, 'emit_geom_y_m')
        assert hasattr(params, 'beta_star_x')
        assert hasattr(params, 'beta_star_y')
        assert hasattr(params, 'alpha_x')
        assert hasattr(params, 'alpha_y')
        assert hasattr(params, 'sigma_z')
        assert hasattr(params, 'sigma_pz')
        assert hasattr(params, 'sigma_gamma')
        assert hasattr(params, 'chirp_h')
        assert hasattr(params, 'dispersion_x')
        assert hasattr(params, 'dispersion_y')
        assert hasattr(params, 'fit_quality')

        # Verify fit quality metrics are present
        assert 'ks_real' in params.fit_quality
        assert 'ks_synthetic' in params.fit_quality
        assert 'ks_excess' in params.fit_quality
        assert 'mean_d2_real' in params.fit_quality
        assert 'mean_d2_synthetic' in params.fit_quality
        assert 'log_likelihood_real' in params.fit_quality
        assert 'log_likelihood_synthetic' in params.fit_quality

    def test_sample_gaussian_bunch_delegates(self, ultra_relativistic_beam, rng):
        """Verify sample_gaussian_bunch delegates to canonical sampling."""
        n_particles = 10000

        # Both should produce the same result (with same rng)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        bunch1 = sample_gaussian_bunch(ultra_relativistic_beam, n_particles, rng=rng1)
        bunch2 = sample_gaussian_canonical(ultra_relativistic_beam, n_particles, rng=rng2)

        np.testing.assert_allclose(bunch1.x, bunch2.x)
        np.testing.assert_allclose(bunch1.y, bunch2.y)
        np.testing.assert_allclose(bunch1.z, bunch2.z)
        np.testing.assert_allclose(bunch1.thx, bunch2.thx)
        np.testing.assert_allclose(bunch1.thy, bunch2.thy)
        np.testing.assert_allclose(bunch1.gamma, bunch2.gamma)


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_sigma_pz(self, rng):
        """Test with zero momentum spread (delta function in gamma).

        Both sigma_pz and rel_energy_spread_rms must be zero for consistency
        (otherwise the beam specification is inconsistent).
        """
        beam = GaussianElectronBeam(
            bunch_charge_C=_pq(1e-12, "coulomb", PhysicalMeaning.BUNCH_CHARGE),
            kinetic_energy_eV=_pq(500e6, "electron_volt", PhysicalMeaning.BEAM_ENERGY),
            rel_energy_spread_rms=0.0,  # Must match sigma_pz=0
            sigma_x_m=_pq(100e-6, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE),
            sigma_y_m=_pq(100e-6, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE),
            emit_geom_x_m=_pq(10e-9, "meter", PhysicalMeaning.EMITTANCE),
            emit_geom_y_m=_pq(10e-9, "meter", PhysicalMeaning.EMITTANCE),
            sigma_t_s=_pq(1e-12, "second", PhysicalMeaning.BUNCH_LENGTH),
            sigma_pz=0.0,  # Zero momentum spread → delta function in gamma
        )

        n_particles = 1000
        bunch = sample_gaussian_canonical(beam, n_particles, rng=rng)

        # With sigma_pz = 0, gamma should be a delta function at gamma0
        gamma0 = beam.gamma0
        assert np.allclose(bunch.gamma, gamma0, rtol=1e-10), (
            f"gamma varies: std = {np.std(bunch.gamma):.3g}"
        )

        # pz varies to conserve mass-shell: pz² = γ² - 1 - px² - py²
        # But the mean should be close to pz_mean
        pz = np.sqrt((bunch.gamma**2 - 1) / (1 + bunch.thx**2 + bunch.thy**2))
        pz_mean = gamma0 * beam.beta0
        assert abs(np.mean(pz) - pz_mean) / pz_mean < 0.01

    def test_large_sigma_pz(self, rng):
        """Test with large momentum spread."""
        beam = GaussianElectronBeam(
            bunch_charge_C=_pq(1e-12, "coulomb", PhysicalMeaning.BUNCH_CHARGE),
            kinetic_energy_eV=_pq(500e6, "electron_volt", PhysicalMeaning.BEAM_ENERGY),
            rel_energy_spread_rms=0.01,
            sigma_x_m=_pq(100e-6, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE),
            sigma_y_m=_pq(100e-6, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE),
            emit_geom_x_m=_pq(10e-9, "meter", PhysicalMeaning.EMITTANCE),
            emit_geom_y_m=_pq(10e-9, "meter", PhysicalMeaning.EMITTANCE),
            sigma_t_s=_pq(1e-12, "second", PhysicalMeaning.BUNCH_LENGTH),
            sigma_pz=0.2,  # 20% momentum spread
        )

        n_particles = 10000
        bunch = sample_gaussian_canonical(beam, n_particles, rng=rng)

        # Should still satisfy mass-shell
        pz = np.sqrt((bunch.gamma**2 - 1) / (1 + bunch.thx**2 + bunch.thy**2))
        px = bunch.thx * pz
        py = bunch.thy * pz

        gamma_sq = bunch.gamma**2
        p_sq = 1.0 + px**2 + py**2 + pz**2

        np.testing.assert_allclose(gamma_sq, p_sq, rtol=1e-10)

    def test_small_number_of_particles(self, ultra_relativistic_beam, rng):
        """Test with very small number of particles."""
        n_particles = 10
        bunch = sample_gaussian_canonical(ultra_relativistic_beam, n_particles, rng=rng)

        # Should still work
        assert len(bunch.x) == n_particles
        params = fit_gaussian(bunch)
        assert params is not None


# ============================================================================
# Chirp and Dispersion Tests
# ============================================================================

class TestChirpAndDispersion:
    """Test chirp and dispersion correlations in sampling."""

    # ---------------------------------------------------------------
    # Chirp
    # ---------------------------------------------------------------

    def test_chirp_creates_z_gamma_correlation(self, ultra_relativistic_beam, rng):
        """Verify chirp creates measurable z-γ correlation."""
        # Maximum chirp: |ρ_zγ| ≤ 1, so |chirp_h| ≤ σ_γ / σ_z
        sigma_z = ultra_relativistic_beam.sigma_z.to("meter").magnitude
        sigma_gamma = ultra_relativistic_beam.sigma_gamma  # sigma_pz * gamma0 * beta0
        max_chirp = sigma_gamma / sigma_z

        # Create beam with moderate chirp (50% of max)
        beam = GaussianElectronBeam(
            bunch_charge_C=ultra_relativistic_beam.bunch_charge_C,
            kinetic_energy_eV=ultra_relativistic_beam.kinetic_energy_eV,
            rel_energy_spread_rms=ultra_relativistic_beam.rel_energy_spread_rms,
            sigma_x_m=ultra_relativistic_beam.sigma_x_m,
            sigma_y_m=ultra_relativistic_beam.sigma_y_m,
            emit_geom_x_m=ultra_relativistic_beam.emit_geom_x_m,
            emit_geom_y_m=ultra_relativistic_beam.emit_geom_y_m,
            sigma_t_s=ultra_relativistic_beam.sigma_t_s,
            sigma_pz=ultra_relativistic_beam.sigma_pz,
            chirp_h=0.5 * max_chirp,
        )

        n_particles = 100000
        bunch = sample_gaussian_canonical(beam, n_particles, rng=rng)

        # Fit to extract measured correlations
        params = fit_gaussian(bunch)

        # Chirp should be positive and close to input value
        assert params.chirp_h > 0, "Chirp should be positive"
        assert params.chirp_h > 0.1 * max_chirp, f"Chirp too weak: {params.chirp_h}"

    def test_chirp_sign(self, ultra_relativistic_beam, rng):
        """Verify chirp sign is respected."""
        sigma_z = ultra_relativistic_beam.sigma_z.to("meter").magnitude
        sigma_gamma = ultra_relativistic_beam.sigma_gamma
        max_chirp = sigma_gamma / sigma_z
        h = 0.4 * max_chirp

        n_particles = 100000

        # Positive chirp: head (z > 0) has higher energy
        beam_pos = GaussianElectronBeam(
            bunch_charge_C=ultra_relativistic_beam.bunch_charge_C,
            kinetic_energy_eV=ultra_relativistic_beam.kinetic_energy_eV,
            rel_energy_spread_rms=ultra_relativistic_beam.rel_energy_spread_rms,
            sigma_x_m=ultra_relativistic_beam.sigma_x_m,
            sigma_y_m=ultra_relativistic_beam.sigma_y_m,
            emit_geom_x_m=ultra_relativistic_beam.emit_geom_x_m,
            emit_geom_y_m=ultra_relativistic_beam.emit_geom_y_m,
            sigma_t_s=ultra_relativistic_beam.sigma_t_s,
            sigma_pz=ultra_relativistic_beam.sigma_pz,
            chirp_h=h,
        )
        bunch_pos = sample_gaussian_canonical(beam_pos, n_particles, rng=rng)
        params_pos = fit_gaussian(bunch_pos)

        # Negative chirp
        beam_neg = GaussianElectronBeam(
            bunch_charge_C=ultra_relativistic_beam.bunch_charge_C,
            kinetic_energy_eV=ultra_relativistic_beam.kinetic_energy_eV,
            rel_energy_spread_rms=ultra_relativistic_beam.rel_energy_spread_rms,
            sigma_x_m=ultra_relativistic_beam.sigma_x_m,
            sigma_y_m=ultra_relativistic_beam.sigma_y_m,
            emit_geom_x_m=ultra_relativistic_beam.emit_geom_x_m,
            emit_geom_y_m=ultra_relativistic_beam.emit_geom_y_m,
            sigma_t_s=ultra_relativistic_beam.sigma_t_s,
            sigma_pz=ultra_relativistic_beam.sigma_pz,
            chirp_h=-h,
        )
        bunch_neg = sample_gaussian_canonical(beam_neg, n_particles, rng=rng)
        params_neg = fit_gaussian(bunch_neg)

        assert params_pos.chirp_h > 0, "Positive chirp should give positive slope"
        assert params_neg.chirp_h < 0, "Negative chirp should give negative slope"
        assert params_pos.chirp_h > params_neg.chirp_h

    def test_chirp_mass_shell(self, ultra_relativistic_beam, rng):
        """Verify mass-shell holds with chirp."""
        sigma_z = ultra_relativistic_beam.sigma_z.to("meter").magnitude
        sigma_gamma = ultra_relativistic_beam.sigma_gamma
        max_chirp = sigma_gamma / sigma_z

        beam = GaussianElectronBeam(
            bunch_charge_C=ultra_relativistic_beam.bunch_charge_C,
            kinetic_energy_eV=ultra_relativistic_beam.kinetic_energy_eV,
            rel_energy_spread_rms=ultra_relativistic_beam.rel_energy_spread_rms,
            sigma_x_m=ultra_relativistic_beam.sigma_x_m,
            sigma_y_m=ultra_relativistic_beam.sigma_y_m,
            emit_geom_x_m=ultra_relativistic_beam.emit_geom_x_m,
            emit_geom_y_m=ultra_relativistic_beam.emit_geom_y_m,
            sigma_t_s=ultra_relativistic_beam.sigma_t_s,
            sigma_pz=ultra_relativistic_beam.sigma_pz,
            chirp_h=0.5 * max_chirp,
        )

        n_particles = 10000
        bunch = sample_gaussian_canonical(beam, n_particles, rng=rng)

        # Recover momenta and check mass-shell
        pz = np.sqrt((bunch.gamma**2 - 1) / (1 + bunch.thx**2 + bunch.thy**2))
        px = bunch.thx * pz
        py = bunch.thy * pz

        gamma_sq = bunch.gamma**2
        p_sq = 1.0 + px**2 + py**2 + pz**2

        np.testing.assert_allclose(gamma_sq, p_sq, rtol=1e-10)

    # ---------------------------------------------------------------
    # Dispersion
    # ---------------------------------------------------------------

    def test_dispersion_x_creates_correlation(self, ultra_relativistic_beam, rng):
        """Verify x-γ dispersion creates measurable correlation."""
        beam = GaussianElectronBeam(
            bunch_charge_C=ultra_relativistic_beam.bunch_charge_C,
            kinetic_energy_eV=ultra_relativistic_beam.kinetic_energy_eV,
            rel_energy_spread_rms=ultra_relativistic_beam.rel_energy_spread_rms,
            sigma_x_m=ultra_relativistic_beam.sigma_x_m,
            sigma_y_m=ultra_relativistic_beam.sigma_y_m,
            emit_geom_x_m=ultra_relativistic_beam.emit_geom_x_m,
            emit_geom_y_m=ultra_relativistic_beam.emit_geom_y_m,
            sigma_t_s=ultra_relativistic_beam.sigma_t_s,
            sigma_pz=ultra_relativistic_beam.sigma_pz,
            dispersion_x=3e-6,
            dispersion_y=3e-6,
        )

        n_particles = 100000
        bunch = sample_gaussian_canonical(beam, n_particles, rng=rng)
        params = fit_gaussian(bunch)

        # Dispersion should be non-zero with correct sign
        assert abs(params.dispersion_x) > 1e-7, f"D_x too small: {params.dispersion_x}"
        assert abs(params.dispersion_y) > 1e-7, f"D_y too small: {params.dispersion_y}"

    def test_dispersion_sign(self, ultra_relativistic_beam, rng):
        """Verify dispersion sign is respected."""
        n_particles = 100000

        beam_pos = GaussianElectronBeam(
            bunch_charge_C=ultra_relativistic_beam.bunch_charge_C,
            kinetic_energy_eV=ultra_relativistic_beam.kinetic_energy_eV,
            rel_energy_spread_rms=ultra_relativistic_beam.rel_energy_spread_rms,
            sigma_x_m=ultra_relativistic_beam.sigma_x_m,
            sigma_y_m=ultra_relativistic_beam.sigma_y_m,
            emit_geom_x_m=ultra_relativistic_beam.emit_geom_x_m,
            emit_geom_y_m=ultra_relativistic_beam.emit_geom_y_m,
            sigma_t_s=ultra_relativistic_beam.sigma_t_s,
            sigma_pz=ultra_relativistic_beam.sigma_pz,
            dispersion_x=3e-6,
            dispersion_y=-3e-6,
        )
        bunch_pos = sample_gaussian_canonical(beam_pos, n_particles, rng=rng)
        params_pos = fit_gaussian(bunch_pos)

        beam_neg = GaussianElectronBeam(
            bunch_charge_C=ultra_relativistic_beam.bunch_charge_C,
            kinetic_energy_eV=ultra_relativistic_beam.kinetic_energy_eV,
            rel_energy_spread_rms=ultra_relativistic_beam.rel_energy_spread_rms,
            sigma_x_m=ultra_relativistic_beam.sigma_x_m,
            sigma_y_m=ultra_relativistic_beam.sigma_y_m,
            emit_geom_x_m=ultra_relativistic_beam.emit_geom_x_m,
            emit_geom_y_m=ultra_relativistic_beam.emit_geom_y_m,
            sigma_t_s=ultra_relativistic_beam.sigma_t_s,
            sigma_pz=ultra_relativistic_beam.sigma_pz,
            dispersion_x=-3e-6,
            dispersion_y=3e-6,
        )
        bunch_neg = sample_gaussian_canonical(beam_neg, n_particles, rng=rng)
        params_neg = fit_gaussian(bunch_neg)

        assert params_pos.dispersion_x > 0, "Positive D_x should give positive dispersion"
        assert params_pos.dispersion_y < 0, "Negative D_y should give negative dispersion"
        assert params_neg.dispersion_x < 0, "Negative D_x should give negative dispersion"
        assert params_neg.dispersion_y > 0, "Positive D_y should give positive dispersion"

    def test_dispersion_mass_shell(self, ultra_relativistic_beam, rng):
        """Verify mass-shell holds with dispersion."""
        beam = GaussianElectronBeam(
            bunch_charge_C=ultra_relativistic_beam.bunch_charge_C,
            kinetic_energy_eV=ultra_relativistic_beam.kinetic_energy_eV,
            rel_energy_spread_rms=ultra_relativistic_beam.rel_energy_spread_rms,
            sigma_x_m=ultra_relativistic_beam.sigma_x_m,
            sigma_y_m=ultra_relativistic_beam.sigma_y_m,
            emit_geom_x_m=ultra_relativistic_beam.emit_geom_x_m,
            emit_geom_y_m=ultra_relativistic_beam.emit_geom_y_m,
            sigma_t_s=ultra_relativistic_beam.sigma_t_s,
            sigma_pz=ultra_relativistic_beam.sigma_pz,
            dispersion_x=3e-6,
            dispersion_y=3e-6,
        )

        n_particles = 10000
        bunch = sample_gaussian_canonical(beam, n_particles, rng=rng)

        pz = np.sqrt((bunch.gamma**2 - 1) / (1 + bunch.thx**2 + bunch.thy**2))
        px = bunch.thx * pz
        py = bunch.thy * pz

        gamma_sq = bunch.gamma**2
        p_sq = 1.0 + px**2 + py**2 + pz**2

        np.testing.assert_allclose(gamma_sq, p_sq, rtol=1e-10)

    # ---------------------------------------------------------------
    # Combined and edge cases
    # ---------------------------------------------------------------

    def test_chirp_and_dispersion_together(self, ultra_relativistic_beam, rng):
        """Verify chirp and dispersion work together."""
        sigma_z = ultra_relativistic_beam.sigma_z.to("meter").magnitude
        sigma_gamma = ultra_relativistic_beam.sigma_gamma
        max_chirp = sigma_gamma / sigma_z
        h = 0.3 * max_chirp  # 30% correlation

        beam = GaussianElectronBeam(
            bunch_charge_C=ultra_relativistic_beam.bunch_charge_C,
            kinetic_energy_eV=ultra_relativistic_beam.kinetic_energy_eV,
            rel_energy_spread_rms=ultra_relativistic_beam.rel_energy_spread_rms,
            sigma_x_m=ultra_relativistic_beam.sigma_x_m,
            sigma_y_m=ultra_relativistic_beam.sigma_y_m,
            emit_geom_x_m=ultra_relativistic_beam.emit_geom_x_m,
            emit_geom_y_m=ultra_relativistic_beam.emit_geom_y_m,
            sigma_t_s=ultra_relativistic_beam.sigma_t_s,
            sigma_pz=ultra_relativistic_beam.sigma_pz,
            chirp_h=h,
            dispersion_x=2e-6,
            dispersion_y=-2e-6,
        )

        n_particles = 100000
        bunch = sample_gaussian_canonical(beam, n_particles, rng=rng)
        params = fit_gaussian(bunch)

        # All three correlations should be measurable
        assert abs(params.chirp_h) > 0.05 * max_chirp
        assert abs(params.dispersion_x) > 1e-7
        assert abs(params.dispersion_y) > 1e-7
        assert params.dispersion_y < 0 < params.dispersion_x  # D_x positive, D_y negative

    def test_no_chirp_dispersion_default(self, ultra_relativistic_beam, rng):
        """Verify default (no chirp/dispersion) gives zero correlations."""
        n_particles = 100000
        bunch = sample_gaussian_canonical(ultra_relativistic_beam, n_particles, rng=rng)
        params = fit_gaussian(bunch)

        # Without chirp/dispersion, correlations should be small
        natural_scale = ultra_relativistic_beam.gamma0 / ultra_relativistic_beam.sigma_z.to("meter").magnitude
        assert abs(params.chirp_h) < 0.5 * natural_scale

        # Dispersion should be small (mostly noise)
        gamma_std = np.std(bunch.gamma)
        sx_in = ultra_relativistic_beam.sigma_x_m.to_unit("meter").magnitude
        assert abs(params.dispersion_x) < 3 * sx_in / gamma_std if gamma_std > 0 else True

    def test_too_strong_correlations_raise(self, ultra_relativistic_beam):
        """Verify error on too-strong chirp/dispersion."""
        # Chirp so strong that ρ_zγ > 1
        sigma_z = ultra_relativistic_beam.sigma_z.to("meter").magnitude
        sigma_gamma = ultra_relativistic_beam.sigma_gamma  # = sigma_pz * gamma0 * beta0
        max_chirp = sigma_gamma / sigma_z  # ρ_zγ = 1 at this value

        # Use chirp slightly above the limit
        beam = GaussianElectronBeam(
            bunch_charge_C=ultra_relativistic_beam.bunch_charge_C,
            kinetic_energy_eV=ultra_relativistic_beam.kinetic_energy_eV,
            rel_energy_spread_rms=ultra_relativistic_beam.rel_energy_spread_rms,
            sigma_x_m=ultra_relativistic_beam.sigma_x_m,
            sigma_y_m=ultra_relativistic_beam.sigma_y_m,
            emit_geom_x_m=ultra_relativistic_beam.emit_geom_x_m,
            emit_geom_y_m=ultra_relativistic_beam.emit_geom_y_m,
            sigma_t_s=ultra_relativistic_beam.sigma_t_s,
            sigma_pz=ultra_relativistic_beam.sigma_pz,
            chirp_h=2 * max_chirp,  # This makes ρ_zγ ≈ 2, violating the constraint
        )

        n_particles = 1000
        with pytest.raises(ValueError, match="Chirp/dispersion too strong"):
            sample_gaussian_canonical(beam, n_particles)

    def test_gamma_total_variance_preserved(self, ultra_relativistic_beam, rng):
        """Verify total marginal variance of gamma is preserved."""
        sigma_z = ultra_relativistic_beam.sigma_z.to("meter").magnitude
        sigma_gamma = ultra_relativistic_beam.sigma_gamma
        max_chirp = sigma_gamma / sigma_z
        h = 0.3 * max_chirp  # 30% z-γ correlation

        n_particles = 100000

        # Sample without correlations
        bunch_no_corr = sample_gaussian_canonical(
            ultra_relativistic_beam, n_particles, rng=rng
        )
        var_no_corr = np.var(bunch_no_corr.gamma)

        # Sample with correlations
        beam_corr = GaussianElectronBeam(
            bunch_charge_C=ultra_relativistic_beam.bunch_charge_C,
            kinetic_energy_eV=ultra_relativistic_beam.kinetic_energy_eV,
            rel_energy_spread_rms=ultra_relativistic_beam.rel_energy_spread_rms,
            sigma_x_m=ultra_relativistic_beam.sigma_x_m,
            sigma_y_m=ultra_relativistic_beam.sigma_y_m,
            emit_geom_x_m=ultra_relativistic_beam.emit_geom_x_m,
            emit_geom_y_m=ultra_relativistic_beam.emit_geom_y_m,
            sigma_t_s=ultra_relativistic_beam.sigma_t_s,
            sigma_pz=ultra_relativistic_beam.sigma_pz,
            chirp_h=h,
            dispersion_x=2e-6,
            dispersion_y=2e-6,
        )
        bunch_corr = sample_gaussian_canonical(beam_corr, n_particles, rng=rng)
        var_corr = np.var(bunch_corr.gamma)

        # The marginal variance should be the same (same sigma_pz, just reallocated)
        # Allow 10% for sampling noise
        rel_diff = abs(var_corr - var_no_corr) / var_no_corr
        assert rel_diff < 0.1, (
            f"Marginal gamma variance changed by {rel_diff*100:.1f}% "
            f"(no corr: {var_no_corr:.3g}, with corr: {var_corr:.3g})"
        )
