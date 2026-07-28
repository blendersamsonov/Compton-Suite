"""Cross-checks for bunch.py's MacroBunch / GaussianElectronBeam /
sample_gaussian_bunch / fit_gaussian.

No cupy/GPU/tkinter needed. Run with `python3 -m pytest tests/` or
`python3 tests/test_bunch.py` directly (plain asserts).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from compton_suite.io.bunch import (  # noqa: E402
    GaussianElectronBeam,
    MacroBunch,
    fit_gaussian,
    sample_gaussian_bunch,
    validate,
)

_EXAMPLE_BEAM = GaussianElectronBeam(
    bunch_charge_C=100.0e-12,
    kinetic_energy_eV=200.0e6,
    rel_energy_spread_rms=0.001,
    sigma_x_m=10.0e-6,
    sigma_y_m=10.0e-6,
    emit_geom_x_m=0.05e-6,
    emit_geom_y_m=0.05e-6,
    sigma_t_s=1.0e-12,
    sigma_pz=0.001,
)


def test_derived_quantities_are_sane():
    assert validate(_EXAMPLE_BEAM) == []
    assert _EXAMPLE_BEAM.gamma0 > 1.0
    assert 0.0 < _EXAMPLE_BEAM.beta0 < 1.0
    assert _EXAMPLE_BEAM.N_e > 0
    assert _EXAMPLE_BEAM.beta_star_x_m > 0
    assert abs(_EXAMPLE_BEAM.beta_star_x_m - _EXAMPLE_BEAM.sigma_x_m**2 / _EXAMPLE_BEAM.emit_geom_x_m) < 1e-30


def test_sample_gaussian_bunch_matches_beam_moments():
    rng = np.random.default_rng(0)
    bunch = sample_gaussian_bunch(_EXAMPLE_BEAM, 200_000, rng=rng)
    assert bunch.n_particles == 200_000
    assert abs(np.std(bunch.x) / _EXAMPLE_BEAM.sigma_x_m - 1.0) < 0.02
    assert abs(np.std(bunch.thx) / _EXAMPLE_BEAM.divergence_x_rad - 1.0) < 0.02
    assert abs(np.mean(bunch.gamma) - _EXAMPLE_BEAM.gamma0) / _EXAMPLE_BEAM.gamma0 < 1e-3
    assert abs(np.std(bunch.gamma) / _EXAMPLE_BEAM.sigma_gamma - 1.0) < 0.02


def test_fit_gaussian_round_trips_at_the_waist():
    # Sampled directly at the waist (alpha=0 by construction): fit_gaussian
    # should recover the original beam's parameters closely.
    rng = np.random.default_rng(1)
    bunch = sample_gaussian_bunch(_EXAMPLE_BEAM, 500_000, rng=rng)
    fit = fit_gaussian(bunch)

    assert abs(fit.sigma_x_m / _EXAMPLE_BEAM.sigma_x_m - 1.0) < 0.02
    assert abs(fit.emit_geom_x_m / _EXAMPLE_BEAM.emit_geom_x_m - 1.0) < 0.03
    assert abs(fit.kinetic_energy_eV / _EXAMPLE_BEAM.kinetic_energy_eV - 1.0) < 1e-3
    assert abs(fit.rel_energy_spread_rms / _EXAMPLE_BEAM.rel_energy_spread_rms - 1.0) < 0.05


def test_fit_gaussian_recovers_waist_after_drift():
    # Sample at the waist, then ballistically drift the macroparticles
    # away from it (pure kinematic drift: x -> x + thx*L). fit_gaussian
    # must still recover the ORIGINAL waist's sigma_x/emittance, since the
    # waist-location algorithm is supposed to be drift-invariant (the
    # concrete "ballistic propagation + Liouville" claim this module's
    # docstring makes).
    rng = np.random.default_rng(2)
    bunch_at_waist = sample_gaussian_bunch(_EXAMPLE_BEAM, 500_000, rng=rng)
    L = 2.5  # meters downstream of the waist
    drifted = MacroBunch(
        x=bunch_at_waist.x + bunch_at_waist.thx * L,
        y=bunch_at_waist.y + bunch_at_waist.thy * L,
        z=bunch_at_waist.z,
        thx=bunch_at_waist.thx,
        thy=bunch_at_waist.thy,
        gamma=bunch_at_waist.gamma,
        weight=bunch_at_waist.weight,
    )
    # Sanity: the drifted bunch's raw sigma_x is now LARGER than the waist's.
    assert np.std(drifted.x) > np.std(bunch_at_waist.x) * 1.5

    fit = fit_gaussian(drifted)
    assert abs(fit.sigma_x_m / _EXAMPLE_BEAM.sigma_x_m - 1.0) < 0.03
    assert abs(fit.emit_geom_x_m / _EXAMPLE_BEAM.emit_geom_x_m - 1.0) < 0.03


def test_sample_gaussian_bunch_chirp_correlates_energy_with_z():
    # Chirp and angle_energy_corr features were removed in favor of canonical sampling.
    # This test is kept as a placeholder to document the change.
    # New sampling uses canonical variables with mass-shell enforcement.
    pass


def test_sample_gaussian_bunch_angle_energy_corr():
    # Chirp and angle_energy_corr features were removed in favor of canonical sampling.
    # This test is kept as a placeholder to document the change.
    # New sampling uses canonical variables with mass-shell enforcement.
    pass


def test_validate_rejects_nonpositive_fields():
    bad = GaussianElectronBeam(
        bunch_charge_C=-1.0, kinetic_energy_eV=1e6, rel_energy_spread_rms=0.001,
        sigma_x_m=1e-5, sigma_y_m=1e-5, emit_geom_x_m=1e-8, emit_geom_y_m=1e-8,
        sigma_t_s=1e-12, sigma_pz=0.001,
    )
    try:
        validate(bad)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_validate_warns_on_suspicious_emittance():
    # emit_geom_x_m > sigma_x_m -- classic mm*mrad-entered-as-m*rad mistake.
    beam = GaussianElectronBeam(
        bunch_charge_C=100e-12, kinetic_energy_eV=200e6, rel_energy_spread_rms=0.001,
        sigma_x_m=10e-6, sigma_y_m=10e-6, emit_geom_x_m=5e-2, emit_geom_y_m=5e-2,
        sigma_t_s=1e-12, sigma_pz=0.001,
    )
    warnings = validate(beam)
    assert any("units" in w.lower() for w in warnings)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
    print("\nAll bunch tests passed.")
