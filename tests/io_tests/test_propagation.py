"""Cross-checks for propagation.py's ballistic drift primitives.

No cupy/GPU/tkinter needed. Run with `python3 -m pytest tests/` or
`python3 tests/test_propagation.py` directly (plain asserts).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from compton_suite.io.bunch import (  # noqa: E402
    Bunch,
    GaussianElectronBeam,
    ballistic_position_z0_reference,
    fit_gaussian,
    propagate,
    sample_gaussian_bunch,
    stream,
)
from compton_suite.io.units import (  # noqa: E402
    C_LIGHT,
    PhysicalMeaning,
    PhysicalQuantity,
    TimeConvention,
    WidthConvention,
)

_EXAMPLE_BEAM = GaussianElectronBeam(
    bunch_charge_C=PhysicalQuantity(100e-12, "coulomb", PhysicalMeaning.BUNCH_CHARGE),
    kinetic_energy_eV=PhysicalQuantity(200e6, "electron_volt", PhysicalMeaning.BEAM_ENERGY),
    rel_energy_spread_rms=0.001,
    sigma_x_m=PhysicalQuantity(10e-6, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE, WidthConvention.SIGMA_INTENSITY_RMS),
    sigma_y_m=PhysicalQuantity(10e-6, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE, WidthConvention.SIGMA_INTENSITY_RMS),
    emit_geom_x_m=PhysicalQuantity(0.05e-6, "meter", PhysicalMeaning.EMITTANCE),
    emit_geom_y_m=PhysicalQuantity(0.05e-6, "meter", PhysicalMeaning.EMITTANCE),
    sigma_t_s=PhysicalQuantity(1e-12, "second", PhysicalMeaning.BUNCH_LENGTH, TimeConvention.SIGMA_INTENSITY_RMS),
    sigma_pz=0.001,
)


def test_propagate_per_particle_positions_match_hand_formula():
    # propagate()'s per-particle push (the old ballistic_position_simultaneous,
    # now folded into drift()'s generalized per-particle-L body -- see
    # bunch.py's propagate() docstring) against a hand-rolled formula.
    x0, y0, z0 = np.array([1.0, -2.0]), np.array([0.5, 0.0]), np.array([0.0, 3.0])
    thx, thy = np.array([0.01, -0.02]), np.array([0.0, 0.01])
    gamma = np.array([500.0, 500.0])
    bunch = Bunch(x=x0, y=y0, z=z0, thx=thx, thy=thy, gamma=gamma, weight=1.0)
    dt = 5.0 / C_LIGHT
    moved = propagate(bunch, dt)
    vz = np.sqrt(1.0 - thx**2 - thy**2)
    L = vz * C_LIGHT * dt
    assert np.allclose(moved.x, x0 + thx * L)
    assert np.allclose(moved.y, y0 + thy * L)
    assert np.allclose(moved.z, z0 + L)
    # dt=0 is a no-op.
    same = propagate(bunch, 0.0)
    assert np.allclose(same.x, x0) and np.allclose(same.y, y0) and np.allclose(same.z, z0)


def test_ballistic_position_z0_reference_matches_hand_formula():
    x0, y0, z0 = np.array([1.0, -2.0]), np.array([0.5, 0.0]), np.array([0.3, -0.1])
    thx, thy = np.array([0.01, -0.02]), np.array([0.0, 0.01])
    t = 2.0
    x, y, z = ballistic_position_z0_reference(x0, y0, z0, thx, thy, t)
    vz = np.sqrt(1.0 - thx**2 - thy**2)
    dt0 = z0 / vz
    assert np.allclose(x, x0 + thx * (t + dt0))
    assert np.allclose(y, y0 + thy * (t + dt0))
    assert np.allclose(z, z0 + vz * t)


def test_propagate_round_trips():
    rng = np.random.default_rng(3)
    bunch = sample_gaussian_bunch(_EXAMPLE_BEAM, 10_000, rng=rng)
    forward = propagate(bunch, 1e-12)
    back = propagate(forward, -1e-12)
    assert np.allclose(back.x, bunch.x, atol=1e-20)
    assert np.allclose(back.y, bunch.y, atol=1e-20)
    assert np.allclose(back.z, bunch.z, atol=1e-15)
    # gamma/thx/thy unchanged (straight-line, no acceleration).
    assert np.array_equal(forward.thx, bunch.thx)
    assert np.array_equal(forward.gamma, bunch.gamma)


def test_propagate_recovers_waist_after_drift():
    # Same spirit as test_bunch.py's test_fit_gaussian_recovers_waist_after_drift,
    # but going through the shared time-based propagate() primitive instead
    # of a hand-rolled `x + thx*L`.
    rng = np.random.default_rng(4)
    bunch_at_waist = sample_gaussian_bunch(_EXAMPLE_BEAM, 500_000, rng=rng)
    dt = 2.5 / C_LIGHT  # time to drift ~2.5 m downstream at ~c
    drifted = propagate(bunch_at_waist, dt)

    assert np.std(drifted.x) > np.std(bunch_at_waist.x) * 1.5

    fit = fit_gaussian(drifted)
    assert abs(fit.sigma_x_m.to_unit("meter").magnitude
               / _EXAMPLE_BEAM.sigma_x_m.to_unit("meter").magnitude - 1.0) < 0.03
    assert abs(fit.emit_geom_x_m.to_unit("meter").magnitude
               / _EXAMPLE_BEAM.emit_geom_x_m.to_unit("meter").magnitude - 1.0) < 0.03


def test_stream_yields_one_snapshot_per_time():
    rng = np.random.default_rng(5)
    bunch = sample_gaussian_bunch(_EXAMPLE_BEAM, 1_000, rng=rng)
    t_grid = np.array([0.0, 1e-12, 2e-12])
    snapshots = list(stream(bunch, t_grid))
    assert len(snapshots) == 3
    assert np.allclose(snapshots[0].x, bunch.x)
    # Each snapshot computed independently from the original bunch, not
    # accumulated -- snapshot[2] should match propagate(bunch, t_grid[2])
    # directly, not propagate(propagate(bunch, t_grid[1]), t_grid[2]-t_grid[1]).
    direct = propagate(bunch, t_grid[2])
    assert np.allclose(snapshots[2].x, direct.x)
    assert np.allclose(snapshots[2].z, direct.z)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
    print("\nAll propagation tests passed.")
