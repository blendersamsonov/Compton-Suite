"""Cross-checks for laser_envelope.py's gaussian_pulse_envelope against the
two independent formulas it replaced (kascade.laser_density's arbitrary-
crossing-angle SI form, xigma_i.particles' k0_las-normalised CGS/
flying-focus form) -- each hand-transcribed here verbatim from what was
deleted, rather than imported, so these tests still catch a regression
after the original code is gone.

No cupy/GPU/tkinter needed. Run with `python3 -m pytest tests/` or
`python3 tests/io_tests/test_laser_envelope.py` directly (plain asserts).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np

from compton_suite.io.laser import gaussian_pulse_envelope  # noqa: E402

rng = np.random.default_rng(0)
_N = 500
_X = rng.normal(0.0, 3.0, _N)
_Y = rng.normal(0.0, 3.0, _N)
_Z = rng.normal(0.0, 5.0, _N)
_CT = rng.normal(0.0, 5.0, _N)

_SIGMA0 = 1.3
_RAYLEIGH = 4.0
_SIGMA_CT = 2.5


def _old_kascade_laser_density(x, y, z, t_ct, *, sigma0_l, R_sf, sigma_par_L,
                                phi=0.0, delta=(0.0, 0.0, 0.0)):
    """Verbatim transcription of the pre-refactor kascade.py laser_density
    body (crossing_angle=phi, delta offset), with ``C_LIGHT * t`` passed in
    directly as ``t_ct`` (this test works in the same ct-as-length
    convention as gaussian_pulse_envelope throughout, so the C_LIGHT
    multiplication that lived at kascade.py's own SI boundary is not
    re-tested here -- see test_laser_envelope's other tests for that
    boundary, and kascade.py's own laser_density docstring)."""
    nx, ny, nz = np.sin(phi), 0.0, -np.cos(phi)
    rx = x - delta[0]
    ry = y - delta[1]
    rz = z - delta[2]
    u = rx * nx + ry * ny + rz * nz
    perp2 = np.clip(rx * rx + ry * ry + rz * rz - u * u, 0.0, None)
    sp2 = sigma0_l ** 2 * (1.0 + (u / R_sf) ** 2)
    norm = 1.0 / ((2.0 * np.pi) ** 1.5 * sp2 * sigma_par_L)
    arg = -perp2 / (2.0 * sp2) - (u - t_ct) ** 2 / (2.0 * sigma_par_L ** 2)
    return norm * np.exp(arg)


def _old_xigma_n_ph_shape(x, y, z, t, *, w0, zT, z_rayleigh, beta_ff=0.0):
    """Verbatim transcription of the pre-refactor xigma_i/particles.py
    _push_and_sample_vectorized's sigma_l_sq/env/n_ph_shape block."""
    sigma_l_sq = w0 * w0 * (1.0 + (z - beta_ff * t) ** 2 / z_rayleigh ** 2)
    env = np.exp(-((z + t) / zT) ** 2 / 2) / np.sqrt(2 * np.pi) / zT
    return np.exp(-(x ** 2 + y ** 2) / sigma_l_sq / 2) / (2 * np.pi) / sigma_l_sq * env


def test_reduces_to_kascade_headon_formula():
    expected = _old_kascade_laser_density(
        _X, _Y, _Z, _CT, sigma0_l=_SIGMA0, R_sf=_RAYLEIGH, sigma_par_L=_SIGMA_CT)
    got = gaussian_pulse_envelope(
        _X, _Y, _Z, _CT, sigma0=_SIGMA0, rayleigh_range=_RAYLEIGH, sigma_ct=_SIGMA_CT)
    assert np.allclose(got, expected, rtol=1e-13, atol=0.0)


def test_reduces_to_xigma_headon_formula():
    expected = _old_xigma_n_ph_shape(
        _X, _Y, _Z, _CT, w0=_SIGMA0, zT=_SIGMA_CT, z_rayleigh=_RAYLEIGH)
    got = gaussian_pulse_envelope(
        _X, _Y, _Z, _CT, sigma0=_SIGMA0, rayleigh_range=_RAYLEIGH, sigma_ct=_SIGMA_CT)
    assert np.allclose(got, expected, rtol=1e-13, atol=0.0)


def test_kascade_and_xigma_formulas_agree_with_each_other_at_headon():
    # The actual unification claim: one call to the shared function
    # simultaneously satisfies both engines' original formulas -- not just
    # individually close, but exactly the same computation.
    kascade_form = _old_kascade_laser_density(
        _X, _Y, _Z, _CT, sigma0_l=_SIGMA0, R_sf=_RAYLEIGH, sigma_par_L=_SIGMA_CT)
    xigma_form = _old_xigma_n_ph_shape(
        _X, _Y, _Z, _CT, w0=_SIGMA0, zT=_SIGMA_CT, z_rayleigh=_RAYLEIGH)
    assert np.allclose(kascade_form, xigma_form, rtol=1e-13, atol=0.0)


def test_beta_ff_matches_xigma_flying_focus():
    beta_ff = 0.3
    z_rayleigh_ff = 2.0 * _SIGMA0 ** 2 * (1.0 + beta_ff)
    expected = _old_xigma_n_ph_shape(
        _X, _Y, _Z, _CT, w0=_SIGMA0, zT=_SIGMA_CT, z_rayleigh=z_rayleigh_ff, beta_ff=beta_ff)
    got = gaussian_pulse_envelope(
        _X, _Y, _Z, _CT, sigma0=_SIGMA0, rayleigh_range=z_rayleigh_ff, sigma_ct=_SIGMA_CT,
        beta_ff=beta_ff)
    assert np.allclose(got, expected, rtol=1e-13, atol=0.0)
    # beta_ff=0 must recover the plain (no flying-focus) formula exactly.
    plain = gaussian_pulse_envelope(
        _X, _Y, _Z, _CT, sigma0=_SIGMA0, rayleigh_range=_RAYLEIGH, sigma_ct=_SIGMA_CT)
    assert np.allclose(
        gaussian_pulse_envelope(_X, _Y, _Z, _CT, sigma0=_SIGMA0, rayleigh_range=_RAYLEIGH,
                                 sigma_ct=_SIGMA_CT, beta_ff=0.0),
        plain, rtol=1e-13, atol=0.0)


def test_crossing_angle_matches_kascade_tilted_axis():
    phi = 0.4
    axis = (np.sin(phi), 0.0, -np.cos(phi))
    expected = _old_kascade_laser_density(
        _X, _Y, _Z, _CT, sigma0_l=_SIGMA0, R_sf=_RAYLEIGH, sigma_par_L=_SIGMA_CT, phi=phi)
    got = gaussian_pulse_envelope(
        _X, _Y, _Z, _CT, sigma0=_SIGMA0, rayleigh_range=_RAYLEIGH, sigma_ct=_SIGMA_CT, axis=axis)
    assert np.allclose(got, expected, rtol=1e-13, atol=0.0)


def test_focus_offset_matches_kascade_delta():
    delta = (0.7, -1.2, 2.1)
    expected = _old_kascade_laser_density(
        _X, _Y, _Z, _CT, sigma0_l=_SIGMA0, R_sf=_RAYLEIGH, sigma_par_L=_SIGMA_CT, delta=delta)
    got = gaussian_pulse_envelope(
        _X, _Y, _Z, _CT, sigma0=_SIGMA0, rayleigh_range=_RAYLEIGH, sigma_ct=_SIGMA_CT, focus=delta)
    assert np.allclose(got, expected, rtol=1e-13, atol=0.0)


def test_density_integrates_to_one_over_space_at_fixed_time():
    # Photon number is conserved as the pulse translates through space, so
    # the SPATIAL integral at any one fixed ct should be 1 -- independent
    # of either engine's own formula, catches both having independently
    # made the same normalisation mistake, which the cross-checks above
    # wouldn't. (Integrating over ct too, on top of space, would
    # double-count and diverge -- NOT what "normalised" means here.)
    n, lim = 120, 20.0  # several sigma0/rayleigh_range widths in every axis
    x = np.linspace(-lim, lim, n)
    y = np.linspace(-lim, lim, n)
    z = np.linspace(-lim, lim, n)
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    for ct0 in (0.0, 1.0, -2.0, 5.0):
        density = gaussian_pulse_envelope(
            X, Y, Z, ct0, sigma0=_SIGMA0, rayleigh_range=_RAYLEIGH, sigma_ct=_SIGMA_CT)
        total = density.sum() * dx * dy * dz
        assert abs(total - 1.0) < 1e-6, (ct0, total)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
    print("\nAll laser_envelope tests passed.")
