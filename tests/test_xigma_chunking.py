"""Coverage for particles.push_and_sample's chunk= streaming path
(gammaforge.models.xigma_i.particles) -- the "query VRAM, cap allocation
at ~70%" backlog item and the CUDA-OOM-with-large-bunches bug (see
CLAUDE.md's "Open items").

Chunking slices the already-normalised per-particle arrays and re-runs
the SAME trajectory-integration math per chunk (no cross-particle
coupling -- every reduction in _integrate_trajectory_core is axis=1, i.e.
per-particle), so chunked vs unchunked output must match either exactly
(gamma/theta_x/theta_y/a0_shape/weight -- pure concatenation, no
cross-chunk arithmetic) or up to ordinary float64 summation-order noise
(diagnostics histograms -- summed across chunks, so tolerance is looser).

Needs the dev-install (see this repo's top-level CLAUDE.md). GPU-only
cases are skipped when cupy/CUDA isn't usable. Run with
`python3 -m pytest tests/test_xigma_chunking.py -v` (the `core` conda env
on this dev machine has cupy).
"""

from __future__ import annotations

import numpy as np
import pytest

from gammaforge.io.bunch import sample_gaussian_bunch
from gammaforge.io.units import LIGHT_TIME_CONTEXT
from gammaforge.misc import available_ram_bytes, available_vram_bytes
from gammaforge.models.xigma_i import particles
from gammaforge.validation.scenarios import BASELINE

_HAS_GPU = available_vram_bytes() is not None
skip_no_gpu = pytest.mark.skipif(not _HAS_GPU, reason="no CUDA-capable GPU/cupy usable")


def _kwargs_for(laser, beam, n_steps=64):
    return dict(
        k0_las=(2.0 * np.pi / laser.wavelength_m.quantity).to("1 / centimeter").magnitude,
        beta_ff=laser.beta_ff,
        sigma_lr0=laser.waist_rms_x_m.to_unit("centimeter").magnitude,
        sigma_lz=laser.duration_rms_s.to_unit("centimeter", context=LIGHT_TIME_CONTEXT).magnitude,
        omega_las=laser.omega0.to("1 / second").magnitude,
        N_l=laser.n_photons, ellipticity=laser.ellipticity, N_e=beam.N_e,
        sigma_ex=beam.sigma_x_m.to_unit("centimeter").magnitude,
        sigma_ey=beam.sigma_y_m.to_unit("centimeter").magnitude,
        n_steps=n_steps,
    )


def _bunch(n_particles, seed=1):
    return sample_gaussian_bunch(BASELINE.beam, n_particles=n_particles,
                                  rng=np.random.default_rng(seed))


def test_numpy_chunked_matches_unchunked_exactly():
    bunch = _bunch(20_000)
    kwargs = _kwargs_for(BASELINE.pulse, BASELINE.beam)

    g1, tx1, ty1, a0_1, w1 = particles.push_and_sample(bunch, backend='numpy', chunk=None, **kwargs)
    g2, tx2, ty2, a0_2, w2 = particles.push_and_sample(bunch, backend='numpy', chunk=3_333, **kwargs)

    np.testing.assert_allclose(a0_1, a0_2, rtol=1e-12)
    np.testing.assert_allclose(w1, w2, rtol=1e-12)
    np.testing.assert_array_equal(g1, g2)
    np.testing.assert_array_equal(tx1, tx2)
    np.testing.assert_array_equal(ty1, ty2)


def test_numpy_chunked_diagnostics_match_unchunked():
    bunch = _bunch(15_000)
    kwargs = _kwargs_for(BASELINE.pulse, BASELINE.beam)

    r1 = particles.push_and_sample(bunch, backend='numpy', chunk=None,
                                    n_time_bins=20, n_spatial_bins=15, **kwargs)
    r2 = particles.push_and_sample(bunch, backend='numpy', chunk=2_500,
                                    n_time_bins=20, n_spatial_bins=15, **kwargs)
    d1, d2 = r1[5], r2[5]

    # Edges compared at float32 precision (rtol=1e-6): the unchunked path's
    # "auto-derive" branch of _resolve_spatial_range inherits whatever
    # dtype sigma_ex/sigma_lr0 already carry (float32 for this scenario's
    # beam), while the chunked path always resolves edges up front via the
    # "given edges" branch, which upcasts through `float()` to float64 --
    # a pre-existing dtype quirk in _resolve_spatial_range, not something
    # chunking introduces; values agree to float32 precision.
    np.testing.assert_allclose(d1.t_edges, d2.t_edges, rtol=1e-6)
    np.testing.assert_allclose(d1.spatial_x_edges, d2.spatial_x_edges, rtol=1e-6)
    np.testing.assert_allclose(d1.spatial_y_edges, d2.spatial_y_edges, rtol=1e-6)
    # Summed across chunks (unlike a0_shape/weight, which are pure
    # concatenation) -- ordinary float64 summation-order noise, not exact.
    np.testing.assert_allclose(d1.time_envelope, d2.time_envelope, rtol=1e-4)
    np.testing.assert_allclose(d1.spatial_envelope, d2.spatial_envelope, rtol=1e-4)


@skip_no_gpu
def test_cupy_chunked_matches_numpy_unchunked():
    bunch = _bunch(20_000)
    kwargs = _kwargs_for(BASELINE.pulse, BASELINE.beam)

    g1, tx1, ty1, a0_1, w1 = particles.push_and_sample(bunch, backend='numpy', chunk=None, **kwargs)
    g2, tx2, ty2, a0_2, w2 = particles.push_and_sample(bunch, backend='cupy', chunk=3_333, **kwargs)

    np.testing.assert_allclose(a0_1, a0_2.get(), rtol=1e-6)
    np.testing.assert_allclose(w1, w2.get(), rtol=1e-6)


@skip_no_gpu
def test_cupy_chunking_fixes_documented_oom_case():
    """Regression test for CLAUDE.md's "CUDA OOM with large electron
    bunches" open item, calibrated on a GTX 1660 Ti (6GB, cupy 14.0.1):
    push_and_sample(backend='cupy') at n_steps=64 OOMs at n_particles=
    800_000 when unchunked, but succeeds when auto-sized chunking is used.
    """
    n = 800_000
    bunch = _bunch(n)
    kwargs = _kwargs_for(BASELINE.pulse, BASELINE.beam)

    chunk = particles.estimate_chunk_size(n, 64, 'cupy')
    if chunk >= n:
        pytest.skip("this GPU has enough free VRAM that n=800_000 wouldn't "
                    "have OOM'd unchunked either -- nothing to regress against")

    g, tx, ty, a0, w = particles.push_and_sample(bunch, backend='cupy', chunk=chunk, **kwargs)
    assert g.shape[0] == n


@skip_no_gpu
def test_cupy_oom_retry_halves_chunk_and_succeeds():
    """A chunk sized too large to fit in memory (but < n_particles, so the
    chunked/retry loop is actually entered) must halve and retry rather
    than propagating the OutOfMemoryError."""
    n = 800_000
    bunch = _bunch(n)
    kwargs = _kwargs_for(BASELINE.pulse, BASELINE.beam)

    g, tx, ty, a0, w = particles.push_and_sample(bunch, backend='cupy', chunk=n - 1, **kwargs)
    assert g.shape[0] == n


@skip_no_gpu
def test_cupy_oom_retry_covers_diagnostics_binning_too():
    """Regression test for a review finding: the OOM-retry halving must
    protect the _bin_temporal/_bin_spatial calls, not just
    _integrate_trajectory_core -- both XigmaAdapter and DirectAdapter
    always request diagnostics (n_time_bins/n_spatial_bins default to
    non-None in adapter.py), so an OOM during binning is the codepath
    real runs actually exercise, not an edge case. Without this coverage,
    a chunk sized fine for the integration step but too large once the
    binners' own temporaries stack on top would crash uncaught instead of
    halving and retrying."""
    n = 800_000
    bunch = _bunch(n)
    kwargs = _kwargs_for(BASELINE.pulse, BASELINE.beam)

    g, tx, ty, a0, w, diagnostics = particles.push_and_sample(
        bunch, backend='cupy', chunk=n - 1, n_time_bins=20, n_spatial_bins=15, **kwargs)
    assert g.shape[0] == n
    assert diagnostics.time_envelope is not None
    assert diagnostics.spatial_envelope is not None


def test_estimate_chunk_size_numba_returns_n_particles_unchanged():
    """numba's own per-particle compiled loop already avoids the
    O(n*n_steps) blowup this function exists to bound -- nothing to
    chunk there, unlike 'numpy' (see the next test)."""
    assert particles.estimate_chunk_size(123_456, 64, 'numba') == 123_456


def test_estimate_chunk_size_numpy_bounded_by_n_particles():
    """Regression test for a follow-on user report: 'numpy' used to be
    exempted from chunking on the assumption that system RAM bounds it --
    wrong at large enough n_particles*n_steps (a related unchunked
    broadcast exhausted 128GB of RAM). 'numpy' is now chunked against
    gammaforge.misc.available_ram_bytes() the same way 'cupy' is chunked
    against available_vram_bytes()."""
    if available_ram_bytes() is None:
        pytest.skip("available_ram_bytes() unsupported on this platform")
    chunk = particles.estimate_chunk_size(5_000_000, 64, 'numpy')
    assert 0 < chunk <= 5_000_000


@skip_no_gpu
def test_estimate_chunk_size_cupy_bounded_by_n_particles():
    chunk = particles.estimate_chunk_size(50_000, 64, 'cupy')
    assert 0 < chunk <= 50_000


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
