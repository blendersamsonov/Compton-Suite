"""Coverage for spectrum_from_particles.angle_integrated_spectrum's chunk=
streaming path -- the CUDA-OOM bug where XigmaAdapter's Stage 2 dN/ds query
(TabulatedEngine.spectrum -> this function) materialised one
(n_particles, len(s))-shaped float64 broadcast in a single unchunked
allocation. Unlike particles.push_and_sample, n_particles here can be
several million while len(s) is only a few hundred, so this is the
opposite chunking axis from test_xigma_chunking.py's coverage.

Needs the dev-install (see this repo's top-level CLAUDE.md). GPU-only
cases are skipped when cupy/CUDA isn't usable. Run with
`python3 -m pytest tests/test_angle_integrated_spectrum_chunking.py -v`.
"""

from __future__ import annotations

import numpy as np
import pytest

from gammaforge.misc import available_vram_bytes
from gammaforge.models.xigma_i.spectrum_from_particles import (
    _estimate_s_chunk,
    angle_integrated_spectrum,
)

_HAS_GPU = available_vram_bytes() is not None
skip_no_gpu = pytest.mark.skipif(not _HAS_GPU, reason="no CUDA-capable GPU/cupy usable")


def _gamma_weight(n, seed=0):
    rng = np.random.default_rng(seed)
    gamma = rng.uniform(50, 500, n)
    weight = rng.uniform(0.5, 1.5, n)
    return gamma, weight


def test_numpy_chunked_matches_unchunked_exactly():
    gamma, weight = _gamma_weight(10_000)
    s = np.linspace(0.0, 1.0, 337) * gamma.max() ** 2

    out_full = angle_integrated_spectrum(gamma, weight, s, backend='numpy', chunk=None)
    out_chunked = angle_integrated_spectrum(gamma, weight, s, backend='numpy', chunk=17)

    np.testing.assert_allclose(out_full, out_chunked, rtol=1e-12)


def test_scalar_s_still_returns_scalar():
    gamma, weight = _gamma_weight(1_000)
    out = angle_integrated_spectrum(gamma, weight, 0.3, backend='numpy')
    assert np.ndim(out) == 0


def test_estimate_s_chunk_numpy_returns_n_s_unchanged():
    assert _estimate_s_chunk(5_000_000, 337, 'numpy') == 337


@skip_no_gpu
def test_estimate_s_chunk_cupy_bounded_by_n_s():
    chunk = _estimate_s_chunk(5_000_000, 337, 'cupy')
    assert 0 < chunk <= 337


@skip_no_gpu
def test_cupy_chunked_matches_numpy_unchunked():
    gamma, weight = _gamma_weight(50_000)
    s = np.linspace(0.0, 1.0, 200) * gamma.max() ** 2

    out_cpu = angle_integrated_spectrum(gamma, weight, s, backend='numpy')
    out_gpu = angle_integrated_spectrum(gamma, weight, s, backend='cupy')

    np.testing.assert_allclose(out_cpu, out_gpu.get(), rtol=1e-6)


@skip_no_gpu
def test_cupy_auto_chunk_avoids_oom_at_reported_scale():
    """Regression test for the reported bug: XigmaAdapter OOM'd allocating
    a 5,000,000 x ~256 float64 array (10.24 GB) in
    angle_integrated_spectrum's un-chunked broadcast. Auto-sized chunking
    (chunk=None) must complete without raising, and must match a CPU
    reference -- computed with an explicit small chunk here too, since an
    *unchunked* 5,000,000 x 256 float64 broadcast is ~30-40GB counting
    temporaries and would exceed system RAM on a modest dev machine (this
    is a test-harness memory bound, not something angle_integrated_spectrum
    itself needs chunking for on 'numpy' -- see _estimate_s_chunk's
    docstring)."""
    n = 5_000_000
    gamma, weight = _gamma_weight(n)
    s = np.linspace(0.0, 1.0, 256) * gamma.max() ** 2

    out_gpu = angle_integrated_spectrum(gamma, weight, s, backend='cupy')
    out_cpu = angle_integrated_spectrum(gamma, weight, s, backend='numpy', chunk=8)

    np.testing.assert_allclose(out_cpu, out_gpu.get(), rtol=1e-6)


@skip_no_gpu
def test_cupy_oom_retry_halves_chunk_and_succeeds():
    """A chunk sized too large to fit in memory (but < len(s), so the
    chunked/retry loop is actually entered) must halve and retry rather
    than propagating the OutOfMemoryError."""
    n = 5_000_000
    gamma, weight = _gamma_weight(n)
    n_s = 256
    s = np.linspace(0.0, 1.0, n_s) * gamma.max() ** 2

    out = angle_integrated_spectrum(gamma, weight, s, backend='cupy', chunk=n_s - 1)
    assert out.shape[0] == n_s


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
