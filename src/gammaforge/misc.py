import os

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False


def detect_device() -> str:
    """Auto-detect which backend to use: a real CUDA GPU via cupy if
    available, else CPU (requires numba). Raises if neither works -- there
    is no third backend."""
    if _HAS_CUPY:
        try:
            if cp.cuda.runtime.getDeviceCount() > 0:
                return 'gpu'
        except Exception:
            pass
    try:
        import numba  # noqa: F401
    except ImportError:
        pass
    else:
        return 'cpu'
    raise RuntimeError(
        "gammaforge.misc.detect_device: no usable backend -- no CUDA-capable GPU "
        "detected (or cupy isn't installed), and numba isn't installed for "
        "the CPU fallback. Install numba (pip install numba) for CPU-only "
        "use, or a working cupy+CUDA setup for GPU use.")


def get_xp(device: str):
    """Array module for ``device`` ('gpu' -> cupy, 'cpu' -> numpy) -- not a
    unit conversion, so it doesn't belong on a physics-parameter bundle;
    every model that dispatches numpy/cupy by device string uses this one
    helper."""
    return cp if device == 'gpu' else np


def to_host(x, device: str):
    """Bring an array back to host (numpy) if it's on-device (cupy);
    a no-op for CPU-resident arrays."""
    return x.get() if device == 'gpu' else x


def available_vram_bytes() -> int | None:
    """Free GPU memory in bytes via cupy, or None if no GPU/cupy usable --
    the one place any model queries VRAM for auto-sizing a chunk/batch."""
    if not _HAS_CUPY:
        return None
    try:
        free, _total = cp.cuda.Device().mem_info
        return int(free)
    except Exception:
        return None


def available_ram_bytes() -> int | None:
    """Available system RAM in bytes (POSIX sysconf), or None if the query
    isn't supported on this platform -- the CPU-backend analogue of
    available_vram_bytes(), used to bound numpy-backend chunk sizing the
    same way cupy-backend chunking is bounded by free VRAM.

    "Plain numpy is bounded by system RAM, so it doesn't need chunking"
    was this codebase's original assumption for numpy-backend paths (see
    particles.estimate_chunk_size's docstring) -- wrong at large enough
    problem sizes: a 5,000,000-particle x 1,024-point broadcast is over
    100GB with only a handful of live float64 temporaries, enough to
    exhaust RAM on a well-provisioned workstation (confirmed by a user
    report). Unlike a cupy OutOfMemoryError, an oversized host allocation
    on Linux can trigger the kernel OOM-killer (or heavy swapping) rather
    than a catchable Python exception, so this proactive estimate is the
    primary defense for the numpy path, not a fallback behind a retry
    loop -- see spectrum_from_particles._estimate_s_chunk/
    particles.estimate_chunk_size.
    """
    try:
        return int(os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGESIZE'))
    except (ValueError, OSError, AttributeError):
        return None
