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
        "compton_suite.misc.detect_device: no usable backend -- no CUDA-capable GPU "
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
