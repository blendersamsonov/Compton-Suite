"""xigma-i-direct: brute-force per-macroparticle resonance-binning model.

Extracted from ``xigma_i.reference.direct_binning_spectrum`` -- no table,
no importance sampling, "assumption-free on both the deposition and the
lookup" (that function's own docstring). Reuses ``xigma_i``'s Stage 0
physics (``particles.bunch_from_macrobunch``/``push_and_sample``)
directly as a library dependency rather than duplicating it -- see this
repo's ``gui_adapter.py`` module docstring and ``CLAUDE.md``. Electron
sampling itself is the caller's job (``gui_adapter.run_simulation``
requires an already-sampled ``electrons`` bunch), so ``particles.
sample_bunch`` is no longer called from anywhere in this repo -- only
from ``xigma_i`` itself, where it remains a legitimate standalone Stage 0
entry point.
"""

from . import _bootstrap

_bootstrap.setup_paths()

from .gui_adapter import DirectConfig, DirectResults, XigmaDirectAdapter  # noqa: E402

__all__ = ["DirectConfig", "DirectResults", "XigmaDirectAdapter"]
