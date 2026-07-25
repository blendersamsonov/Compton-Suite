"""xigma-i-direct: brute-force per-macroparticle resonance-binning model.

Extracted from ``xigma_i.reference.direct_binning_spectrum`` -- no table,
no importance sampling, "assumption-free on both the deposition and the
lookup" (that function's own docstring). Reuses ``xigma_i``'s Stage 0
physics (``particles.sample_bunch``/``push_and_sample``) directly as a
library dependency rather than duplicating it -- see this repo's
``gui_adapter.py`` module docstring and ``CLAUDE.md``.
"""

from . import _bootstrap

_bootstrap.setup_paths()

from .gui_adapter import DirectConfig, DirectResults, XigmaDirectAdapter  # noqa: E402

__all__ = ["DirectConfig", "DirectResults", "XigmaDirectAdapter"]
