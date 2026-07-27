"""delta: brute-force per-macroparticle resonance-binning model.

Extracted from ``xigma_i.spectrum_from_particles.direct_binning_spectrum`` -- no table,
no importance sampling, "assumption-free on both the deposition and the
lookup" (that function's own docstring). Reuses ``xigma_i``'s Stage 0
physics (``particles.push_and_sample``) directly as a library dependency,
rather than duplicating it -- see this
repo's ``gui_adapter.py`` module docstring and ``CLAUDE.md``. Electron
sampling itself is the caller's job (``gui_adapter.run_simulation``
requires an already-sampled ``electrons`` bunch) -- there is no internal
sampler anywhere in this package.
"""

from .gui_adapter import DirectConfig, DeltaAdapter

__all__ = ["DirectConfig", "DeltaAdapter"]
