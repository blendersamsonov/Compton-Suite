"""Physics engine models for ComptonSuite.

Each model implements the ``ModelAdapter`` protocol (``compton_suite.models.api``)
and can be used independently or through the GUI.

Available models:
- kascade: Sequential multi-photon Monte Carlo (CPU, SI units)
- xigma_i: Tabulated overlap table pipeline (GPU/CPU, CGS units); its
  brute-force per-macroparticle mode is registered separately as "delta"
  (see ``xigma_i.adapter.DirectAdapter``) -- not its own subpackage.
- analytical: Fast closed-form estimates (CPU, SI units)

Deliberately does NOT eagerly import the model subpackages here: xigma_i
needs cupy+CUDA or numba, which may not be installed -- ``models.api.
discover_models()`` imports each one lazily and simply skips registering
"xigma-i"/"delta" if neither is usable, instead of one missing optional
dependency breaking `import compton_suite.models` entirely.
"""

from __future__ import annotations

__all__: list[str] = []