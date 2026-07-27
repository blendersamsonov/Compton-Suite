"""Physics engine models for ComptonSuite.

Each model implements the ModelAdapter protocol (gui.model_api) and can be
used independently or through the GUI.

Available models:
- kascade: Sequential multi-photon Monte Carlo (CPU, SI units)
- xigma_i: Tabulated overlap table pipeline (GPU/CPU, CGS units)
- delta: Brute-force per-macroparticle binning (GPU/CPU, CGS units)
- analytical: Fast closed-form estimates (CPU, SI units)
"""

from __future__ import annotations

from . import kascade, xigma_i, delta, analytical

__all__ = ["kascade", "xigma_i", "delta", "analytical"]