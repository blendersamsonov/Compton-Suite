"""GammaForge: unified package for inverse-Compton scattering simulation.

Subpackages:
- gammaforge.io: shared constants, units, parameter conventions, beam/laser representations
- gammaforge.gui: Tkinter GUI (gammaforge.gui)
- gammaforge.models.kascade: sequential multi-photon Monte Carlo engine
- gammaforge.models.xigma_i: GPU/CPU tabulated overlap-table engine
- gammaforge.models.delta: brute-force per-macroparticle binning engine
- gammaforge.models.analytical: fast closed-form estimates
"""

from __future__ import annotations

from .models.api import discover_models

__all__ = ["discover_models", "run_gui"]


def run_gui() -> None:
    from .gui import run_gui as _run_gui
    _run_gui()