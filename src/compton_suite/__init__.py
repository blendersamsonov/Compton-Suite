"""ComptonSuite: unified package for inverse-Compton scattering simulation.

Subpackages:
- compton_suite.io: shared constants, units, parameter conventions, beam/laser representations
- compton_suite.gui: Tkinter GUI (compton_suite.gui)
- compton_suite.models.kascade: sequential multi-photon Monte Carlo engine
- compton_suite.models.xigma_i: GPU/CPU tabulated overlap-table engine
- compton_suite.models.delta: brute-force per-macroparticle binning engine
- compton_suite.models.analytical: fast closed-form estimates
"""

from __future__ import annotations

from .models.api import discover_models

__all__ = ["discover_models", "run_gui"]


def run_gui() -> None:
    from .gui import run_gui as _run_gui
    _run_gui()