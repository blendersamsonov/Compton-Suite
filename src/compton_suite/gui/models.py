"""Model registry -- direct imports since all packages ship together."""

from __future__ import annotations

from compton_suite.gui.model_api import UnavailableAdapter, register, registered_models
from compton_suite.models.kascade import kascade_adapter
from compton_suite.models.xigma_i import gui_adapter as _xigma_gui
from compton_suite.models.xigma_direct import gui_adapter as _xigma_direct_gui
from compton_suite.models.analytical import analytical_adapter as _analytical


def discover_models() -> dict:
    """Populate the model registry with direct imports."""
    register("kascade", kascade_adapter.KascadeAdapter())
    register("xigma-i", _xigma_gui.XigmaAdapter())
    register("xigma-i-direct", _xigma_direct_gui.XigmaDirectAdapter())
    register("analytical", _analytical.AnalyticalAdapter())
    return registered_models()