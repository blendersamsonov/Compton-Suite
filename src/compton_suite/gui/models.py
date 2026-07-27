"""Model registry -- direct imports since all packages ship together."""

from __future__ import annotations

from compton_suite.gui.model_api import UnavailableAdapter, register, registered_models
from compton_suite.models.kascade import kascade_adapter
from compton_suite.models.xigma_i import gui_adapter as _xigma_gui
from compton_suite.models.delta import gui_adapter as _delta_gui
from compton_suite.models.analytical import analytical_adapter as _analytical


def discover_models() -> dict:
    """Populate the model registry with direct imports."""
    register("kascade", kascade_adapter.KascadeAdapter())
    register("xigma-i", _xigma_gui.XigmaAdapter())
    register("delta", _delta_gui.DeltaAdapter())
    register("analytical", _analytical.AnalyticalAdapter())
    return registered_models()