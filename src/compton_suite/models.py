"""Headless-safe model discovery for scripting -- no tkinter/matplotlib
import, same as ``compton_guide.models`` itself (which this re-exports).

    import compton_suite
    models = compton_suite.discover_models()
    adapter = models["kascade"]
    cfg, extra = adapter.params_to_config(...)
"""

from __future__ import annotations

from . import _bootstrap

_bootstrap.setup_paths()

from compton_guide.models import discover_models  # noqa: E402

__all__ = ["discover_models"]
