"""ComptonSuite: the umbrella meta-package for a toolkit of independent,
sibling-repo Compton-scattering physics tools.

Glues together (each its own git repo, importable independently, not
pip-published anywhere -- see ``ComptonSuite/CLAUDE.md`` for the full
topology):

  * ``compton_io`` -- shared units/constants/parameter-convention
    framework, electron-bunch and laser-pulse representations, external
    I/O (elegant ``.ele``, this package's own YAML formats).
  * ``compton_guide`` -- the Tkinter GUI. Contains no physics of its own;
    discovers and runs whichever model adapters are available.
  * ``kascade`` / ``xigma_i`` / ``xigma_direct`` -- physics engines/models,
    each pluggable into the GUI via a duck-typed ``ModelAdapter`` and each
    independently usable outside it.

This package itself additionally hosts the "analytical" model
(``analytical.py``/``analytical_adapter.py``) -- closed-form yield/
spectrum/width estimates, fast enough that the GUI runs it automatically
alongside whichever model is selected, for real-time preview and base
validation. It lives here rather than in ``compton_guide`` (which must
stay physics-free) or in an engine-specific repo (since it's meant to
validate every model, not just one).

Typical scripting use (no tkinter/matplotlib import):

    import compton_suite
    models = compton_suite.discover_models()
    cfg, extra = models["kascade"].params_to_config(fields, quantum=False)

Or from the command line, after ``pip install -e ComptonSuite/``:

    comptonsuite-gui
"""

from __future__ import annotations

from . import _bootstrap

_bootstrap.setup_paths()

from . import analytical  # noqa: E402
from .models import discover_models  # noqa: E402

__all__ = ["analytical", "discover_models", "run_gui"]


def run_gui() -> None:
    from .gui import run_gui as _run_gui
    _run_gui()
