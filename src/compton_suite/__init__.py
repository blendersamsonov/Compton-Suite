"""ComptonSuite: the umbrella package tying together the toolkit's pieces
(all now part of this single repo):

  * ``compton_io`` (``IO/``) -- shared units/constants/parameter-convention
    framework, electron-bunch and laser-pulse representations, external
    I/O (elegant ``.ele``, this package's own YAML formats).
  * ``compton_guide`` (``GUIde/``) -- the Tkinter GUI. Contains no physics
    of its own; discovers and runs whichever model adapters are available.
  * ``kascade`` / ``xigma_i`` / ``xigma_direct`` / ``analytical`` (all
    under ``models/``) -- physics engines/models, each pluggable into the
    GUI via a duck-typed ``ModelAdapter`` and each independently usable
    outside it.

Typical scripting use (no tkinter/matplotlib import):

    import compton_suite
    models = compton_suite.discover_models()
    cfg, extra = models["kascade"].params_to_config(fields, quantum=False)

Or from the command line, after the dev-install (see this repo's
top-level ``CLAUDE.md``):

    comptonsuite-gui
"""

from __future__ import annotations

from .models import discover_models

__all__ = ["discover_models", "run_gui"]


def run_gui() -> None:
    from .gui import run_gui as _run_gui
    _run_gui()
