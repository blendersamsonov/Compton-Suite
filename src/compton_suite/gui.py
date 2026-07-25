"""``comptonsuite-gui`` console-script entry point.

Thin wrapper: wires ``compton_io``/``compton_guide`` onto sys.path (this
package's own bootstrap), then launches the same ``compton_guide.app.main``
that ``GUIde/scripts/run_gui.py`` calls directly.
"""

from __future__ import annotations

from . import _bootstrap


def run_gui() -> None:
    _bootstrap.setup_paths()
    from compton_guide.app import main
    main()


if __name__ == "__main__":
    run_gui()
