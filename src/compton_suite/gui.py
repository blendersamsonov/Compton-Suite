"""``comptonsuite-gui`` console-script entry point.

Thin wrapper launching the same ``compton_guide.app.main`` that
``GUIde/scripts/run_gui.py`` calls directly.
"""

from __future__ import annotations


def run_gui() -> None:
    from compton_guide.app import main
    main()


if __name__ == "__main__":
    run_gui()
