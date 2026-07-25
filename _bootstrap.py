"""sys.path wiring so ``kascade`` can import ``compton_suite`` without it
being pip-installed.

Same content-based sibling-directory autodiscovery pattern already used by
``compton_guide.bootstrap`` (for ``kascade``/``xigma_i``) and
``xigma_i._bootstrap`` (for ``compton_suite``), physically duplicated here
for the same reason: the discovery code has to run before the thing it's
discovering is importable, and this module must not depend on
``compton_guide``/``xigma_i`` either way.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
# .../MC-Kost -> .../XIGMA (sibling dir) -- kascade.py has no src/ layout,
# so this is a one-level walk, not two like xigma_i's.
_XIGMA_ROOT = _THIS_DIR.parent

_COMPTON_SUITE_MARKER = "src/compton_suite/constants.py"
_ENV_VAR = "KASCADE_COMPTON_SUITE_SRC"


def _find_siblings(root: Path, marker: str) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        entry for entry in root.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
        and (entry / marker).exists()
    )


def setup_paths() -> None:
    """Insert ``compton_suite``'s ``src/`` directory into ``sys.path`` if
    it isn't already importable. Safe to call more than once.

    Resolution order, highest priority first:
      1. ``KASCADE_COMPTON_SUITE_SRC`` env var, if set.
      2. Autodiscovery: the (alphabetically first, if several) sibling of
         this repo containing ``src/compton_suite/constants.py``.

    Raises ``ImportError`` if neither resolves -- compton_suite is
    load-bearing here (this module's own physical constants come from it),
    not an optional extra.
    """
    override = os.environ.get(_ENV_VAR)
    if override:
        suite_src = Path(override)
    else:
        matches = _find_siblings(_XIGMA_ROOT, _COMPTON_SUITE_MARKER)
        if not matches:
            raise ImportError(
                f"kascade._bootstrap: could not find compton_suite under "
                f"{_XIGMA_ROOT} (looked for a sibling directory containing "
                f"{_COMPTON_SUITE_MARKER!r}). Set {_ENV_VAR} to its src/ "
                f"directory if it's checked out elsewhere."
            )
        if len(matches) > 1:
            print(
                f"kascade._bootstrap: multiple candidates found under "
                f"{_XIGMA_ROOT} (all contain {_COMPTON_SUITE_MARKER!r}): "
                f"{[str(m) for m in matches]}; using {matches[0]} -- set "
                f"{_ENV_VAR} to pin a specific one.",
                file=sys.stderr,
            )
        suite_src = matches[0] / "src"

    suite_src_str = str(suite_src)
    if suite_src.is_dir() and suite_src_str not in sys.path:
        sys.path.insert(0, suite_src_str)
