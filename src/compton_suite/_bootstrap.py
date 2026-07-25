"""sys.path wiring so ``compton_suite`` can import ``compton_io`` and
``compton_guide`` without either being pip-installed.

Same content-based marker-file autodiscovery pattern used throughout this
project (``compton_guide.bootstrap``, ``kascade._bootstrap``,
``xigma_i._bootstrap``), just scanning this project's own *children*
rather than its siblings -- ``compton_suite`` is the root repo that
contains ``GUIde``/``IO``/``Kaskade``/``Xigma`` as subdirectories, not a
sibling of them.

Only ``compton_io`` and ``compton_guide`` need discovering directly here:
``compton_io`` is what ``analytical.py``/``analytical_adapter.py`` need,
and ``compton_guide`` is what ``models.py``/``gui.py`` need -- once
``compton_guide`` itself is importable, its own ``bootstrap.setup_paths()``
(called internally by ``compton_guide.models.discover_models()``) finds
``kascade``/``xigma_i``/``compton_suite`` (itself) in turn, so nothing
here needs to reach past ``compton_guide`` to those directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
# .../ComptonSuite/src/compton_suite -> .../ComptonSuite (this repo's own root)
_SUITE_ROOT = _THIS_DIR.parents[1]

_COMPTON_IO_MARKER = "src/compton_io/constants.py"
_COMPTON_GUIDE_MARKER = "src/compton_guide/models.py"


def _find_children(root: Path, marker: str) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        entry for entry in root.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
        and (entry / marker).exists()
    )


def _discover(root: Path, marker: str, env_var: str) -> Path | None:
    matches = _find_children(root, marker)
    if not matches:
        return None
    if len(matches) > 1:
        print(
            f"compton_suite._bootstrap: multiple candidates found under {root} "
            f"(all contain {marker!r}): {[str(m) for m in matches]}; using "
            f"{matches[0]} -- set {env_var} to pin a specific one.",
            file=sys.stderr,
        )
    return matches[0]


def setup_paths() -> None:
    """Insert ``compton_io``'s and ``compton_guide``'s ``src/`` directories
    into ``sys.path`` if they aren't already importable. Safe to call more
    than once.

    Resolution order for each, highest priority first:
      1. ``COMPTON_SUITE_COMPTON_IO_SRC`` / ``COMPTON_SUITE_COMPTON_GUIDE_SRC``
         env vars, if set.
      2. Autodiscovery: the (alphabetically first, if several) child of
         this repo containing the package's marker file.

    Raises ``ImportError`` if ``compton_io`` can't be found (load-bearing
    for ``analytical.py``). ``compton_guide`` is treated as optional here
    (only ``models.py``/``gui.py`` need it) -- a warning to stderr is
    enough, matching how ``compton_guide.bootstrap`` itself treats its own
    optional physics engines.
    """
    io_override = os.environ.get("COMPTON_SUITE_COMPTON_IO_SRC")
    io_src = (
        Path(io_override) if io_override
        else (lambda d: d / "src" if d is not None else None)(
            _discover(_SUITE_ROOT, _COMPTON_IO_MARKER, "COMPTON_SUITE_COMPTON_IO_SRC")
        )
    )
    if io_src is None:
        raise ImportError(
            f"compton_suite._bootstrap: could not find compton_io under "
            f"{_SUITE_ROOT} (looked for a child directory containing "
            f"{_COMPTON_IO_MARKER!r}). Set COMPTON_SUITE_COMPTON_IO_SRC to "
            f"its src/ directory if it's checked out elsewhere."
        )
    io_src_str = str(io_src)
    if io_src.is_dir() and io_src_str not in sys.path:
        sys.path.insert(0, io_src_str)

    guide_override = os.environ.get("COMPTON_SUITE_COMPTON_GUIDE_SRC")
    guide_src = (
        Path(guide_override) if guide_override
        else (lambda d: d / "src" if d is not None else None)(
            _discover(_SUITE_ROOT, _COMPTON_GUIDE_MARKER, "COMPTON_SUITE_COMPTON_GUIDE_SRC")
        )
    )
    if guide_src is None:
        print(
            f"compton_suite._bootstrap: could not find compton_guide under "
            f"{_SUITE_ROOT} -- set COMPTON_SUITE_COMPTON_GUIDE_SRC to its "
            f"location if it's installed elsewhere. models.py/gui.py will "
            f"fail to import until this is resolved.",
            file=sys.stderr,
        )
        return
    guide_src_str = str(guide_src)
    if guide_src.is_dir() and guide_src_str not in sys.path:
        sys.path.insert(0, guide_src_str)
