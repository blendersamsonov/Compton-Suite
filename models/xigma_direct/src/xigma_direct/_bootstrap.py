"""sys.path wiring so ``xigma_direct`` can import ``compton_io`` and
``xigma_i`` without either being pip-installed.

Same content-based sibling-directory autodiscovery pattern used throughout
this project (see ``compton_guide.bootstrap``'s module docstring for the
full rationale). ``xigma_i`` is load-bearing here (this repo reuses its
``particles``/``reference`` modules directly, by design -- see this
repo's CLAUDE.md, "don't extract a shared home for push_and_sample now").
``compton_io`` is load-bearing too (bunch/laser representations, units).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
# .../XigmaDirect/src/xigma_direct -> .../XigmaDirect -> .../ComptonSuite (sibling dir)
_SUITE_ROOT = _THIS_DIR.parents[2]

_COMPTON_IO_MARKER = "src/compton_io/constants.py"
_XIGMA_MARKER = "src/xigma_i/gui_adapter.py"


def _find_siblings(root: Path, marker: str) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        entry for entry in root.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
        and (entry / marker).exists()
    )


def _discover(root: Path, marker: str, env_var: str) -> Path | None:
    matches = _find_siblings(root, marker)
    if not matches:
        return None
    if len(matches) > 1:
        print(
            f"xigma_direct._bootstrap: multiple candidates found under {root} "
            f"(all contain {marker!r}): {[str(m) for m in matches]}; using "
            f"{matches[0]} -- set {env_var} to pin a specific one.",
            file=sys.stderr,
        )
    return matches[0]


def setup_paths() -> None:
    """Insert ``compton_io``'s and ``xigma_i``'s ``src/`` directories into
    ``sys.path`` if they aren't already importable. Safe to call more than
    once.

    Resolution order for each, highest priority first:
      1. ``XIGMA_DIRECT_COMPTON_IO_SRC`` / ``XIGMA_DIRECT_XIGMA_SRC`` env
         vars, if set.
      2. Autodiscovery: the (alphabetically first, if several) sibling of
         this repo containing the package's marker file.

    Raises ``ImportError`` if either can't be found -- both are
    load-bearing here, not optional extras.
    """
    for marker, env_var, label in (
        (_COMPTON_IO_MARKER, "XIGMA_DIRECT_COMPTON_IO_SRC", "compton_io"),
        (_XIGMA_MARKER, "XIGMA_DIRECT_XIGMA_SRC", "xigma_i"),
    ):
        override = os.environ.get(env_var)
        src = (
            Path(override) if override
            else (lambda d: d / "src" if d is not None else None)(
                _discover(_SUITE_ROOT, marker, env_var)
            )
        )
        if src is None:
            raise ImportError(
                f"xigma_direct._bootstrap: could not find {label} under "
                f"{_SUITE_ROOT} (looked for a sibling directory containing "
                f"{marker!r}). Set {env_var} to its src/ directory if it's "
                f"checked out elsewhere."
            )
        src_str = str(src)
        if src.is_dir() and src_str not in sys.path:
            sys.path.insert(0, src_str)
