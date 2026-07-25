"""sys.path wiring so ``xigma_i`` can import ``compton_suite`` without it
being pip-installed.

``compton_suite`` lives in a sibling checkout whose directory name and
exact path aren't a stable package-manager path (same situation
``compton_guide.bootstrap`` already solves for ``xigma_i``/``kascade`` --
see that module's docstring). This is a smaller, single-target version of
the same content-based-autodiscovery pattern: physically duplicated here
rather than imported from ``compton_guide``, because the discovery code
has to run *before* the thing it's discovering is importable, and this
package must not depend on ``compton_guide`` either way (see
``gui_adapter.py``'s decoupling notes).

Unlike ``compton_guide.bootstrap``'s lazy, failure-tolerant discovery of
optional physics engines (a missing engine just greys out a GUI menu
entry), ``compton_suite`` is load-bearing for ``xigma_i`` itself --
``config.py``'s physical constants and ``params/``'s parameter-semantics
framework both come from it. ``setup_paths()`` here raises
``ImportError`` with a clear message if ``compton_suite`` can't be found
and no override is given, rather than silently continuing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
# .../xigma_i-repo/src/xigma_i -> .../xigma_i-repo -> .../XIGMA (sibling dir)
_XIGMA_ROOT = _THIS_DIR.parents[2]

_COMPTON_SUITE_MARKER = "src/compton_suite/constants.py"
_ENV_VAR = "XIGMA_COMPTON_SUITE_SRC"


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
    it isn't already importable. Safe to call more than once (idempotent
    -- subsequent calls are cheap no-ops once the path is already present).

    Resolution order, highest priority first:
      1. ``XIGMA_COMPTON_SUITE_SRC`` env var, if set -- used verbatim, no
         autodiscovery, for checkouts autodiscovery can't find (e.g. this
         package itself running from inside a git worktree, where the
         sibling-directory assumption below doesn't hold).
      2. Autodiscovery: the (alphabetically first, if several) sibling of
         this repo containing ``src/compton_suite/constants.py``.

    Raises ``ImportError`` if neither resolves -- compton_suite is
    load-bearing for this package, not an optional extra.
    """
    override = os.environ.get(_ENV_VAR)
    if override:
        suite_src = Path(override)
    else:
        matches = _find_siblings(_XIGMA_ROOT, _COMPTON_SUITE_MARKER)
        if not matches:
            raise ImportError(
                f"xigma_i._bootstrap: could not find compton_suite under "
                f"{_XIGMA_ROOT} (looked for a sibling directory containing "
                f"{_COMPTON_SUITE_MARKER!r}). Set {_ENV_VAR} to its src/ "
                f"directory if it's checked out elsewhere."
            )
        if len(matches) > 1:
            print(
                f"xigma_i._bootstrap: multiple candidates found under "
                f"{_XIGMA_ROOT} (all contain {_COMPTON_SUITE_MARKER!r}): "
                f"{[str(m) for m in matches]}; using {matches[0]} -- set "
                f"{_ENV_VAR} to pin a specific one.",
                file=sys.stderr,
            )
        suite_src = matches[0] / "src"

    suite_src_str = str(suite_src)
    if suite_src.is_dir() and suite_src_str not in sys.path:
        sys.path.insert(0, suite_src_str)
