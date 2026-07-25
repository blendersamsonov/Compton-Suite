"""sys.path wiring so ``compton_guide`` can import ``kascade``, ``xigma_i``,
``xigma_direct``, ``compton_io``, and the root ``compton_suite``
meta-package (for its analytical model) without any of them being
pip-installed.

These packages currently live in locations that aren't stable
package-manager paths: ``kascade`` is a loose script directory
(``MC-Kost/``, sibling of this project) with no packaging at all,
``xigma_i`` lives in a sibling checkout whose directory name and exact
path are not stable across machines -- e.g. it may be a plain repo clone
on one machine and a git-worktree checkout on another, under whatever
name that machine's checkout happens to have (this has already changed
once, when ``xigma_i``'s GUI adapter moved out of a git worktree and into
that repo's ``main`` branch) -- and ``compton_io`` (shared physical
constants, pint registry, parameter-convention framework, electron-bunch/
laser-pulse representations) is a newer sibling repo with the same
instability risk despite being deliberately created rather than
historically drifted. A hardcoded ``pyproject.toml`` path-dependency, or
even a hardcoded sibling directory *name*, would break on the next machine
or the next reorganization -- so this is a runtime ``sys.path`` bootstrap
that autodiscovers all four locations by content (looking for a marker
file inside each candidate directory) rather than by name, with all four
still overridable via environment variables for anyone whose checkout
layout doesn't match autodiscovery at all.

The root ``compton_suite`` meta-package is the one exception to "sibling
directory": its source sits at this project's *parent* directory (it is
the repo that contains ``GUIde``/``IO``/``Kaskade``/``Xigma`` as
subdirectories, not a sibling of them), so its discovery step also checks
one level up, not just siblings.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
# .../compton-gui/src/compton_guide -> .../compton-gui -> .../ComptonSuite (parent/sibling-root dir)
_SUITE_ROOT = _THIS_DIR.parents[2]

# Marker files used to recognize each package, relative to the candidate
# directory itself.
_KASCADE_MARKER = "kascade.py"
_XIGMA_MARKER = "src/xigma_i/gui_adapter.py"
_XIGMA_DIRECT_MARKER = "src/xigma_direct/gui_adapter.py"
_COMPTON_IO_MARKER = "src/compton_io/constants.py"
_COMPTON_SUITE_MARKER = "src/compton_suite/analytical_adapter.py"


def _find_siblings(root: Path, marker: str) -> list[Path]:
    """Sibling directories of this project (children of ``root``) that
    contain ``marker``, sorted by name for a deterministic pick among
    ties."""
    if not root.is_dir():
        return []
    return sorted(
        entry for entry in root.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
        and (entry / marker).exists()
    )


def _discover(root: Path, marker: str, env_var: str) -> Path | None:
    """A candidate sibling directory containing ``marker``, or ``None``.
    Warns (but does not fail) if more than one match is found -- pin the
    right one via ``env_var`` in that case."""
    matches = _find_siblings(root, marker)
    if not matches:
        return None
    if len(matches) > 1:
        print(
            f"compton_guide.bootstrap: multiple candidates found under {root} "
            f"(all contain {marker!r}): {[str(m) for m in matches]}; using "
            f"{matches[0]} -- set {env_var} to pin a specific one.",
            file=sys.stderr,
        )
    return matches[0]


def _discover_root_package(root: Path, marker: str, env_var: str) -> Path | None:
    """Like ``_discover``, but for the root ``compton_suite`` meta-package,
    whose source is not a sibling of this project but sits at ``root``
    itself (this project's parent directory contains it, rather than
    being contained alongside it) -- checks ``root`` directly (does its
    own marker check, not ``root``'s children) before falling back to a
    sibling-style search, in case the layout ever changes."""
    if (root / marker).exists():
        return root
    return _discover(root, marker, env_var)


def setup_paths() -> None:
    """Insert the kascade, xigma_i, compton_io, and root compton_suite
    source directories into ``sys.path`` if they aren't already
    importable. Safe to call more than once.

    Resolution order for each, highest priority first:
      1. The relevant ``COMPTON_GUIDE_*`` environment variable, if set --
         used verbatim, no autodiscovery, for checkouts autodiscovery
         can't find.
      2. Autodiscovery: the (alphabetically first, if several) sibling of
         this project containing the package's marker file (or, for the
         root ``compton_suite`` package only, this project's own parent
         directory).

    Never raises -- kascade/xigma_i/compton_suite are optional (a missing
    one just greys out a GUI menu entry / disables the analytical preview
    panel), so a warning to stderr is enough. ``compton_io`` is *not*
    optional for anything that actually imports it
    (``physics_params``/``physics_constants``), but the natural
    ``ImportError`` those imports raise on their own if ``compton_io``
    isn't on ``sys.path`` already does the "fail loudly" job -- this
    function's contract (never raises) stays uniform across all four
    targets.
    """
    kascade_override = os.environ.get("COMPTON_GUIDE_KASCADE_PATH")
    kascade_path = (
        Path(kascade_override) if kascade_override
        else _discover(_SUITE_ROOT, _KASCADE_MARKER, "COMPTON_GUIDE_KASCADE_PATH")
    )

    xigma_override = os.environ.get("COMPTON_GUIDE_XIGMA_SRC")
    xigma_src = (
        Path(xigma_override) if xigma_override
        else (lambda d: d / "src" if d is not None else None)(
            _discover(_SUITE_ROOT, _XIGMA_MARKER, "COMPTON_GUIDE_XIGMA_SRC")
        )
    )

    xigma_direct_override = os.environ.get("COMPTON_GUIDE_XIGMA_DIRECT_SRC")
    xigma_direct_src = (
        Path(xigma_direct_override) if xigma_direct_override
        else (lambda d: d / "src" if d is not None else None)(
            _discover(_SUITE_ROOT, _XIGMA_DIRECT_MARKER, "COMPTON_GUIDE_XIGMA_DIRECT_SRC")
        )
    )

    compton_io_override = os.environ.get("COMPTON_GUIDE_COMPTON_IO_SRC")
    compton_io_src = (
        Path(compton_io_override) if compton_io_override
        else (lambda d: d / "src" if d is not None else None)(
            _discover(_SUITE_ROOT, _COMPTON_IO_MARKER, "COMPTON_GUIDE_COMPTON_IO_SRC")
        )
    )

    compton_suite_override = os.environ.get("COMPTON_GUIDE_COMPTON_SUITE_SRC")
    compton_suite_src = (
        Path(compton_suite_override) if compton_suite_override
        else (lambda d: d / "src" if d is not None else None)(
            _discover_root_package(
                _SUITE_ROOT, _COMPTON_SUITE_MARKER, "COMPTON_GUIDE_COMPTON_SUITE_SRC")
        )
    )

    for p, label, env_var in (
        (kascade_path, "kascade", "COMPTON_GUIDE_KASCADE_PATH"),
        (xigma_src, "xigma_i", "COMPTON_GUIDE_XIGMA_SRC"),
        (xigma_direct_src, "xigma_direct", "COMPTON_GUIDE_XIGMA_DIRECT_SRC"),
        (compton_io_src, "compton_io", "COMPTON_GUIDE_COMPTON_IO_SRC"),
        (compton_suite_src, "compton_suite", "COMPTON_GUIDE_COMPTON_SUITE_SRC"),
    ):
        if p is None:
            print(
                f"compton_guide.bootstrap: could not find {label} under "
                f"{_SUITE_ROOT} -- set {env_var} to its location if it's "
                f"installed elsewhere.",
                file=sys.stderr,
            )
            continue
        p_str = str(p)
        if p.is_dir() and p_str not in sys.path:
            sys.path.insert(0, p_str)
