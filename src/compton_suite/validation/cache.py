"""Commit-hash-keyed result cache for the cross-model validation suite.

Motivation: re-running every model on every scenario every time you touch
any file is wasteful -- if nothing has changed since a (model, scenario)
pair was last computed, its cached result is still correct. This module
lets ``validation/runners.py`` skip recomputing a model's result for a
scenario when the repo's commit state hasn't changed since it was last
computed, and transparently recompute (and re-cache) it otherwise.

Cache key = (model name, a content fingerprint of the scenario's actual
beam/pulse parameter VALUES, and the current HEAD commit hash of this
repo). The scenario fingerprint hashes the ``Scenario`` dataclass's
actual field values (via ``dataclasses.asdict`` -> stable JSON -> sha256),
not its ``.name``. Editing ``BASELINE``'s numbers in place without
renaming it must not silently reuse a stale cache entry for the old
numbers.

Dirty working tree: if this repo has uncommitted changes
(``git status --porcelain`` non-empty), the cache is neither read from
nor written to -- a commit hash is only a meaningful fingerprint of
*committed* code; silently caching against (or serving a stale hit that
ignores) uncommitted changes would make cache hits actively misleading
rather than just occasionally wasteful.

Storage: ``validation/.cache/<key>.pkl`` (the full result object, via
pickle) + a ``validation/.cache/<key>.json`` sidecar (model name,
scenario name, commit hash, timestamp -- human-inspectable without
unpickling, and what ``list_cache_entries()`` reads). The whole directory
is gitignored -- regenerate, don't commit. Pickling the result object
as-is means any GPU-resident (cupy) arrays a result still carries (e.g.
xigma-i/xigma-i-direct's private on-demand-recompute caches when run with
``device='gpu'``) need the same cupy+CUDA environment to unpickle as to
pickle -- fine for this single-machine dev cache, not intended to be
portable across machines.

Not yet wired into ``runners.py``'s ``run_kascade``/``run_xigma``/
``run_xigma_direct``/``run_analytical`` -- wiring lands as a follow-up,
via the ``get_or_compute`` entry point below.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_head_hash(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                              capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def _git_is_dirty(repo_root: Path) -> bool:
    """True if uncommitted changes exist, OR if cleanliness can't be
    determined at all (missing git, not a repo, ...) -- fail safe towards
    "don't trust the cache", not towards "trust it anyway"."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root,
                              capture_output=True, text=True, check=True)
        return bool(out.stdout.strip())
    except Exception:
        return True


def _scenario_fingerprint(scenario: Any) -> str:
    payload = asdict(scenario)
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def repo_hash() -> str | None:
    """Current HEAD commit hash of this repo, or None if it's dirty/
    unhashable (see module docstring)."""
    if _git_is_dirty(REPO_ROOT):
        return None
    return _git_head_hash(REPO_ROOT)


def cache_key(model_name: str, scenario: Any) -> str | None:
    """Stable cache key for (model_name, scenario) under the CURRENT
    commit state of this repo, or None if that state can't be trusted
    right now (a dirty tree -- see repo_hash)."""
    h = repo_hash()
    if h is None:
        return None
    payload = {
        "model": model_name,
        "scenario_fp": _scenario_fingerprint(scenario),
        "repo_hash": h,
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _paths(key: str) -> tuple[Path, Path]:
    return CACHE_DIR / f"{key}.pkl", CACHE_DIR / f"{key}.json"


def load(key: str) -> Any | None:
    pkl_path, _ = _paths(key)
    if not pkl_path.exists():
        return None
    try:
        with pkl_path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        # A corrupt/incompatible cache entry (e.g. pickled under a
        # different cupy/numpy version) should degrade to "recompute",
        # never crash the suite.
        return None


def save(key: str, model_name: str, scenario: Any, result: Any) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    pkl_path, json_path = _paths(key)
    with pkl_path.open("wb") as f:
        pickle.dump(result, f)
    sidecar = {
        "model": model_name,
        "scenario": getattr(scenario, "name", repr(scenario)),
        "repo_hash": repo_hash(),
        "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with json_path.open("w") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)


def get_or_compute(model_name: str, scenario: Any,
                    compute_fn: Callable[[], Any]) -> tuple[Any, bool]:
    """Return (result, was_cached). Looks up a cache entry for
    (model_name, scenario) under the current dependency commit state;
    on a miss (or an untrustworthy/dirty dependency state), calls
    compute_fn() and -- if the dependency state was trustworthy -- caches
    the result for next time.

    This is the intended integration point for runners.py's run_kascade/
    run_xigma/run_xigma_direct/run_analytical: wrap each model's existing
    compute body in a zero-arg closure and call this instead of calling
    the model directly. Not wired in yet -- see module docstring.
    """
    key = cache_key(model_name, scenario)
    if key is not None:
        cached = load(key)
        if cached is not None:
            return cached, True
    result = compute_fn()
    if key is not None:
        save(key, model_name, scenario, result)
    return result, False


def list_cache_entries() -> list[dict]:
    """Sidecar metadata for every entry currently in the cache, newest
    first -- for manual inspection, not used by get_or_compute itself."""
    if not CACHE_DIR.is_dir():
        return []
    entries = []
    for json_path in CACHE_DIR.glob("*.json"):
        try:
            with json_path.open() as f:
                entries.append(json.load(f))
        except Exception:
            continue
    entries.sort(key=lambda e: e.get("cached_at", ""), reverse=True)
    return entries


def clear_cache() -> int:
    """Delete every cache entry. Returns the number of entries removed."""
    if not CACHE_DIR.is_dir():
        return 0
    n = 0
    for path in CACHE_DIR.glob("*"):
        path.unlink()
        n += 1
    return n // 2  # each entry is a .pkl + .json pair


if __name__ == "__main__":
    print(f"cache dir: {CACHE_DIR}")
    dirty = _git_is_dirty(REPO_ROOT)
    h = _git_head_hash(REPO_ROOT)
    status = "DIRTY" if dirty else "clean"
    print(f"repo root: {REPO_ROOT}  HEAD={h}  [{status}]")
    print()
    entries = list_cache_entries()
    print(f"{len(entries)} cached entries:")
    for e in entries:
        print(f"  {e.get('cached_at')}  {e.get('model'):14s} {e.get('scenario')}")
