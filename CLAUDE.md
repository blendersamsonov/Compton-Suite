# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`ComptonSuite` is a parent directory holding four **independently-versioned
git repositories** for an inverse-Compton-scattering physics simulation +
GUI suite — it is not itself a git repo and has no root `pyproject.toml`.
Each subdirectory has its own `.git`, its own `pyproject.toml`, and its own
`CLAUDE.md` with deep implementation detail; read the relevant one before
working in that subtree. This file only covers what spans all four.

## Directory → repo/package name mapping

The folder names on disk do **not** match the repo or Python package names —
easy to get wrong when writing bootstrap/import code or talking about "the
X repo":

| Directory | Repo name        | Python package  | Role |
|-----------|-------------------|-----------------|------|
| `GUIde/`  | `Compton-GUIde`   | `compton_guide` | Tkinter desktop GUI, model-agnostic |
| `IO/`     | `compton-suite`   | `compton_suite` | Shared constants/units/parameter-convention framework |
| `Kaskade/`| `KASCADE` (was `MC-Kost`/`dfe5`) | flat module `kascade.py` | CPU Monte Carlo physics engine (event generator) |
| `Xigma/`  | `xigma_i`         | `xigma_i`       | GPU (CuPy, numba CPU fallback) physics engine (tabulated/semi-analytic) |

Note in particular: the `IO/` directory *is* the `compton-suite` repo
(package `compton_suite`) — not to be confused with `compton_guide`
(`GUIde/`), a one-word difference that's easy to typo or autocomplete-wrong
across repos.

## Architecture

`GUIde` (`compton_guide`) is a model-agnostic GUI that plugs in physics
engines through a `ModelAdapter` protocol (`compton_guide/model_api.py`)
instead of hardcoded imports, so the GUI and each physics engine don't
depend on each other's internals:

- **`kascade`** (`Kaskade/kascade.py`) — sequential multi-photon inverse-Compton
  MC event generator, SI units, CPU-only (numpy), always available. Adapter:
  `GUIde/src/compton_guide/adapters/kascade_adapter.py` (lives in the GUI
  repo, wraps `kascade.py` with zero changes to that file).
- **`xigma-i`** (`Xigma/src/xigma_i`) — tabulated-overlap-table pipeline
  (particle push → 4D deposition → spectrum kernel), CGS units, GPU
  (CuPy/`cupyx.jit`) with a numba CPU fallback, greyed out in the GUI if
  cupy/CUDA isn't usable. Adapter: `Xigma/src/xigma_i/gui_adapter.py`
  (lives in the *physics* repo, not the GUI repo — already-completed
  integration work at the time `GUIde` was split out).

**`compton_suite`** (`IO/`) is the shared-nothing dependency underneath all
three others: physical constants (`constants.py`, derived from a pint
registry rather than hand-typed literals), the pint unit registry
(`units.py`), and a parameter-semantics/convention framework
(`enums.py`/`quantities.py`/`canonical.py`/`schema.py`/`validation.py`) so
"a FWHM in µm" or "a duration in ps" gets converted the same way everywhere.
It exists because each consumer used to hand-maintain its own copy of both
constants (with a real numeric disagreement between them) and the
convention framework (structurally identical but not the same Python
classes across copies, which broke `isinstance`/spec-validation checks).

**No pip installs between these repos.** Every cross-repo import goes
through a small, physically-duplicated-per-consumer `sys.path` bootstrap
(`GUIde/src/compton_guide/bootstrap.py`, `Xigma/src/xigma_i/_bootstrap.py`,
`Kaskade/_bootstrap.py`) that scans **sibling directories** for a marker
file (`kascade.py`, `src/xigma_i/gui_adapter.py`,
`src/compton_suite/constants.py`) rather than assuming a fixed directory
name — this is why the on-disk names (`GUIde`/`IO`/`Kaskade`/`Xigma`) are
free to differ from the repo names. Override with the env var each
bootstrap module defines (`COMPTON_GUIDE_KASCADE_PATH`,
`COMPTON_GUIDE_XIGMA_SRC`, `COMPTON_GUIDE_COMPTON_SUITE_SRC`,
`XIGMA_COMPTON_SUITE_SRC`, `KASCADE_COMPTON_SUITE_SRC`) if a checkout lives
outside this parent directory, or if autodiscovery finds more than one
candidate.

## Cross-repo gotchas

- **Never `isinstance()` against both sides of a GUI/engine data
  boundary.** `xigma_i/gui_adapter.py` deliberately defines its own
  structurally-identical-but-not-the-same local dataclasses
  (`BinnedSpectrum`, etc.) instead of importing `compton_guide.model_api`'s,
  so the physics repo doesn't have to depend on the GUI repo. Code that does
  `isinstance(x, A): ... elif isinstance(x, B): ...` against both silently
  breaks for whichever side wasn't imported from. Use duck typing instead
  (e.g. `hasattr(x, "weight")` vs `hasattr(x, "dNdE_per_eV")`) — this exact
  bug already hit `validate_results()` and several GUI render methods once.
- **Units differ per engine.** `kascade` is SI (m, s, J); `xigma_i` is CGS
  (cm, erg), normalized internally to the laser wavenumber. Each adapter
  converts at its own boundary — don't assume a value crossing into the GUI
  is in a particular unit system without checking which engine produced it.
- **Physical constants have one source of truth**: `compton_suite.constants`
  (`IO/`). The other three repos re-export it rather than hand-copying
  literals; if you find a hand-typed constant in `Kaskade`, `GUIde`, or
  `Xigma`, that's a regression, not a pattern to follow.
- **Each subdirectory is its own git repo** — there is no root `.git`, so
  `git status`/`git log`/commits must be run from inside the specific
  subdirectory (`GUIde/`, `IO/`, `Kaskade/`, `Xigma/`), and a change that
  spans repos needs a commit in each. `IO/` (`compton-suite`) is currently
  local-only with no GitHub remote.

## Commands per component

Each repo's own `CLAUDE.md` has the full picture (dev conda env details,
GPU requirements, etc.); short version:

```bash
# GUIde — run the GUI (needs numpy, matplotlib, tkinter, pint; cupy optional for xigma-i)
cd GUIde && python3 scripts/run_gui.py

# GUIde — headless smoke test (discover_models -> params_to_config -> run -> validate_results)
cd GUIde && python3 scripts/headless_test.py

# IO (compton_suite) — framework self-checks, no GPU/tkinter needed
cd IO && python3 tests/test_constants.py && python3 tests/test_conversions.py

# Kaskade — standalone CLI run (independent of the GUI)
cd Kaskade && python3 kascade.py -c kascade_config.toml

# Xigma — validation/comparison scripts (needs GPU+cupy, or numba CPU fallback)
cd Xigma && python3 compare_direct_vs_table.py --grid-integrate
```

On this dev machine, system Python lacks pip/cupy/matplotlib; GPU-dependent
work (`Xigma`, `xigma-i`-enabled GUI runs) needs the `core` conda env:

```bash
conda run -n core --no-capture-output python3 <script>
```

(`--no-capture-output` is required — plain `conda run` silently swallows
stdout.)

## Where to look next

- `GUIde/CLAUDE.md` — GUI layout, `ModelAdapter` contract, adding a new
  observable, the `extra_params()` mechanism, known gaps.
- `IO/CLAUDE.md` — `compton_suite` layout, why it exists, naming.
- `Kaskade/CLAUDE.md` — engine internals, `Config`/`Results` fields,
  `.ele` file I/O, units.
- `Xigma/CLAUDE.md` — the four-stage pipeline (particle push → deposition →
  spectrum kernel → validation), physics conventions, a long list of
  previously-made mistakes documented as traps (a0 trajectory-averaging,
  the `1/(1+a0)` Jacobian, shared-memory aliasing, etc.) — read before
  touching `Xigma/src/xigma_i`.
