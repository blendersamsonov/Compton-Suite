# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`ComptonSuite` is a single git repository for an inverse-Compton-scattering
physics simulation + GUI suite. It used to be six independent git repos
(one per component); they were merged here via `git subtree`, so each
component's full commit history is preserved under its new path (use
`git log --full-history -- <path>` to see pre-merge history, since the
path changed at the merge). There is no more per-component `pyproject.toml`
autodiscovery-by-sys.path trick — every component below is a real,
pip-installable package; see "Dev install" below.

## Layout

| Directory | Python package | Role |
|-----------|-----------------|------|
| `IO/` | `compton_io` | Shared constants/units/parameter-convention framework, electron-bunch (`bunch.py`) and laser-pulse (`laser.py`) representations, output/observable dataclasses (`photons.py`), external I/O (`io_formats/`). Depended on by everything else; depends on nothing in this repo. |
| `GUIde/` | `compton_guide` | Tkinter desktop GUI, model-agnostic via the `ModelAdapter` protocol (`compton_guide/model_api.py`). |
| `models/kaskade/` | flat module `kascade` | CPU Monte Carlo physics engine (sequential multi-photon event generator), SI units, always available. |
| `models/xigma/` | `xigma_i` | GPU (CuPy, numba CPU fallback) tabulated-overlap-table physics engine, CGS units. Greyed out in the GUI if cupy/CUDA isn't usable. |
| `models/xigma_direct/` | `xigma_direct` | Brute-force per-macroparticle resonance-binning model, reuses `xigma_i`'s Stage 0 physics directly. |
| `models/analytical/` | flat modules `analytical`/`analytical_adapter` | Fast, closed-form yield/spectrum/width estimates; always-on GUI preview alongside whichever other model is selected. |
| `src/compton_suite/` | `compton_suite` | Thin umbrella package: `discover_models()` re-export and the `comptonsuite-gui` console-script entry point. |
| `validation/` | (not a package) | Cross-model validation suite — shared `Scenario`s, per-model runners, tiered comparisons (`run_cross_validation.py`), plotting (`visualize.py`), a commit-hash-keyed result cache (`cache.py`). |
| `tests/` | (not a package) | Root-level tests (currently `test_analytical.py`). |

All four physics engines (`kascade`, `xigma_i`, `xigma_direct`, `analytical`)
live under `models/` — they're the pluggable, `ModelAdapter`-shaped pieces;
`IO/` (shared framework) and `GUIde/` (GUI) are not models and stay at the
top level.

## Architecture

`GUIde` (`compton_guide`) plugs in physics engines through a `ModelAdapter`
protocol (`compton_guide/model_api.py`) instead of hardcoded imports, so the
GUI and each physics engine don't depend on each other's internals. Every
model's `run()` requires a pre-sampled `electrons: MacroBunch` (keyword-only)
— sampling the electron bunch is `compton_io`'s job (`compton_io.bunch`),
not any individual model's; no model has its own internal bunch sampler.

- **`kascade`** (`models/kaskade/kascade.py`) — sequential multi-photon
  inverse-Compton MC event generator, SI units, CPU-only (numpy). Adapter:
  `GUIde/src/compton_guide/adapters/kascade_adapter.py`.
- **`xigma-i`** (`models/xigma/src/xigma_i`) — tabulated-overlap-table
  pipeline (particle push → 4D deposition → spectrum kernel), CGS units,
  GPU (CuPy/`cupyx.jit`) with a numba CPU fallback. Adapter:
  `models/xigma/src/xigma_i/gui_adapter.py`.
- **`xigma-i-direct`** (`models/xigma_direct/src/xigma_direct`) —
  brute-force per-macroparticle binning, no table/kernel; reuses `xigma_i`'s
  Stage 0 (`particles.bunch_from_macrobunch`/`push_and_sample`) as a library
  dependency. Adapter: `models/xigma_direct/src/xigma_direct/gui_adapter.py`.
- **`analytical`** (`models/analytical/`) — closed-form yield/spectrum/width
  estimates, fast enough that the GUI runs it automatically alongside
  whichever other model is selected. Adapter: `analytical_adapter.py`.

**`compton_io`** (`IO/`) is the shared-nothing dependency underneath all
four models and the GUI: physical constants (`constants.py`, derived from a
pint registry rather than hand-typed literals), the pint unit registry
(`units.py`), a parameter-semantics/convention framework
(`enums.py`/`quantities.py`/`canonical.py`/`schema.py`/`validation.py`) so
"a FWHM in µm" or "a duration in ps" gets converted the same way
everywhere, electron-bunch/laser-pulse representations (`bunch.py`/
`laser.py`), and shared output dataclasses (`photons.py`).

## Dev install

Every component keeps its own `pyproject.toml` (this is a monorepo holding
several installable packages, not one big package) — install them all
editable in one go:

```bash
pip install -e ./IO -e ./GUIde -e ./models/kaskade -e ./models/xigma \
            -e ./models/xigma_direct -e ./models/analytical -e .
```

On this dev machine, system Python lacks pip/cupy/matplotlib; GPU-dependent
work (`xigma-i`, `xigma-i-direct`, GUI runs with `xigma-i` enabled) needs the
`core` conda env:

```bash
conda run -n core --no-capture-output pip install -e ./IO -e ./GUIde \
    -e ./models/kaskade -e ./models/xigma -e ./models/xigma_direct \
    -e ./models/analytical -e .
conda run -n core --no-capture-output python3 <script>
```

(`--no-capture-output` is required — plain `conda run` silently swallows
stdout.)

## Cross-repo gotchas (still apply post-merge)

- **Never `isinstance()` against both sides of a GUI/engine data
  boundary.** `xigma_i/gui_adapter.py` and `xigma_direct/gui_adapter.py`
  deliberately define their own structurally-identical-but-not-the-same
  local dataclasses (`BinnedSpectrum`, etc.) instead of importing
  `compton_guide.model_api`'s, so the physics packages don't have to depend
  on the GUI package. Code that does `isinstance(x, A): ... elif
  isinstance(x, B): ...` against both silently breaks for whichever side
  wasn't imported from. Use duck typing instead (e.g. `hasattr(x, "weight")`
  vs `hasattr(x, "dNdE_per_eV")`) — this exact bug already hit
  `validate_results()` and several GUI render methods once.
- **Units differ per engine.** `kascade`/`analytical` are SI (m, s, J);
  `xigma_i`/`xigma_direct` are CGS (cm, erg), normalized internally to the
  laser wavenumber. Each adapter converts at its own boundary — don't
  assume a value crossing into the GUI is in a particular unit system
  without checking which engine produced it.
- **Physical constants have one source of truth**: `compton_io.constants`
  (`IO/`). Every model/GUI package re-exports it rather than hand-copying
  literals; a hand-typed physical constant anywhere else is a regression,
  not a pattern to follow.
- **No model-local physics-parameter configs.** Electron-beam and laser
  parameters come from `compton_io.bunch`/`compton_io.laser` (see
  `beam_from_shared_fields`/`laser_from_shared_fields`); a model's own
  `Config` should only carry parameters with no shared cross-model meaning
  (grid/step/bin counts, chunk sizes, and similar numerics).

## Commands per component

Each component's own `CLAUDE.md` has the full picture (dev conda env
details, GPU requirements, etc.); short version:

```bash
# GUIde -- run the GUI (needs numpy, matplotlib, tkinter, pint; cupy optional for xigma-i)
python3 GUIde/scripts/run_gui.py

# GUIde -- headless smoke test (discover_models -> params_to_config -> run -> validate_results)
python3 GUIde/scripts/headless_test.py

# IO (compton_io) -- framework self-checks, no GPU/tkinter needed
python3 IO/tests/test_constants.py && python3 IO/tests/test_conversions.py \
    && python3 IO/tests/test_bunch.py && python3 IO/tests/test_laser.py \
    && python3 IO/tests/test_io_formats.py

# Kaskade -- pure library, no CLI (see models/kaskade/CLAUDE.md for a
# minimal run_simulation() snippet)

# Xigma -- pure library, GPU+cupy or numba CPU fallback (see
# models/xigma/CLAUDE.md for a minimal TabulatedEngine snippet)

# Cross-model validation suite (all four models, tiered comparisons)
python3 validation/run_cross_validation.py
```

## Where to look next

- `GUIde/CLAUDE.md` -- GUI layout, `ModelAdapter` contract, adding a new
  observable, the `extra_params()` mechanism, known gaps.
- `IO/CLAUDE.md` -- `compton_io` layout, why it exists, naming.
- `models/kaskade/CLAUDE.md` -- engine internals, `Config`/`Results` fields,
  `.ele` file I/O, units.
- `models/xigma/CLAUDE.md` -- the four-stage pipeline (particle push →
  deposition → spectrum kernel → validation), physics conventions, a long
  list of previously-made mistakes documented as traps (a0
  trajectory-averaging, the `1/(1+a0)` Jacobian, shared-memory aliasing,
  etc.) -- read before touching `models/xigma/src/xigma_i`.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
