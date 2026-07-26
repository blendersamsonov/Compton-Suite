# AGENTS.md

This file provides guidance to AI agents (Claude Code, etc.) when working with code in this repository.

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
| `gui/` | `compton_guide` | Tkinter desktop GUI, model-agnostic via the `ModelAdapter` protocol (`compton_guide/model_api.py`). |
| `models/kaskade/` | flat module `kascade` | CPU Monte Carlo physics engine (sequential multi-photon event generator), SI units, always available. |
| `models/xigma/` | `xigma_i` | GPU (CuPy, numba CPU fallback) tabulated-overlap-table physics engine, CGS units. Greyed out in the GUI if cupy/CUDA isn't usable. |
| `models/xigma_direct/` | `xigma_direct` | Brute-force per-macroparticle resonance-binning model, reuses `xigma_i`'s Stage 0 physics directly. |
| `models/analytical/` | flat modules `analytical`/`analytical_adapter` | Fast, closed-form yield/spectrum/width estimates; always-on GUI preview alongside whichever other model is selected. |
| `src/compton_suite/` | `compton_suite` | Thin umbrella package: `discover_models()` re-export and the `comptonsuite-gui` console-script entry point. |
| `validation/` | (not a package) | Cross-model validation suite — shared `Scenario`s, per-model runners, tiered comparisons (`run_cross_validation.py`), plotting (`visualize.py`), a commit-hash-keyed result cache (`cache.py`). |
| `tests/` | (not a package) | Root-level tests (currently `test_analytical.py`). |

All four physics engines (`kascade`, `xigma_i`, `xigma_direct`, `analytical`)
live under `models/` — they're the pluggable, `ModelAdapter`-shaped pieces;
`IO/` (shared framework) and `gui/` (GUI) are not models and stay at the
top level.

## Architecture

`gui` (`compton_guide`) plugs in physics engines through a `ModelAdapter`
protocol (`compton_guide/model_api.py`) instead of hardcoded imports, so the
GUI and each physics engine don't depend on each other's internals. Every
model's `run()` requires a pre-sampled `electrons: MacroBunch` (keyword-only)
— sampling the electron bunch is `compton_io`'s job (`compton_io.bunch`),
not any individual model's; no model has its own internal bunch sampler.

- **`kascade`** (`models/kaskade/kascade.py`) — sequential multi-photon
  inverse-Compton MC event generator, SI units, CPU-only (numpy). Adapter:
  `gui/src/compton_guide/adapters/kascade_adapter.py`.
- **`xigma-i`** (`models/xigma/src/xigma_i`) — tabulated-overlap-table
  pipeline (particle push → 4D deposition → spectrum kernel), CGS units,
  GPU (CuPy/`cupyx.jit`) with a numba CPU fallback. Adapter:
  `models/xigma/src/xigma_i/gui_adapter.py`.
- **`xigma-i-direct`** (`models/xigma_direct/src/xigma_direct`) —
  brute-force per-macroparticle binning, no table/kernel; reuses `xigma_i`'s
  Stage 0 (`particles.push_and_sample`) as a library dependency. Adapter:
  `models/xigma_direct/src/xigma_direct/gui_adapter.py`.
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
pip install -e ./IO -e ./gui -e ./models/kaskade -e ./models/xigma \
            -e ./models/xigma_direct -e ./models/analytical -e .
```

On this dev machine, system Python lacks pip/cupy/matplotlib; GPU-dependent
work (`xigma-i`, `xigma-i-direct`, GUI runs with `xigma-i` enabled) needs the
`core` conda env:

```bash
conda run -n core --no-capture-output pip install -e ./IO -e ./gui \
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
- **No model-local physics-parameter configs, no model-local particle
  sampling, no model-local result contract.** Electron-beam and laser
  parameters come from `compton_io.bunch`/`compton_io.laser` (see
  `beam_from_shared_fields`/`laser_from_shared_fields`); a model's own
  `Config` should only carry parameters with no shared cross-model meaning
  (grid/step/bin counts, chunk sizes, and similar numerics). No model may
  sample its own electrons -- `compton_io.bunch.sample_gaussian_bunch` is
  the only place that happens. A model that needs a derived, unit-converted
  scalar bundle for its own kernels (e.g. `models/xigma`'s CGS/`k0_las`-
  normalised `CollisionParams`) builds it fresh, once, via a pure function
  from `compton_io`'s beam/laser/geometry description -- never a stateful,
  `set_*`-mutated object the model owns across a run. Every model's `run()`
  returns `compton_io.results.CommonResults` directly (leaving unsupported
  fields `None`), not a model-local lookalike class.

## Commands per component

Each component's own `AGENTS.md` has the full picture (dev conda env
details, GPU requirements, etc.); short version:

```bash
# gui -- run the GUI (needs numpy, matplotlib, tkinter, pint; cupy optional for xigma-i)
python3 gui/scripts/run_gui.py

# gui -- headless smoke test (discover_models -> params_to_config -> run -> validate_results)
python3 gui/scripts/headless_test.py

# IO (compton_io) -- framework self-checks, no GPU/tkinter needed
python3 IO/tests/test_constants.py && python3 IO/tests/test_conversions.py \
    && python3 IO/tests/test_bunch.py && python3 IO/tests/test_laser.py \
    && python3 IO/tests/test_io_formats.py

# Kaskade -- pure library, no CLI (see models/kaskade/AGENTS.md for a
# minimal run_simulation() snippet)

# Xigma -- pure library, GPU+cupy or numba CPU fallback (see
# models/xigma/AGENTS.md for a minimal TabulatedEngine snippet)

# Cross-model validation suite (all four models, tiered comparisons)
python3 validation/run_cross_validation.py
```

## Where to look next

- `gui/AGENTS.md` -- GUI layout, `ModelAdapter` contract, adding a new
  observable, the `extra_params()` mechanism, known gaps.
- `IO/AGENTS.md` -- `compton_io` layout, why it exists, naming.
- `models/kaskade/AGENTS.md` -- engine internals, `Config`/`Results` fields,
  `.ele` file I/O, units.
- `models/xigma/AGENTS.md` -- the four-stage pipeline (particle push →
  deposition → spectrum kernel → validation), physics conventions, a long
  list of previously-made mistakes documented as traps (a0
  trajectory-averaging, the `1/(1+a0)` Jacobian, shared-memory aliasing,
  etc.) -- read before touching `models/xigma/src/xigma_i`.

## Roadmap: follow-up work

Not urgent, not blocking anything — pick any one of these independently.
Each has enough pointers to start without re-deriving context.

### 1. Dead-code / unused-config sweep

Run a fresh sweep (grep for unused imports/fields, or a tool like
`vulture`) across `models/*/` and `IO/` — the last full manual pass was
during the 2026-07-26 repo-merge session and wasn't exhaustive; a
follow-up pass (2026-07-26, same day) resolved the two items below.
Resolved:
- `models/xigma/src/xigma_i/gui_adapter.py`'s `Config.emulate_nonlinearity`
  field has been removed (confirmed zero computational effect anywhere in
  `xigma_i`, never set from any caller, only echoed into diagnostics).
  `Config.quantum` is kept as a documented no-op: it's part of the shared
  `ModelAdapter.params_to_config(fields, quantum)` contract that `kascade`
  actually uses, so dropping it would mean changing the shared protocol
  across all four models -- a decision left for item 5 below, not a
  dead-code deletion.
- `models/xigma/src/xigma_i/params/spec.py`'s `XIGMA_SPEC`/
  `XIGMA_DIAGNOSTIC_SPEC` are still not wired into `params_to_config` --
  left as-is (still exercised by `gui/scripts/physics_params_demo.py`).
  Finishing that wiring is real design work tied to item 5, still open.

### 2. Move `CollisionParams`/`build_params` into `compton_io`

Flagged in `models/xigma/src/xigma_i/config.py`'s own module docstring.
**Partially done (2026-07-26):** the electron-side derivation (`beta_x`/
`beta_y`/`sigma_thx`/`sigma_thy`) was arithmetically identical to
`compton_io.bunch.GaussianElectronBeam.beta_star_x_m`/`beta_star_y_m`/
`divergence_x_rad`/`divergence_y_rad` (just needed `* 100` for the two
length-based `beta_x`/`beta_y` fields; `sigma_thx`/`sigma_thy` are
dimensionless angles needing no conversion). Two new module-level
functions, `compton_io.bunch.beta_star_from_sigma_emit`/
`divergence_from_sigma_emit`, now back both `GaussianElectronBeam`'s
properties and every place that used to re-derive the same formula
independently: `xigma_i/config.py::build_params` (now calls
`beam.beta_star_x_m`/`divergence_x_rad` directly instead of recomputing),
`xigma_i/gui_adapter.py`'s own `Config.__post_init__` SI pre-step,
`xigma_direct/gui_adapter.py`'s `DirectConfig.__post_init__`, and
`models/kaskade/kascade.py`'s `Config.__post_init__` -- one formula
instead of four independent copies.

Still open, deliberately untouched: the `a0` formula, which has its own
already-flagged, unresolved ~49% discrepancy against `GaussianParaxialLaser.
a0_focus` (`validation/tier0_wiring.py`'s `check_a0_formula_agreement`,
`FORMULA_TOL`) -- moving *this* piece needs either resolving the physics
discrepancy first (a real investigation, not a refactor) or moving it with
the discrepancy intact and clearly documented as "xigma's own convention,
not yet reconciled with compton_io.laser's" (this is what `config.py`'s
module docstring now says). `beta_ff`/`ellipticity` also stay xigma-only,
no shared-representation analogue (`compton_io.laser`'s own module
docstring explains why they're excluded from `GaussianParaxialLaser`).

### 3. GUI: reconsider "experimental" trust levels/warnings

**Done (2026-07-26).** A fresh run of `validation/run_cross_validation.py`
confirmed kascade/xigma-i/xigma-i-direct agree to <1% on `total_yield`
and to a few percent (weighted-L1) on angle-integrated spectrum shape at
baseline configs, with the a0-formula discrepancy above and the `~2*pi`
angular-spectrum residual (`reference.py`'s module docstring) as the two
remaining concrete open items rather than blanket "experimental" status.
`xigma-i` and `xigma-i-direct` were graduated to
`trust_level="production"` (`display_name` no longer says "experimental"),
and `_TRUST_NOTE` in both `models/xigma/src/xigma_i/gui_adapter.py` and
`models/xigma_direct/src/xigma_direct/gui_adapter.py` was rewritten to
name those two open items explicitly, plus the observed spectrum-shape
degradation (~24% weighted-L1) near `a0_max`. Note: `trust_note` only
renders in the GUI when `trust_level != "production"` (`app.py` ~line 350),
so this text is now dormant there but still worth keeping accurate for
anyone reading the adapter source directly.
`models/xigma/passport.md` (a separate, human-authored formal document)
was deliberately left untouched -- it still self-rates trust level "C";
reconciling it with the graduated GUI trust_level is a call for whoever
owns that document, not something to auto-edit.

### 4. GUI: per-model sample count instead of a misleading global field

`app.py`'s shared "Number of macroelectrons" field (`n_mc`, ~line 534) is
silently ignored by xigma-i/xigma-i-direct, which size Stage 0/1 from
their own `extra_params()` field (`n_particles_01`) instead --
`params_to_config` already raises a warning about this (see `models/xigma/AGENTS.md`'s "GUI integration"), but the GUI still shows the shared field
as if it mattered for every model, which is exactly the kind of "annoying"
inconsistency worth fixing properly: either grey out/hide "Number of
macroelectrons" when the active model doesn't use it (`ModelCapabilities`
would need a new flag, e.g. `uses_shared_sample_count`), or fold sample
count into each model's own `extra_params()` uniformly and remove the
shared field entirely, so there's one obvious place per model to control
"how many particles", not a shared field that's a no-op half the time.

### 5. Unify the `ModelAdapter` interface properly

The protocol (`gui/src/compton_guide/model_api.py`) is already shared,
but per-model wiring is still inconsistent in ways worth finishing, not
just documenting as "not yet done":
- `params_to_config` still does FWHM/waist/duration arithmetic by hand in
  both `kascade_adapter.py` and `xigma_i/gui_adapter.py`, instead of going
  through `compton_io`'s canonical-conversion framework
  (`adapt_to_model`/`ModelSpec`) that `xigma_i.params.XIGMA_SPEC` already
declares but doesn't use (see item 1). `gui/scripts/
physics_params_demo.py` already demonstrates what the wired-up version
  would look like end to end.
- `kascade` has no `ModelSpec`/`ParameterSpec` schema at all (unlike
  `xigma_i.params`) -- `gui/src/compton_guide/physics_params/schemas/
  kascade.py`'s `KASCADE_SPEC` is GUI-owned, not model-owned, which is the
  same asymmetry `compton_io.results`/`compton_io.photons` already fixed
  for results -- move it into `models/kaskade/` for real model-contract
  ownership.
- `extra_params()` only supports numeric fields (`list[tuple[str, float,
  str]]`) -- item 6 below needs a choice/enum-typed field, which doesn't
  fit this shape yet and would need a small protocol extension (or a
  separate `extra_choices()` method) rather than overloading a float.

### 6. Manual CPU/GPU selection for xigma-i

**As a library**: already possible -- `xigma_i.config.build_params(...,
device="cpu"|"gpu")` and `TabulatedEngine` take an explicit device, no
code changes needed; `_detect_device()` is only the *default* when
`device=None`.

**In the GUI**: not wired up at all. `models/xigma/src/xigma_i/
gui_adapter.py`'s `run_simulation` always calls `_detect_device()`
unconditionally (~line 437), ignoring any user preference, and there's no
GUI control for it. To fix: add a device choice somewhere in `Config`
(e.g. `device_preference: str = "auto"`), surface it via a GUI control
(needs the `extra_params()` extension from item 5, since this is a
string/enum choice, not a float), and pass it through to `build_params`
instead of always auto-detecting. Same applies to `xigma_direct`'s
`gui_adapter.py`, which has the identical `_detect_device()` call.

### 7. How to add a new model

1. Create `models/<name>/` with its own `pyproject.toml` (see
   `models/kaskade/pyproject.toml` for a flat single-module package, or
   `models/xigma/pyproject.toml` for a `src/` layout) and add it to the
   dev-install command in this file's "Dev install" section.
2. Implement the `ModelAdapter` protocol (`gui/src/compton_guide/
   model_api.py`): `capabilities()`, `available()`, `extra_params()`,
   `params_to_config(fields, quantum)`, `run(cfg, n_mc, seed, *,
   electrons: MacroBunch) -> CommonResults`, `load_ele_file`,
   `ele_file_summary`, `spectrum_in_angular_range`. Either as a module of
   free functions (xigma_i's style: `gui_adapter.py` has module-level
   functions plus a thin `XigmaAdapter` class delegating to them) or a
   single class (kascade's `KascadeAdapter` style) -- both work, `models/
   xigma_direct/` and `models/xigma/` use the module-functions style,
   `gui/src/compton_guide/adapters/kascade_adapter.py` uses the class
   style directly.
3. **Follow this session's architecture rules** (see "No model-local
   physics-parameter configs..." above): build electron/laser objects via
   `compton_io.bunch.beam_from_shared_fields`/`compton_io.laser.
   laser_from_shared_fields`, never sample particles yourself (require
   `electrons: MacroBunch` in `run()`, no internal fallback), construct
   `compton_io.results.CommonResults` directly (don't define a local
   lookalike), and keep only genuinely model-specific numerics (grid
   sizes, step counts, ...) on your own `Config`.
4. Register in `gui/src/compton_guide/models.py`'s `discover_models()`:
   `try: from <name>_pkg import gui_adapter as _x; register("<name>",
   _x.SomeAdapter()) except Exception as e: register("<name>",
   UnavailableAdapter(...))` -- wrap in try/except so a missing optional
   dependency (e.g. no GPU) greys the model out instead of crashing GUI
   startup, matching every existing model.
5. Extend `gui/scripts/headless_test.py`'s model loop to exercise the
   new adapter, and add a `build_<name>_config`/`run_<name>` pair to
   `validation/scenarios.py`/`validation/runners.py` if it should
   participate in the cross-model validation suite.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
