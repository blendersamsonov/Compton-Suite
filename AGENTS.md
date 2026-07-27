# AGENTS.md

This file provides guidance to AI agents (Claude Code, etc.) when working with code in this repository.

## What this is

`ComptonSuite` is a single git repository for an inverse-Compton-scattering
physics simulation + GUI suite. It used to be six independent git repos
(one per component); they were merged here via `git subtree`, so each
component's full commit history is preserved under its new path (use
`git log --full-history -- <path>` to see pre-merge history, since the
path changed at the merge). The components have since been consolidated
into a single unified package (`compton_suite`) with sub-packages for
io, gui, and models — see "Layout" below.

## Layout

| Directory | Python package | Role |
|-----------|-----------------|------|
| `src/compton_suite/io/` | `compton_suite.io` | Model-agnostic shared layer: physical constants (`constants.py`, pint-derived), unit registry (`units.py`), parameter-convention framework (`enums.py`/`quantities.py`/`canonical.py`/`schema.py`/`validation.py`), electron-bunch (`bunch.py`) and laser-pulse (`laser.py`) representations, collision parameters (`collision.py`), interaction parameters (`interaction.py`), propagation (`propagation.py`), laser envelope (`laser_envelope.py`), output dataclasses (`photons.py`/`results.py`), external I/O (`io_formats/`). **Depended on by everything else; depends on nothing in this repo.** |
| `src/compton_suite/gui/` | `compton_suite.gui` | Tkinter desktop GUI. Thin consumer of `io/` — no physics computation, only rendering and field parsing. Model-agnostic via the `ModelAdapter` protocol (`model_api.py`). |
| `src/compton_suite/models/kascade/` | `compton_suite.models.kascade` | CPU Monte Carlo physics engine (sequential multi-photon event generator), SI units, always available. |
| `src/compton_suite/models/xigma_i/` | `compton_suite.models.xigma_i` | GPU (CuPy, numba CPU fallback) tabulated-overlap-table physics engine, CGS units. Greyed out in the GUI if cupy/CUDA isn't usable. |
| `src/compton_suite/models/delta/` | `compton_suite.models.delta` | Brute-force per-macroparticle resonance-binning model, reuses `xigma_i`'s Stage 0 physics directly. |
| `src/compton_suite/models/analytical/` | `compton_suite.models.analytical` | Fast, closed-form yield/spectrum/width estimates; always-on GUI preview alongside whichever other model is selected. |
| `src/compton_suite/` | `compton_suite` | Unified package: `discover_models()` re-export and the `comptonsuite-gui` console-script entry point. |
| `src/compton_suite/validation/` | `compton_suite.validation` | Cross-model validation suite — shared `Scenario`s, per-model runners, tiered comparisons (`run_cross_validation.py`), plotting (`visualize.py`), a commit-hash-keyed result cache (`cache.py`). |
| `tests/` | (not a package) | Root-level tests (currently `test_analytical.py`). |

## Architecture

### Dependency flow

```
io/  (shared layer — no deps in this repo)
 ↑
 ├── models/kascade/     (SI, CPU, numpy)
 ├── models/xigma_i/     (CGS, GPU/CPU, cupy/numba)
 ├── models/delta/       (CGS, reuses xigma_i Stage 0)
 ├── models/analytical/  (SI, closed-form)
 └── gui/                (Tkinter, thin consumer)
```

`io/` is the single source of truth for physical constants, unit conventions,
beam/laser representations, collision parameters, and output dataclasses.
Every model and the GUI import from `io/` — no model depends on another
model, and no model depends on the GUI.

### Model registration

The GUI plugs in physics engines through a `ModelAdapter` protocol
(`model_api.py`) instead of hardcoded imports. All four adapters implement
the same interface: `capabilities()`, `available()`, `extra_params()`,
`extra_choices()`, `params_to_config(fields, quantum)`, `run(cfg, n_mc,
seed, *, electrons: MacroBunch) -> CommonResults`, `load_ele_file`,
`ele_file_summary`, `spectrum_in_angular_range`.

Adapters live in each model's own package:
- `models/kascade/kascade_adapter.py` — class-based (`KascadeAdapter`)
- `models/xigma_i/gui_adapter.py` — module-functions + thin delegating class (`XigmaAdapter`)
- `models/delta/gui_adapter.py` — module-functions + thin delegating class (`DeltaAdapter`)
- `models/analytical/analytical_adapter.py` — class-based (`AnalyticalAdapter`)

### Electron sampling

Every model's `run()` requires a pre-sampled `electrons: MacroBunch`
(keyword-only) — sampling the electron bunch is `io.bunch`'s job, not any
individual model's. The GUI draws ONE canonical `MacroBunch` via
`io.bunch.sample_gaussian_bunch` and passes it to every model uniformly.
No model has its own internal bunch sampler.

### Key `io/` functions

| Function | Module | Purpose |
|----------|--------|---------|
| `beam_from_shared_fields` | `bunch.py` | Build `GaussianElectronBeam` from flat SI fields |
| `laser_from_shared_fields` | `laser.py` | Build `GaussianParaxialLaser` from flat SI fields |
| `sample_gaussian_bunch` | `bunch.py` | Draw macroparticles from a `GaussianElectronBeam` |
| `fit_gaussian` | `bunch.py` | Fit `GaussianElectronBeam` from raw macroparticles |
| `build_params` | `collision.py` | Build CGS `CollisionParams` for xigma-i/delta |
| `a0_from_fields` | `laser.py` | Peak a0 from raw SI laser fields (single source of truth) |
| `focal_radii_m` | `laser.py` | RMS/FWHM/e^{-1/2} focal radii |
| `sigma_from_emittance` | `bunch.py` | Transverse rms beam size from emittance/beta/gamma |
| `recoil_parameter` | `interaction.py` | Quantum recoil parameter q = 4γℏω/m_ec² |
| `adapt_to_model` / `params_to_floats` | `adapter.py` | Canonical unit/conversion for duration inputs |

### Units

- `kascade`/`analytical` are SI (m, s, J)
- `xigma_i`/`delta` are CGS (cm, erg), normalized to the laser wavenumber
- Each adapter converts at its own boundary — never assume a value crossing
  into the GUI is in a particular unit system without checking which engine
  produced it

## Dev install

The entire suite is a single installable package with a unified `pyproject.toml` at the repo root:

```bash
pip install -e .
```

On this dev machine, system Python lacks pip/cupy/matplotlib; GPU-dependent
work (`xigma-i`, `delta`, GUI runs with `xigma-i` enabled) needs the
`core` conda env:

```bash
conda run -n core --no-capture-output pip install -e .
conda run -n core --no-capture-output python3 <script>
```

(`--no-capture-output` is required — plain `conda run` silently swallows
stdout.)

## Cross-repo gotchas (still apply post-merge)

- **Never `isinstance()` against both sides of a GUI/engine data
  boundary.** Use duck typing instead (e.g. `hasattr(x, "weight")`
  vs `hasattr(x, "dNdE_per_eV")`).
- **Units differ per engine.** Each adapter converts at its own boundary.
- **Physical constants have one source of truth**: `compton_suite.io.constants`.
- **No model-local physics-parameter configs, no model-local particle
  sampling, no model-local result contract.** Electron-beam and laser
  parameters come from `io.bunch`/`io.laser`; a model's own `Config`
  carries only model-specific numerics (grid sizes, step counts). Every
  model's `run()` returns `io.results.CommonResults` directly.

## Commands

```bash
# GUI (needs numpy, matplotlib, tkinter; cupy optional for xigma-i)
python3 scripts/run_gui.py

# Headless smoke test (all 4 models + preview)
python3 scripts/headless_test.py

# Cross-model validation suite
python3 src/compton_suite/validation/run_cross_validation.py
```

## Where to look next

- `models/kascade/AGENTS.md` — engine internals, Config/Results fields,
  .ele file I/O, units.
- `models/xigma_i/AGENTS.md` — the four-stage pipeline, physics
  conventions, documented traps (a0 trajectory-averaging, 1/(1+a0)
  Jacobian, shared-memory aliasing).
- `docs/refactor/` — historical refactor notes (parameter framework,
  core-simulation-api status).

## Roadmap: completed items

All items below are done. Kept as a record of what was accomplished.

### 1. Dead-code / unused-config sweep ✅
Resolved: `Config.emulate_nonlinearity` removed, `XIGMA_SPEC` partially
wired (`sigma_par_e`/`sigma_par_L` through `adapt_to_model` in all 4
adapters). `sigma0_x`/`sigma0_y`/`sigma0_l` deliberately still
hand-derived (unambiguous fields, not raw convention-ambiguous inputs).

### 2. Move `CollisionParams`/`build_params` into `io/` ✅
Lives in `io/collision.py`. All 4 adapters import from there. The `a0`
formula has a known ~49% CGS vs SI discrepancy (flagged in
`io/laser.py`'s `a0_from_fields` docstring) — a physics investigation,
not a refactor.

### 3. GUI: trust levels/warnings ✅
xigma-i/delta graduated to `trust_level="production"`. Trust notes name
the two concrete open items (a0 formula, ~2π angular residual).

### 4. GUI: per-model sample count ✅
`uses_shared_sample_count` on `ModelCapabilities`. Grey-out works for
xigma-i/delta/analytical.

### 5. Unify the `ModelAdapter` interface ✅
All 4 adapters aligned: `adapt_to_model` for durations, `beam_from_shared_fields`/`laser_from_shared_fields`, `extra_choices()`, `ModelCapabilities` return type, canonical constants. Analytical adapter routed through `io/` shared infrastructure.

### 6. Manual CPU/GPU selection ✅
`device_preference` in xigma-i/delta `extra_params()` + `extra_choices()`
returning `["auto", "gpu", "cpu"]`.

### 7. How to add a new model ✅
Documented in this file (see Architecture > Model registration).

### 8. GUI-as-thin-consumer ✅
All GUI physics moved to `io/`: `peak_a0` → `io.laser.a0_from_fields`,
`sigma_e` → `io.bunch.sigma_from_emittance`, `laser_focal_radii` →
`io.laser.focal_radii_m`, `recoil_q` → `io.interaction.recoil_parameter`.
`app.py` (1174 lines) is now pure GUI rendering with no physics
computation. Only trivial unit conversions (gamma → MeV) remain.

## Open items (physics investigation, not refactoring)

- **a0 formula**: CGS `collision.py` formula produces ~half the SI
  `laser.py` value (`validation/tier0_wiring.py`'s
  `check_a0_formula_agreement`). Needs human verification of the CGS
  derivation.
- **~2π angular-spectrum residual**: documented in `reference.py`'s
  module docstring.
- **Stale top-level `validation/` directory**: contains only runtime
  artifacts (`__pycache__/`, `.cache/`, `.ele`, `plots/`) — safe to
  `rm -rf` from the main repo.

---

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
