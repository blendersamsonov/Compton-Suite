# ComptonSuite model-agnostic-core refactor: status

**Status**: Superseded by what actually happened — kept as a historical
record of the goal and a running log of progress, not a package plan to
execute literally.
**Location**: `docs/refactor/core-simulation-api.md`

---

## What changed since this doc was first written

The original version of this doc (2026-07-26) proposed extracting a new
`compton_suite.core` package (`core/protocol.py`, `core/collision.py`,
`core/simulation.py`, `core/adapters/`) between `io/` and `gui/`/`models/`.
**That package was never built.** Instead, the repo consolidated directly
into a single `src/compton_suite` package with `io/`, `gui/`, `models/`,
`validation/` submodules — `io/` (née `compton_io`) *is* the model-agnostic
shared layer this doc originally wanted `core/` to be. Individual pieces
of physics/config logic have been moved out of models and into `io/` one
at a time, generalized just enough for their actual consumers, each with
a docstring explaining what stayed local and why. This doc now tracks that
list instead of a speculative package-creation plan.

## Current package layout

```
src/compton_suite/
├── io/            # shared-nothing dependency layer (was compton_io)
│   ├── bunch.py, laser.py          # GaussianElectronBeam / GaussianParaxialLaser (SI, v0.1 contracts)
│   ├── laser_envelope.py           # gaussian_pulse_envelope (see "Landed" below)
│   ├── propagation.py              # ballistic drift + laser_overlap_time_window
│   ├── collision.py                # CollisionParams / build_params (CGS/k0_las)
│   ├── interaction.py, photons.py, results.py, constants.py, units.py, ...
│   └── io_formats/                 # elegant .ele, YAML spec I/O
├── gui/           # Tkinter GUI (compton_guide) — still a thick consumer, see "Still open" below
├── models/
│   ├── kascade/       # SI, sequential multi-photon MC, arbitrary crossing angle
│   ├── xigma_i/        # CGS/k0_las-normalised, tabulated-overlap GPU/CPU pipeline
│   ├── xigma_direct/   # brute-force per-macroparticle binning, reuses xigma_i's Stage 0
│   └── analytical/     # closed-form yield/spectrum estimates
└── validation/    # shared Scenarios, per-model runners, tiered cross-model comparisons
```

## Landed — moves that fulfilled the original goal without a `core/` package

Each of these started as logic duplicated in (or private to) one model,
and got promoted to a shared, unit-convention-agnostic function in `io/`
once a second consumer needed the same thing:

- **`CollisionParams`/`build_params`** → `io/collision.py` (was
  `xigma_i/config.py`, the only consumer at the time; `xigma_direct` now
  reuses it directly too).
- **`ballistic_position_simultaneous`/`ballistic_position_z0_reference`/
  `propagate`/`stream`/`laser_overlap_time_window`** → `io/propagation.py`
  (`laser_overlap_time_window` was originally `xigma_i.particles.
  _time_window`).
- **`gaussian_pulse_envelope`** → `io/laser_envelope.py` (this refactor).
  `models/kascade/kascade.py`'s `laser_density`/`laser_a0sq` (SI,
  arbitrary crossing angle) and `models/xigma_i/particles.py`'s inline
  spatiotemporal envelope math (CGS/`k0_las`-normalised, flying-focus)
  independently reimplemented the same Gaussian-pulse-envelope physics —
  verified algebraically and numerically identical at
  `crossing_angle=0, beta_ff=0` (`tests/io_tests/test_laser_envelope.py`).
  Both models now call the one shared function; `io/laser.py`'s
  `GaussianParaxialLaser` itself is intentionally untouched (still
  on-axis-peak-only by design, see its own module docstring) since a
  crossing-angle/flying-focus-capable evaluator would contradict that
  module's v0.1 contract.
- **electron-side beam-derivation formulas** (`beta_star_x_m`/
  `beta_star_y_m`/`divergence_x_rad`/`divergence_y_rad`) → module-level
  functions in `io/bunch.py`, backing both `GaussianElectronBeam`'s own
  properties and `collision.build_params`, instead of four independent
  re-derivations (`xigma_i/config.py`, `xigma_i/gui_adapter.py`,
  `xigma_direct/gui_adapter.py`, `models/kascade/kascade.py` all used to
  have their own copy).

## Still open

- **GUI-as-thin-consumer.** Still a real, largely unmet goal — the
  original doc's literal prescription ("delete `gui/model_api.py`/
  `adapters/`, import `core.*` only") no longer applies since there's no
  `core/` package to import from, but the underlying goal (GUI shouldn't
  carry its own physics-layer knowledge) is still valid and still unmet.
  `gui/app.py` is ~1200 lines and directly imports `matplotlib`/`tkinter`
  alongside `compton_suite.io.bunch` and its own `gui.model_api`/
  `gui.models` adapter layer — that's the concrete baseline to shrink
  from, not a specific package migration.
- **`xigma_i.config.py`'s `a0` formula** — still has the documented,
  unreconciled ~49% discrepancy against `GaussianParaxialLaser.a0_focus`
  (`validation/tier0_wiring.py`'s `check_a0_formula_agreement`). Explicitly
  out of scope for the `gaussian_pulse_envelope` move above (envelope
  *shape*, not the `a0` normalization, was what was duplicated) — a real
  physics investigation, not a refactor.
- **Two copies of `validation/`** — a top-level `validation/` directory
  (still importing the pre-consolidation `compton_io` module name directly
  and failing to import under the current package) alongside the live
  `src/compton_suite/validation/` that everything above actually runs
  against. Noticed while verifying this refactor; not touched here —
  worth a deliberate cleanup pass (likely just deleting the stale
  top-level copy) as its own follow-up.

## Explicitly dropped from the original plan

- The `core/protocol.py`/`ModelProtocol` abstraction and `core/adapters/`
  — no second package layer was introduced; each model's adapter
  (`models/*/gui_adapter.py` or `kascade_adapter.py`) still talks to the
  GUI's own `gui/model_api.py` directly.
- The backward-compatibility section's "`compton_io.collision.build_params`
  kept (deprecated, delegates to core)" — not applicable; `build_params`
  in `io/collision.py` already *is* the landed target, not a shim around
  something else.
- `SimulationConfig`/`run_simulation()` as a single unified entry point —
  never built; each model is still driven through its own
  `ModelAdapter.run()` via `gui/model_api.py` and `validation/runners.py`
  independently.
