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

**A future rename to "GammaForge"** (import alias `gfrg`) is planned but
not yet started — the package is still `compton_suite` everywhere. Don't
introduce partial `gfrg` aliasing outside a deliberate, complete rename
pass.

## Layout

| Directory | Python package | Role |
|-----------|-----------------|------|
| `src/compton_suite/io/` | `compton_suite.io` | Model-agnostic shared layer: `units.py` (pint registry, physical constants, the width/time/amplitude convention enums, `PhysicalQuantity`, and the raw `convert_width`/`convert_time`/`convert_amplitude` conversion functions — all in one file), electron-bunch (`bunch.py`) and laser-pulse (`laser.py`) representations, the shared (beam, laser) bundle (`interaction.py`), output/results dataclasses (`photons.py`), external I/O (`io_formats/`). **Depended on by everything else; depends on nothing in this repo.** There is no generic "ParameterSpec"/"ModelSpec"/`adapt_to_model` framework — each model converts a `PhysicalQuantity` to whatever convention/unit it needs directly via `convert_width`/`convert_time`/`convert_amplitude` at its own boundary. |
| `src/compton_suite/gui/` | `compton_suite.gui` | Tkinter desktop GUI (`app.py`, `calculations.py`). Thin consumer of `io/` and `models/api.py` — no physics computation, only rendering, field parsing, and compiling a `Job` to hand to whichever adapter is selected. |
| `src/compton_suite/models/api.py` | `compton_suite.models.api` | The `ModelAdapter` protocol, `Job`/`OutputSpec` dataclasses, the model registry (`register`/`registered_models`/`discover_models`). Every model plugs in here — see "Model registration" below. |
| `src/compton_suite/models/kascade/` | `compton_suite.models.kascade` | CPU Monte Carlo physics engine (sequential multi-photon event generator), SI units, always available. |
| `src/compton_suite/models/xigma_i/` | `compton_suite.models.xigma_i` | GPU (CuPy, numba CPU fallback) tabulated-overlap-table physics engine, CGS units, `k0_las`-normalised. `adapter.py` exposes **two** adapters: `XigmaAdapter` (registered as `"xigma-i"`, the full Stage 0/1/2 tabulated pipeline) and `DirectAdapter` (registered as `"delta"`, a brute-force per-macroparticle resonance-binning mode reusing this package's own Stage 0 directly — **not a separate model package**, just a different Stage-2 evaluation of the same engine). Greyed out (`UnavailableAdapter`) in the GUI if neither cupy/CUDA nor numba is usable. |
| `src/compton_suite/models/analytical.py` | `compton_suite.models.analytical` | Fast, closed-form yield/spectrum/width estimates, in a single flat module (`estimate_yield`, `estimate_spectrum_width`, `angle_integrated_spectrum`, `Adapter`). No `Config` class — `Adapter`'s one numeric knob (`theta_col_rad`) lives directly on the adapter instance. Always-on GUI preview alongside whichever other model is selected (hardcoded in `gui/app.py` as `self.analytical_adapter`). |
| `src/compton_suite/misc.py` | — | `detect_device()` — the one shared cupy/numba backend-detection helper every model that needs GPU/CPU dispatch imports. |
| `src/compton_suite/` | `compton_suite` | Unified package: `discover_models` re-export (via `models.api`) and the `run_gui`/console-script entry point. |
| `src/compton_suite/validation/` | `compton_suite.validation` | Cross-model validation suite — shared `Scenario`s (`scenarios.py`), per-model runners (`runners.py`, `Job`-based), tiered comparisons (`tier0_wiring.py`..`tier4_regime_boundary.py`, `run_cross_validation.py`), plotting (`visualize.py`), a commit-hash-keyed result cache (`cache.py`). |
| `tests/` | (not a package) | Root-level tests (`test_analytical.py`, `test_bunch_improvements.py`) plus `tests/io_tests/` (per-`io/`-module tests). |

## Architecture

### Dependency flow

```
io/  (shared layer — no deps in this repo)
 ↑
 ├── models/kascade/     (SI, CPU, numpy)
 ├── models/xigma_i/     (CGS, GPU/CPU, cupy/numba; "delta" is a mode of this package)
 ├── models/analytical/  (SI, closed-form)
 ├── models/api.py       (ModelAdapter protocol + registry — depends on io/ only)
 └── gui/                (Tkinter, thin consumer of io/ + models/api.py)
```

`io/` is the single source of truth for physical constants, unit conventions,
beam/laser representations, and output dataclasses. Every model and the GUI
import from `io/` — no model depends on another model, and no model depends
on the GUI. `models/api.py` depends only on `io/`; each model subpackage
depends on `io/` and `models/api.py`, never on a sibling model.

### Model registration

The GUI plugs in physics engines through the `ModelAdapter` protocol
(`models/api.py`) instead of hardcoded imports. Every adapter implements:

- `model_params() -> list[tuple[str, float | str, str]]` — model-specific
  numeric/choice fields as `(label, default, key)` triples, for the GUI's
  "Model Parameters" panel.
- `model_choices() -> dict[str, list[str]]` — optional; allowed values for any
  choice-type keys from `model_params()` (renders as a dropdown). Return `{}` if the model has no choice fields.
- `run(job: Job) -> Photons` — the one call that actually runs the model, taking a pre-compiled config and returning unbinned or binned spectrum results.

`Job` (`models/api.py`) is the single compiled-from-UI config object —
"GUI just calls model's adapter run with the config compiled from ui". It
bundles `interaction: InteractionParameters` (the shared beam+laser),
`electrons: Bunch` (pre-sampled, see below), `output: OutputSpec`
(resolution knobs), `seed: int`, and `extra: dict` (this model's own
`model_params()` values, read live from the GUI). There is no
`params_to_config(fields, quantum)` step and no per-adapter flat
`(cfg, n_mc, seed, electrons, output)` call signature — those predate the
`Job` convention and are gone.

Adapters live in each model's own package:
- `models/kascade/kascade_adapter.py` — `KascadeAdapter`
- `models/xigma_i/adapter.py` — `XigmaAdapter` (registered `"xigma-i"`),
  `DirectAdapter` (registered `"delta"`)
- `models/analytical.py` — `Adapter`

`models/api.py`'s `discover_models()` registers kascade and analytical
unconditionally, and wraps xigma_i's registration in a `try/except`,
falling back to `UnavailableAdapter` for both `"xigma-i"` and `"delta"` if
neither cupy/CUDA nor numba is usable — one missing optional dependency
never breaks `import compton_suite.models`.

**Only kascade has a standalone `Config` class.** xigma-i (`XigmaAdapter`),
delta (`DirectAdapter`), and analytical (`Adapter`) don't have a `Config`
dataclass at all — each adapter instance holds its own model-specific
numeric knobs directly as plain `self` attributes (`self.n_steps_0`,
`self.theta_col_rad`, etc.), updated from `job.extra` at the top of
`run()`. `XigmaAdapter` additionally holds its `TabulatedEngine`/
`CollisionParams` as `self.engine`/`self.params` after a run, so
`spectrum_in_angular_range()` can recompute an on-demand angular-range
query without rerunning the whole simulation; `DirectAdapter` (no
persistent table) instead keeps the raw per-particle arrays it needs for
the same purpose. kascade's own `Config` is real physics-engine state (used
throughout `kascade.py`'s ~1400 lines), not a redundant adapter-side
duplicate, so it stays — but `KascadeAdapter` itself still reads
`job.extra` straight into that `Config` on every call, with no adapter-level
caching (there's no functional need: no persistent engine/table to justify
it, and the GUI's own widgets already remember typed values across model
switches).

**No `Config`/adapter carries derived-value properties** (`eps0`, `N_e`,
`lambda_L`, etc.) that just rename/duplicate what's already on
`interaction.beam`/`interaction.laser` — those aren't "model state", they're
available directly from the input parameters. Read
`cfg.interaction.beam.gamma0` (etc.) at the point of use, or via a small
module-level helper function if the derivation is genuinely model-specific
(e.g. kascade's `_eps_L`/`_sigma0_l`/`_R_sf` — a relativistic-units
convention and a round-beam collapse, neither of which belongs on the
shared `GaussianParaxialLaser`). Physically meaningful quantities that
*are* generic (not model-specific) belong on the shared `io/` class itself
— e.g. `GaussianParaxialLaser.omega0`/`.a0_focus`/`.n_photons`/`.beta_ff`/
`.phi_pol`/`.ellipticity`. xigma_i's own CGS `CollisionParams.a0`/`.N_l`/
`.beta_ff`/`.ellipticity` (`models/xigma_i/collision.py`) are a straight
pass-through of the shared laser's own fields, not independently re-derived
or caller-supplied — see "Open items" below for why the a0/N_l case
mattered.

### Electron sampling

Every model's `Job.electrons` must be pre-sampled — sampling the electron
bunch is `io.bunch`'s job, not any individual model's. The GUI draws ONE
canonical `Bunch` via `io.bunch.sample_gaussian_bunch` (which delegates to
`sample_gaussian_canonical`) and passes it to every model uniformly. No
model has its own internal bunch sampler.

`Bunch` holds flat arrays directly (`x, y, z, thx, thy, gamma, weight,
meta`) — no nested `.particles` sub-object — **plus** `gaussian_fit:
GaussianElectronBeam | None`, the analytic description of that same
population. `GaussianElectronBeam` is a single type doing double duty: the
analytic *input* description (charge, energy, sizes, emittances, duration,
`alpha_x`/`alpha_y` Twiss tilt, plus optional `chirp_h`/`dispersion_x`/
`dispersion_y`) **and** the output of a structured fit (`fit_quality` is
`None` for a pure input, populated when `fit_gaussian` produced it) — there
is no separate `BeamFittedParams` type; that three-way split
(`Bunch`/`GaussianElectronBeam`/`BeamFittedParams`) was tried and
explicitly rejected — real consumers (analytical's yield/width estimates,
xigma's table-boundary sizing) need beam-level parameters attached to
whatever electrons are actually in play, not just at the moment of initial
sampling.

The sampling uses **canonical variables** (x, y, z, thx, thy, gamma) with
mass-shell enforcement: `pz = sqrt((gamma**2-1)/(1+thx**2+thy**2))`, `px =
thx*pz`, `py = thy*pz` is the *only* way `pz` is ever derived (never
independently sampled), so `gamma**2 = 1 + px**2 + py**2 + pz**2` holds
automatically by construction — there's no separate "mass-shell
enforcement" step to get wrong. `sample_gaussian_canonical`/
`sample_gaussian_bunch` attach the exact input `beam` as `gaussian_fit`
(a freshly-drawn sample matches it exactly, no fit needed).

After sampling, beams can be propagated using `drift(bunch, L)` (ballistic,
`x → x + thx·L`, naturally producing Twiss tilt from waist sampling) or
`propagate(bunch, dt)`/`stream(bunch, t_grid)` (light-travel-time,
built on `drift`'s same per-particle-`L` push internally — no duplicated
ballistic-position math between them). Both analytically update the
attached `gaussian_fit`'s `alpha_x`/`alpha_y` in lockstep with the
macroparticles (closed-form Twiss-drift relation, `alpha_new = alpha_old -
L/beta*` — no refit needed; waist-referenced sizes are already
drift-invariant). A bunch with no known analytic description (loaded from
a `.ele` file) carries `gaussian_fit=None` until explicitly fit.

For fitting macroparticles back to a beam description from scratch, use
`fit_gaussian(bunch)` — a full covariance-based Twiss/chirp/dispersion fit
(handles nonzero `alpha_x`/`alpha_y`, unlike the old waist-only trick),
returning a `GaussianElectronBeam` with `fit_quality` populated
(Mahalanobis distance, KS statistics, log-likelihood via
`evaluate_fit_quality`).

### Results contract

Every model's `run()` returns `io.photons.Photons` (**not**
`CommonResults` — that name doesn't exist anywhere in this codebase).
`Photons.spectrum` (and `.temporal_envelope`/`.spatial_distribution`) come
in two duck-typed shapes: `Sampled*` (unbinned per-macroparticle arrays +
a uniform `weight`, from kascade) or `Binned*` (smooth pre-binned density
arrays, from xigma_i/delta/analytical). Check shape with
`hasattr(spectrum, "weight")` vs `hasattr(spectrum, "dNdE_per_eV")` —
**never `isinstance()`** against a GUI/engine boundary (see "Cross-repo
gotchas"). `Photons` has no `cfg` field — no model carries a standalone
`Config` object to stash there anymore (see "Model registration" above);
an adapter needing its own state for later on-demand recompute holds it as
`self.<attr>` instead (e.g. `XigmaAdapter.engine`/`.params`).

### Key `io/` functions

| Function | Module | Purpose |
|----------|--------|---------|
| `sample_gaussian_bunch` / `sample_gaussian_canonical` | `bunch.py` | Draw macroparticles from a `GaussianElectronBeam` (attaches it as `Bunch.gaussian_fit`) |
| `fit_gaussian` | `bunch.py` | Fit a `GaussianElectronBeam` (Twiss, chirp, dispersion, fit quality) from raw macroparticles |
| `drift` / `propagate` / `stream` | `bunch.py` | Ballistic (distance) / light-travel-time (duration, time grid) propagation — both analytically update `gaussian_fit`'s Twiss tilt |
| `gaussian_pulse_envelope` | `laser.py` | (x, y, z, t) Gaussian-pulse photon-density evaluator, shared by kascade and xigma_i |
| `sigma_from_emittance` | `bunch.py` | Transverse rms beam size from emittance/beta/gamma |
| `recoil_parameter` | `interaction.py` | Quantum recoil parameter q = 4γℏω/m_ec² |
| `convert_width` / `convert_time` / `convert_amplitude` | `units.py` | Convert a bare float between conventions (FWHM ↔ RMS ↔ 1/e², etc.) — the model-boundary conversion primitive; no generic spec/adapter framework wraps these |
| `build_params` | `models/xigma_i/collision.py` | Build CGS `CollisionParams` for xigma-i/delta — **model-owned, not shared io/** (kascade has no use for a CGS bundle) |

### Units

- `kascade`/`analytical` are SI (m, s, J)
- `xigma_i`/`delta` are CGS (cm, erg), normalized to the laser wavenumber —
  `models/xigma_i/collision.py`'s `CollisionParams`/`build_params`
- Each model converts at its own boundary — never assume a value crossing
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
- **Units differ per engine.** Each model converts at its own boundary.
- **Physical constants have one source of truth**: `compton_suite.io.units`
  (constants, pint registry, and the convention enums all live in this one
  file now — there's no separate `constants.py`/`enums.py`/`quantities.py`).
- **No model-local particle sampling, no model-local result contract.**
  Electron-beam and laser parameters come from `io.bunch`/`io.laser`;
  model-specific numerics (grid sizes, step counts) and any
  geometry/crossing-angle/quantum-toggle fields that not every model
  supports are model-owned (as plain attributes on the adapter, or —
  kascade only — on its own real `Config`), never part of the shared
  bundle. Every model's `run()` returns `io.photons.Photons` directly.
- **No derived-value properties, and (except kascade) no `Config` class at
  all**, duplicating what's already on `interaction.beam`/
  `interaction.laser` — see "Model registration" above.
- **No `*_from_shared_fields` factory functions.** `GaussianElectronBeam`/
  `GaussianParaxialLaser` fields are already `PhysicalQuantity`-wrapped —
  a caller (the GUI) builds them directly, wrapping its own raw floats at
  the call site; don't reintroduce an io-level indirection function whose
  only job is "take flat floats, build the dataclass" or the reverse.

## Commands

```bash
# GUI (needs numpy, matplotlib, tkinter; cupy optional for xigma-i)
python3 scripts/run_gui.py

# Headless smoke test (all 4 registered models + always-on preview)
python3 scripts/headless_test.py

# Cross-model validation suite
python3 src/compton_suite/validation/run_cross_validation.py
```

## Where to look next

- `models/kascade/AGENTS.md` — engine internals, Config/Results fields,
  .ele file I/O, units.
- `docs/models/xigma.md` — the four-stage pipeline, physics
  conventions, documented traps (shared-memory aliasing, etc. — the a0
  trajectory-averaging note there predates this session's a0-formula fix,
  see "Open items" below).
- `docs/refactor/` — historical refactor notes (parameter framework,
  core-simulation-api status) — predate the `Job`-based `ModelAdapter`
  rewrite; read as history, not current architecture.

## Design decisions (not to be revisited without good reason)

- **`Bunch` unifies samples + analytic fit**: `GaussianElectronBeam` is both the input specification and the output of `fit_gaussian()`. No separate `BeamFittedParams` type. `Bunch.gaussian_fit` is analytically updated through `drift`/`propagate` (no refit needed).
- **No `*_from_shared_fields` factory functions**: the GUI builds `GaussianElectronBeam`/`GaussianParaxialLaser` directly; callers do the dataclass construction at the boundary.
- **Adapters hold model-specific state as `self` attributes**: numeric knobs (`n_particles_01`, `n_steps_0`, etc.) and recompute caches (`engine`, `params`) live as plain instance attributes, not in a `Config` dataclass. Exception: `KascadeAdapter` has its own `Config` class (a real physics-engine state object, not an adapter-side wrapper).
- **Laser properties on shared `GaussianParaxialLaser`**: `beta_ff`, `phi_pol`, `ellipticity` are model-agnostic and belong on the laser, not duplicated in model-specific config.
- **No `quantum`, `crossing_angle`, `Theta_x`, `Theta_y` fields**: these were declared but never wired. Crossing-angle support is planned; until then, `job.interaction.crossing_angle` must be zero.
- **No `Photons.cfg` field**: results carry no back-reference to the configuration that produced them.

## Open items (physics investigation, not refactoring)

- **~2π angular-spectrum residual** (xigma_i/delta's raw kernel normalization): `spectrum4d.py`'s `calculate_angular_spectrum_4d` and `spectrum_from_particles.direct_binning_spectrum` have an unexplained absolute normalization factor at the kernel level. User-facing: `angular_spectrum` now integrates to `total_yield` exactly for both xigma-i and delta, matching `spectrum_in_angular_range` queries. The kernel-level discrepancy remains open.

- **a0/N_l pass-through**: `CollisionParams.a0`/`.N_l` are now a direct pass-through of `GaussianParaxialLaser.a0_focus`/`.n_photons` (a0/N_l are model-agnostic, available from the input laser). Verified by `validation/tier0_wiring.py`'s `check_a0_N_l_passthrough`.

- **CUDA OOM with large electron bunches in xigma_i/delta**: GPU memory exhaustion when running large n_mc values (e.g. 200k+). `particles.push_and_sample` processes all electrons at once with no batching. Workaround: chunked GPU processing or automatic CPU fallback needed.

---

## User-specified tasks (pending)

In priority order, as recorded in `docs/gui/tasks.md` and `docs/models/tasks.md`.

### GUI

- [ ] **Abandon angular-range tab.**
- [ ] **Remove hard-coded seed and number of macroelectrons** from compton photons tab into model specific parameters.
- [ ] **Grey out inputs after simulation done** (except charge). For XIGMA leave pulse energy and gamma active. Add a "release" button to change parameters again.
- [ ] **Dropdown per-input unit selection** (cm, m, mm, µm, etc.) with automatic PhysicalQuantity conversion. Must work even when values are read-only (e.g. from a macrobunch).
- [ ] **Multiple self-consistent spatial-scale definitions:** waist vs Rayleigh range / beta function.
- [ ] **2D/3D interaction geometry sketches:** axes, 3D ellipses for e⁻ and laser, polarization arrows, "ghost" foci at different time delays.
- [ ] **Sliders for inputs.**
- [ ] **Parameter scans/ranges.**
- [ ] **Save to file:** graphs, photon representations.

### Xigma-i

- [ ] **Streaming GPU usage:** query VRAM, cap allocation at ~70%.
- [ ] **Crossing angle support** (should only change the polarization factor).
- [ ] **Rename:** drop "-i" suffix → just "XIGMA". (Note: the adapter's display name is already `"XIGMA"`; the registry key and package name themselves still say `xigma_i`/`xigma-i`.)
- [ ] **Gamma-axis rescaling** analogous to a0 rescaling, to vary mean energy without recomputing Stages 0-1.

### Analytical model

- [ ] **Foci displacement** in the analytical model.
- [ ] **Non-round beam** closed-form total yield.
- [ ] **Collimated spectrum** from total yield, collimation angle, and spectrum width (convolution with energy distribution + a₀).

### All models

- [ ] **Jitter and shot averaging.**

---

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
