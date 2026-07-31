# KASCADE

Sequential multi-photon inverse-Compton Monte Carlo (event generator).
Samples individual macro-electrons through a multi-photon emission chain
and returns unbinned per-macro-photon/per-macro-electron arrays. This is
one of two independent physics engines plugged into the `Compton-GUIde`
project (Python package `compton_suite.gui`) via its
`ModelAdapter` contract — see "Relationship to sibling repos" below.

The name: K stands for Klein-Nishina — the quantum recoil correction this
engine implements that the other engine (XIGMA) doesn't — and the engine's
core mechanic is a cascading chain of sequential photon emissions per
electron (tracked via the `ph_gen` generation index), hence KASCADE.

## Files

- `kascade.py` (formerly `dfe5_compton_mc.py`) — the entire engine (~1400 lines). Key pieces:
  - `Config` — SI-unit dataclass (meters, seconds, Joules) for beam/laser/geometry params.
  - `run_multiphoton_chain(x_w, thx, y_w, thy, z0, eps, cfg, rng)` — simulates the emission chain for one chunk of electrons. Samples emission time via inverse-CDF (`invert_lambda`), computes photon energy/angle via Thomson or Klein-Nishina cross section depending on `cfg.quantum`, applies recoil kick to the electron.
  - `run_simulation(cfg, n_mc, seed, *, electrons)` — the main driver; `electrons` is a required, keyword-only dict (`eps`/`z0`/`x_w`/`y_w`/`thx`/`thy` per-particle arrays) — sampling it is the caller's job (see below), not this function's; passing `electrons=None` raises `TypeError`. Chunks over `cfg.chunk` electrons at a time, concatenates all per-chunk photon arrays, computes summary stats, and returns a `Results` object. Also **always writes a "final 6D electron distribution" `.ele` file to disk** (default `final_distribution.ele` in the CWD, overridable via `electrons["final_ele_path"]` or the `KASCADE_FINAL_ELE_PATH` env var) — this happens on every call, including test/smoke runs, which is why that filename is gitignored.
  - `Results` — per-photon fields: `ph_E_eV`, `ph_thx`/`ph_thy` (emission angle rel. to electron), `ph_thx_lab`/`ph_thy_lab` (lab-frame angle), `ph_t` (emission time, seconds), `ph_x`/`ph_y` (transverse position at emission, meters), `ph_gen`, `ph_parent`; per-electron fields: `eps_i`/`eps_f`, `thx_i/f`, `thy_i/f`, `z0`, `n_phot`, `lambda_total`. `ph_t`/`ph_x`/`ph_y` were added in 2026-07 (additive — threaded through `run_multiphoton_chain` → `run_simulation` → `Results`, mirroring the pre-existing `ph_thx_lab` pattern; no existing field or function signature changed).
  - `load_ele_file`/`ele_file_summary`/`save_ele_file` — SDDS `.ele` format I/O for loading/saving 6-D electron bunches.

This module is a pure physics library — no `main()`, no CLI, no TOML config
loading, no plotting. Call `run_simulation(cfg, n_mc, seed, electrons=...)`
directly:

```python
from compton_suite.models.kascade import kascade as _kascade
from compton_suite.models.kascade.kascade_adapter import _bunch_to_kascade_electrons
from compton_suite.io.bunch import GaussianElectronBeam, sample_gaussian_bunch
from compton_suite.io.interaction import InteractionParameters

beam = GaussianElectronBeam(...)
laser = GaussianParaxialLaser(...)
cfg = _kascade.Config(interaction=InteractionParameters(beam=beam, laser=laser))
macrobunch = sample_gaussian_bunch(beam, n_particles=20_000)
electrons = _bunch_to_kascade_electrons(macrobunch)
results = _kascade.run_simulation(cfg, n_mc=20_000, seed=0, electrons=electrons)
```

`electrons` (a dict of `eps`/`z0`/`x_w`/`y_w`/`thx`/`thy` per-particle
arrays, converted from a `compton_suite.io.bunch.Bunch`) is required —
sampling the electron bunch is `compton_suite.io`'s job, not this module's; see
"Relationship to other components" below and the root `AGENTS.md`'s
Architecture section. `Config` has no derived-value properties of its own
(`eps0`/`eps_L`/`N_L`/`sigma0_l`/`R_sf`/`sigma_par_L` are computed on demand
from `cfg.interaction.beam`/`cfg.interaction.laser` by the physics
functions, or by the module-level `_eps_L`/`_sigma0_l`/`_R_sf` helpers —
not stored as Config state).

Internally, docstrings/comments still refer to a prior "dfe4"/"dfe5" code
lineage this engine's physics generalizes (arbitrary crossing angle +
Klein-Nishina on top of a "dfe4" baseline) — that's genuine historical
narrative about earlier code generations, left as-is; only this engine's
*own* self-references were renamed to kascade/KASCADE.

## Units and conventions

- Everything in `Config`/`Results` is **SI** (m, s, J, rad) — this is different from `xigma_i`'s CGS-like (cm, erg) convention. `compton_suite.gui`'s `model_api.py`/adapters were designed around this engine's SI convention; `xigma_i/gui_adapter.py` converts to/from CGS at its own boundary.
- `cfg.quantum` toggles Thomson (classical) vs Klein-Nishina (quantum recoil) cross section — this is a genuine physics toggle; `xigma_i` accepts the same field for interface symmetry but its differential cross section has no classical/quantum switch, so it's a no-op there.
- Supports arbitrary `crossing_angle` (unlike `xigma_i`, which is head-on only).

## Physical constants (`compton_suite.io`)

`kascade.py`'s "Physical constants (SI)" block (`C_LIGHT`/`E_CHARGE`/
`HBAR`/`EPS0`/`SIGMA_T`/`MEC2_EV`/`MEC2_J`) comes from `compton_suite.io.units`
(pint-derived, not hand-typed literals) instead of local literals. This was
a **zero-numeric-change** dedup: this module's previous literals already
agreed with `compton_suite.io`'s canonical values to their quoted precision
(unlike `xigma_i`'s config.py, whose hbar/electron-mass were on an older
CODATA vintage and needed an actual, deliberate numeric update as part of
the same migration).

There is no generic `ParameterSpec`/`ModelSpec`/`adapt_to_model` framework
anymore (the earlier `params/spec.py`/`KASCADE_SPEC` this section used to
describe is gone) — `Config.crossing_angle`/`.delta_x/y/z`/`.quantum` are
plain kascade-owned fields (not part of the shared `InteractionParameters`
bundle, since xigma_i/delta don't support them), populated by
`kascade_adapter.KascadeAdapter.model_params()`/`.model_choices()` reading
straight off the GUI's fields — no convention conversion needed for these
(none of them are width/duration-ambiguous).

## No GPU dependency

Pure NumPy/CPU — no cupy needed. This is why `kascade` is always the available/enabled model in `Compton-GUIde`'s registry, while `xigma-i` can be greyed out if cupy/CUDA isn't set up.

## Testing

No unit test suite lives in this package. It's validated via `scripts/headless_test.py`, which builds a `Job` and calls `KascadeAdapter.run(job) → validate_results` and checks the temporal/spatial/angular-range fields too. Run that script after changing anything here.

## Relationship to other components

- `gui/` (Python package `compton_suite.gui`) — the Tkinter GUI. Consumes
  this engine exclusively through
  `models/kascade/kascade_adapter.py` — this package has
  zero knowledge of the GUI/adapter layer by design ("as little
  interruption as possible" was the explicit goal when the GUI was split
  out). If you add a new `Results` field the GUI should see, thread it
  through here additively, then read it from `kascade_adapter.py` on the
  other side.
- `models/xigma_i/` — the other physics engine plugged into the same GUI,
  entirely independent of this code.
- `io/` (package `compton_suite.io`) — shared physical constants/pint
  registry/parameter-convention framework, depended on by this package
  (constants only, see above), the reverse direction from the gui
  relationship above.
