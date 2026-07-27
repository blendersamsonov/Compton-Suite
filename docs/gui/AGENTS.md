# Compton-GUIde

Standalone, model-agnostic Tkinter GUI for Compton-scattering physics
engines. Plugs in physics packages through a shared `ModelAdapter`
contract (`model_api.py`) instead of a hardcoded import, so this project
and the physics packages it drives don't depend on each other's
internals — only on the adapter shape. Python package name: `compton_guide`
(repo/brand name stylized `Compton-GUIde` — "GUI" capitalized on purpose).

Four models are currently registered (`compton_guide/models.py`):
- `kascade` (`models/kaskade/kascade.py`), an event-generator MC, always available (CPU only).
- `xigma-i` (the `xigma_i` package, GPU/cupy-only `Compton` class), shown greyed-out in the Model menu if cupy/CUDA isn't usable.
- `xigma-i-direct` (the `xigma_direct` package), brute-force per-particle binning, same GPU/cupy availability story as `xigma-i`.
- `analytical` (`models/analytical/`), a fast closed-form estimate, always available; also runs automatically as a preview alongside whichever other model is active.

## Layout

```
src/compton_guide/
  app.py              # ComptonGuideApp(tk.Tk) — the actual GUI. ~1180 lines.
  models.py           # discover_models() — model registry setup. Deliberately
                       # has NO tkinter/matplotlib import, so it (and anything
                       # built on it) can be exercised headlessly.
  model_api.py         # The contract: CommonResults, ModelCapabilities,
                       # ModelAdapter protocol, SampledSpectrum/BinnedSpectrum
                       # and their temporal/spatial/angular-range counterparts,
                       # validate_results(), MODEL_REGISTRY.
  physics_constants.py  # Re-export of compton_io.constants (C_LIGHT, HBAR,
                         # MEC2_EV, ...) for the GUI's local formula helpers.
  adapters/kascade_adapter.py  # KascadeAdapter — wraps kascade.py with zero
                                # changes to that package's own code.
scripts/
  run_gui.py           # Entry point: python3 scripts/run_gui.py
  headless_test.py     # No-display smoke test — see "Testing" below.
docs/new-features-plan.md  # Status table of which observable is ready /
                            # needs-adapter-change / needs-core-change, per
                            # model. Keep updated when adding new observables.
```

Note: `xigma_i`'s adapter (`gui_adapter.py`) lives in the `xigma_i` repo itself,
not here — it was already-completed integration work at the time this repo
was split out, and moving it would have touched that repo more than
necessary. `compton_guide` only ever does `from xigma_i import gui_adapter`;
decoupling is achieved without relocating that file.

## The ModelAdapter contract, and the bug it caused

`model_api.py` defines `SampledSpectrum`/`BinnedSpectrum`/`BinnedTemporalEnvelope`/
etc. `xigma_i/gui_adapter.py` (in the *other* repo) deliberately defines its
own **structurally-identical but not-the-same-class** local dataclasses,
rather than importing these — so that `xigma_i` doesn't have to depend on
this GUI project. This is intentional and documented in both files.

**Consequence: never use `isinstance(x, SampledSpectrum)` / `isinstance(x, BinnedSpectrum)`
etc. to distinguish "sampled" from "binned" results if both branches check a
specific class.** A single `isinstance(x, SampledSpectrum)` check with an
`else` for everything else is fine (real for kascade, and correctly false for
anything from xigma-i). But `if isinstance(x, A): ... elif isinstance(x, B): ...`
against *both* `model_api` classes will silently do nothing for xigma-i's
duck-typed equivalents. This exact bug hit `validate_results()` (raised a
bogus "unexpected type" error, reported as "error on unexpected
BinnedSpectrum") and three `app.py` render methods (silently rendered blank
tabs). Fixed by switching to duck-typing: check `hasattr(x, "weight")` (sampled)
vs. `hasattr(x, "dNdE_per_eV"/"rate"/"density")` (binned) instead. If you add
a new paired Sampled/Binned dataclass, follow the same pattern.

## Model-specific parameters (`extra_params()`)

Beyond the shared Electrons/Laser/Compton-photons panels (common to every
model), an adapter can declare extra numeric fields with no shared-panel
analogue via `ModelAdapter.extra_params() -> list[(label, default, key)]`
(same shape `add_field_grid` already consumes). `app.py`'s grey
"MODEL PARAMETERS" panel (`_build_model_params_panel`/`_rebuild_model_params_panel`)
rebuilds itself from this whenever the active model changes, feeding the
resulting values into the same flat `fields` dict passed to
`params_to_config`. `kascade` currently declares none (`[]`); `xigma-i`
declares `beta_ff`/`phi_pol` (its own extras with no `kascade` analogue).
Return `[]` if a model has nothing extra to add.

## Parameter semantics & units (`physics_params/`), and `compton_io`

`src/compton_guide/physics_params/` implements `Conventions-and-units.md`'s
design, but **the framework itself lives in `IO/` (package `compton_io`),
not here** — this package is a thin re-export
(`physics_params/__init__.py` imports `compton_io`'s `PhysicalQuantity`,
the `PhysicalMeaning`/`WidthConvention`/`TimeConvention`/
`AmplitudeConvention` enums, `canonical.py`'s conversion machinery,
`ParameterSpec`/`ModelSpec`, `adapt_to_model`, verbatim, so
`compton_guide.physics_params.PhysicalQuantity` is the literal same class
as `xigma_i.params.PhysicalQuantity`, not an independently-defined
look-alike). An earlier version of this gave each consumer its own copy
of the framework — structurally identical, not the same Python classes, so
a `PhysicalQuantity` built with one copy's enums failed
`validate_against_spec`'s `meaning` check against the other's `ModelSpec`.
`compton_io` exists specifically to eliminate that; see its own
`CLAUDE.md`.

Physical constants also moved: `physics_constants.py` is a thin re-export
of `compton_io.constants` (was its own hand-typed literal block before).
`adapters/kascade_adapter.py`'s `params_to_config` reads
`compton_io.constants.MEC2_EV`/`E_CHARGE`/`C_LIGHT` directly too, rather
than the live `kascade` module's own copies — one less independent
representation of `c` floating around this codebase.

**Both models' schemas now live in their own packages, not here.**
`xigma_i/params/spec.py`'s `XIGMA_SPEC`/`XIGMA_DIAGNOSTIC_SPEC` (see
`models/xigma/CLAUDE.md`, "Parameter semantics & units") and
`models/kascade/params/spec.py`'s `KASCADE_SPEC`/`KASCADE_DIAGNOSTIC_SPEC`
(moved out of this package's own `physics_params/schemas/`, which no
longer exists) — each model declares its own parameter contract instead of
the GUI declaring it on the model's behalf.

Both schemas were written by reading each engine's actual source
(`kascade.py`'s `laser_density`/`Config` field comments for `KASCADE_SPEC`;
`xigma_i/config.py`'s `set_laser_parameters` comment for `XIGMA_SPEC`), not
guessed — and turn out to declare the *same* convention (RMS of the
density/intensity profile, lengths standing in for durations, peak — not
RMS — `a0`) on every field both `Config`s share, which is why
`KASCADE_SPEC`'s docstring calls this "machine-checking" a fact rather
than fixing a real mismatch: both adapters' hand-written `params_to_config`
already agreed, this just makes that agreement explicit and enforced
instead of implicit and silently breakable.

**Partially wired into `params_to_config`** in both adapters:
`sigma_par_e`/`sigma_par_L` (the two genuinely raw, convention-ambiguous
duration inputs) go through `adapt_to_model`/`params_to_floats` against
each model's own spec; `sigma0_x`/`sigma0_y`/`sigma0_l` still do the
emittance/beta/Rayleigh-length arithmetic by hand, since they're derived
from other unambiguous fields rather than typed directly in an ambiguous
convention (see `docs/refactor/parameter-framework-and-collision-params.md`'s
"sigma0_x/sigma0_l wrinkle"). `scripts/physics_params_demo.py` is a
GPU-free, tkinter-free demo/self-check: builds **one** raw-input dict in
mixed conventions (a FWHM in µm, a duration in ps), adapts it to both
`XIGMA_SPEC` and `KASCADE_SPEC`, asserts the two models' outputs agree and
that the FWHM→sigma and duration→length conversions actually ran, and
asserts `compton_suite.models.kascade.params.PhysicalQuantity is
compton_suite.models.xigma_i.params.PhysicalQuantity` as a permanent
regression guard against the two re-export shims silently re-diverging
into independent copies again.

```bash
python3 scripts/physics_params_demo.py
```

Needs the dev-install (see root `CLAUDE.md`) so `xigma_i` and `compton_io`
are importable; needs neither cupy nor a GPU.

## Running it

```bash
python3 scripts/run_gui.py
```

Needs `numpy`, `matplotlib`, system Tk (`tkinter`), `pint` (for
`compton_io`), and — only if you want `xigma-i`/`xigma-i-direct` enabled
rather than greyed-out — `cupy` + a working CUDA setup.

On this dev machine specifically: system Python has no pip/cupy/matplotlib.
There's a conda env (`miniforge3`, env name `core`) that already has
cupy 14.0.1 + numpy + matplotlib + tkinter working against the local GPU:

```bash
conda run -n core --no-capture-output python3 scripts/run_gui.py
```

(`conda run` without `--no-capture-output` silently swallows stdout — always
pass it when you want to see anything.)

## Testing (headless, no display needed)

```bash
python3 scripts/headless_test.py                       # kascade only, cupy not required
conda run -n core --no-capture-output python3 scripts/headless_test.py   # + xigma-i, needs GPU
```

This calls the *exact* sequence the GUI's `on_start()` calls:
`discover_models() → params_to_config() → run() → validate_results()`, plus
the temporal/spatial/angular-distribution fields and
`spectrum_in_angular_range()`. It's the tool that caught the isinstance bug
above — run it after touching either adapter or either physics engine.
`xigma-i` reporting "unavailable" is expected and not a failure on a machine
without cupy/GPU.

## Adding a new GUI observable

Pattern established by the existing four (temporal envelope, spatial
distribution, angular distribution, angular-range spectrum):
1. Add a `Sampled*`/`Binned*` dataclass pair to `model_api.py` if the shape
   differs between an event-generator model (kascade) and a semi-analytic one
   (xigma-i); add a `supports_*` flag to `ModelCapabilities` (default `False`).
2. Populate it in `KascadeAdapter.run()` (usually cheap — kascade already has the
   raw per-photon arrays) and in `xigma_i/gui_adapter.py`'s `run_simulation`
   (check whether `core.Compton` already computes what you need internally
   before assuming a new kernel is required — `time_envelope` and the
   angular-range spectrum both turned out to need zero `core.py` changes;
   only spatial distribution genuinely needed new kernel work).
3. Add a tab in `app.py`'s `_build_plot_area`, a `_render_*` method following
   the existing duck-typing convention (not `isinstance` against both variants),
   and gate it via `_apply_model_capabilities`'s tab-disabling loop.
4. Extend `headless_test.py`'s `test_model()` to check the new field.
5. Update `docs/new-features-plan.md`'s status table.

## Known gaps

- `xigma_i`'s spatial-distribution normalization is self-consistently
  rescaled against `calculate_total()` rather than derived from first
  principles (see that repo's CLAUDE.md) — fine for visualization, not
  independently validated as an absolute physical calibration.
- No automated test for `app.py`'s actual Tkinter rendering (only the
  adapter/model layer is covered by `headless_test.py`) — testing the real
  widget tree needs a display (or Xvfb), which hasn't been set up.
- `Conventions-and-units.md` in this directory is implemented as `IO/`
  (framework + constants), re-exported here as
  `physics_params/`/`physics_constants.py`, plus, for xigma-i
  specifically, `xigma_i.params` (its own `XIGMA_SPEC`, built on
  `compton_io`) in that model's own package — but not yet wired into
  either adapter's `params_to_config`, so it doesn't change any current
  behavior yet. `kascade`'s own schema (`KASCADE_SPEC`) hasn't had the
  same schema-ownership move as xigma-i's, and still lives here.
