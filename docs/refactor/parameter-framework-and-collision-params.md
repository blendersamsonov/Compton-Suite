# CollisionParams pruning + finishing the PhysicalQuantity/ModelSpec migration

**Status**: Planned -- not yet implemented.
**Location**: `docs/refactor/parameter-framework-and-collision-params.md`

## Context

Surfaced during the 2026-07-27 config-unification session (PR #1,
`worktree-core-simulation-api-refactor`): while explaining why
`compton_suite.io.quantities.PhysicalQuantity` is only exercised in tests/
`scripts/physics_params_demo.py`, and why `compton_suite.io.collision.
CollisionParams` carries fields that look redundant with `GaussianElectronBeam`,
two genuinely separate, independently-actionable follow-ups fell out:

1. **`CollisionParams` carries 9 fields nothing downstream reads.** Verified
   by exhaustive grep across `src/`, `tests/`, `scripts/` for `params.<field>`
   -- see Phase 1.
2. **The `PhysicalQuantity`/`ModelSpec` parameter-convention framework
   (`io/quantities.py`, `io/schema.py`, `io/adapter.py`'s `adapt_to_model`/
   `params_to_floats`) is fully built, tested, and demonstrated
   end-to-end in `scripts/physics_params_demo.py` -- but no real GUI adapter
   (`kascade_adapter.py`, `xigma_i/gui_adapter.py`, `delta/gui_adapter.py`)
   actually calls it.** This is a known, tracked gap (root `AGENTS.md` roadmap
   items 1 and 5 both flag it explicitly), not something abandoned -- see
   Phase 2.

These two are independent: Phase 1 is a small, mechanical, low-risk deletion
confined to one file (`io/collision.py`). Phase 2 is real design work with an
honest nuance that needs deciding before writing code (see Phase 2's "the
sigma0_x/sigma0_l wrinkle" section) -- do not treat it as a mechanical
find-and-replace.

## Phase 1: Prune `CollisionParams`'s dead fields

### Evidence

`grep -rn "\bparams\.<field>\b" src/ tests/ scripts/` for each of
`CollisionParams`'s 20 fields (`io/collision.py:83-118`) found **zero**
downstream reads, anywhere in the repo, for:

- `emit_x`, `emit_y` -- electron geometric emittance
- `sigma_ez` -- electron bunch length (CGS)
- `beta_x`, `beta_y` -- electron beta functions (CGS)
- `sigma_thx`, `sigma_thy` -- electron angular divergence
- `lambda_l` -- laser wavelength itself (only its derivatives `omega_las`/
  `k0_las`/`Wph` are read)
- `delta_x`, `delta_y`, `delta_z` -- foci displacement. `particles.py:328`
  has its own comment admitting this: "axis/focus left at their defaults
  (head-on, no offset), matching this pipeline's own convention (see
  `gaussian_pulse_envelope`'s docstring on why `CollisionParams.delta_x/y/z`
  aren't threaded through here)."

Everything else (`N_e`, `sigma_ex`, `sigma_ey`, `sigma_lr0`, `sigma_lz`,
`omega_las`, `k0_las`, `Wph`, `N_l`, `a0`, `beta_ff`, `ellipticity`, `device`/
`xp`/`asnumpy`) is genuinely load-bearing -- confirmed real reads in
`particles.py`, `deposition.py`, `spectrum4d.py`, `tabulated_engine.py`.
`N_e` backs macroparticle weighting (`weight = N_e / n_particles`,
`particles.py:170`); `sigma_ex`/`sigma_ey` back a grid-sizing heuristic bound
(`particles.py:98-99`, `max(params.sigma_ex, params.sigma_lr0)`), not real
physics.

**Root cause** (for whoever picks this up, so it isn't re-litigated): the
laser side of `CollisionParams` is a continuous analytic field the
electrons fly through, so its ensemble-level shape parameters are real
physics input. The electron side used to presumably be modeled the same way
(an analytic Gaussian overlap); once `particles.py`'s `_normalise_bunch`
switched to consuming a real, already-sampled `MacroBunch` (reading
`macrobunch.x/y/z/thx/thy/gamma` directly), the bunch's own *statistical
shape* parameters became fossils that nobody pruned.

### Also confirmed safe

- `CollisionParams(` is constructed in exactly one place, `build_params()`
  itself (`io/collision.py:177`) -- no test or other caller constructs it
  directly with these kwargs.
- `geometry.crossing_angle_rad` is never read inside `build_params` either
  (only `delta_x_m`/`delta_y_m`/`delta_z_m` are, at lines 184-185) -- so once
  `delta_x/y/z` are dropped, the `geometry: InteractionGeometry` parameter
  to `build_params` becomes fully inert *for this convention*.

### Design decision: keep the `geometry` parameter, drop the fields it feeds

Do **not** remove `build_params`'s `geometry` parameter -- that would be a
bigger, more disruptive signature change across 3 call sites
(`xigma_i/gui_adapter.py::run_simulation`, `delta/gui_adapter.py::
run_simulation`, `validation/scenarios.py::build_params_for_xigma`) for a
benefit (removing one unused parameter) that isn't really motivated by
dead-field pruning alone, and a future crossing-offset feature would need
the parameter back anyway. Just stop deriving/storing `delta_x/y/z` (and the
other 8 fields) as `CollisionParams` fields. If/when xigma_i actually grows
offset support, reintroduce these with a real consumer, not preemptively.

### Steps

Touches exactly one file, `src/compton_suite/io/collision.py`:

1. Delete these fields from the `CollisionParams` dataclass (lines
   ~89-116): `emit_x`, `emit_y`, `sigma_ez`, `beta_x`, `beta_y`,
   `sigma_thx`, `sigma_thy`, `lambda_l`, `delta_x`, `delta_y`, `delta_z`.
2. In `build_params()`, delete the now-unused local computations that fed
   only those fields (`emit_x`, `emit_y`, `sigma_ez`, `beta_x`, `beta_y`,
   `sigma_thx`, `sigma_thy`, `lambda_l` locals -- but keep `lambda_l` as a
   *local* variable if `omega_las`/`k0_las` still need it for their own
   formula, just don't store it on `CollisionParams`) and the
   `delta_x=...`/`delta_y=...`/`delta_z=...` kwargs in the final
   `CollisionParams(...)` call.
3. No other file changes -- every call site passes `beam`/`laser`/
   `geometry` positionally/by keyword into `build_params`, never touches
   the dropped fields on the returned `CollisionParams` (confirmed by the
   grep above), so nothing else moves.

### Verification

- `python3 scripts/headless_test.py` -- all 4 models ALL PASS (no changes
  expected, `CollisionParams` is xigma_i/delta-internal).
- `python3 src/compton_suite/validation/run_cross_validation.py` -- diff
  full stdout before/after; expect byte-identical (same technique used to
  verify the config-unification PR: `git stash` the change, capture
  output, pop, re-run, `diff`).

## Phase 2: Finish wiring `PhysicalQuantity`/`ModelSpec` into real adapters

### What the framework actually is (for context, see prior session's
explanation in conversation history if resuming without it)

- `PhysicalQuantity(value, unit, meaning, convention)` -- a value tagged
  with what physical quantity it is and which convention it's expressed in
  (e.g. is "a width" a sigma, a FWHM, or a 1/e^2 radius?).
- `ModelSpec = dict[str, ParameterSpec]` -- a model's declared contract:
  "my field X means Y, in convention Z, unit W." `XIGMA_SPEC`
  (`models/xigma_i/params/spec.py`) and `KASCADE_SPEC`
  (`gui/physics_params/schemas/kascade.py`) currently declare this for 5
  fields: `sigma0_x`, `sigma0_y`, `sigma_par_e`, `sigma0_l`, `sigma_par_L`
  (all `WidthConvention.SIGMA_INTENSITY_RMS`/`TimeConvention.
  SIGMA_INTENSITY_RMS`).
- `adapt_to_model(canonical_params: dict[str, PhysicalQuantity], spec) ->
  dict[str, PhysicalQuantity]` + `params_to_floats(...) -> dict[str, float]`
  (`io/adapter.py`) round-trip each `PhysicalQuantity` through a canonical
  representation and back out in the spec's declared convention/unit,
  stripping to a plain float at the end.
- `scripts/physics_params_demo.py` demonstrates the intended full pipeline
  end to end and is the reference to copy the *pattern* from -- but see the
  wrinkle below before copying it verbatim.

### The sigma0_x/sigma0_l wrinkle -- read before writing code

`physics_params_demo.py`'s `raw_inputs` treats `sigma0_x`/`sigma0_y`/
`sigma0_l` as if they were raw, directly-user-typed, convention-ambiguous
widths (e.g. `sigma0_l` given as a **FWHM in micrometers**). That is *not*
how any real adapter's GUI fields work today:

- `sigma_par_e`/`sigma_par_L` **are** genuinely raw, convention-ambiguous
  inputs today: each adapter's `params_to_config` takes a duration in
  picoseconds off a GUI field (`bunch_duration_ps`/`pulse_duration_ps`) and
  converts it to a length via `C_LIGHT * duration_s`
  (`kascade_adapter.py:116-117`, `xigma_i/gui_adapter.py:342,352`,
  `delta/gui_adapter.py:246,256` -- current line numbers, re-grep
  before executing since they will have shifted again). This is a
  mechanical, direct fit for wrapping the raw `*_ps` GUI value as
  `PhysicalQuantity(value, "picosecond", MEANING, TimeConvention.
  SIGMA_INTENSITY_RMS)` and calling `adapt_to_model`/`params_to_floats` --
  same conversion, now framework-verified instead of hand-derived.
- `sigma0_x`/`sigma0_y` are **not** raw inputs at all -- every adapter
  computes them from two *other*, unambiguous GUI fields:
  `sigma0_x = sqrt(emit_x * beta_x)` where `emit_x`/`beta_x` come from
  `emit_x_mmmrad`/`beta_x_m` fields (normalized emittance and a beta
  function, neither of which has a "which convention" choice -- they're
  well-defined physical quantities on their own). The result is already in
  the SI meters/sigma convention by construction of the formula.
- `sigma0_l` similarly comes from a **Rayleigh length** GUI field
  (`rayleigh_length_m`, itself unambiguous) via the fixed inverse-Rayleigh
  formula `sigma0_l = 0.5*sqrt(R_sf*lambda_L/pi)`, not from a directly-typed
  width in some convention.

So `physics_params_demo.py`'s framing of `sigma0_x`/`sigma0_l` as raw
convention-ambiguous PhysicalQuantity inputs describes a **hypothetical**
alternate GUI (e.g. "type the beam width directly, pick your convention")
that doesn't exist in the current field set -- it is not describing today's
`rayleigh_length_m`/`emit_x_mmmrad`/`beta_x_m`-based GUI.

**This is a real decision to make, not a detail to gloss over**, before
writing Phase 2 code:

- **Option A (narrower, lower-risk, recommended first slice)**: wire
  `PhysicalQuantity`/`adapt_to_model` only for `sigma_par_e`/`sigma_par_L`
  (the two fields with genuine raw-convention ambiguity today), across all
  three adapters plus a new `KASCADE_SPEC`-equivalent living in
  `models/kascade/` (see below). Leave `sigma0_x`/`sigma0_y`/`sigma0_l` as
  hand-derived-then-optionally-*asserted*-against-the-spec (a lighter,
  non-value-changing use: after computing `sigma0_x` the old way, optionally
  wrap it and confirm `adapt_to_model` round-trips it to the same number --
  a machine-checked consistency assertion, not a real conversion).
- **Option B (larger scope)**: also change the GUI to expose a
  directly-typed width field (with a convention selector) for
  `sigma0_x`/`sigma0_y`/`sigma0_l`, making the `physics_params_demo.py`
  framing literally true for those fields too. This is a real UI-facing
  design change (new field type, new adapter logic to choose between
  "derive from emit/beta" vs "type directly"), well beyond "finish wiring
  the existing framework" -- do not do this as part of Phase 2 without a
  separate, explicit go-ahead.

This plan recommends **Option A** as Phase 2's actual scope.

### kascade has no `ModelSpec` at all yet

`AGENTS.md` item 5 already flags this: `KASCADE_SPEC`
(`gui/physics_params/schemas/kascade.py`) is GUI-owned, not model-owned --
unlike `XIGMA_SPEC`, which already moved into `models/xigma_i/params/` (see
that module's own docstring: "moved here so this model declares its own
parameter contract directly"). Before kascade's `params_to_config` can use
`adapt_to_model`, either:
- move `KASCADE_SPEC` from `gui/physics_params/schemas/kascade.py` into a
  new `models/kascade/params/spec.py` (matching xigma_i's already-moved
  pattern), or
- keep it GUI-owned for now and just start calling it from
  `kascade_adapter.py` (smaller diff, defers the ownership move to a
  separate cleanup).

Recommend the move (first option) -- `models/kascade/AGENTS.md` already
documents this as "a materially bigger lift... stays a documented, undone
follow-up," and doing the move now means all four models finally have
symmetric ownership of their own parameter contracts.

### Steps (Option A scope)

For each of `kascade_adapter.py`, `xigma_i/gui_adapter.py`,
`delta/gui_adapter.py`'s `params_to_config`:

1. Replace the two-line hand conversion for `sigma_par_e`/`sigma_par_L`
   (`C_LIGHT * (g("..._ps") * 1e-12)`) with:
   ```python
   raw = {
       "sigma_par_e": PhysicalQuantity(g("bunch_duration_ps"), "picosecond",
                                       PhysicalMeaning.BUNCH_LENGTH,
                                       TimeConvention.SIGMA_INTENSITY_RMS),
       "sigma_par_L": PhysicalQuantity(g("pulse_duration_ps"), "picosecond",
                                        PhysicalMeaning.PULSE_DURATION,
                                        TimeConvention.SIGMA_INTENSITY_RMS),
   }
   adapted = params_to_floats(adapt_to_model(raw, <THIS_MODEL>_SPEC))
   sigma_par_e, sigma_par_L = adapted["sigma_par_e"], adapted["sigma_par_L"]
   ```
   (exact `PhysicalMeaning`/convention enum names to confirm against
   `io/quantities.py` at execution time -- this plan sketches the pattern,
   not a verbatim diff, since the file has moved/changed several times
   already this project's lifetime).
2. Leave every other field's arithmetic untouched (magic-literal cleanup
   for `eps0`/`N_e`/`lambda_L`/`pulse_energy_J`/`delta_x/y/z` is a separate,
   lower-priority cleanup -- see "Explicitly out of scope" below).
3. Move `KASCADE_SPEC` to `models/kascade/params/spec.py` (see prior
   section) and update `kascade_adapter.py`'s import.
4. Update `scripts/physics_params_demo.py` if the two specs' field sets
   changed shape, and add `sigma_par_e`/`sigma_par_L` PhysicalQuantity
   round-trip assertions there if not already covered (they already are,
   per the existing demo -- just re-verify after any spec changes).

### Verification

- `python3 scripts/physics_params_demo.py` -- still passes, now also
  exercising the real adapters' code path indirectly if the demo is
  extended to call `params_to_config` itself (optional, stronger check)
  rather than only the spec objects directly.
- `python3 scripts/headless_test.py` -- all 4 models ALL PASS.
- `python3 src/compton_suite/validation/run_cross_validation.py` -- diff
  before/after; expect byte-identical (this is meant to be a pure
  restructuring: same numeric conversion, now framework-verified instead of
  hand-derived).
- Manually diff a `params_to_config` call's output `Config` before/after for
  a couple of representative field dicts, confirming
  `sigma_par_e`/`sigma_par_L` match to floating-point precision, not just
  "close enough."

## Explicitly out of scope for both phases

- The remaining magic-literal unit arithmetic in `params_to_config`
  (`1e-6`, `1e-9`, `510_998.950`/`2.99792458e8` hardcoded instead of
  imported from `compton_suite.io.constants`, `eps0`/`N_e`/`lambda_L`/
  `pulse_energy_J`/`delta_x/y/z` conversions) -- these are simple,
  unambiguous unit conversions, not convention choices, so they don't need
  `PhysicalQuantity`; replacing them with `pint`-based conversions is a
  separate, lower-priority cleanup already noted in the previously-executed
  config-unification plan (PR #1) as deliberately deferred.
- The GUI-exposing-a-directly-typed-convention-choice-field idea (Option B
  above) -- a real UI/UX decision, not a refactor.
- The `xigma_i` `a0`-formula discrepancy against `GaussianParaxialLaser.
  a0_focus` (~49%, `validation/tier0_wiring.py`) -- unrelated, already
  tracked as its own open item in `docs/refactor/core-simulation-api.md`.

## Where this fits relative to other in-flight/recent work

- Builds on PR #1 (`worktree-core-simulation-api-refactor`, 2026-07-27):
  Config-unification onto `InteractionParameters` + GUI spread-box removal
  + stale top-level `validation/` deletion.
- Complements (does not duplicate) `docs/refactor/core-simulation-api.md`'s
  "Still open" list (GUI-as-thin-consumer, the a0-formula discrepancy) --
  this doc's two phases are additional, more narrowly-scoped items that
  surfaced from the same investigation, not a restatement of that doc.
