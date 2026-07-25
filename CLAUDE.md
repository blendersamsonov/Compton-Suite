# compton-suite

Shared physical constants, pint unit registry, and parameter-semantics/
convention framework for the Compton-scattering GUI + physics-engine suite:
`compton-gui` (`compton_guide`), `xigma_i` (`../git-repo-claude` or
wherever that checkout lives), and `kascade` (`../MC-Kost`). See
`src/compton_suite/__init__.py`'s module docstring for the full design
rationale.

## Why this exists

Each of the three consumer repos used to hand-maintain its own copy of
physical constants (with a real ~1.6e-8 relative numeric disagreement
between `xigma_i`'s older-CODATA-vintage copy and the other two, which
already agreed with each other), and, once more than one repo had a
parameter-semantics/convention layer, its own copy of the enums/dataclasses
describing it -- structurally identical between copies but not the same
Python classes, so a `PhysicalQuantity` built with one copy's enums failed
validation against another copy's `ModelSpec`. This package exists so
there's exactly one copy of both, that every consumer imports directly.

## Layout

```
src/compton_suite/
  __init__.py     # aggregate re-export
  constants.py    # physical constants, derived from units.py's pint
                   # registry (not hand-typed literals) -- SI block for
                   # compton_guide/kascade, CGS-Gaussian views for xigma_i
  units.py        # the one shared pint.UnitRegistry + "light_time" context
  enums.py        # PhysicalMeaning, WidthConvention, TimeConvention,
                   # AmplitudeConvention
  quantities.py   # PhysicalQuantity (value + unit + meaning + convention)
  canonical.py     # one canonical convention+unit per PhysicalMeaning,
                   # to_canonical/from_canonical
  converters.py    # pure scalar-factor conversions (FWHM<->sigma, etc.)
  validation.py     # fail-fast error types + validate_quantity/
                     # validate_against_spec
  schema.py          # ParameterSpec/ModelSpec *types* (not any instance --
                       # each model owns its own ModelSpec in its own repo)
tests/
  test_constants.py    # cross-checks constants.py's SI/CGS pairs
  test_conversions.py  # self-contained framework round-trip checks
```

**What does NOT live here**: any specific model's `ModelSpec` *instance*
(`xigma_i.params.spec.XIGMA_SPEC`/`XIGMA_DIAGNOSTIC_SPEC`,
`compton_guide.physics_params.schemas.kascade.KASCADE_SPEC`) -- those are
model contracts owned by each model's own repo, not shared framework.

## Naming

Package is `compton_suite` (import name) / `compton-suite` (pyproject
name, directory name). Note the one-word difference from `compton_guide`
(the GUI) -- easy to typo/autocomplete-confuse; double-check import
statements when working across both.

## How consumers find this package

**Not pip-installed by anyone.** Every cross-repo import in this whole
suite (kascade/xigma_i into compton-gui, and now compton-suite into all
three) goes through a small, physically-duplicated-per-consumer, content-
based sys.path bootstrap -- looks for a marker file inside sibling
directories rather than assuming a stable directory name, because sibling
checkout names genuinely aren't stable across machines (a worktree, a
fork, a differently-named clone). This package's own marker file is
`src/compton_suite/constants.py`. See:

- `compton_guide/src/compton_guide/bootstrap.py` -- the original pattern,
  extended with a third marker/env-var pair (`COMPTON_GUIDE_COMPTON_SUITE_SRC`).
- `xigma_i/src/xigma_i/_bootstrap.py` -- same algorithm, env var
  `XIGMA_COMPTON_SUITE_SRC`. Called eagerly (not lazily) from both
  `xigma_i/config.py` and `xigma_i/params/__init__.py`, since `xigma_i`'s
  own physical constants now come from here -- unlike compton-gui's
  degrade-to-greyed-out story for a missing physics engine, there's no
  graceful fallback for missing constants, so this raises a clear
  `ImportError` rather than failing silently.
- `MC-Kost/_bootstrap.py` -- same pattern, env var
  `KASCADE_COMPTON_SUITE_SRC`.

## Local-only, no GitHub remote (yet)

This repo is `git init`-only at present -- no GitHub remote, no push.
Local development only until that's explicitly requested.

## Testing

```bash
python3 tests/test_constants.py
python3 tests/test_conversions.py
```

No cupy/GPU/tkinter needed -- pure `compton_suite` + `pint` + `numpy`.
