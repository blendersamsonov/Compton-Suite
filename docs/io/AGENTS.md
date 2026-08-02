# compton_suite.io

Shared physical constants, pint unit registry, parameter-semantics/
convention framework, and electron-bunch/laser-pulse representations for
the ComptonSuite toolkit: `compton_suite.gui` (`gui/`), `compton_suite.models.xigma_i`/
`compton_suite.models.delta`/`compton_suite.models.analytical` (`models/`), and `compton_suite.models.kascade` (`models/kascade/`).
See `src/compton_suite/io/__init__.py`'s module docstring for the full design
rationale.

## Why this exists

Each consumer used to hand-maintain its own copy of physical constants
(with a real ~1.6e-8 relative numeric disagreement between `xigma_i`'s
older-CODATA-vintage copy and the others, which already agreed with each
other), and, once more than one consumer had a parameter-semantics/
convention layer, its own copy of the enums/dataclasses describing it --
structurally identical between copies but not the same Python classes, so
a `PhysicalQuantity` built with one copy's enums failed validation against
another copy's `ModelSpec`. This package exists so there's exactly one
copy of each, that every consumer imports directly.

## Layout

```
src/compton_suite/io/
  __init__.py     # aggregate re-export
  constants.py    # physical constants, derived from units.py's pint
                   # registry (not hand-typed literals) -- SI block for
                   # compton_suite.gui/kascade, CGS-Gaussian views for xigma_i
  units.py        # the one shared pint.UnitRegistry + "light_time" context
  enums.py        # PhysicalMeaning, WidthConvention, TimeConvention,
                   # AmplitudeConvention
  quantities.py   # PhysicalQuantity (value + unit + meaning + convention)
  canonical.py     # one canonical convention+unit per PhysicalMeaning,
                    # to_canonical/from_canonical
  converters.py     # pure scalar-factor conversions (FWHM<->sigma, etc.)
  schema.py          # ParameterSpec/ModelSpec *types* (not any instance --
                       # each model owns its own ModelSpec in its own package)
  validation.py       # fail-fast error types + validate_quantity/
                       # validate_against_spec
  adapter.py           # adapt_to_model, params_to_floats
  bunch.py             # Bunch (raw macroparticle arrays, plus the
                        # GaussianElectronBeam it was sampled from/fit to,
                        # as Bunch.gaussian_fit), GaussianElectronBeam
                        # (gaussian_6d_waist v0.1 analytic contract -- also
                        # what a structured fit returns, no separate
                        # fit-output type), sample_gaussian_bunch/
                        # sample_gaussian_canonical (canonical sampling
                        # with mass-shell enforcement), fit_gaussian
                        # (structured Gaussian fitting with Twiss, chirp,
                        # dispersion, fit quality), drift/propagate/stream
                        # (vacuum propagation, analytically updating an
                        # attached gaussian_fit), evaluate_fit_quality
                        # (Mahalanobis, KS, log-likelihood metrics). No
                        # `*_from_shared_fields` factories -- callers build
                        # GaussianElectronBeam/GaussianParaxialLaser
                        # directly (PhysicalQuantity already does the
                        # conversion work).
  laser.py              # GaussianParaxialLaser (gaussian_paraxial v0.1
                         # analytic contract)
  photons.py             # Sampled*/Binned* spectrum, angular-spectrum,
                         # temporal-envelope, spatial-distribution
                         # dataclasses every model reports results through
  io_formats/
    sdds.py              # elegant .ele load/save for MacroBunch
    yaml_spec.py          # this package's own YAML I/O for beam/laser
specs/
  electron_beam_io_v0.1_full.md / _short.md   # gaussian_6d_waist spec
  gaussian_paraxial_laser_io_v0.1.md / _short.md  # gaussian_paraxial spec
tests/
  test_constants.py    # cross-checks constants.py's SI/CGS pairs
  test_conversions.py  # self-contained framework round-trip checks
  test_bunch.py         # MacroBunch/GaussianElectronBeam round trips
  test_laser.py          # GaussianParaxialLaser derived-quantity checks
  test_io_formats.py      # .ele and YAML round trips
```

**What does NOT live here**: any specific model's `ModelSpec` *instance*
(`compton_suite.models.xigma_i.params.spec.XIGMA_SPEC`/
`XIGMA_DIAGNOSTIC_SPEC`, `compton_suite.models.kascade.params.spec.
KASCADE_SPEC`/`KASCADE_DIAGNOSTIC_SPEC`) -- those are model contracts
owned by each model's own package, not shared framework.

## Naming

Package is `compton_suite.io` (import name) / `compton-suite` (pyproject name).
Note the difference from `compton_suite.gui` (the GUI) -- easy to typo/
autocomplete-confuse; double-check import statements when working across
both.

## Testing

```bash
python3 tests/test_constants.py
python3 tests/test_conversions.py
python3 tests/test_bunch.py
python3 tests/test_bunch_improvements.py  # New: canonical sampling, fitting, drift
python3 tests/test_laser.py
python3 tests/test_io_formats.py
```

No cupy/GPU/tkinter needed -- pure `compton_suite.io` + `pint` + `numpy` + `pyyaml`.
