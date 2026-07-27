# Graph Report - param-framework-collision-params-refactor  (2026-07-27)

## Corpus Check
- 94 files · ~101,830 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1072 nodes · 2187 edges · 47 communities (42 shown, 5 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 194 edges (avg confidence: 0.53)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d681118a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Scenario
- compton_io/__init__.py
- ComptonGuideApp
- deposition.py
- particles.py
- kaskade/kascade.py
- CommonResults
- XigmaDirectAdapter
- GaussianElectronBeam
- xigma_i/gui_adapter.py
- MacroBunch
- cache.py
- headless_test.py
- CollisionParams pruning + finishing the PhysicalQuantity/ModelSpec migration
- propagation.py
- KascadeAdapter
- InteractionGeometry
- AnalyticalConfig
- BinnedSpectrum
- sample_gaussian_bunch
- laser.py
- test_laser.py
- bunch.py
- UnavailableAdapter
- run_multiphoton_chain
- analytical.py
- compton_suite/__init__.py
- detect_device
- test_constants.py
- AnalyticalAdapter
- XigmaAdapter
- test_analytical.py
- InteractionParameters
- compton-io
- schemas/__init__.py
- adapters/__init__.py
- compton_guide/__init__.py
- kascade
- xigma-direct
- xigma-i
- XigmaAdapter
- Compton-GUIde
- New GUI observables: status per model
- opencode.json
- graphify.js
- ComptonSuite model-agnostic-core refactor: status

## God Nodes (most connected - your core abstractions)
1. `MacroBunch` - 54 edges
2. `GaussianElectronBeam` - 44 edges
3. `GaussianParaxialLaser` - 41 edges
4. `Config` - 40 edges
5. `Scenario` - 39 edges
6. `ComptonGuideApp` - 38 edges
7. `DirectConfig` - 38 edges
8. `CommonResults` - 37 edges
9. `Config` - 37 edges
10. `ModelCapabilities` - 29 edges

## Surprising Connections (you probably didn't know these)
- `run()` --calls--> `check()`  [INFERRED]
  src/compton_suite/validation/tier0_wiring.py → scripts/headless_test.py
- `test_derived_quantities_are_sane()` --calls--> `validate()`  [EXTRACTED]
  tests/io_tests/test_bunch.py → src/compton_suite/io/bunch.py
- `test_laser_from_shared_fields_round_trips()` --calls--> `laser_from_shared_fields()`  [EXTRACTED]
  tests/io_tests/test_laser.py → src/compton_suite/io/laser.py
- `sample_electrons_for()` --calls--> `beam_from_shared_fields()`  [EXTRACTED]
  scripts/headless_test.py → src/compton_suite/io/bunch.py
- `sample_electrons_for()` --calls--> `sample_gaussian_bunch()`  [EXTRACTED]
  scripts/headless_test.py → src/compton_suite/io/bunch.py

## Import Cycles
- None detected.

## Communities (47 total, 5 thin omitted)

### Community 0 - "Scenario"
Cohesion: 0.08
Nodes (6): DirectConfig, ndarray, ``electrons`` is required: electron sampling is the caller's job,     not this a, SI-unit physics config, trimmed to what Stage 0 (particles.py)     needs -- no S, run_simulation(), _theta_grid()

### Community 1 - "compton_io/__init__.py"
Cohesion: 0.08
Nodes (57): Enum, ModelSpec, Quantity, Thin re-export of ``compton_suite.io``'s parameter-semantics/unit normalisation, adapt_to_model(), params_to_floats(), Canonical params -> a specific model's own convention/units.  This is the last s, ``canonical_params`` may be in any convention/unit as long as the     meaning ma (+49 more)

### Community 2 - "ComptonGuideApp"
Cohesion: 0.07
Nodes (74): Shared comparison metric for the cross-model validation suite.  Promoted (moved, s, spec, spec_ref: 1D arrays of equal shape (same grid for both     spectra -- r, Linear-interpolate spec_src(s_src) onto s_ref, zero outside s_src's range., resample_to(), window_integrated_relative_error(), User-curated bank of additional cross-validation parameter sets.  Deliberately e, Cross-model validation suite -- entry point.  Runs kascade/xigma-i/xigma-i-direc, Tier 0 (config-only, no run needed) + Tier 1/2 (share one run of     each model) (+66 more)

### Community 3 - "deposition.py"
Cohesion: 0.07
Nodes (28): BinnedSpectrum, Binned dN/dE spectrum (semi-analytic/tabulated models)., _attach_private_cache(), capabilities(), extra_choices(), extra_params(), _float(), ParamError (+20 more)

### Community 4 - "particles.py"
Cohesion: 0.06
Nodes (26): Raise NotImplementedError if capabilities().supports_ele_file_io is False., Raise NotImplementedError if capabilities().supports_ele_file_io is False., fit_gaussian(), MacroBunch, Draw macroparticles from a :class:`GaussianElectronBeam`.      Independent facto, Fit a :class:`GaussianElectronBeam` (at its waist) from raw macroparticles., Raw, engine-agnostic macroparticle representation. SI units.      ``x``/``y``/``, sample_gaussian_bunch() (+18 more)

### Community 5 - "kaskade/kascade.py"
Cohesion: 0.07
Nodes (37): _array_module(), build_table(), build_table_streaming(), _cell_indices(), check_accumulation_precision(), _deposit(), deposit_cic(), deposit_nearest() (+29 more)

### Community 6 - "CommonResults"
Cohesion: 0.06
Nodes (55): gaussian_pulse_envelope(), Shared spatiotemporal Gaussian-pulse envelope -- the piece ``compton_io.laser.Ga, Photon-density envelope of a Gaussian laser pulse at an arbitrary     point in s, ballistic_position_simultaneous(), ballistic_position_z0_reference(), laser_overlap_time_window(), propagate(), Ballistic (straight-line, no-acceleration) electron-beam propagation, and a time (+47 more)

### Community 7 - "XigmaDirectAdapter"
Cohesion: 0.15
Nodes (16): ModelAdapter for the fast, always-on analytical model.  Duck-typed against ``com, angle_integrated_spectrum(), estimate_spectrum_width(), estimate_yield(), ndarray, Fast, closed-form Compton-source physics: total yield, angle-integrated spectrum, Cheap analytic total-photon-yield estimate from an overlap integral     between, Cheap analytic estimate of the collimated-spectrum FWHM (in units of     the Com (+8 more)

### Community 8 - "GaussianElectronBeam"
Cohesion: 0.09
Nodes (10): beta_star_from_sigma_emit(), divergence_from_sigma_emit(), GaussianElectronBeam, Validate a :class:`GaussianElectronBeam` per spec Sec. 16.      Raises ``ValueEr, Beta function at the waist, ``sigma**2 / emit_geom`` -- same units in     as out, RMS divergence at the waist, ``emit_geom / sigma`` -- a dimensionless     angle,, The ``gaussian_6d_waist`` v0.1 I/O contract.      A 6D factorized Gaussian defin, validate() (+2 more)

### Community 9 - "xigma_i/gui_adapter.py"
Cohesion: 0.12
Nodes (15): detect_device(), Auto-detect which backend to use: a real CUDA GPU via cupy if     available, els, _attach_private_cache(), available(), extra_choices(), _float(), ParamError, Exception (+7 more)

### Community 10 - "MacroBunch"
Cohesion: 0.05
Nodes (35): KASCADE: Sequential multi-photon inverse-Compton Monte Carlo engine., beta_of(), build_lambda_grid(), Config, cos_collision(), doppler_D(), invert_lambda(), kn_over_thomson() (+27 more)

### Community 12 - "headless_test.py"
Cohesion: 0.07
Nodes (29): 0. Краткое резюме, 10.1. Что уже работает, 10.2. Что нужно сделать для первого сравнения кодов, 10.3. Что должно войти в следующую версию паспорта, 10. Текущий статус и ближайшие шаги, 11. Итоговая оценка готовности, 1. Карточка кода / метода, 2. Назначение и роль в проекте (+21 more)

### Community 13 - "CollisionParams pruning + finishing the PhysicalQuantity/ModelSpec migration"
Cohesion: 0.12
Nodes (16): Also confirmed safe, CollisionParams pruning + finishing the PhysicalQuantity/ModelSpec migration, Context, Design decision: keep the `geometry` parameter, drop the fields it feeds, Evidence, Explicitly out of scope for both phases, kascade has no `ModelSpec` at all yet, Phase 1: Prune `CollisionParams`'s dead fields (+8 more)

### Community 14 - "propagation.py"
Cohesion: 0.07
Nodes (26): 10. Error Handling, 11. Extensibility, 12. Non-Goals, 1. Enums (enums.py), 2. Quantity Wrapper (quantities.py), 3. Canonical Representation (canonical.py), 4. Conversion Engine (converters.py), 5. Schema Definition (schema.py) (+18 more)

### Community 15 - "KascadeAdapter"
Cohesion: 0.09
Nodes (21): 10. Энергетический разброс, 11. Заряд, число электронов и пиковый ток, 12. Пиковая плотность, 13. 6D гауссово распределение v0.1, 14. Минимальный пример входного файла, 15. Рекомендуемые derived output-параметры, 16. Проверки валидности input, 17. Не входит в v0.1 (+13 more)

### Community 16 - "InteractionGeometry"
Cohesion: 0.10
Nodes (20): AGENTS.md, Architecture, Conventions, Convergence testing, Current state, Environment, GUI-facing engine -- `tabulated_engine.py`, GUI integration (`gui_adapter.py`) (+12 more)

### Community 17 - "AnalyticalConfig"
Cohesion: 0.18
Nodes (20): cache_key(), clear_cache(), get_or_compute(), _git_head_hash(), _git_is_dirty(), list_cache_entries(), load(), _paths() (+12 more)

### Community 18 - "BinnedSpectrum"
Cohesion: 0.10
Nodes (19): 10. Минимальный пример входного файла, 11. Рекомендуемые derived output-параметры, 12. Проверки валидности input, 13. Не входит в v0.1, 14. Mermaid-схема v0.1, 1. Назначение, 2. Геометрическая конвенция, 3.1. Таблица входных параметров (+11 more)

### Community 19 - "sample_gaussian_bunch"
Cohesion: 0.14
Nodes (6): GaussianParaxialLaser, ``a0_at(z_m)**2`` -- the mean-square amplitude.          Consumers that only eve, A paraxial Gaussian laser pulse. SI units.      ``waist_rms_x_m``/``waist_rms_y_, Photon count in the pulse: N_L = pulse_energy_J / (hbar*omega0)., Normalized vector potential at ``z_m``, from the exact SI         plane-wave rel, _baseline()

### Community 20 - "laser.py"
Cohesion: 0.24
Nodes (7): calculate_angular_spectrum_4d(), _get_numba(), get_spectrum_kernel_4d_cpu(), CPU/numba equivalent of spectrum4d.py's GPU rawkernel (spectrum_kernel_4d), used, Lazily compiles and caches spectrum_kernel_4d_cpu (see module docstring)., Stage 2: `spectrum_kernel_4d`, the GPU kernel that turns a Stage 1 4D overlap ta, Host-side driver for spectrum_kernel_4d, the Stage-2 analogue of     Compton.cal

### Community 21 - "test_laser.py"
Cohesion: 0.20
Nodes (13): check(), FakeVar, main(), make_fields(), Same mechanism app.py's on_start() uses: whichever model is     'selected', the, Stand-in for tk.StringVar -- params_to_config only ever calls .get()., sample_electrons_for(), test_model() (+5 more)

### Community 22 - "bunch.py"
Cohesion: 0.14
Nodes (11): CODATA-style physical constants used by the GUI's local formula helpers (peak_a0, Physical constants, derived from ``units.py``'s pint registry rather than hand-t, Laser-pulse representation: the ``gaussian_paraxial`` v0.1 I/O contract (``specs, Validate a :class:`GaussianParaxialLaser` per spec Sec. 12.      Raises ``ValueE, validate(), Cross-checks for laser.py's GaussianParaxialLaser.  No cupy/GPU/tkinter needed., test_defocused_interaction_intensity_is_lower(), test_derived_quantities_are_sane() (+3 more)

### Community 23 - "UnavailableAdapter"
Cohesion: 0.17
Nodes (22): Protocol, ModelAdapter, ModelCapabilities, Any, Model-agnostic contract between app.py and physics-engine adapters.  This module, Return (True, "") if the model can actually be run right now, else         (Fals, Model-specific fields with no shared-panel analogue         (kascade's Electrons, Optional: return a dict mapping parameter keys to allowed string values (+14 more)

### Community 24 - "run_multiphoton_chain"
Cohesion: 0.18
Nodes (11): Electron-bunch representations.  Two distinct types, matching the distinction th, External-format I/O for compton_io's bunch/laser representations., Elegant / SDDS ``.ele`` file I/O for :class:`compton_io.bunch.MacroBunch`.  Relo, load_electron_beam(), load_laser(), YAML I/O for the ``gaussian_6d_waist``/``gaussian_paraxial`` v0.1 formats define, save_electron_beam(), save_laser() (+3 more)

### Community 25 - "analytical.py"
Cohesion: 0.15
Nodes (12): 10. Проверки валидности, 11. Не входит в v0.1, 1. Scope, 2. Геометрическая конвенция, 3. Input-параметры, 4. Математическое определение, 5. Эмиттанс и расходимость, 6. Энергия и заряд (+4 more)

### Community 26 - "compton_suite/__init__.py"
Cohesion: 0.06
Nodes (25): add_field_grid(), ComptonGuideApp, _float_or_none(), laser_focal_radii(), main(), peak_a0(), Peak normalised vector potential a_0 from the laser fields (or None)., Transverse rms bunch size [m] from normalized emittance, beta [m] and gamma. (+17 more)

### Community 27 - "detect_device"
Cohesion: 0.15
Nodes (12): 10. Проверки валидности input, 11. Не входит в v0.1, 1. Scope, 2. Геометрическая конвенция, 3. Input-параметры, 4. Математическое определение профиля, 5. Конверсии rms / FWHM / $1/e^2$, 6. Длина Рэлея и размер в точке взаимодействия (+4 more)

### Community 28 - "test_constants.py"
Cohesion: 0.20
Nodes (5): AnalyticalConfig, _float(), ParamError, Exception, Wraps beam/pulse, but ALSO exposes a handful of flat SI fields every     other m

### Community 29 - "AnalyticalAdapter"
Cohesion: 0.18
Nodes (7): register(), registered_models(), discover_models(), Model registry -- direct imports since all packages ship together., Populate the model registry with direct imports., XigmaDirectAdapter, xigma-i-direct: brute-force per-macroparticle resonance-binning model.  Extracte

### Community 30 - "XigmaAdapter"
Cohesion: 0.11
Nodes (18): 1. Dead-code / unused-config sweep, 2. Move `CollisionParams`/`build_params` into `compton_io`, 3. GUI: reconsider "experimental" trust levels/warnings, 4. GUI: per-model sample count instead of a misleading global field, 5. Unify the `ModelAdapter` interface properly, 6. Manual CPU/GPU selection for xigma-i, 7. How to add a new model, 8. Model-agnostic simulation core — superseded by direct `io/` consolidation (+10 more)

### Community 31 - "test_analytical.py"
Cohesion: 0.15
Nodes (18): beam_from_shared_fields(), Build a :class:`GaussianElectronBeam` from the flat SI field set every     model, InteractionGeometry, InteractionParameters, The shared "interaction parameters" bundle: an electron beam, a laser pulse, and, Collision geometry: foci displacement and crossing angle. SI units., One canonical (beam, pulse, geometry) triple -- the physics-parameter     subset, laser_from_shared_fields() (+10 more)

### Community 32 - "InteractionParameters"
Cohesion: 0.19
Nodes (7): build_params(), CollisionParams, The CGS "collision parameters" bundle for tabulated-overlap-style GPU/CPU pipeli, Derive this convention's CGS :class:`CollisionParams` from     ``compton_io``'s, CGS scalars for one laser-electron collision -- immutable, built     once by :fu, Physics engine models for ComptonSuite.  Each model implements the ModelAdapter, Physics constants and GPU kernel sizing constants for this pipeline (particles.p

### Community 34 - "schemas/__init__.py"
Cohesion: 0.20
Nodes (9): Adding a new GUI observable, Compton-GUIde, Known gaps, Layout, Model-specific parameters (`extra_params()`), Parameter semantics & units (`physics_params/`), and `compton_io`, Running it, Testing (headless, no display needed) (+1 more)

### Community 35 - "adapters/__init__.py"
Cohesion: 0.25
Nodes (7): Angular Distribution (energy-integrated), Angular-Range-Restricted Spectrum, Headless testing, New GUI observables: status per model, Sequencing, Spatial (Transverse-Plane) Distribution, Temporal Envelope

### Community 36 - "compton_guide/__init__.py"
Cohesion: 0.25
Nodes (7): Files, KASCADE, No GPU dependency, Physical constants (`compton_io`), Relationship to other components, Testing, Units and conventions

### Community 37 - "kascade"
Cohesion: 0.25
Nodes (7): Files, KASCADE, No GPU dependency, Physical constants (`compton_io`), Relationship to other components, Testing, Units and conventions

### Community 39 - "xigma-i"
Cohesion: 0.33
Nodes (5): compton_io, Layout, Naming, Testing, Why this exists

### Community 40 - "XigmaAdapter"
Cohesion: 0.17
Nodes (4): Optional capability -- check         capabilities().supports_angular_range_spect, AngularRangeSpectrumResult, Mask the cached per-photon lab angles against the picked range and         histo, XigmaAdapter

### Community 41 - "Compton-GUIde"
Cohesion: 0.20
Nodes (10): Evaluate direct_binning_spectrum at a grid of OBSERVATION angles     spanning th, spectrum_in_angular_range(), angle_integrated_spectrum(), direct_binning_spectrum(), Table-free spectrum paths computed directly from Stage 0/1 macroparticles -- no, Resolves backend='numpy'|'cupy' to its array module. cupy is imported     lazily, dN/ds integrated over all emission solid angle, from real Stage 0/1     macropar, `xigma-i-direct`'s actual computation: for each real macroparticle,     compute (+2 more)

### Community 46 - "opencode.json"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

### Community 53 - "ComptonSuite model-agnostic-core refactor: status"
Cohesion: 0.29
Nodes (6): ComptonSuite model-agnostic-core refactor: status, Current package layout, Explicitly dropped from the original plan, Landed — moves that fulfilled the original goal without a `core/` package, Still open, What changed since this doc was first written

## Knowledge Gaps
- **184 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `compton-suite`, `What this is`, `Layout` (+179 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MacroBunch` connect `particles.py` to `Scenario`, `deposition.py`, `CommonResults`, `XigmaDirectAdapter`, `XigmaAdapter`, `xigma_i/gui_adapter.py`, `New GUI observables: status per model`, `MacroBunch`, `cache.py`, `UnavailableAdapter`, `run_multiphoton_chain`, `test_constants.py`, `AnalyticalAdapter`, `test_analytical.py`?**
  _High betweenness centrality (0.100) - this node is a cross-community bridge._
- **Why does `ComptonGuideApp` connect `compton_suite/__init__.py` to `UnavailableAdapter`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Why does `sample_gaussian_bunch()` connect `particles.py` to `ComptonGuideApp`, `kaskade/kascade.py`, `CommonResults`, `GaussianElectronBeam`, `test_laser.py`, `run_multiphoton_chain`, `compton_suite/__init__.py`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `MacroBunch` (e.g. with `ModelAdapter` and `ModelCapabilities`) actually correct?**
  _`MacroBunch` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `GaussianElectronBeam` (e.g. with `CollisionParams` and `InteractionGeometry`) actually correct?**
  _`GaussianElectronBeam` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `GaussianParaxialLaser` (e.g. with `CollisionParams` and `InteractionGeometry`) actually correct?**
  _`GaussianParaxialLaser` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Config` (e.g. with `MacroBunch` and `InteractionParameters`) actually correct?**
  _`Config` has 2 INFERRED edges - model-reasoned connections that need verification._