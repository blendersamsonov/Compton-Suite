# Graph Report - ComptonSuite  (2026-07-27)

## Corpus Check
- 102 files · ~103,153 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1099 nodes · 2336 edges · 46 communities (40 shown, 6 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 213 edges (avg confidence: 0.55)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d5f0d253`
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
- GaussianParaxialLaser
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
- _interp4d
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
- ModelCapabilities
- Compton-GUIde
- New GUI observables: status per model
- opencode.json
- graphify.js

## God Nodes (most connected - your core abstractions)
1. `MacroBunch` - 54 edges
2. `GaussianElectronBeam` - 46 edges
3. `GaussianParaxialLaser` - 46 edges
4. `ComptonGuideApp` - 38 edges
5. `CommonResults` - 37 edges
6. `ModelCapabilities` - 29 edges
7. `ModelAdapter` - 27 edges
8. `sample_gaussian_bunch()` - 27 edges
9. `BinnedSpectrum` - 25 edges
10. `UnavailableAdapter` - 24 edges

## Surprising Connections (you probably didn't know these)
- `run()` --calls--> `check()`  [INFERRED]
  src/compton_suite/validation/tier0_wiring.py → scripts/headless_test.py
- `run()` --calls--> `check()`  [INFERRED]
  validation/tier0_wiring.py → scripts/headless_test.py
- `_baseline()` --calls--> `GaussianElectronBeam`  [INFERRED]
  validation/scenarios.py → src/compton_suite/io/bunch.py
- `Scenario` --uses--> `GaussianElectronBeam`  [INFERRED]
  validation/scenarios.py → src/compton_suite/io/bunch.py
- `run_analytical()` --calls--> `sample_gaussian_bunch()`  [INFERRED]
  validation/runners.py → src/compton_suite/io/bunch.py

## Import Cycles
- None detected.

## Communities (46 total, 6 thin omitted)

### Community 0 - "Scenario"
Cohesion: 0.08
Nodes (57): Shared comparison metric for the cross-model validation suite.  Promoted (moved, s, spec, spec_ref: 1D arrays of equal shape (same grid for both     spectra -- r, Linear-interpolate spec_src(s_src) onto s_ref, zero outside s_src's range., resample_to(), window_integrated_relative_error(), Cross-model validation suite -- entry point.  Runs kascade/xigma-i/xigma-i-direc, Tier 0 (config-only, no run needed) + Tier 1/2 (share one run of     each model), run() (+49 more)

### Community 1 - "compton_io/__init__.py"
Cohesion: 0.07
Nodes (56): Enum, ModelSpec, Quantity, Thin re-export of ``compton_io``'s parameter-semantics/unit normalisation framew, ModelSpec for kascade (formerly dfe5_compton_mc), at the boundary its own adapte, adapt_to_model(), params_to_floats(), Canonical params -> a specific model's own convention/units.  This is the last s (+48 more)

### Community 2 - "ComptonGuideApp"
Cohesion: 0.08
Nodes (57): Shared comparison metric for the cross-model validation suite.  Promoted (moved, s, spec, spec_ref: 1D arrays of equal shape (same grid for both     spectra -- r, Linear-interpolate spec_src(s_src) onto s_ref, zero outside s_src's range., resample_to(), window_integrated_relative_error(), Cross-model validation suite -- entry point.  Runs kascade/xigma-i/xigma-i-direc, Tier 0 (config-only, no run needed) + Tier 1/2 (share one run of     each model), run() (+49 more)

### Community 3 - "deposition.py"
Cohesion: 0.06
Nodes (44): beam_from_shared_fields(), Build a :class:`GaussianElectronBeam` from the flat SI field set every     model, build_params(), detect_device(), Derive this convention's CGS :class:`CollisionParams` from     ``compton_io``'s, Auto-detect which backend to use: a real CUDA GPU via cupy if     available, els, InteractionGeometry, Collision geometry: foci displacement and crossing angle. SI units. (+36 more)

### Community 4 - "particles.py"
Cohesion: 0.07
Nodes (25): Raise NotImplementedError if capabilities().supports_ele_file_io is False., Raise NotImplementedError if capabilities().supports_ele_file_io is False., fit_gaussian(), MacroBunch, Electron-bunch representations.  Two distinct types, matching the distinction th, Fit a :class:`GaussianElectronBeam` (at its waist) from raw macroparticles., Raw, engine-agnostic macroparticle representation. SI units.      ``x``/``y``/``, load_elegant_ele() (+17 more)

### Community 5 - "kaskade/kascade.py"
Cohesion: 0.07
Nodes (37): _array_module(), build_table(), build_table_streaming(), _cell_indices(), check_accumulation_precision(), _deposit(), deposit_cic(), deposit_nearest() (+29 more)

### Community 6 - "CommonResults"
Cohesion: 0.07
Nodes (40): ballistic_position_simultaneous(), ballistic_position_z0_reference(), laser_overlap_time_window(), propagate(), Ballistic (straight-line, no-acceleration) electron-beam propagation, and a time, Ballistically drift every macroparticle in ``bunch`` by a time     offset ``dt``, Per-particle time window ``[t0, t1]`` (normalised length units,     e.g. ``k0_la, Yield a propagated :class:`MacroBunch` snapshot at each time in     ``t_grid`` ( (+32 more)

### Community 7 - "XigmaDirectAdapter"
Cohesion: 0.10
Nodes (35): KASCADE: Sequential multi-photon inverse-Compton Monte Carlo engine., beta_of(), build_lambda_grid(), Config, cos_collision(), doppler_D(), invert_lambda(), kn_over_thomson() (+27 more)

### Community 8 - "GaussianElectronBeam"
Cohesion: 0.07
Nodes (19): beta_star_from_sigma_emit(), divergence_from_sigma_emit(), GaussianElectronBeam, Validate a :class:`GaussianElectronBeam` per spec Sec. 16.      Raises ``ValueEr, Draw macroparticles from a :class:`GaussianElectronBeam`.      Independent facto, Beta function at the waist, ``sigma**2 / emit_geom`` -- same units in     as out, RMS divergence at the waist, ``emit_geom / sigma`` -- a dimensionless     angle,, The ``gaussian_6d_waist`` v0.1 I/O contract.      A 6D factorized Gaussian defin (+11 more)

### Community 9 - "xigma_i/gui_adapter.py"
Cohesion: 0.08
Nodes (25): BinnedSpectrum, Binned dN/dE spectrum (semi-analytic/tabulated models)., _attach_private_cache(), available(), DirectConfig, extra_choices(), _float(), ParamError (+17 more)

### Community 10 - "MacroBunch"
Cohesion: 0.16
Nodes (28): InteractionParameters, The shared "interaction parameters" bundle: an electron beam, a laser pulse, and, One canonical (beam, pulse, geometry) triple -- the physics-parameter     subset, User-curated bank of additional cross-validation parameter sets.  Deliberately e, _baseline(), build_analytical_config(), build_kascade_config(), build_xigma_config() (+20 more)

### Community 11 - "cache.py"
Cohesion: 0.18
Nodes (20): cache_key(), clear_cache(), get_or_compute(), _git_head_hash(), _git_is_dirty(), list_cache_entries(), load(), _paths() (+12 more)

### Community 12 - "headless_test.py"
Cohesion: 0.07
Nodes (29): 0. Краткое резюме, 10.1. Что уже работает, 10.2. Что нужно сделать для первого сравнения кодов, 10.3. Что должно войти в следующую версию паспорта, 10. Текущий статус и ближайшие шаги, 11. Итоговая оценка готовности, 1. Карточка кода / метода, 2. Назначение и роль в проекте (+21 more)

### Community 13 - "GaussianParaxialLaser"
Cohesion: 0.11
Nodes (29): AnalyticalConfig, Wraps beam/pulse, but ALSO exposes the flat SI fields every other     model's Co, User-curated bank of additional cross-validation parameter sets.  Deliberately e, _baseline(), build_analytical_config(), build_kascade_config(), build_params_for_xigma(), build_xigma_config() (+21 more)

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
Cohesion: 0.15
Nodes (5): GaussianParaxialLaser, ``a0_at(z_m)**2`` -- the mean-square amplitude.          Consumers that only eve, A paraxial Gaussian laser pulse. SI units.      ``waist_rms_x_m``/``waist_rms_y_, Photon count in the pulse: N_L = pulse_energy_J / (hbar*omega0)., Normalized vector potential at ``z_m``, from the exact SI         plane-wave rel

### Community 20 - "laser.py"
Cohesion: 0.13
Nodes (13): calculate_angular_spectrum_4d(), _get_numba(), get_spectrum_kernel_4d_cpu(), CPU/numba equivalent of spectrum4d.py's GPU rawkernel (spectrum_kernel_4d), used, Lazily compiles and caches spectrum_kernel_4d_cpu (see module docstring)., Stage 2: `spectrum_kernel_4d`, the GPU kernel that turns a Stage 1 4D overlap ta, Host-side driver for spectrum_kernel_4d, the Stage-2 analogue of     Compton.cal, angle_integrated_spectrum() (+5 more)

### Community 21 - "test_laser.py"
Cohesion: 0.20
Nodes (13): check(), FakeVar, main(), make_fields(), Same mechanism app.py's on_start() uses: whichever model is     'selected', the, Stand-in for tk.StringVar -- params_to_config only ever calls .get()., sample_electrons_for(), test_model() (+5 more)

### Community 22 - "bunch.py"
Cohesion: 0.15
Nodes (7): CODATA-style physical constants used by the GUI's local formula helpers (peak_a0, CollisionParams, The CGS "collision parameters" bundle for tabulated-overlap-style GPU/CPU pipeli, CGS scalars for one laser-electron collision -- immutable, built     once by :fu, Physical constants, derived from ``units.py``'s pint registry rather than hand-t, Physics engine models for ComptonSuite.  Each model implements the ModelAdapter, Physics constants and GPU kernel sizing constants for this pipeline (particles.p

### Community 23 - "UnavailableAdapter"
Cohesion: 0.13
Nodes (24): Protocol, ModelAdapter, ModelCapabilities, Model-agnostic contract between app.py and physics-engine adapters.  This module, Return (True, "") if the model can actually be run right now, else         (Fals, Model-specific fields with no shared-panel analogue         (kascade's Electrons, Optional: return a dict mapping parameter keys to allowed string values, Optional capability -- check         capabilities().supports_angular_range_spect (+16 more)

### Community 24 - "_interp4d"
Cohesion: 0.20
Nodes (10): External-format I/O for compton_io's bunch/laser representations., load_electron_beam(), load_laser(), YAML I/O for the ``gaussian_6d_waist``/``gaussian_paraxial`` v0.1 formats define, save_electron_beam(), save_laser(), Laser-pulse representation: the ``gaussian_paraxial`` v0.1 I/O contract (``specs, Cross-checks for io_formats/sdds.py and io_formats/yaml_spec.py.  No cupy/GPU/tk (+2 more)

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
Nodes (6): register(), registered_models(), discover_models(), Model registry -- direct imports since all packages ship together., Populate the model registry with direct imports., XigmaAdapter

### Community 29 - "AnalyticalAdapter"
Cohesion: 0.21
Nodes (8): Validate a :class:`GaussianParaxialLaser` per spec Sec. 12.      Raises ``ValueE, validate(), Cross-checks for laser.py's GaussianParaxialLaser.  No cupy/GPU/tkinter needed., test_defocused_interaction_intensity_is_lower(), test_derived_quantities_are_sane(), test_laser_from_shared_fields_round_trips(), test_validate_rejects_nonpositive_fields(), test_validate_warns_on_astigmatism()

### Community 30 - "XigmaAdapter"
Cohesion: 0.11
Nodes (17): 1. Dead-code / unused-config sweep, 2. Move `CollisionParams`/`build_params` into `compton_io`, 3. GUI: reconsider "experimental" trust levels/warnings, 4. GUI: per-model sample count instead of a misleading global field, 5. Unify the `ModelAdapter` interface properly, 6. Manual CPU/GPU selection for xigma-i, 7. How to add a new model, AGENTS.md (+9 more)

### Community 31 - "test_analytical.py"
Cohesion: 0.16
Nodes (15): angle_integrated_spectrum(), estimate_spectrum_width(), estimate_yield(), ndarray, Fast, closed-form Compton-source physics: total yield, angle-integrated spectrum, Cheap analytic total-photon-yield estimate from an overlap integral     between, Cheap analytic estimate of the collimated-spectrum FWHM (in units of     the Com, dN/ds integrated over all emission solid angle, from the standard     angle-inde (+7 more)

### Community 32 - "InteractionParameters"
Cohesion: 0.18
Nodes (3): KascadeAdapter, Mask the cached per-photon lab angles against the picked range and         histo, Convert the GUI fields into a physics Config plus a driver dict.          Ported

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

### Community 46 - "opencode.json"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

## Knowledge Gaps
- **165 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `compton-suite`, `What this is`, `Layout` (+160 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `sample_gaussian_bunch()` connect `GaussianElectronBeam` to `Scenario`, `ComptonGuideApp`, `particles.py`, `kaskade/kascade.py`, `CommonResults`, `test_laser.py`, `compton_suite/__init__.py`?**
  _High betweenness centrality (0.113) - this node is a cross-community bridge._
- **Why does `MacroBunch` connect `particles.py` to `InteractionParameters`, `deposition.py`, `CommonResults`, `XigmaDirectAdapter`, `ModelCapabilities`, `GaussianElectronBeam`, `xigma_i/gui_adapter.py`, `GaussianParaxialLaser`, `UnavailableAdapter`, `test_constants.py`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Why does `GaussianElectronBeam` connect `GaussianElectronBeam` to `deposition.py`, `particles.py`, `MacroBunch`, `GaussianParaxialLaser`, `bunch.py`, `_interp4d`, `test_analytical.py`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `MacroBunch` (e.g. with `ModelAdapter` and `ModelCapabilities`) actually correct?**
  _`MacroBunch` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `GaussianElectronBeam` (e.g. with `CollisionParams` and `InteractionGeometry`) actually correct?**
  _`GaussianElectronBeam` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `GaussianParaxialLaser` (e.g. with `CollisionParams` and `InteractionGeometry`) actually correct?**
  _`GaussianParaxialLaser` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `CommonResults` (e.g. with `ModelAdapter` and `ModelCapabilities`) actually correct?**
  _`CommonResults` has 22 INFERRED edges - model-reasoned connections that need verification._