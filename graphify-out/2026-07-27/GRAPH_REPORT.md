# Graph Report - ComptonSuite  (2026-07-27)

## Corpus Check
- 105 files · ~105,828 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1124 nodes · 2379 edges · 55 communities (47 shown, 8 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 213 edges (avg confidence: 0.55)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b61875d3`
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
- propagation.py
- UnavailableAdapter
- opencode.json
- AnalyticalAdapter
- Grid4D
- spectrum_in_angular_range
- graphify.js
- build_table_streaming
- _interp4d
- ComptonSuite model-agnostic-core refactor: status
- beta_star_from_sigma_emit

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

## Communities (55 total, 8 thin omitted)

### Community 0 - "Scenario"
Cohesion: 0.08
Nodes (57): Shared comparison metric for the cross-model validation suite.  Promoted (moved, s, spec, spec_ref: 1D arrays of equal shape (same grid for both     spectra -- r, Linear-interpolate spec_src(s_src) onto s_ref, zero outside s_src's range., resample_to(), window_integrated_relative_error(), Cross-model validation suite -- entry point.  Runs kascade/xigma-i/xigma-i-direc, Tier 0 (config-only, no run needed) + Tier 1/2 (share one run of     each model), run() (+49 more)

### Community 1 - "compton_io/__init__.py"
Cohesion: 0.07
Nodes (57): Enum, ModelSpec, Quantity, CODATA-style physical constants used by the GUI's local formula helpers (peak_a0, Thin re-export of ``compton_io``'s parameter-semantics/unit normalisation framew, ModelSpec for kascade (formerly dfe5_compton_mc), at the boundary its own adapte, adapt_to_model(), params_to_floats() (+49 more)

### Community 2 - "ComptonGuideApp"
Cohesion: 0.08
Nodes (57): Shared comparison metric for the cross-model validation suite.  Promoted (moved, s, spec, spec_ref: 1D arrays of equal shape (same grid for both     spectra -- r, Linear-interpolate spec_src(s_src) onto s_ref, zero outside s_src's range., resample_to(), window_integrated_relative_error(), Cross-model validation suite -- entry point.  Runs kascade/xigma-i/xigma-i-direc, Tier 0 (config-only, no run needed) + Tier 1/2 (share one run of     each model), run() (+49 more)

### Community 3 - "deposition.py"
Cohesion: 0.06
Nodes (44): beam_from_shared_fields(), Build a :class:`GaussianElectronBeam` from the flat SI field set every     model, build_params(), detect_device(), Derive this convention's CGS :class:`CollisionParams` from     ``compton_io``'s, Auto-detect which backend to use: a real CUDA GPU via cupy if     available, els, InteractionGeometry, Collision geometry: foci displacement and crossing angle. SI units. (+36 more)

### Community 4 - "particles.py"
Cohesion: 0.10
Nodes (20): Raise NotImplementedError if capabilities().supports_ele_file_io is False., Raise NotImplementedError if capabilities().supports_ele_file_io is False., fit_gaussian(), MacroBunch, Fit a :class:`GaussianElectronBeam` (at its waist) from raw macroparticles., Raw, engine-agnostic macroparticle representation. SI units.      ``x``/``y``/``, Physical constants, derived from ``units.py``'s pint registry rather than hand-t, load_elegant_ele() (+12 more)

### Community 5 - "kaskade/kascade.py"
Cohesion: 0.19
Nodes (14): _array_module(), _cell_indices(), check_accumulation_precision(), deposit_cic(), deposit_nearest(), occupancy_diagnostics(), Stage 1: 4D deposition of macroparticle samples into H[gamma, theta_x, theta_y,, Continuous (float) cell coordinates along each axis, i.e. how many     bin-width (+6 more)

### Community 6 - "CommonResults"
Cohesion: 0.07
Nodes (43): gaussian_pulse_envelope(), Shared spatiotemporal Gaussian-pulse envelope -- the piece ``compton_io.laser.Ga, Photon-density envelope of a Gaussian laser pulse at an arbitrary     point in s, ballistic_position_z0_reference(), laser_overlap_time_window(), Per-particle time window ``[t0, t1]`` (normalised length units,     e.g. ``k0_la, Straight-line position at time offset ``t``, given a per-particle     reference, _bin_spatial() (+35 more)

### Community 7 - "XigmaDirectAdapter"
Cohesion: 0.10
Nodes (35): KASCADE: Sequential multi-photon inverse-Compton Monte Carlo engine., beta_of(), build_lambda_grid(), Config, cos_collision(), doppler_D(), invert_lambda(), kn_over_thomson() (+27 more)

### Community 8 - "GaussianElectronBeam"
Cohesion: 0.12
Nodes (4): divergence_from_sigma_emit(), GaussianElectronBeam, RMS divergence at the waist, ``emit_geom / sigma`` -- a dimensionless     angle,, The ``gaussian_6d_waist`` v0.1 I/O contract.      A 6D factorized Gaussian defin

### Community 9 - "xigma_i/gui_adapter.py"
Cohesion: 0.12
Nodes (18): _attach_private_cache(), available(), DirectConfig, extra_choices(), _float(), ParamError, params_to_config(), Exception (+10 more)

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
Cohesion: 0.12
Nodes (27): AnalyticalConfig, Wraps beam/pulse, but ALSO exposes the flat SI fields every other     model's Co, User-curated bank of additional cross-validation parameter sets.  Deliberately e, _baseline(), build_analytical_config(), build_kascade_config(), build_xigma_config(), build_xigma_direct_config() (+19 more)

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
Cohesion: 0.09
Nodes (16): load_laser(), save_laser(), GaussianParaxialLaser, ``a0_at(z_m)**2`` -- the mean-square amplitude.          Consumers that only eve, Validate a :class:`GaussianParaxialLaser` per spec Sec. 12.      Raises ``ValueE, A paraxial Gaussian laser pulse. SI units.      ``waist_rms_x_m``/``waist_rms_y_, Photon count in the pulse: N_L = pulse_energy_J / (hbar*omega0)., Normalized vector potential at ``z_m``, from the exact SI         plane-wave rel (+8 more)

### Community 20 - "laser.py"
Cohesion: 0.20
Nodes (8): calculate_angular_spectrum_4d(), _get_numba(), get_spectrum_kernel_4d_cpu(), CPU/numba equivalent of spectrum4d.py's GPU rawkernel (spectrum_kernel_4d), used, Lazily compiles and caches spectrum_kernel_4d_cpu (see module docstring)., Stage 2: `spectrum_kernel_4d`, the GPU kernel that turns a Stage 1 4D overlap ta, Host-side driver for spectrum_kernel_4d, the Stage-2 analogue of     Compton.cal, GUI-facing engine on the tabulated-energy pipeline (particles.py/ deposition.py/

### Community 21 - "test_laser.py"
Cohesion: 0.20
Nodes (13): check(), FakeVar, main(), make_fields(), Same mechanism app.py's on_start() uses: whichever model is     'selected', the, Stand-in for tk.StringVar -- params_to_config only ever calls .get()., sample_electrons_for(), test_model() (+5 more)

### Community 22 - "bunch.py"
Cohesion: 0.20
Nodes (5): CollisionParams, The CGS "collision parameters" bundle for tabulated-overlap-style GPU/CPU pipeli, CGS scalars for one laser-electron collision -- immutable, built     once by :fu, Physics engine models for ComptonSuite.  Each model implements the ModelAdapter, Physics constants and GPU kernel sizing constants for this pipeline (particles.p

### Community 23 - "UnavailableAdapter"
Cohesion: 0.17
Nodes (22): Protocol, ModelAdapter, ModelCapabilities, Any, Model-agnostic contract between app.py and physics-engine adapters.  This module, Return (True, "") if the model can actually be run right now, else         (Fals, Model-specific fields with no shared-panel analogue         (kascade's Electrons, Optional: return a dict mapping parameter keys to allowed string values (+14 more)

### Community 24 - "_interp4d"
Cohesion: 0.24
Nodes (7): External-format I/O for compton_io's bunch/laser representations., load_electron_beam(), YAML I/O for the ``gaussian_6d_waist``/``gaussian_paraxial`` v0.1 formats define, save_electron_beam(), Laser-pulse representation: the ``gaussian_paraxial`` v0.1 I/O contract (``specs, Cross-checks for io_formats/sdds.py and io_formats/yaml_spec.py.  No cupy/GPU/tk, test_spec_example_electron_beam_yaml_round_trips()

### Community 25 - "analytical.py"
Cohesion: 0.15
Nodes (12): 10. Проверки валидности, 11. Не входит в v0.1, 1. Scope, 2. Геометрическая конвенция, 3. Input-параметры, 4. Математическое определение, 5. Эмиттанс и расходимость, 6. Энергия и заряд (+4 more)

### Community 26 - "compton_suite/__init__.py"
Cohesion: 0.06
Nodes (21): add_field_grid(), ComptonGuideApp, _float_or_none(), laser_focal_radii(), peak_a0(), Peak normalised vector potential a_0 from the laser fields (or None)., Transverse rms bunch size [m] from normalized emittance, beta [m] and gamma., Return radial RMS, FWHM, and exp(-1/2) focal radii [m]. (+13 more)

### Community 27 - "detect_device"
Cohesion: 0.15
Nodes (12): 10. Проверки валидности input, 11. Не входит в v0.1, 1. Scope, 2. Геометрическая конвенция, 3. Input-параметры, 4. Математическое определение профиля, 5. Конверсии rms / FWHM / $1/e^2$, 6. Длина Рэлея и размер в точке взаимодействия (+4 more)

### Community 28 - "test_constants.py"
Cohesion: 0.15
Nodes (10): main(), Compton-GUIde: Tkinter GUI for pluggable Compton-scattering physics models., register(), registered_models(), discover_models(), Model registry -- direct imports since all packages ship together., Populate the model registry with direct imports., ComptonSuite: unified package for inverse-Compton scattering simulation.  Subpac (+2 more)

### Community 29 - "AnalyticalAdapter"
Cohesion: 0.13
Nodes (8): Optional capability -- check         capabilities().supports_angular_range_spect, AngularRangeSpectrumResult, BinnedSpectrum, Binned dN/dE spectrum (semi-analytic/tabulated models)., Mask the cached per-photon lab angles against the picked range and         histo, XigmaDirectAdapter, Fresh, on-demand spectrum over an arbitrary user-picked angular     sub-range, u, spectrum_in_angular_range()

### Community 30 - "XigmaAdapter"
Cohesion: 0.11
Nodes (18): 1. Dead-code / unused-config sweep, 2. Move `CollisionParams`/`build_params` into `compton_io`, 3. GUI: reconsider "experimental" trust levels/warnings, 4. GUI: per-model sample count instead of a misleading global field, 5. Unify the `ModelAdapter` interface properly, 6. Manual CPU/GPU selection for xigma-i, 7. How to add a new model, 8. Extract model-agnostic simulation core to `compton_suite.core` (+10 more)

### Community 31 - "test_analytical.py"
Cohesion: 0.16
Nodes (15): angle_integrated_spectrum(), estimate_spectrum_width(), estimate_yield(), ndarray, Fast, closed-form Compton-source physics: total yield, angle-integrated spectrum, Cheap analytic total-photon-yield estimate from an overlap integral     between, Cheap analytic estimate of the collimated-spectrum FWHM (in units of     the Com, dN/ds integrated over all emission solid angle, from the standard     angle-inde (+7 more)

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

### Community 40 - "ModelCapabilities"
Cohesion: 0.19
Nodes (14): Electron-bunch representations.  Two distinct types, matching the distinction th, Validate a :class:`GaussianElectronBeam` per spec Sec. 16.      Raises ``ValueEr, Draw macroparticles from a :class:`GaussianElectronBeam`.      Independent facto, sample_gaussian_bunch(), validate(), Cross-checks for bunch.py's MacroBunch / GaussianElectronBeam / sample_gaussian_, test_derived_quantities_are_sane(), test_fit_gaussian_recovers_waist_after_drift() (+6 more)

### Community 44 - "propagation.py"
Cohesion: 0.22
Nodes (12): ballistic_position_simultaneous(), propagate(), Ballistic (straight-line, no-acceleration) electron-beam propagation, and a time, Ballistically drift every macroparticle in ``bunch`` by a time     offset ``dt``, Yield a propagated :class:`MacroBunch` snapshot at each time in     ``t_grid`` (, Straight-line position at time offset ``dt`` later, given a     per-particle ref, stream(), Cross-checks for propagation.py's ballistic drift primitives.  No cupy/GPU/tkint (+4 more)

### Community 46 - "opencode.json"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

### Community 47 - "AnalyticalAdapter"
Cohesion: 0.18
Nodes (5): AnalyticalAdapter, _float(), ParamError, Exception, Fast closed-form model: total yield, angle-integrated spectrum, and     an estim

### Community 48 - "Grid4D"
Cohesion: 0.20
Nodes (6): Grid4D, Uniform 4D grid over (gamma, theta_x, theta_y, a0)., A deposited H table plus everything Stage 2 needs to consume it     without re-d, Turn an a0_kind='shape' table into a spectrum-ready a0_kind='ahat'     table for, retarget_a0(), Table

### Community 49 - "spectrum_in_angular_range"
Cohesion: 0.24
Nodes (9): Evaluate direct_binning_spectrum at a grid of OBSERVATION angles     spanning th, spectrum_in_angular_range(), angle_integrated_spectrum(), direct_binning_spectrum(), Table-free spectrum paths computed directly from Stage 0/1 macroparticles -- no, Resolves backend='numpy'|'cupy' to its array module. cupy is imported     lazily, dN/ds integrated over all emission solid angle, from real Stage 0/1     macropar, `xigma-i-direct`'s actual computation: for each real macroparticle,     compute (+1 more)

### Community 51 - "build_table_streaming"
Cohesion: 0.24
Nodes (9): build_table(), build_table_streaming(), _deposit(), gamma_bracket(), Derive grid extents from the populated data range, plus a margin         express, Dispatches to deposit_nearest/deposit_cic, optionally batching: if     batch_siz, Lowest/highest gamma at which the table has non-negligible content,     as the q, Orchestrates Stage 1: grid derivation (if not supplied) + deposition +     diagn (+1 more)

### Community 52 - "_interp4d"
Cohesion: 0.27
Nodes (8): _interp4d(), Cross-validation-only tooling: a brute-force, non-GPU-kernel way to turn a Stage, Brute-force quadrature of dN/(ds dOmega) at a single observation point     (x0,, Resolves backend='numpy'|'cupy' to its array module. cupy is imported     lazily, Array-module-agnostic core of interp4d: takes H already converted to     xp's mo, Quadrilinear interpolation of table.H at query points (arrays of equal     shape, spectrum_from_table(), _xp_for()

### Community 53 - "ComptonSuite model-agnostic-core refactor: status"
Cohesion: 0.29
Nodes (6): ComptonSuite model-agnostic-core refactor: status, Current package layout, Explicitly dropped from the original plan, Landed — moves that fulfilled the original goal without a `core/` package, Still open, What changed since this doc was first written

## Knowledge Gaps
- **171 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `compton-suite`, `What this is`, `Layout` (+166 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `sample_gaussian_bunch()` connect `ModelCapabilities` to `Scenario`, `ComptonGuideApp`, `particles.py`, `kaskade/kascade.py`, `GaussianElectronBeam`, `propagation.py`, `build_table_streaming`, `test_laser.py`, `compton_suite/__init__.py`, `test_constants.py`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Why does `MacroBunch` connect `particles.py` to `InteractionParameters`, `deposition.py`, `XigmaDirectAdapter`, `ModelCapabilities`, `xigma_i/gui_adapter.py`, `propagation.py`, `UnavailableAdapter`, `GaussianParaxialLaser`, `AnalyticalAdapter`, `UnavailableAdapter`, `test_constants.py`, `AnalyticalAdapter`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Why does `ComptonGuideApp` connect `compton_suite/__init__.py` to `test_constants.py`, `UnavailableAdapter`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `MacroBunch` (e.g. with `ModelAdapter` and `ModelCapabilities`) actually correct?**
  _`MacroBunch` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `GaussianElectronBeam` (e.g. with `CollisionParams` and `InteractionGeometry`) actually correct?**
  _`GaussianElectronBeam` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `GaussianParaxialLaser` (e.g. with `CollisionParams` and `InteractionGeometry`) actually correct?**
  _`GaussianParaxialLaser` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `CommonResults` (e.g. with `ModelAdapter` and `ModelCapabilities`) actually correct?**
  _`CommonResults` has 22 INFERRED edges - model-reasoned connections that need verification._