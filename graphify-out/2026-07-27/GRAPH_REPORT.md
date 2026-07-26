# Graph Report - ComptonSuite  (2026-07-26)

## Corpus Check
- 100 files · ~87,258 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1001 nodes · 1925 edges · 53 communities (40 shown, 13 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 225 edges (avg confidence: 0.55)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `506f2583`
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
- kascade
- xigma-direct
- xigma-i
- ModelCapabilities
- Compton-GUIde
- New GUI observables: status per model
- KASCADE
- kascade_adapter.py
- compton_io
- opencode.json
- beam_from_shared_fields
- capabilities
- capabilities
- graphify.js
- tasks.md

## God Nodes (most connected - your core abstractions)
1. `MacroBunch` - 55 edges
2. `Scenario` - 43 edges
3. `GaussianElectronBeam` - 41 edges
4. `GaussianParaxialLaser` - 40 edges
5. `ComptonGuideApp` - 38 edges
6. `CommonResults` - 38 edges
7. `ModelCapabilities` - 26 edges
8. `ModelAdapter` - 26 edges
9. `UnavailableAdapter` - 24 edges
10. `BinnedSpectrum` - 21 edges

## Surprising Connections (you probably didn't know these)
- `run()` --calls--> `check()`  [INFERRED]
  validation/tier0_wiring.py → GUIde/scripts/headless_test.py
- `ModelCapabilities` --uses--> `MacroBunch`  [INFERRED]
  GUIde/src/compton_guide/model_api.py → IO/src/compton_io/bunch.py
- `ModelCapabilities` --uses--> `AngularRangeSpectrumResult`  [INFERRED]
  GUIde/src/compton_guide/model_api.py → IO/src/compton_io/photons.py
- `ModelCapabilities` --uses--> `BinnedAngularSpectrum`  [INFERRED]
  GUIde/src/compton_guide/model_api.py → IO/src/compton_io/photons.py
- `ModelCapabilities` --uses--> `BinnedSpatialDistribution`  [INFERRED]
  GUIde/src/compton_guide/model_api.py → IO/src/compton_io/photons.py

## Import Cycles
- None detected.

## Communities (53 total, 13 thin omitted)

### Community 0 - "Scenario"
Cohesion: 0.07
Nodes (77): Shared comparison metric for the cross-model validation suite.  Promoted (moved, s, spec, spec_ref: 1D arrays of equal shape (same grid for both     spectra -- r, Linear-interpolate spec_src(s_src) onto s_ref, zero outside s_src's range., resample_to(), window_integrated_relative_error(), User-curated bank of additional cross-validation parameter sets.  Deliberately e, Cross-model validation suite -- entry point.  Runs kascade/xigma-i/xigma-i-direc, Tier 0 (config-only, no run needed) + Tier 1/2 (share one run of     each model) (+69 more)

### Community 1 - "compton_io/__init__.py"
Cohesion: 0.05
Nodes (59): Enum, Thin re-export of ``compton_io``'s parameter-semantics/unit normalisation framew, ModelSpec for kascade (formerly dfe5_compton_mc), at the boundary its own adapte, adapt_to_model(), params_to_floats(), Canonical params -> a specific model's own convention/units.  This is the last s, ``canonical_params`` may be in any convention/unit as long as the     meaning ma, Strip the semantic wrapper for the final call into model code, which     wants p (+51 more)

### Community 2 - "ComptonGuideApp"
Cohesion: 0.06
Nodes (23): add_field_grid(), ComptonGuideApp, _float_or_none(), laser_focal_radii(), main(), peak_a0(), Peak normalised vector potential a_0 from the laser fields (or None)., Transverse rms bunch size [m] from normalized emittance, beta [m] and gamma. (+15 more)

### Community 3 - "deposition.py"
Cohesion: 0.05
Nodes (44): _array_module(), build_table(), build_table_streaming(), _cell_indices(), check_accumulation_precision(), _deposit(), deposit_cic(), deposit_nearest() (+36 more)

### Community 4 - "particles.py"
Cohesion: 0.07
Nodes (40): ballistic_position_simultaneous(), ballistic_position_z0_reference(), laser_overlap_time_window(), propagate(), Ballistic (straight-line, no-acceleration) electron-beam propagation, and a time, Ballistically drift every macroparticle in ``bunch`` by a time     offset ``dt``, Per-particle time window ``[t0, t1]`` (normalised length units,     e.g. ``k0_la, Yield a propagated :class:`MacroBunch` snapshot at each time in     ``t_grid`` ( (+32 more)

### Community 5 - "kaskade/kascade.py"
Cohesion: 0.10
Nodes (34): beta_of(), build_lambda_grid(), Config, cos_collision(), doppler_D(), invert_lambda(), kn_over_thomson(), kn_sigma_ratio() (+26 more)

### Community 6 - "CommonResults"
Cohesion: 0.13
Nodes (27): AngularRangeSpectrumResult, ElectronFinalState, PhotonMultiplicity, Output-side observable representations: the spectrum/angular-spectrum/ temporal-, Unbinned per-macrophoton spectrum (event-generator models)., SampledSpatialDistribution, SampledSpectrum, SampledTemporalEnvelope (+19 more)

### Community 7 - "XigmaDirectAdapter"
Cohesion: 0.10
Nodes (25): build_params(), Derive this convention's CGS :class:`CollisionParams` from     ``compton_io``'s, InteractionGeometry, Collision geometry: foci displacement and crossing angle. SI units., BinnedAngularSpectrum, BinnedSpatialDistribution, BinnedTemporalEnvelope, _attach_private_cache() (+17 more)

### Community 8 - "GaussianElectronBeam"
Cohesion: 0.09
Nodes (6): beta_star_from_sigma_emit(), divergence_from_sigma_emit(), GaussianElectronBeam, Beta function at the waist, ``sigma**2 / emit_geom`` -- same units in     as out, RMS divergence at the waist, ``emit_geom / sigma`` -- a dimensionless     angle,, The ``gaussian_6d_waist`` v0.1 I/O contract.      A 6D factorized Gaussian defin

### Community 9 - "xigma_i/gui_adapter.py"
Cohesion: 0.08
Nodes (24): _attach_private_cache(), Config, extra_params(), _float(), ParamError, params_to_config(), Exception, ndarray (+16 more)

### Community 10 - "MacroBunch"
Cohesion: 0.09
Nodes (8): MacroBunch, Raw, engine-agnostic macroparticle representation. SI units.      ``x``/``y``/``, load_elegant_ele(), Parse a 6-D electron-bunch ``.ele`` file in SDDS ASCII format.      Required col, KascadeAdapter, _macrobunch_to_kascade_electrons(), Convert a MacroBunch into the plain dict kascade.run_simulation's     ``electron, XigmaAdapter

### Community 11 - "cache.py"
Cohesion: 0.18
Nodes (20): cache_key(), clear_cache(), get_or_compute(), _git_head_hash(), _git_is_dirty(), list_cache_entries(), load(), _paths() (+12 more)

### Community 12 - "headless_test.py"
Cohesion: 0.20
Nodes (13): check(), FakeVar, main(), make_fields(), Same mechanism app.py's on_start() uses: whichever model is     'selected', the, Stand-in for tk.StringVar -- params_to_config only ever calls .get()., sample_electrons_for(), test_model() (+5 more)

### Community 13 - "GaussianParaxialLaser"
Cohesion: 0.06
Nodes (24): CollisionParams, CGS scalars for one laser-electron collision -- immutable, built     once by :fu, InteractionParameters, One canonical (beam, pulse, geometry) triple -- the physics-parameter     subset, GaussianParaxialLaser, laser_from_shared_fields(), ``a0_at(z_m)**2`` -- the mean-square amplitude.          Consumers that only eve, Validate a :class:`GaussianParaxialLaser` per spec Sec. 12.      Raises ``ValueE (+16 more)

### Community 14 - "propagation.py"
Cohesion: 0.07
Nodes (29): 0. Краткое резюме, 10.1. Что уже работает, 10.2. Что нужно сделать для первого сравнения кодов, 10.3. Что должно войти в следующую версию паспорта, 10. Текущий статус и ближайшие шаги, 11. Итоговая оценка готовности, 1. Карточка кода / метода, 2. Назначение и роль в проекте (+21 more)

### Community 15 - "KascadeAdapter"
Cohesion: 0.07
Nodes (26): 10. Error Handling, 11. Extensibility, 12. Non-Goals, 1. Enums (enums.py), 2. Quantity Wrapper (quantities.py), 3. Canonical Representation (canonical.py), 4. Conversion Engine (converters.py), 5. Schema Definition (schema.py) (+18 more)

### Community 16 - "InteractionGeometry"
Cohesion: 0.09
Nodes (21): 10. Энергетический разброс, 11. Заряд, число электронов и пиковый ток, 12. Пиковая плотность, 13. 6D гауссово распределение v0.1, 14. Минимальный пример входного файла, 15. Рекомендуемые derived output-параметры, 16. Проверки валидности input, 17. Не входит в v0.1 (+13 more)

### Community 18 - "BinnedSpectrum"
Cohesion: 0.29
Nodes (6): BinnedSpectrum, Binned dN/dE spectrum (semi-analytic/tabulated models)., ParamError, Exception, Evaluate direct_binning_spectrum at a grid of OBSERVATION angles     spanning th, spectrum_in_angular_range()

### Community 19 - "sample_gaussian_bunch"
Cohesion: 0.17
Nodes (15): fit_gaussian(), Validate a :class:`GaussianElectronBeam` per spec Sec. 16.      Raises ``ValueEr, Draw macroparticles from a :class:`GaussianElectronBeam`.      Independent facto, Fit a :class:`GaussianElectronBeam` (at its waist) from raw macroparticles., sample_gaussian_bunch(), validate(), Cross-checks for bunch.py's MacroBunch / GaussianElectronBeam / sample_gaussian_, test_derived_quantities_are_sane() (+7 more)

### Community 20 - "laser.py"
Cohesion: 0.15
Nodes (14): External-format I/O for compton_io's bunch/laser representations., Elegant / SDDS ``.ele`` file I/O for :class:`compton_io.bunch.MacroBunch`.  Relo, Write a :class:`MacroBunch` in SDDS ASCII ``.ele`` format.      Mirrors :func:`l, save_elegant_ele(), load_electron_beam(), load_laser(), YAML I/O for the ``gaussian_6d_waist``/``gaussian_paraxial`` v0.1 formats define, save_electron_beam() (+6 more)

### Community 21 - "test_laser.py"
Cohesion: 0.10
Nodes (20): AGENTS.md, Architecture, Conventions, Convergence testing, Current state, Environment, GUI-facing engine -- `tabulated_engine.py`, GUI integration (`gui_adapter.py`) (+12 more)

### Community 22 - "bunch.py"
Cohesion: 0.10
Nodes (19): 10. Минимальный пример входного файла, 11. Рекомендуемые derived output-параметры, 12. Проверки валидности input, 13. Не входит в v0.1, 14. Mermaid-схема v0.1, 1. Назначение, 2. Геометрическая конвенция, 3.1. Таблица входных параметров (+11 more)

### Community 23 - "UnavailableAdapter"
Cohesion: 0.08
Nodes (17): ModelAdapter, Any, Model-agnostic contract between app.py and physics-engine adapters.  This module, Return (True, "") if the model can actually be run right now, else         (Fals, Model-specific numeric fields with no shared-panel analogue         (kascade's E, ``electrons`` is required: electron sampling is the IO layer's         (caller's, Raise NotImplementedError if capabilities().supports_ele_file_io is False., Raise NotImplementedError if capabilities().supports_ele_file_io is False. (+9 more)

### Community 24 - "_interp4d"
Cohesion: 0.27
Nodes (8): _interp4d(), Cross-validation-only tooling: a brute-force, non-GPU-kernel way to turn a Stage, Brute-force quadrature of dN/(ds dOmega) at a single observation point     (x0,, Resolves backend='numpy'|'cupy' to its array module. cupy is imported     lazily, Array-module-agnostic core of interp4d: takes H already converted to     xp's mo, Quadrilinear interpolation of table.H at query points (arrays of equal     shape, spectrum_from_table(), _xp_for()

### Community 25 - "analytical.py"
Cohesion: 0.24
Nodes (7): Electron-bunch representations.  Two distinct types, matching the distinction th, Physical constants, derived from ``units.py``'s pint registry rather than hand-t, ModelAdapter for the fast, always-on analytical model.  Duck-typed against ``com, angle_integrated_spectrum(), ndarray, Fast, closed-form Compton-source physics: total yield, angle-integrated spectrum, dN/ds integrated over all emission solid angle, from the standard     angle-inde

### Community 26 - "compton_suite/__init__.py"
Cohesion: 0.32
Nodes (5): ``comptonsuite-gui`` console-script entry point.  Thin wrapper launching the sam, run_gui(), ComptonSuite: the umbrella package tying together the toolkit's pieces (all now, run_gui(), Headless-safe model discovery for scripting -- no tkinter/matplotlib import, sam

### Community 27 - "detect_device"
Cohesion: 0.29
Nodes (7): detect_device(), Auto-detect which backend to use: a real CUDA GPU via cupy if     available, els, available(), available(), _backend_note(), True if either supported backend can actually run: a real CUDA GPU     (cupy + a, Which backend run_simulation will actually use, for display/logging     -- not p

### Community 29 - "AnalyticalAdapter"
Cohesion: 0.20
Nodes (3): AnalyticalAdapter, _float(), Fast closed-form model: total yield, angle-integrated spectrum, and     an estim

### Community 30 - "XigmaAdapter"
Cohesion: 0.11
Nodes (16): 1. Dead-code / unused-config sweep, 2. Move `CollisionParams`/`build_params` into `compton_io`, 3. GUI: reconsider "experimental" trust levels/warnings, 4. GUI: per-model sample count instead of a misleading global field, 5. Unify the `ModelAdapter` interface properly, 6. Manual CPU/GPU selection for xigma-i, 7. How to add a new model, Architecture (+8 more)

### Community 32 - "InteractionParameters"
Cohesion: 0.15
Nodes (12): 10. Проверки валидности, 11. Не входит в v0.1, 1. Scope, 2. Геометрическая конвенция, 3. Input-параметры, 4. Математическое определение, 5. Эмиттанс и расходимость, 6. Энергия и заряд (+4 more)

### Community 33 - "compton-io"
Cohesion: 0.50
Nodes (4): compton-analytical, compton-guide, compton-io, compton-suite

### Community 35 - "adapters/__init__.py"
Cohesion: 0.15
Nodes (12): 10. Проверки валидности input, 11. Не входит в v0.1, 1. Scope, 2. Геометрическая конвенция, 3. Input-параметры, 4. Математическое определение профиля, 5. Конверсии rms / FWHM / $1/e^2$, 6. Длина Рэлея и размер в точке взаимодействия (+4 more)

### Community 41 - "Compton-GUIde"
Cohesion: 0.20
Nodes (9): Adding a new GUI observable, Compton-GUIde, Known gaps, Layout, Model-specific parameters (`extra_params()`), Parameter semantics & units (`physics_params/`), and `compton_io`, Running it, Testing (headless, no display needed) (+1 more)

### Community 42 - "New GUI observables: status per model"
Cohesion: 0.25
Nodes (7): Angular Distribution (energy-integrated), Angular-Range-Restricted Spectrum, Headless testing, New GUI observables: status per model, Sequencing, Spatial (Transverse-Plane) Distribution, Temporal Envelope

### Community 43 - "KASCADE"
Cohesion: 0.25
Nodes (7): Files, KASCADE, No GPU dependency, Physical constants (`compton_io`), Relationship to other components, Testing, Units and conventions

### Community 44 - "kascade_adapter.py"
Cohesion: 0.33
Nodes (5): _float(), ParamError, Exception, ModelAdapter wrapping the existing kascade engine.  This is the "control" adapte, Convert the GUI fields into a physics Config plus a driver dict.          Ported

### Community 45 - "compton_io"
Cohesion: 0.33
Nodes (5): compton_io, Layout, Naming, Testing, Why this exists

### Community 46 - "opencode.json"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

## Knowledge Gaps
- **165 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `compton-guide`, `compton-analytical`, `kascade` (+160 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MacroBunch` connect `MacroBunch` to `particles.py`, `kaskade/kascade.py`, `CommonResults`, `XigmaDirectAdapter`, `ModelCapabilities`, `xigma_i/gui_adapter.py`, `kascade_adapter.py`, `AnalyticalConfig`, `BinnedSpectrum`, `sample_gaussian_bunch`, `laser.py`, `UnavailableAdapter`, `analytical.py`, `AnalyticalAdapter`?**
  _High betweenness centrality (0.102) - this node is a cross-community bridge._
- **Why does `ComptonGuideApp` connect `ComptonGuideApp` to `UnavailableAdapter`, `beam_from_shared_fields`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Why does `Scenario` connect `Scenario` to `kaskade/kascade.py`, `CommonResults`, `XigmaDirectAdapter`, `GaussianElectronBeam`, `GaussianParaxialLaser`, `AnalyticalConfig`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Are the 19 inferred relationships involving `MacroBunch` (e.g. with `ModelAdapter` and `ModelCapabilities`) actually correct?**
  _`MacroBunch` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `Scenario` (e.g. with `GaussianElectronBeam` and `InteractionGeometry`) actually correct?**
  _`Scenario` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `GaussianElectronBeam` (e.g. with `CollisionParams` and `InteractionGeometry`) actually correct?**
  _`GaussianElectronBeam` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `GaussianParaxialLaser` (e.g. with `CollisionParams` and `InteractionGeometry`) actually correct?**
  _`GaussianParaxialLaser` has 9 INFERRED edges - model-reasoned connections that need verification._