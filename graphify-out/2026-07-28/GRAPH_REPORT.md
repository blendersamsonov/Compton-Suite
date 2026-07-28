# Graph Report - ComptonSuite  (2026-07-28)

## Corpus Check
- 95 files · ~101,125 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1128 nodes · 2263 edges · 62 communities (49 shown, 13 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 194 edges (avg confidence: 0.53)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6575f45a`
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
- .run
- .available
- opencode.json
- ._apply_model_capabilities
- ._update_derived
- BinnedSpectrum
- graphify.js
- peak_a0
- ._load_ele
- ComptonSuite model-agnostic-core refactor: status
- models.py
- test_laser.py
- UnavailableAdapter
- AnalyticalAdapter
- detect_device
- .run
- beta_star_from_sigma_emit
- divergence_from_sigma_emit

## God Nodes (most connected - your core abstractions)
1. `MacroBunch` - 54 edges
2. `GaussianElectronBeam` - 43 edges
3. `Config` - 40 edges
4. `Scenario` - 39 edges
5. `ComptonGuideApp` - 38 edges
6. `DirectConfig` - 38 edges
7. `Config` - 38 edges
8. `GaussianParaxialLaser` - 37 edges
9. `CommonResults` - 37 edges
10. `ModelCapabilities` - 34 edges

## Surprising Connections (you probably didn't know these)
- `run()` --calls--> `check()`  [INFERRED]
  src/compton_suite/validation/tier0_wiring.py → scripts/headless_test.py
- `test_laser_from_shared_fields_round_trips()` --calls--> `laser_from_shared_fields()`  [EXTRACTED]
  tests/io_tests/test_laser.py → src/compton_suite/io/laser.py
- `sample_electrons_for()` --calls--> `beam_from_shared_fields()`  [EXTRACTED]
  scripts/headless_test.py → src/compton_suite/io/bunch.py
- `sample_electrons_for()` --calls--> `sample_gaussian_bunch()`  [EXTRACTED]
  scripts/headless_test.py → src/compton_suite/io/bunch.py
- `test_model()` --calls--> `validate_results()`  [INFERRED]
  scripts/headless_test.py → src/compton_suite/io/results.py

## Import Cycles
- None detected.

## Communities (62 total, 13 thin omitted)

### Community 0 - "Scenario"
Cohesion: 0.05
Nodes (39): 10. Future extensions, 11. Key insight, (1) Always sample in canonical variables, 1. Goal, (2) Enforce mass-shell, 2. Representation, (3) Center data before fitting, 3. Sampling (Gaussian at waist) (+31 more)

### Community 1 - "compton_io/__init__.py"
Cohesion: 0.08
Nodes (57): Enum, ModelSpec, Quantity, Thin re-export of ``compton_suite.io``'s parameter-semantics/unit normalisation, adapt_to_model(), params_to_floats(), Canonical params -> a specific model's own convention/units.  This is the last s, ``canonical_params`` may be in any convention/unit as long as the     meaning ma (+49 more)

### Community 2 - "ComptonGuideApp"
Cohesion: 0.07
Nodes (75): Shared comparison metric for the cross-model validation suite.  Promoted (moved, s, spec, spec_ref: 1D arrays of equal shape (same grid for both     spectra -- r, Linear-interpolate spec_src(s_src) onto s_ref, zero outside s_src's range., resample_to(), window_integrated_relative_error(), User-curated bank of additional cross-validation parameter sets.  Deliberately e, Cross-model validation suite -- entry point.  Runs kascade/xigma-i/delta/analyti, Tier 0 (config-only, no run needed) + Tier 1/2 (share one run of     each model) (+67 more)

### Community 3 - "deposition.py"
Cohesion: 0.08
Nodes (22): _attach_private_cache(), capabilities(), extra_choices(), extra_params(), ndarray, GUI-facing adapter exposing the xigma_i pipeline through a dfe5-shaped Config/ru, Stash this adapter's own private recompute cache on a     ``compton_io.results.C, Model-specific numeric fields with no dfe5 analogue, for the GUI to     render i (+14 more)

### Community 4 - "particles.py"
Cohesion: 0.10
Nodes (18): Raise NotImplementedError if capabilities().supports_ele_file_io is False., Raise NotImplementedError if capabilities().supports_ele_file_io is False., fit_gaussian(), MacroBunch, Fit a :class:`GaussianElectronBeam` (at its waist) from raw macroparticles., Raw, engine-agnostic macroparticle representation. SI units.      ``x``/``y``/``, load_elegant_ele(), Elegant / SDDS ``.ele`` file I/O for :class:`compton_io.bunch.MacroBunch`.  Relo (+10 more)

### Community 5 - "kaskade/kascade.py"
Cohesion: 0.07
Nodes (37): _array_module(), build_table(), build_table_streaming(), _cell_indices(), check_accumulation_precision(), _deposit(), deposit_cic(), deposit_nearest() (+29 more)

### Community 6 - "CommonResults"
Cohesion: 0.07
Nodes (43): gaussian_pulse_envelope(), Shared spatiotemporal Gaussian-pulse envelope -- the piece ``compton_io.laser.Ga, Photon-density envelope of a Gaussian laser pulse at an arbitrary     point in s, ballistic_position_z0_reference(), laser_overlap_time_window(), Per-particle time window ``[t0, t1]`` (normalised length units,     e.g. ``k0_la, Straight-line position at time offset ``t``, given a per-particle     reference, _bin_spatial() (+35 more)

### Community 7 - "XigmaDirectAdapter"
Cohesion: 0.07
Nodes (8): _attach_private_cache(), DirectConfig, ndarray, Stash this adapter's own private recompute cache on a     ``compton_io.results.C, ``electrons`` is required: electron sampling is the caller's job,     not this a, SI-unit physics config, trimmed to what Stage 0 (particles.py)     needs -- no S, run_simulation(), _theta_grid()

### Community 8 - "GaussianElectronBeam"
Cohesion: 0.12
Nodes (7): GaussianElectronBeam, The ``gaussian_6d_waist`` v0.1 I/O contract.      A 6D factorized Gaussian defin, Validate a :class:`GaussianElectronBeam` per spec Sec. 16.      Raises ``ValueEr, validate(), test_derived_quantities_are_sane(), test_validate_rejects_nonpositive_fields(), test_validate_warns_on_suspicious_emittance()

### Community 9 - "xigma_i/gui_adapter.py"
Cohesion: 0.19
Nodes (11): Electron-bunch representations.  Two distinct types, matching the distinction th, Draw macroparticles from a :class:`GaussianElectronBeam`.      Independent facto, sample_gaussian_bunch(), Physical constants, derived from ``units.py``'s pint registry rather than hand-t, The shared "interaction parameters" bundle: an electron beam, a laser pulse, and, Cross-checks for bunch.py's MacroBunch / GaussianElectronBeam / sample_gaussian_, test_fit_gaussian_recovers_waist_after_drift(), test_fit_gaussian_round_trips_at_the_waist() (+3 more)

### Community 10 - "MacroBunch"
Cohesion: 0.10
Nodes (33): KASCADE: Sequential multi-photon inverse-Compton Monte Carlo engine., beta_of(), build_lambda_grid(), cos_collision(), doppler_D(), invert_lambda(), kn_over_thomson(), kn_sigma_ratio() (+25 more)

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
Cohesion: 0.15
Nodes (5): GaussianParaxialLaser, ``a0_at(z_m)**2`` -- the mean-square amplitude.          Consumers that only eve, A paraxial Gaussian laser pulse. SI units.      ``waist_rms_x_m``/``waist_rms_y_, Photon count in the pulse: N_L = pulse_energy_J / (hbar*omega0)., Normalized vector potential at ``z_m``, from the exact SI         plane-wave rel

### Community 20 - "laser.py"
Cohesion: 0.16
Nodes (15): angle_integrated_spectrum(), estimate_spectrum_width(), estimate_yield(), ndarray, Fast, closed-form Compton-source physics: total yield, angle-integrated spectrum, Cheap analytic total-photon-yield estimate from an overlap integral     between, Cheap analytic estimate of the collimated-spectrum FWHM (in units of     the Com, dN/ds integrated over all emission solid angle, from the standard     angle-inde (+7 more)

### Community 21 - "test_laser.py"
Cohesion: 0.20
Nodes (13): check(), FakeVar, main(), make_fields(), Same mechanism app.py's on_start() uses: whichever model is     'selected', the, Stand-in for tk.StringVar -- params_to_config only ever calls .get()., sample_electrons_for(), test_model() (+5 more)

### Community 22 - "bunch.py"
Cohesion: 0.22
Nodes (12): ballistic_position_simultaneous(), propagate(), Ballistic (straight-line, no-acceleration) electron-beam propagation, and a time, Ballistically drift every macroparticle in ``bunch`` by a time     offset ``dt``, Yield a propagated :class:`MacroBunch` snapshot at each time in     ``t_grid`` (, Straight-line position at time offset ``dt`` later, given a     per-particle ref, stream(), Cross-checks for propagation.py's ballistic drift primitives.  No cupy/GPU/tkint (+4 more)

### Community 23 - "UnavailableAdapter"
Cohesion: 0.17
Nodes (23): Protocol, ModelAdapter, ModelCapabilities, Model-agnostic contract between app.py and physics-engine adapters.  This module, Return (True, "") if the model can actually be run right now, else         (Fals, Model-specific fields with no shared-panel analogue         (kascade's Electrons, Optional: return a dict mapping parameter keys to allowed string values, Optional capability -- check         capabilities().supports_angular_range_spect (+15 more)

### Community 24 - "run_multiphoton_chain"
Cohesion: 0.17
Nodes (12): External-format I/O for compton_io's bunch/laser representations., load_electron_beam(), load_laser(), YAML I/O for the ``gaussian_6d_waist``/``gaussian_paraxial`` v0.1 formats define, save_electron_beam(), save_laser(), a0_from_fields(), Laser-pulse representation: the ``gaussian_paraxial`` v0.1 I/O contract (``specs (+4 more)

### Community 25 - "analytical.py"
Cohesion: 0.15
Nodes (12): 10. Проверки валидности, 11. Не входит в v0.1, 1. Scope, 2. Геометрическая конвенция, 3. Input-параметры, 4. Математическое определение, 5. Эмиттанс и расходимость, 6. Энергия и заряд (+4 more)

### Community 26 - "compton_suite/__init__.py"
Cohesion: 0.18
Nodes (3): ComptonGuideApp, Update the preview panel from self.preview_res (set by         _poll_queue). Nev, Return (total_flux, collimated_flux, cmask_or_None) [ph/s].          cmask is on

### Community 27 - "detect_device"
Cohesion: 0.15
Nodes (12): 10. Проверки валидности input, 11. Не входит в v0.1, 1. Scope, 2. Геометрическая конвенция, 3. Input-параметры, 4. Математическое определение профиля, 5. Конверсии rms / FWHM / $1/e^2$, 6. Длина Рэлея и размер в точке взаимодействия (+4 more)

### Community 28 - "test_constants.py"
Cohesion: 0.22
Nodes (4): AnalyticalConfig, _float(), ModelAdapter for the fast, always-on analytical model.  Duck-typed against ``com, Wraps beam/pulse, but ALSO exposes a handful of flat SI fields every     other m

### Community 29 - "AnalyticalAdapter"
Cohesion: 0.18
Nodes (3): DeltaAdapter, delta: brute-force per-macroparticle resonance-binning model.  Extracted from ``, Physics engine models for ComptonSuite.  Each model implements the ModelAdapter

### Community 30 - "XigmaAdapter"
Cohesion: 0.08
Nodes (24): 1. Dead-code / unused-config sweep ✅, 2. Move `CollisionParams`/`build_params` into `io/` ✅, 3. GUI: trust levels/warnings ✅, 4. GUI: per-model sample count ✅, 5. Unify the `ModelAdapter` interface ✅, 6. Manual CPU/GPU selection ✅, 7. How to add a new model ✅, 8. GUI-as-thin-consumer ✅ (+16 more)

### Community 31 - "test_analytical.py"
Cohesion: 0.14
Nodes (19): beam_from_shared_fields(), Build a :class:`GaussianElectronBeam` from the flat SI field set every     model, InteractionParameters, One canonical (beam, pulse, geometry) triple -- the physics-parameter     subset, laser_from_shared_fields(), Build a :class:`GaussianParaxialLaser` from the flat SI field set every     mode, extra_choices(), _float() (+11 more)

### Community 32 - "InteractionParameters"
Cohesion: 0.21
Nodes (8): Physics constants and GPU kernel sizing constants for this pipeline (particles.p, calculate_angular_spectrum_4d(), _get_numba(), get_spectrum_kernel_4d_cpu(), CPU/numba equivalent of spectrum4d.py's GPU rawkernel (spectrum_kernel_4d), used, Lazily compiles and caches spectrum_kernel_4d_cpu (see module docstring)., Stage 2: `spectrum_kernel_4d`, the GPU kernel that turns a Stage 1 4D overlap ta, Host-side driver for spectrum_kernel_4d, the Stage-2 analogue of     Compton.cal

### Community 34 - "schemas/__init__.py"
Cohesion: 0.20
Nodes (9): Adding a new GUI observable, Compton-GUIde, Known gaps, Layout, Model-specific parameters (`extra_params()`), Parameter semantics & units (`physics_params/`), and `compton_suite.io`, Running it, Testing (headless, no display needed) (+1 more)

### Community 35 - "adapters/__init__.py"
Cohesion: 0.25
Nodes (7): Angular Distribution (energy-integrated), Angular-Range-Restricted Spectrum, Headless testing, New GUI observables: status per model, Sequencing, Spatial (Transverse-Plane) Distribution, Temporal Envelope

### Community 36 - "compton_guide/__init__.py"
Cohesion: 0.25
Nodes (7): Files, KASCADE, No GPU dependency, Physical constants (`compton_suite.io`), Relationship to other components, Testing, Units and conventions

### Community 37 - "kascade"
Cohesion: 0.25
Nodes (7): Files, KASCADE, No GPU dependency, Physical constants (`compton_suite.io`), Relationship to other components, Testing, Units and conventions

### Community 39 - "xigma-i"
Cohesion: 0.33
Nodes (5): compton_suite.io, Layout, Naming, Testing, Why this exists

### Community 41 - "Compton-GUIde"
Cohesion: 0.20
Nodes (10): Evaluate direct_binning_spectrum at a grid of OBSERVATION angles     spanning th, spectrum_in_angular_range(), angle_integrated_spectrum(), direct_binning_spectrum(), Table-free spectrum paths computed directly from Stage 0/1 macroparticles -- no, Resolves backend='numpy'|'cupy' to its array module. cupy is imported     lazily, dN/ds integrated over all emission solid angle, from real Stage 0/1     macropar, `delta`'s actual computation: for each real macroparticle,     compute the photo (+2 more)

### Community 42 - "New GUI observables: status per model"
Cohesion: 0.13
Nodes (15): _float_or_none(), laser_focal_radii(), peak_a0(), Transverse rms bunch size [m] from normalized emittance, beta [m] and gamma., Return radial RMS, FWHM, and exp(-1/2) focal radii [m].      Delegates to :func:, Refresh electron and laser values derived from the current fields., Peak normalised vector potential a_0 from the laser fields (or None).      Deleg, sigma_e() (+7 more)

### Community 44 - ".run"
Cohesion: 0.25
Nodes (4): add_field_grid(), Place (label, default, key) triples in an n_cols-wide grid inside a     coloured, Small always-visible panel showing the fast analytical model's         estimate,, StringVar

### Community 46 - "opencode.json"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

### Community 47 - "._apply_model_capabilities"
Cohesion: 0.20
Nodes (4): Grey out / restore controls the active model doesn't support, and         refres, Drop the loaded ``.ele`` bunch and restore the Electrons panel         to its no, Restore the Electron-panel entries to editable input mode and         re-attach, Repopulate the Model Parameters panel from         ``self.active_adapter.extra_p

### Community 49 - "BinnedSpectrum"
Cohesion: 0.40
Nodes (4): BinnedSpectrum, Binned dN/dE spectrum (semi-analytic/tabulated models)., ParamError, Exception

### Community 51 - "peak_a0"
Cohesion: 0.22
Nodes (8): build_params(), CollisionParams, The CGS "collision parameters" bundle for tabulated-overlap-style GPU/CPU pipeli, Derive this convention's CGS :class:`CollisionParams` from     ``compton_io``'s, CGS scalars for one laser-electron collision -- immutable, built     once by :fu, InteractionGeometry, Collision geometry: foci displacement and crossing angle. SI units., This scenario's (beam, pulse, geometry) as one shared         compton_suite.io.i

### Community 53 - "ComptonSuite model-agnostic-core refactor: status"
Cohesion: 0.29
Nodes (6): ComptonSuite model-agnostic-core refactor: status, Current package layout, Explicitly dropped from the original plan, Landed — moves that fulfilled the original goal without a `core/` package, Still open, What changed since this doc was first written

### Community 54 - "models.py"
Cohesion: 0.24
Nodes (9): main(), Compton-GUIde: Tkinter GUI for pluggable Compton-scattering physics models., register(), registered_models(), discover_models(), Model registry -- direct imports since all packages ship together., Populate the model registry with direct imports., ComptonSuite: unified package for inverse-Compton scattering simulation.  Subpac (+1 more)

### Community 55 - "test_laser.py"
Cohesion: 0.21
Nodes (8): Validate a :class:`GaussianParaxialLaser` per spec Sec. 12.      Raises ``ValueE, validate(), Cross-checks for laser.py's GaussianParaxialLaser.  No cupy/GPU/tkinter needed., test_defocused_interaction_intensity_is_lower(), test_derived_quantities_are_sane(), test_laser_from_shared_fields_round_trips(), test_validate_rejects_nonpositive_fields(), test_validate_warns_on_astigmatism()

### Community 58 - "detect_device"
Cohesion: 0.29
Nodes (7): detect_device(), Auto-detect which backend to use: a real CUDA GPU via cupy if     available, els, available(), available(), _backend_note(), True if either supported backend can actually run: a real CUDA GPU     (cupy + a, Which backend run_simulation will actually use, for display/logging     -- not p

## Knowledge Gaps
- **220 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `compton-suite`, `What this is`, `Layout` (+215 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MacroBunch` connect `particles.py` to `deposition.py`, `XigmaDirectAdapter`, `XigmaAdapter`, `xigma_i/gui_adapter.py`, `MacroBunch`, `cache.py`, `.available`, `._update_derived`, `BinnedSpectrum`, `bunch.py`, `UnavailableAdapter`, `UnavailableAdapter`, `AnalyticalAdapter`, `.run`, `test_constants.py`, `AnalyticalAdapter`, `test_analytical.py`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Why does `Config` connect `._update_derived` to `MacroBunch`, `ComptonGuideApp`, `particles.py`, `test_analytical.py`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Why does `sample_gaussian_bunch()` connect `xigma_i/gui_adapter.py` to `ComptonGuideApp`, `particles.py`, `kaskade/kascade.py`, `GaussianElectronBeam`, `New GUI observables: status per model`, `test_laser.py`, `bunch.py`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `MacroBunch` (e.g. with `ModelAdapter` and `ModelCapabilities`) actually correct?**
  _`MacroBunch` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `GaussianElectronBeam` (e.g. with `CollisionParams` and `InteractionGeometry`) actually correct?**
  _`GaussianElectronBeam` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Config` (e.g. with `MacroBunch` and `InteractionParameters`) actually correct?**
  _`Config` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `Scenario` (e.g. with `GaussianElectronBeam` and `InteractionGeometry`) actually correct?**
  _`Scenario` has 4 INFERRED edges - model-reasoned connections that need verification._