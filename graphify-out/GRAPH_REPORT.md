# Graph Report - ComptonSuite  (2026-07-28)

## Corpus Check
- 92 files · ~81,373 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1114 nodes · 2272 edges · 52 communities (43 shown, 9 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 208 edges (avg confidence: 0.53)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f13d223c`
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
- KASCADE
- .run
- .available
- opencode.json
- ._apply_model_capabilities
- ._update_derived
- graphify.js
- ._load_ele
- ComptonSuite model-agnostic-core refactor: status
- models.py

## God Nodes (most connected - your core abstractions)
1. `MacroBunch` - 55 edges
2. `GaussianElectronBeam` - 43 edges
3. `ComptonGuideApp` - 41 edges
4. `Config` - 40 edges
5. `Scenario` - 40 edges
6. `CommonResults` - 38 edges
7. `DirectConfig` - 38 edges
8. `Config` - 38 edges
9. `GaussianParaxialLaser` - 37 edges
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

## Communities (52 total, 9 thin omitted)

### Community 0 - "Scenario"
Cohesion: 0.05
Nodes (39): 10. Future extensions, 11. Key insight, (1) Always sample in canonical variables, 1. Goal, (2) Enforce mass-shell, 2. Representation, (3) Center data before fitting, 3. Sampling (Gaussian at waist) (+31 more)

### Community 1 - "compton_io/__init__.py"
Cohesion: 0.08
Nodes (57): Enum, ModelSpec, Quantity, Thin re-export of ``compton_suite.io``'s parameter-semantics/unit normalisation, adapt_to_model(), params_to_floats(), Canonical params -> a specific model's own convention/units.  This is the last s, ``canonical_params`` may be in any convention/unit as long as the     meaning ma (+49 more)

### Community 2 - "ComptonGuideApp"
Cohesion: 0.07
Nodes (77): Shared comparison metric for the cross-model validation suite.  Promoted (moved, s, spec, spec_ref: 1D arrays of equal shape (same grid for both     spectra -- r, Linear-interpolate spec_src(s_src) onto s_ref, zero outside s_src's range., resample_to(), window_integrated_relative_error(), User-curated bank of additional cross-validation parameter sets.  Deliberately e, Cross-model validation suite -- entry point.  Runs kascade/xigma-i/delta/analyti, Tier 0 (config-only, no run needed) + Tier 1/2 (share one run of     each model) (+69 more)

### Community 3 - "deposition.py"
Cohesion: 0.07
Nodes (27): detect_device(), Auto-detect which backend to use: a real CUDA GPU via cupy if     available, els, laser_from_shared_fields(), Build a :class:`GaussianParaxialLaser` from the flat SI field set every     mode, available(), available(), _backend_note(), capabilities() (+19 more)

### Community 4 - "particles.py"
Cohesion: 0.07
Nodes (20): Raise NotImplementedError if capabilities().supports_ele_file_io is False., Raise NotImplementedError if capabilities().supports_ele_file_io is False., fit_gaussian(), MacroBunch, Electron-bunch representations.  Two distinct types, matching the distinction th, Fit a :class:`GaussianElectronBeam` (at its waist) from raw macroparticles., Raw, engine-agnostic macroparticle representation. SI units.      ``x``/``y``/``, load_elegant_ele() (+12 more)

### Community 5 - "kaskade/kascade.py"
Cohesion: 0.07
Nodes (37): _array_module(), build_table(), build_table_streaming(), _cell_indices(), check_accumulation_precision(), _deposit(), deposit_cic(), deposit_nearest() (+29 more)

### Community 6 - "CommonResults"
Cohesion: 0.06
Nodes (55): gaussian_pulse_envelope(), Shared spatiotemporal Gaussian-pulse envelope -- the piece ``compton_suite.io.la, Photon-density envelope of a Gaussian laser pulse at an arbitrary     point in s, ballistic_position_simultaneous(), ballistic_position_z0_reference(), laser_overlap_time_window(), propagate(), Ballistic (straight-line, no-acceleration) electron-beam propagation, and a time (+47 more)

### Community 8 - "GaussianElectronBeam"
Cohesion: 0.05
Nodes (34): beta_star_from_sigma_emit(), divergence_from_sigma_emit(), GaussianElectronBeam, The ``gaussian_6d_waist`` v0.1 I/O contract.      A 6D factorized Gaussian defin, Validate a :class:`GaussianElectronBeam` per spec Sec. 16.      Raises ``ValueEr, Draw macroparticles from a :class:`GaussianElectronBeam`.      Independent facto, Beta function at the waist, ``sigma**2 / emit_geom`` -- same units in     as out, RMS divergence at the waist, ``emit_geom / sigma`` -- a dimensionless     angle, (+26 more)

### Community 9 - "xigma_i/gui_adapter.py"
Cohesion: 0.13
Nodes (7): Stub registered when a model's real adapter couldn't be imported/used     (e.g., register(), registered_models(), UnavailableAdapter, discover_models(), Model registry -- direct imports since all packages ship together., Populate the model registry with direct imports.

### Community 10 - "MacroBunch"
Cohesion: 0.19
Nodes (8): CODATA-style physical constants used by the GUI's local formula helpers (peak_a0, The CGS "collision parameters" bundle for tabulated-overlap-style GPU/CPU pipeli, Physical constants, derived from ``units.py``'s pint registry rather than hand-t, The shared "interaction parameters" bundle: an electron beam, a laser pulse, and, Quantum recoil parameter ``q = 4 * gamma * hbar * omega / (m_e * c^2)``.      Di, recoil_parameter(), Laser-pulse representation: the ``gaussian_paraxial`` v0.1 I/O contract (``specs, Physics constants and GPU kernel sizing constants for this pipeline (particles.p

### Community 11 - "cache.py"
Cohesion: 0.08
Nodes (5): Config, ndarray, A generous fixed window around the current collimation angle, wide     enough th, SI-unit physics config for the xigma-i model.      Field names/units mirror ``df, _theta_grid()

### Community 12 - "headless_test.py"
Cohesion: 0.07
Nodes (29): 0. Краткое резюме, 10.1. Что уже работает, 10.2. Что нужно сделать для первого сравнения кодов, 10.3. Что должно войти в следующую версию паспорта, 10. Текущий статус и ближайшие шаги, 11. Итоговая оценка готовности, 1. Карточка кода / метода, 2. Назначение и роль в проекте (+21 more)

### Community 13 - "CollisionParams pruning + finishing the PhysicalQuantity/ModelSpec migration"
Cohesion: 0.12
Nodes (16): Also confirmed safe, CollisionParams pruning + finishing the PhysicalQuantity/ModelSpec migration, Context, Design decision: keep the `geometry` parameter, drop the fields it feeds, Evidence, Explicitly out of scope for both phases, kascade has no `ModelSpec` at all yet, Phase 1: Prune `CollisionParams`'s dead fields ✅ (+8 more)

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
Cohesion: 0.07
Nodes (22): External-format I/O for compton_suite.io's bunch/laser representations., load_electron_beam(), load_laser(), YAML I/O for the ``gaussian_6d_waist``/``gaussian_paraxial`` v0.1 formats define, save_electron_beam(), save_laser(), GaussianParaxialLaser, ``a0_at(z_m)**2`` -- the mean-square amplitude.          Consumers that only eve (+14 more)

### Community 20 - "laser.py"
Cohesion: 0.22
Nodes (4): CollisionParams, CGS scalars for one laser-electron collision -- immutable, built     once by :fu, delta: brute-force per-macroparticle resonance-binning model.  Extracted from ``, Physics engine models for ComptonSuite.  Each model implements the ModelAdapter

### Community 21 - "test_laser.py"
Cohesion: 0.20
Nodes (13): check(), FakeVar, main(), make_fields(), Same mechanism app.py's on_start() uses: whichever model is     'selected', the, Stand-in for tk.StringVar -- params_to_config only ever calls .get()., sample_electrons_for(), test_model() (+5 more)

### Community 22 - "bunch.py"
Cohesion: 0.22
Nodes (11): Write a :class:`MacroBunch` in SDDS ASCII ``.ele`` format.      Mirrors :func:`l, save_elegant_ele(), KASCADE: Sequential multi-photon inverse-Compton Monte Carlo engine., kn_sigma_ratio(), L_theta(), Total Klein--Nishina cross section relative to Thomson, sigma_KN(k)/sigma_T., Normalised (linearly-polarised) Thomson angular distribution., Run the multi-photon inverse-Compton Monte-Carlo.      Parameters     ---------- (+3 more)

### Community 23 - "UnavailableAdapter"
Cohesion: 0.14
Nodes (32): Protocol, ModelAdapter, ModelCapabilities, OutputSpec, Any, Model-agnostic contract between app.py and physics-engine adapters.  This module, Model-specific fields with no shared-panel analogue         (kascade's Electrons, Optional: return a dict mapping parameter keys to allowed string values (+24 more)

### Community 25 - "analytical.py"
Cohesion: 0.15
Nodes (12): 10. Проверки валидности, 11. Не входит в v0.1, 1. Scope, 2. Геометрическая конвенция, 3. Input-параметры, 4. Математическое определение, 5. Эмиттанс и расходимость, 6. Энергия и заряд (+4 more)

### Community 26 - "compton_suite/__init__.py"
Cohesion: 0.18
Nodes (3): ComptonGuideApp, Update the preview panel from self.preview_res (set by         _poll_queue). Nev, Return (total_flux, collimated_flux, cmask_or_None) [ph/s].          cmask is on

### Community 27 - "detect_device"
Cohesion: 0.15
Nodes (12): 10. Проверки валидности input, 11. Не входит в v0.1, 1. Scope, 2. Геометрическая конвенция, 3. Input-параметры, 4. Математическое определение профиля, 5. Конверсии rms / FWHM / $1/e^2$, 6. Длина Рэлея и размер в точке взаимодействия (+4 more)

### Community 29 - "AnalyticalAdapter"
Cohesion: 0.12
Nodes (9): _attach_private_cache(), DeltaAdapter, ndarray, Stash this adapter's own private recompute cache on a     ``compton_suite.io.res, ``electrons`` is required: electron sampling is the caller's job,     not this a, run_simulation(), _theta_grid(), direct_binning_spectrum() (+1 more)

### Community 30 - "XigmaAdapter"
Cohesion: 0.08
Nodes (24): 1. Dead-code / unused-config sweep ✅, 2. Move `CollisionParams`/`build_params` into `io/` ✅, 3. GUI: trust levels/warnings ✅, 4. GUI: per-model sample count ✅, 5. Unify the `ModelAdapter` interface ✅, 6. Manual CPU/GPU selection ✅, 7. How to add a new model ✅, 8. GUI-as-thin-consumer ✅ (+16 more)

### Community 31 - "test_analytical.py"
Cohesion: 0.12
Nodes (22): beam_from_shared_fields(), Build a :class:`GaussianElectronBeam` from the flat SI field set every     model, build_params(), Derive this convention's CGS :class:`CollisionParams` from     ``compton_suite.i, InteractionGeometry, InteractionParameters, Collision geometry: foci displacement and crossing angle. SI units., One canonical (beam, pulse, geometry) triple -- the physics-parameter     subset (+14 more)

### Community 32 - "InteractionParameters"
Cohesion: 0.24
Nodes (7): calculate_angular_spectrum_4d(), _get_numba(), get_spectrum_kernel_4d_cpu(), CPU/numba equivalent of spectrum4d.py's GPU rawkernel (spectrum_kernel_4d), used, Lazily compiles and caches spectrum_kernel_4d_cpu (see module docstring)., Stage 2: `spectrum_kernel_4d`, the GPU kernel that turns a Stage 1 4D overlap ta, Host-side driver for spectrum_kernel_4d, the Stage-2 analogue of     Compton.cal

### Community 35 - "adapters/__init__.py"
Cohesion: 0.24
Nodes (10): beta_of(), build_lambda_grid(), cos_collision(), doppler_D(), v/c from the normalised energy eps (= Lorentz gamma), clipped physical., cos(angle between +z electron direction and laser propagation n_L).      = n_L ., Doppler up-shift factor D = (1 - beta cos alpha)/(1 - beta).      On-axis Thomso, Recoil parameter X = D eps_L/eps = 2k (k = rest-frame photon energy).      The r (+2 more)

### Community 36 - "compton_guide/__init__.py"
Cohesion: 0.25
Nodes (9): invert_lambda(), kn_over_thomson(), ndarray, Klein--Nishina / Thomson differential-cross-section ratio R_KN in [0,1].      ``, Sample (thx, thy) from L_theta (classical) or L_theta * R_KN (quantum).      Pro, Solve Lambda(tau_emit) = threshold row-by-row (Lambda monotonic in tau)., Simulate the emission chain for a chunk of electrons.      Records, per photon,, run_multiphoton_chain() (+1 more)

### Community 37 - "kascade"
Cohesion: 0.25
Nodes (7): Files, KASCADE, No GPU dependency, Physical constants (`compton_suite.io`), Relationship to other components, Testing, Units and conventions

### Community 39 - "xigma-i"
Cohesion: 0.33
Nodes (5): compton_suite.io, Layout, Naming, Testing, Why this exists

### Community 41 - "Compton-GUIde"
Cohesion: 0.29
Nodes (6): angle_integrated_spectrum(), Table-free spectrum paths computed directly from Stage 0/1 macroparticles -- no, Resolves backend='numpy'|'cupy' to its array module. cupy is imported     lazily, dN/ds integrated over all emission solid angle, from real Stage 0/1     macropar, _xp_for(), GUI-facing engine on the tabulated-energy pipeline (particles.py/ deposition.py/

### Community 42 - "New GUI observables: status per model"
Cohesion: 0.22
Nodes (7): _float_or_none(), peak_a0(), Build an OutputSpec from the GUI fields., Refresh electron and laser values derived from the current fields., Peak normalised vector potential a_0 from the laser fields (or None).      Deleg, a0_from_fields(), Peak normalised vector potential a_0 from raw SI laser fields.      This is the

### Community 43 - "KASCADE"
Cohesion: 0.33
Nodes (6): laser_a0sq(), laser_axis(), laser_density(), Laser propagation unit vector n_L = (sin phi, 0, -cos phi).      phi = crossing_, Gaussian laser photon density propagating along n_L, focus at delta.      Reduce, Normalised laser vector-potential squared a_0^2(r,t) (as in dfe4).

### Community 44 - ".run"
Cohesion: 0.17
Nodes (6): add_field_grid(), Place (label, default, key) triples in an n_cols-wide grid inside a     coloured, Repopulate the Model Parameters panel from         ``self.active_adapter.extra_p, Panel for configuring output resolution (spectrum, temporal,         spatial, an, Small always-visible panel showing the fast analytical model's         estimate,, StringVar

### Community 45 - ".available"
Cohesion: 0.17
Nodes (5): _attach_private_cache(), Stash this adapter's own private recompute cache on a     ``compton_suite.io.res, ``n_mc`` is accepted for ``ModelAdapter.run`` signature compatibility     but un, run_simulation(), XigmaAdapter

### Community 46 - "opencode.json"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

### Community 47 - "._apply_model_capabilities"
Cohesion: 0.29
Nodes (3): Grey out / restore controls the active model doesn't support, and         refres, Drop the loaded ``.ele`` bunch and restore the Electrons panel         to its no, Restore the Electron-panel entries to editable input mode and         re-attach

### Community 53 - "ComptonSuite model-agnostic-core refactor: status"
Cohesion: 0.29
Nodes (6): ComptonSuite model-agnostic-core refactor: status, Current package layout, Explicitly dropped from the original plan, Landed — moves that fulfilled the original goal without a `core/` package, Still open, What changed since this doc was first written

### Community 54 - "models.py"
Cohesion: 0.17
Nodes (12): laser_focal_radii(), main(), Transverse rms bunch size [m] from normalized emittance, beta [m] and gamma., Return radial RMS, FWHM, and exp(-1/2) focal radii [m].      Delegates to :func:, sigma_e(), Compton-GUIde: Tkinter GUI for pluggable Compton-scattering physics models., ComptonSuite: unified package for inverse-Compton scattering simulation.  Subpac, run_gui() (+4 more)

## Knowledge Gaps
- **201 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `compton-suite`, `What this is`, `Layout` (+196 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MacroBunch` connect `particles.py` to `deposition.py`, `CommonResults`, `XigmaDirectAdapter`, `GaussianElectronBeam`, `xigma_i/gui_adapter.py`, `XigmaAdapter`, `cache.py`, `.available`, `._update_derived`, `bunch.py`, `UnavailableAdapter`, `AnalyticalAdapter`, `test_analytical.py`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **Why does `sample_gaussian_bunch()` connect `GaussianElectronBeam` to `ComptonGuideApp`, `particles.py`, `kaskade/kascade.py`, `CommonResults`, `New GUI observables: status per model`, `test_laser.py`, `models.py`, `bunch.py`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Why does `Config` connect `._update_derived` to `ComptonGuideApp`, `adapters/__init__.py`, `particles.py`, `compton_guide/__init__.py`, `KASCADE`, `bunch.py`, `test_analytical.py`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Are the 17 inferred relationships involving `MacroBunch` (e.g. with `ModelAdapter` and `ModelCapabilities`) actually correct?**
  _`MacroBunch` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `GaussianElectronBeam` (e.g. with `CollisionParams` and `InteractionGeometry`) actually correct?**
  _`GaussianElectronBeam` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `ComptonGuideApp` (e.g. with `ModelAdapter` and `OutputSpec`) actually correct?**
  _`ComptonGuideApp` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Config` (e.g. with `MacroBunch` and `InteractionParameters`) actually correct?**
  _`Config` has 2 INFERRED edges - model-reasoned connections that need verification._