# Graph Report - ComptonSuite  (2026-07-28)

## Corpus Check
- 93 files · ~85,248 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1185 nodes · 2420 edges · 69 communities (56 shown, 13 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 228 edges (avg confidence: 0.52)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `93270180`
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
- collision.py
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
- fit_beam_full
- graphify.js
- sample_gaussian_bunch
- ._load_ele
- ComptonSuite model-agnostic-core refactor: status
- models.py
- sample_gaussian_canonical
- evaluate_fit_quality
- drift
- AnalyticalConfig
- delta/gui_adapter.py
- test_bunch_improvements.py
- TestEdgeCases
- yaml_spec.py
- compton_suite/__init__.py
- TestIntegration
- _theta_grid
- recoil_parameter
- .params_to_config
- extra_choices

## God Nodes (most connected - your core abstractions)
1. `MacroBunch` - 66 edges
2. `GaussianElectronBeam` - 55 edges
3. `ComptonGuideApp` - 43 edges
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
- `TestCanonicalSampling` --uses--> `MacroBunch`  [INFERRED]
  tests/test_bunch_improvements.py → src/compton_suite/io/bunch.py
- `TestDrift` --uses--> `MacroBunch`  [INFERRED]
  tests/test_bunch_improvements.py → src/compton_suite/io/bunch.py
- `TestEdgeCases` --uses--> `MacroBunch`  [INFERRED]
  tests/test_bunch_improvements.py → src/compton_suite/io/bunch.py
- `TestFitQuality` --uses--> `MacroBunch`  [INFERRED]
  tests/test_bunch_improvements.py → src/compton_suite/io/bunch.py

## Import Cycles
- None detected.

## Communities (69 total, 13 thin omitted)

### Community 0 - "Scenario"
Cohesion: 0.05
Nodes (39): 10. Future extensions, 11. Key insight, (1) Always sample in canonical variables, 1. Goal, (2) Enforce mass-shell, 2. Representation, (3) Center data before fitting, 3. Sampling (Gaussian at waist) (+31 more)

### Community 1 - "compton_io/__init__.py"
Cohesion: 0.08
Nodes (56): Enum, ModelSpec, Quantity, Thin re-export of ``compton_suite.io``'s parameter-semantics/unit normalisation, adapt_to_model(), params_to_floats(), Canonical params -> a specific model's own convention/units.  This is the last s, ``canonical_params`` may be in any convention/unit as long as the     meaning ma (+48 more)

### Community 2 - "ComptonGuideApp"
Cohesion: 0.07
Nodes (75): Shared comparison metric for the cross-model validation suite.  Promoted (moved, s, spec, spec_ref: 1D arrays of equal shape (same grid for both     spectra -- r, Linear-interpolate spec_src(s_src) onto s_ref, zero outside s_src's range., resample_to(), window_integrated_relative_error(), User-curated bank of additional cross-validation parameter sets.  Deliberately e, Cross-model validation suite -- entry point.  Runs kascade/xigma-i/delta/analyti, Tier 0 (config-only, no run needed) + Tier 1/2 (share one run of     each model) (+67 more)

### Community 3 - "deposition.py"
Cohesion: 0.17
Nodes (7): table.total_weight -- retarget_a0 preserves it exactly., (t_seconds, rate) bin-center arrays -- photon-emission rate vs         time. Non, (x_centers, y_centers, density) -- transverse areal density         [photons/cm^, d2N/(ds dOmega) grid -- spectrum4d.calculate_angular_spectrum_4d         on this, Drives Stage 0/1/2 of the new path for one `params` (CollisionParams)     config, Stage 0 (particles.push_and_sample) + Stage 1         (deposition.build_table, a, TabulatedEngine

### Community 4 - "particles.py"
Cohesion: 0.11
Nodes (10): fit_gaussian(), MacroBunch, Fit a :class:`GaussianElectronBeam` (at its waist) from raw macroparticles., Raw, engine-agnostic macroparticle representation. SI units.      ``x``/``y``/``, load_elegant_ele(), Parse a 6-D electron-bunch ``.ele`` file in SDDS ASCII format.      Required col, AnalyticalAdapter, Fast closed-form model: total yield, angle-integrated spectrum, and     an estim (+2 more)

### Community 5 - "kaskade/kascade.py"
Cohesion: 0.07
Nodes (37): _array_module(), build_table(), build_table_streaming(), _cell_indices(), check_accumulation_precision(), _deposit(), deposit_cic(), deposit_nearest() (+29 more)

### Community 6 - "CommonResults"
Cohesion: 0.06
Nodes (55): gaussian_pulse_envelope(), Shared spatiotemporal Gaussian-pulse envelope -- the piece ``compton_suite.io.la, Photon-density envelope of a Gaussian laser pulse at an arbitrary     point in s, ballistic_position_simultaneous(), ballistic_position_z0_reference(), laser_overlap_time_window(), propagate(), Ballistic (straight-line, no-acceleration) electron-beam propagation, and a time (+47 more)

### Community 7 - "XigmaDirectAdapter"
Cohesion: 0.08
Nodes (4): DirectConfig, ndarray, SI-unit physics config, trimmed to what Stage 0 (particles.py)     needs -- no S, _theta_grid()

### Community 8 - "GaussianElectronBeam"
Cohesion: 0.09
Nodes (7): beta_star_from_sigma_emit(), divergence_from_sigma_emit(), GaussianElectronBeam, The ``gaussian_6d_waist`` v0.1 I/O contract.      A 6D factorized Gaussian defin, Absolute RMS dispersion for longitudinal momentum (normalized to mc)., Beta function at the waist, ``sigma**2 / emit_geom`` -- same units in     as out, RMS divergence at the waist, ``emit_geom / sigma`` -- a dimensionless     angle,

### Community 10 - "MacroBunch"
Cohesion: 0.13
Nodes (18): Physical constants, derived from ``units.py``'s pint registry rather than hand-t, Laser-pulse representation: the ``gaussian_paraxial`` v0.1 I/O contract (``specs, ModelAdapter for the fast, always-on analytical model.  Duck-typed against ``com, angle_integrated_spectrum(), estimate_spectrum_width(), estimate_yield(), ndarray, Fast, closed-form Compton-source physics: total yield, angle-integrated spectrum (+10 more)

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
Cohesion: 0.08
Nodes (17): load_laser(), save_laser(), GaussianParaxialLaser, ``a0_at(z_m)**2`` -- the mean-square amplitude.          Consumers that only eve, Validate a :class:`GaussianParaxialLaser` per spec Sec. 12.      Raises ``ValueE, A paraxial Gaussian laser pulse. SI units.      ``waist_rms_x_m``/``waist_rms_y_, Photon count in the pulse: N_L = pulse_energy_J / (hbar*omega0)., Normalized vector potential at ``z_m``, from the exact SI         plane-wave rel (+9 more)

### Community 20 - "laser.py"
Cohesion: 0.33
Nodes (3): Model registry -- direct imports since all packages ship together., delta: brute-force per-macroparticle resonance-binning model.  Extracted from ``, Physics engine models for ComptonSuite.  Each model implements the ModelAdapter

### Community 21 - "test_laser.py"
Cohesion: 0.20
Nodes (13): check(), FakeVar, main(), make_fields(), Same mechanism app.py's on_start() uses: whichever model is     'selected', the, Stand-in for tk.StringVar -- params_to_config only ever calls .get()., sample_electrons_for(), test_model() (+5 more)

### Community 22 - "bunch.py"
Cohesion: 0.17
Nodes (13): Elegant / SDDS ``.ele`` file I/O for :class:`compton_suite.io.bunch.MacroBunch`., Write a :class:`MacroBunch` in SDDS ASCII ``.ele`` format.      Mirrors :func:`l, save_elegant_ele(), KASCADE: Sequential multi-photon inverse-Compton Monte Carlo engine., kn_sigma_ratio(), L_theta(), Total Klein--Nishina cross section relative to Thomson, sigma_KN(k)/sigma_T., Normalised (linearly-polarised) Thomson angular distribution. (+5 more)

### Community 23 - "UnavailableAdapter"
Cohesion: 0.23
Nodes (23): ModelCapabilities, OutputSpec, Model-agnostic contract between app.py and physics-engine adapters.  This module, Model-agnostic output resolution specification.      Models receive this in thei, AngularRangeSpectrumResult, BinnedAngularSpectrum, BinnedSpatialDistribution, BinnedTemporalEnvelope (+15 more)

### Community 24 - "run_multiphoton_chain"
Cohesion: 0.10
Nodes (14): Protocol, ModelAdapter, Any, Return (True, "") if the model can actually be run right now, else         (Fals, Model-specific fields with no shared-panel analogue         (kascade's Electrons, Optional: return a dict mapping parameter keys to allowed string values, ``electrons`` is required: electron sampling is the IO layer's         (caller's, Raise NotImplementedError if capabilities().supports_ele_file_io is False. (+6 more)

### Community 25 - "analytical.py"
Cohesion: 0.15
Nodes (12): 10. Проверки валидности, 11. Не входит в v0.1, 1. Scope, 2. Геометрическая конвенция, 3. Input-параметры, 4. Математическое определение, 5. Эмиттанс и расходимость, 6. Энергия и заряд (+4 more)

### Community 26 - "compton_suite/__init__.py"
Cohesion: 0.18
Nodes (3): ComptonGuideApp, Return (total_flux, collimated_flux, cmask_or_None) [ph/s].          cmask is on, Update the preview panel from self.preview_res (set by         _poll_queue). Nev

### Community 27 - "detect_device"
Cohesion: 0.15
Nodes (12): 10. Проверки валидности input, 11. Не входит в v0.1, 1. Scope, 2. Геометрическая конвенция, 3. Input-параметры, 4. Математическое определение профиля, 5. Конверсии rms / FWHM / $1/e^2$, 6. Длина Рэлея и размер в точке взаимодействия (+4 more)

### Community 28 - "collision.py"
Cohesion: 0.14
Nodes (13): build_params(), CollisionParams, detect_device(), The CGS "collision parameters" bundle for tabulated-overlap-style GPU/CPU pipeli, Derive this convention's CGS :class:`CollisionParams` from     ``compton_suite.i, Auto-detect which backend to use: a real CUDA GPU via cupy if     available, els, CGS scalars for one laser-electron collision -- immutable, built     once by :fu, available() (+5 more)

### Community 29 - "AnalyticalAdapter"
Cohesion: 0.11
Nodes (13): BinnedSpectrum, Binned dN/dE spectrum (semi-analytic/tabulated models)., _attach_private_cache(), DeltaAdapter, Stash this adapter's own private recompute cache on a     ``compton_suite.io.res, ``electrons`` is required: electron sampling is the caller's job,     not this a, Evaluate direct_binning_spectrum at a grid of OBSERVATION angles     spanning th, run_simulation() (+5 more)

### Community 30 - "XigmaAdapter"
Cohesion: 0.08
Nodes (24): 1. Dead-code / unused-config sweep ✅, 2. Move `CollisionParams`/`build_params` into `io/` ✅, 3. GUI: trust levels/warnings ✅, 4. GUI: per-model sample count ✅, 5. Unify the `ModelAdapter` interface ✅, 6. Manual CPU/GPU selection ✅, 7. How to add a new model ✅, 8. GUI-as-thin-consumer ✅ (+16 more)

### Community 31 - "test_analytical.py"
Cohesion: 0.13
Nodes (22): beam_from_shared_fields(), Electron-bunch representations.  Two distinct types, matching the distinction th, Build a :class:`GaussianElectronBeam` from the flat SI field set every     model, InteractionGeometry, InteractionParameters, The shared "interaction parameters" bundle: an electron beam, a laser pulse, and, Collision geometry: foci displacement and crossing angle. SI units., One canonical (beam, pulse, geometry) triple -- the physics-parameter     subset (+14 more)

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

### Community 43 - "KASCADE"
Cohesion: 0.33
Nodes (6): laser_a0sq(), laser_axis(), laser_density(), Laser propagation unit vector n_L = (sin phi, 0, -cos phi).      phi = crossing_, Gaussian laser photon density propagating along n_L, focus at delta.      Reduce, Normalised laser vector-potential squared a_0^2(r,t) (as in dfe4).

### Community 44 - ".run"
Cohesion: 0.23
Nodes (5): add_field_grid(), Place (label, default, key) triples in an n_cols-wide grid inside a     coloured, Panel for configuring output resolution (spectrum, temporal,         spatial, an, Small always-visible panel showing the fast analytical model's         estimate,, StringVar

### Community 45 - ".available"
Cohesion: 0.17
Nodes (5): _attach_private_cache(), Stash this adapter's own private recompute cache on a     ``compton_suite.io.res, ``n_mc`` is accepted for ``ModelAdapter.run`` signature compatibility     but un, run_simulation(), XigmaAdapter

### Community 46 - "opencode.json"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

### Community 47 - "._apply_model_capabilities"
Cohesion: 0.20
Nodes (4): Grey out / restore controls the active model doesn't support, and         refres, Drop the loaded ``.ele`` bunch and restore the Electrons panel         to its no, Restore the Electron-panel entries to editable input mode and         re-attach, Repopulate the Model Parameters panel from         ``self.active_adapter.extra_p

### Community 49 - "fit_beam_full"
Cohesion: 0.19
Nodes (10): BeamFittedParams, fit_beam_full(), Physically meaningful beam parameters from fitting.      Contains all parameters, Fit structured Gaussian model with physical correlations.      Extracts physical, Test structured Gaussian fitting., Verify fitted parameters match input beam., Verify fitting works correctly with drifted beam., Verify sigma_gamma is correctly calculated from fit. (+2 more)

### Community 51 - "sample_gaussian_bunch"
Cohesion: 0.19
Nodes (11): Validate a :class:`GaussianElectronBeam` per spec Sec. 16.      Raises ``ValueEr, Draw macroparticles from a :class:`GaussianElectronBeam`.      Uses canonical sa, sample_gaussian_bunch(), validate(), Cross-checks for bunch.py's MacroBunch / GaussianElectronBeam / sample_gaussian_, test_derived_quantities_are_sane(), test_fit_gaussian_recovers_waist_after_drift(), test_fit_gaussian_round_trips_at_the_waist() (+3 more)

### Community 53 - "ComptonSuite model-agnostic-core refactor: status"
Cohesion: 0.29
Nodes (6): ComptonSuite model-agnostic-core refactor: status, Current package layout, Explicitly dropped from the original plan, Landed — moves that fulfilled the original goal without a `core/` package, Still open, What changed since this doc was first written

### Community 54 - "models.py"
Cohesion: 0.13
Nodes (19): _float_or_none(), laser_focal_radii(), peak_a0(), Transverse rms bunch size [m] from normalized emittance, beta [m] and gamma., Return radial RMS, FWHM, and exp(-1/2) focal radii [m].      Delegates to :func:, Peak normalised vector potential a_0 from the laser fields (or None).      Deleg, Refresh electron and laser values derived from the current fields., sigma_e() (+11 more)

### Community 55 - "sample_gaussian_canonical"
Cohesion: 0.21
Nodes (8): Draw macroparticles from a :class:`GaussianElectronBeam` using canonical     var, sample_gaussian_canonical(), Verify sampled statistics match input parameters., Verify gamma is correctly calculated from momenta., Test canonical sampling with mass-shell enforcement., Verify mass-shell constraint: gamma^2 = 1 + px^2 + py^2 + pz^2., Verify pz > 1 for all particles (physical constraint)., TestCanonicalSampling

### Community 56 - "evaluate_fit_quality"
Cohesion: 0.22
Nodes (8): evaluate_fit_quality(), ndarray, Evaluate Gaussian fit quality with sampling-noise baseline.      Compares the re, Test fit quality evaluation metrics., Verify Gaussian data produces low KS excess., Verify non-Gaussian data produces high KS excess., Verify log-likelihood comparison between real and synthetic., TestFitQuality

### Community 57 - "drift"
Cohesion: 0.24
Nodes (7): drift(), Propagate beam in vacuum over distance L.      Ballistic propagation:     - x ->, Test vacuum propagation., Verify x -> x + x' * L., Verify z, thx, thy, gamma unchanged by drift., Verify Twiss alpha emerges from waist + drift., TestDrift

### Community 58 - "AnalyticalConfig"
Cohesion: 0.20
Nodes (5): AnalyticalConfig, _float(), ParamError, Exception, Wraps beam/pulse, but ALSO exposes a handful of flat SI fields every     other m

### Community 59 - "delta/gui_adapter.py"
Cohesion: 0.22
Nodes (7): extra_choices(), _float(), params_to_config(), ModelAdapter for the brute-force particle-binning model.  Reuses ``xigma_i.parti, Allowed values for choice/enum fields in extra_params()., Same shared GUI fields as xigma-i's params_to_config -- identical     parsing, t, # TODO: pass output spec to run_simulation when it's refactored

### Community 60 - "test_bunch_improvements.py"
Cohesion: 0.25
Nodes (7): mildly_relativistic_beam(), Tests for the improved electron bunch sampling, fitting, and evaluation.  This m, Random number generator for reproducible tests., Ultra-relativistic beam (gamma >> 1) for testing.      Physical parameters: 500, Mildly relativistic beam (gamma ~ 19) for testing.      Physical parameters: 9 M, rng(), ultra_relativistic_beam()

### Community 61 - "TestEdgeCases"
Cohesion: 0.25
Nodes (5): Test edge cases and error handling., Test with zero momentum spread (delta function in pz)., Test with large momentum spread., Test with very small number of particles., TestEdgeCases

### Community 62 - "yaml_spec.py"
Cohesion: 0.33
Nodes (5): External-format I/O for compton_suite.io's bunch/laser representations., load_electron_beam(), YAML I/O for the ``gaussian_6d_waist``/``gaussian_paraxial`` v0.1 formats define, save_electron_beam(), test_spec_example_electron_beam_yaml_round_trips()

### Community 63 - "compton_suite/__init__.py"
Cohesion: 0.47
Nodes (4): main(), Compton-GUIde: Tkinter GUI for pluggable Compton-scattering physics models., ComptonSuite: unified package for inverse-Compton scattering simulation.  Subpac, run_gui()

### Community 64 - "TestIntegration"
Cohesion: 0.33
Nodes (4): End-to-end integration tests., Test complete pipeline: sample -> drift -> fit -> evaluate., Verify sample_gaussian_bunch delegates to canonical sampling., TestIntegration

### Community 65 - "_theta_grid"
Cohesion: 0.67
Nodes (3): ndarray, A generous fixed window around the current collimation angle, wide     enough th, _theta_grid()

## Knowledge Gaps
- **201 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `compton-suite`, `What this is`, `Layout` (+196 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MacroBunch` connect `particles.py` to `CommonResults`, `XigmaDirectAdapter`, `xigma_i/gui_adapter.py`, `MacroBunch`, `cache.py`, `bunch.py`, `UnavailableAdapter`, `run_multiphoton_chain`, `AnalyticalAdapter`, `test_analytical.py`, `XigmaAdapter`, `.available`, `._update_derived`, `fit_beam_full`, `sample_gaussian_bunch`, `sample_gaussian_canonical`, `evaluate_fit_quality`, `drift`, `AnalyticalConfig`, `delta/gui_adapter.py`, `TestEdgeCases`, `TestIntegration`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Why does `sample_gaussian_bunch()` connect `sample_gaussian_bunch` to `TestIntegration`, `ComptonGuideApp`, `particles.py`, `kaskade/kascade.py`, `CommonResults`, `GaussianElectronBeam`, `New GUI observables: status per model`, `test_laser.py`, `models.py`, `sample_gaussian_canonical`, `bunch.py`, `test_analytical.py`?**
  _High betweenness centrality (0.102) - this node is a cross-community bridge._
- **Why does `ComptonGuideApp` connect `compton_suite/__init__.py` to `compton_io/__init__.py`, `New GUI observables: status per model`, `.run`, `._apply_model_capabilities`, `._load_ele`, `models.py`, `UnavailableAdapter`, `run_multiphoton_chain`, `compton_suite/__init__.py`?**
  _High betweenness centrality (0.063) - this node is a cross-community bridge._
- **Are the 23 inferred relationships involving `MacroBunch` (e.g. with `ModelAdapter` and `ModelCapabilities`) actually correct?**
  _`MacroBunch` has 23 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `GaussianElectronBeam` (e.g. with `CollisionParams` and `InteractionGeometry`) actually correct?**
  _`GaussianElectronBeam` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `ComptonGuideApp` (e.g. with `ModelAdapter` and `OutputSpec`) actually correct?**
  _`ComptonGuideApp` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Config` (e.g. with `MacroBunch` and `InteractionParameters`) actually correct?**
  _`Config` has 2 INFERRED edges - model-reasoned connections that need verification._