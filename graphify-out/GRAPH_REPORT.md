# Graph Report - ComptonSuite  (2026-07-31)

## Corpus Check
- 76 files · ~81,532 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1117 nodes · 2163 edges · 53 communities (47 shown, 6 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 252 edges (avg confidence: 0.54)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `fbc96cc2`
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
- opencode.json
- fit_beam_full
- graphify.js
- sample_gaussian_bunch
- ComptonSuite model-agnostic-core refactor: status
- test_bunch_improvements.py
- yaml_spec.py
- compton_suite/__init__.py

## God Nodes (most connected - your core abstractions)
1. `GaussianElectronBeam` - 77 edges
2. `GaussianParaxialLaser` - 50 edges
3. `Bunch` - 46 edges
4. `ComptonGUIApp` - 44 edges
5. `Job` - 43 edges
6. `Scenario` - 43 edges
7. `PhysicalQuantity` - 37 edges
8. `sample_gaussian_canonical()` - 34 edges
9. `Photons` - 33 edges
10. `ModelCapabilities` - 31 edges

## Surprising Connections (you probably didn't know these)
- `run()` --calls--> `check()`  [INFERRED]
  src/compton_suite/validation/tier0_wiring.py → scripts/headless_test.py
- `TestCanonicalSampling` --uses--> `Bunch`  [INFERRED]
  tests/test_bunch_improvements.py → src/compton_suite/io/bunch.py
- `TestChirpAndDispersion` --uses--> `Bunch`  [INFERRED]
  tests/test_bunch_improvements.py → src/compton_suite/io/bunch.py
- `TestDrift` --uses--> `Bunch`  [INFERRED]
  tests/test_bunch_improvements.py → src/compton_suite/io/bunch.py
- `TestEdgeCases` --uses--> `Bunch`  [INFERRED]
  tests/test_bunch_improvements.py → src/compton_suite/io/bunch.py

## Import Cycles
- None detected.

## Communities (53 total, 6 thin omitted)

### Community 0 - "Scenario"
Cohesion: 0.05
Nodes (39): 10. Future extensions, 11. Key insight, (1) Always sample in canonical variables, 1. Goal, (2) Enforce mass-shell, 2. Representation, (3) Center data before fitting, 3. Sampling (Gaussian at waist) (+31 more)

### Community 1 - "compton_io/__init__.py"
Cohesion: 0.11
Nodes (23): Enum, Shared physical constants, pint unit registry, parameter-semantics vocabulary, a, AmplitudeConvention, _convert(), convert_amplitude(), convert_time(), convert_width(), NoConvention (+15 more)

### Community 2 - "ComptonGuideApp"
Cohesion: 0.07
Nodes (71): Shared comparison metric for the cross-model validation suite.  Promoted (moved, s, spec, spec_ref: 1D arrays of equal shape (same grid for both     spectra -- r, Linear-interpolate spec_src(s_src) onto s_ref, zero outside s_src's range., resample_to(), window_integrated_relative_error(), User-curated bank of additional cross-validation parameter sets.  Deliberately e, Cross-model validation suite -- entry point.  Runs kascade/xigma-i/delta/analyti, Tier 0 (config-only, no run needed) + Tier 1/2 (share one run of     each model) (+63 more)

### Community 3 - "deposition.py"
Cohesion: 0.12
Nodes (9): Engine on the tabulated-energy pipeline (particles.py/deposition.py/ spectrum4d., table.total_weight -- retarget_a0 preserves it exactly., (t_seconds, rate) bin-center arrays -- photon-emission rate vs         time. Non, (x_centers, y_centers, density) -- transverse areal density         [photons/cm^, dN/ds, angle-integrated over all emission solid angle --         spectrum_from_p, d2N/(ds dOmega) grid -- spectrum4d.calculate_angular_spectrum_4d         on this, Drives Stage 0/1/2 of the new path for one `params` (CollisionParams)     config, Stage 0 (particles.push_and_sample) + Stage 1         (deposition.build_table, a (+1 more)

### Community 4 - "particles.py"
Cohesion: 0.13
Nodes (18): Bunch, fit_gaussian(), Macroparticle electron bunch. SI units, flat arrays.      ``x``/``y``/``z`` are, Total number of physical electrons., Fit a :class:`GaussianElectronBeam` (at its waist) from raw macroparticles., Draw macroparticles from ``beam``. Delegates to     :func:`sample_gaussian_canon, sample_gaussian_bunch(), load_elegant_ele() (+10 more)

### Community 5 - "kaskade/kascade.py"
Cohesion: 0.06
Nodes (45): _array_module(), build_table(), build_table_streaming(), _cell_indices(), check_accumulation_precision(), _deposit(), deposit_cic(), deposit_nearest() (+37 more)

### Community 6 - "CommonResults"
Cohesion: 0.08
Nodes (42): ballistic_position_z0_reference(), laser_overlap_time_window(), Straight-line position at time offset ``t``, given a per-particle     reference, Per-particle time window ``[t0, t1]`` (normalised length units,     e.g. ``k0_la, gaussian_pulse_envelope(), Photon-density envelope of a Gaussian laser pulse at an arbitrary     point in s, _bin_spatial(), _bin_temporal() (+34 more)

### Community 7 - "XigmaDirectAdapter"
Cohesion: 0.06
Nodes (35): Acceptance Criteria, Acceptance Criteria, Acceptance Criteria, Acceptance Criteria, Acceptance Criteria, Acceptance Criteria, Description, Description (+27 more)

### Community 8 - "GaussianElectronBeam"
Cohesion: 0.05
Nodes (14): beta_star_from_sigma_emit(), divergence_from_sigma_emit(), GaussianElectronBeam, The ``gaussian_6d_waist`` v0.1 I/O contract.      A 6D factorized Gaussian defin, Absolute RMS dispersion for longitudinal momentum (normalized to mc)., Correlation coefficient ρ_zγ = chirp_h · σ_z / σ_γ., Correlation coefficient ρ_xγ = D_x · σ_γ / σ_x., Correlation coefficient ρ_yγ = D_y · σ_γ / σ_y. (+6 more)

### Community 9 - "xigma_i/gui_adapter.py"
Cohesion: 0.12
Nodes (11): ModelSelectionPanel, StringVar, Show/hide parameter frame when checkbox toggles., Called by the Calculate button -- override in the main app or         connect to, Return names of models whose checkbox is checked., Overlay spectra from all models on *ax* with colour-coded legend., Render 2D angular distribution for a specific model., Overlay temporal envelopes from all models. (+3 more)

### Community 11 - "cache.py"
Cohesion: 0.15
Nodes (12): InteractionParameters, One canonical (beam, pulse, geometry) triple -- the physics-parameter     subset, Photons, What every model's ``run()`` must return (shape-compatibly).      Only ``model_n, Adapter, AnalyticalConfig, angle_integrated_spectrum(), ndarray (+4 more)

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
Nodes (14): GaussianParaxialLaser, Rayleigh range for the x waist: ``pi * w0^2 / lambda`` with         ``w0 = 2 * w, RMS intensity-profile width in x at absolute position ``z_m``,         from stan, Angular frequency, ``2*pi*c/wavelength`` [rad/s]., Photon count in the pulse: N_L = pulse_energy / (hbar*omega0)., Peak power at the pulse's temporal center, at focus:         ``E / (sqrt(2*pi) *, On-axis, peak-in-time intensity [W/m^2] at absolute position         ``z_m``, fr, Normalized vector potential at ``z_m``, from the exact SI         plane-wave rel (+6 more)

### Community 20 - "laser.py"
Cohesion: 0.23
Nodes (11): ballistic_position_simultaneous(), propagate(), Straight-line position at time offset ``dt`` later, given a     per-particle ref, Ballistically drift every macroparticle in ``bunch`` by a time     offset ``dt``, Yield a propagated :class:`Bunch` snapshot at each time in     ``t_grid`` (SI se, stream(), Cross-checks for propagation.py's ballistic drift primitives.  No cupy/GPU/tkint, test_ballistic_position_simultaneous_straight_line() (+3 more)

### Community 21 - "test_laser.py"
Cohesion: 0.29
Nodes (10): check(), main(), Mirrors app.py's on_start(): the always-on analytical preview     (ModelCapabili, test_model(), test_preview_alongside(), Any, Defensive duck-type check, run right after a model's ``run()``     returns., validate_results() (+2 more)

### Community 22 - "bunch.py"
Cohesion: 0.16
Nodes (17): KASCADE: Sequential multi-photon inverse-Compton Monte Carlo engine., ModelAdapter wrapping the existing kascade engine.  Moves today's ``Config``/``r, beta_of(), doppler_D(), _eps_L(), kn_sigma_ratio(), L_theta(), Laser photon energy, in units of m_e c^2 -- a kascade-specific     relativistic- (+9 more)

### Community 23 - "UnavailableAdapter"
Cohesion: 0.19
Nodes (11): PhotonMultiplicity, Output-side observable representations: the spectrum/angular-spectrum/ temporal-, SampledSpatialDistribution, SampledSpectrum, SampledTemporalEnvelope, Job, The single compiled-from-UI config object the GUI passes to     ``ModelAdapter.r, Stand-in ``ModelAdapter`` for a model whose runtime dependencies     (e.g. cupy/ (+3 more)

### Community 24 - "run_multiphoton_chain"
Cohesion: 0.20
Nodes (10): Protocol, calculations.py ===============  Multi-model calculation support for the Compton, discover_models(), ModelAdapter, Model-agnostic contract between app.py and physics-engine adapters.  This module, Model-specific parameters as (label, default, key) triples.          Default can, Optional: return a dict mapping parameter keys to allowed string         values, Populate the model registry with direct imports. kascade and     analytical have (+2 more)

### Community 25 - "analytical.py"
Cohesion: 0.15
Nodes (12): 10. Проверки валидности, 11. Не входит в v0.1, 1. Scope, 2. Геометрическая конвенция, 3. Input-параметры, 4. Математическое определение, 5. Эмиттанс и расходимость, 6. Энергия и заряд (+4 more)

### Community 26 - "compton_suite/__init__.py"
Cohesion: 0.17
Nodes (11): CommonResults (from io/results.py), Current GUI Architecture, Dependencies, Description, Feature: GUI Calculations Section, Model Adapters, Out of Scope, Requirements (+3 more)

### Community 27 - "detect_device"
Cohesion: 0.15
Nodes (12): 10. Проверки валидности input, 11. Не входит в v0.1, 1. Scope, 2. Геометрическая конвенция, 3. Input-параметры, 4. Математическое определение профиля, 5. Конверсии rms / FWHM / $1/e^2$, 6. Длина Рэлея и размер в точке взаимодействия (+4 more)

### Community 28 - "collision.py"
Cohesion: 0.24
Nodes (4): AngularRangeSpectrumResult, Evaluate direct_binning_spectrum at a grid of OBSERVATION angles     spanning th, spectrum_in_angular_range_direct(), XigmaAdapter

### Community 29 - "AnalyticalAdapter"
Cohesion: 0.17
Nodes (24): BinnedAngularSpectrum, BinnedSpatialDistribution, BinnedSpectrum, BinnedTemporalEnvelope, _attach_delta_cache(), _attach_xigma_cache(), Config, DirectConfig (+16 more)

### Community 30 - "XigmaAdapter"
Cohesion: 0.08
Nodes (24): AGENTS.md, All models, Analytical model, Architecture, Commands, Cross-repo gotchas (still apply post-merge), Dependency flow, Dev install (+16 more)

### Community 31 - "test_analytical.py"
Cohesion: 0.09
Nodes (24): The shared "interaction parameters" bundle: an electron beam, a laser pulse, and, Quantum recoil parameter ``q = 4 * gamma * hbar * omega / (m_e * c^2)``.      Di, recoil_parameter(), a0_from_fields(), laser_from_shared_fields(), _pq(), Laser-pulse representation: the ``gaussian_paraxial`` v0.1 I/O contract.  Every, Validate a :class:`GaussianParaxialLaser`.      Raises ``ValueError`` on hard re (+16 more)

### Community 32 - "InteractionParameters"
Cohesion: 0.21
Nodes (8): Physics constants and GPU kernel sizing constants for this pipeline (particles.p, calculate_angular_spectrum_4d(), _get_numba(), get_spectrum_kernel_4d_cpu(), CPU/numba equivalent of spectrum4d.py's GPU rawkernel (spectrum_kernel_4d), used, Lazily compiles and caches spectrum_kernel_4d_cpu (see module docstring)., Stage 2: `spectrum_kernel_4d`, the GPU kernel that turns a Stage 1 4D overlap ta, Host-side driver for spectrum_kernel_4d, the Stage-2 analogue of     Compton.cal

### Community 35 - "adapters/__init__.py"
Cohesion: 0.25
Nodes (5): detect_device(), Auto-detect which backend to use: a real CUDA GPU via cupy if     available, els, CollisionParams, The CGS "collision parameters" bundle for this package's tabulated- overlap-styl, CGS scalars for one laser-electron collision -- immutable, built     once by :fu

### Community 36 - "compton_guide/__init__.py"
Cohesion: 0.25
Nodes (9): invert_lambda(), kn_over_thomson(), ndarray, Klein--Nishina / Thomson differential-cross-section ratio R_KN in [0,1].      ``, Sample (thx, thy) from L_theta (classical) or L_theta * R_KN (quantum).      Pro, Solve Lambda(tau_emit) = threshold row-by-row (Lambda monotonic in tau)., Simulate the emission chain for a chunk of electrons.      Records, per photon,, run_multiphoton_chain() (+1 more)

### Community 37 - "kascade"
Cohesion: 0.25
Nodes (7): Files, KASCADE, No GPU dependency, Physical constants (`compton_suite.io`), Relationship to other components, Testing, Units and conventions

### Community 39 - "xigma-i"
Cohesion: 0.33
Nodes (5): compton_suite.io, Layout, Naming, Testing, Why this exists

### Community 40 - "XigmaAdapter"
Cohesion: 0.15
Nodes (5): ModelCapabilities, What the GUI needs to know about a model to render it, independent     of runnin, Static, cheap-to-call metadata for the GUI's model-selection         panel -- no, KascadeAdapter, DirectAdapter

### Community 41 - "Compton-GUIde"
Cohesion: 0.40
Nodes (4): All models, Analytical model, Model tasks, Xigma-i

### Community 42 - "New GUI observables: status per model"
Cohesion: 0.50
Nodes (3): Quantity, _const(), A CODATA constant from pint's own table, as a plain float in SI base     units -

### Community 43 - "KASCADE"
Cohesion: 0.17
Nodes (16): build_lambda_grid(), Config, cos_collision(), laser_a0sq(), laser_axis(), laser_density(), _R_sf(), Laser transverse RMS (photon-density) width -- round-beam collapse     of the la (+8 more)

### Community 46 - "opencode.json"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

### Community 49 - "fit_beam_full"
Cohesion: 0.05
Nodes (31): drift(), evaluate_fit_quality(), fit_beam_full(), ndarray, Evaluate Gaussian fit quality with sampling-noise baseline.      Compares the re, Fit a structured Gaussian model with physical correlations.      Extracts physic, Draw macroparticles from a :class:`GaussianElectronBeam` using     canonical var, Ballistically propagate a bunch by a longitudinal distance ``L`` (SI     metres) (+23 more)

### Community 51 - "sample_gaussian_bunch"
Cohesion: 0.24
Nodes (7): Validate a :class:`GaussianElectronBeam`.      Raises ``ValueError`` on hard req, validate(), Cross-checks for bunch.py's Bunch / GaussianElectronBeam / sample_gaussian_bunch, test_derived_quantities_are_sane(), test_sample_gaussian_bunch_matches_beam_moments(), test_validate_rejects_nonpositive_fields(), test_validate_warns_on_suspicious_emittance()

### Community 53 - "ComptonSuite model-agnostic-core refactor: status"
Cohesion: 0.29
Nodes (6): ComptonSuite model-agnostic-core refactor: status, Current package layout, Explicitly dropped from the original plan, Landed — moves that fulfilled the original goal without a `core/` package, Still open, What changed since this doc was first written

### Community 60 - "test_bunch_improvements.py"
Cohesion: 0.09
Nodes (33): beam_from_shared_fields(), BeamFittedParams, _pq(), Electron-bunch representations.  Two distinct types, matching the distinction be, The output of :func:`fit_beam_full`: a structured Gaussian fit to a     macropar, Build a :class:`GaussianElectronBeam` from flat SI fields a model's     own ``Co, Shortcut to build a PhysicalQuantity., PhysicalMeaning (+25 more)

### Community 62 - "yaml_spec.py"
Cohesion: 0.23
Nodes (9): External-format I/O for compton_suite.io's bunch/laser representations., load_electron_beam(), load_laser(), YAML I/O for the ``gaussian_6d_waist``/``gaussian_paraxial`` v0.1 formats define, save_electron_beam(), save_laser(), Cross-checks for io_formats/sdds.py and io_formats/yaml_spec.py.  No cupy/GPU/tk, test_spec_example_electron_beam_yaml_round_trips() (+1 more)

### Community 63 - "compton_suite/__init__.py"
Cohesion: 0.05
Nodes (31): add_field_grid(), ComptonGUIApp, _float_or_none(), main(), StringVar, Return (total_flux, collimated_flux, cmask_or_None) [ph/s].          cmask is on, Load a 6-D electron bunch from an SDDS ``.ele`` file and turn the         Electr, Drop the loaded ``.ele`` bunch and restore the Electrons panel         to its no (+23 more)

## Knowledge Gaps
- **240 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `compton-suite`, `Description`, `UI Structure` (+235 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GaussianElectronBeam` connect `GaussianElectronBeam` to `compton_io/__init__.py`, `ComptonGuideApp`, `adapters/__init__.py`, `particles.py`, `cache.py`, `fit_beam_full`, `sample_gaussian_bunch`, `sample_gaussian_bunch`, `test_bunch_improvements.py`, `AnalyticalAdapter`, `yaml_spec.py`, `test_analytical.py`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Why does `Bunch` connect `particles.py` to `compton_io/__init__.py`, `XigmaAdapter`, `cache.py`, `KASCADE`, `fit_beam_full`, `laser.py`, `test_laser.py`, `bunch.py`, `UnavailableAdapter`, `run_multiphoton_chain`, `test_bunch_improvements.py`, `test_analytical.py`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Why does `sample_gaussian_bunch()` connect `particles.py` to `ComptonGuideApp`, `kaskade/kascade.py`, `GaussianElectronBeam`, `fit_beam_full`, `sample_gaussian_bunch`, `laser.py`, `test_laser.py`, `test_bunch_improvements.py`, `compton_suite/__init__.py`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Are the 17 inferred relationships involving `GaussianElectronBeam` (e.g. with `NoConvention` and `PhysicalMeaning`) actually correct?**
  _`GaussianElectronBeam` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `GaussianParaxialLaser` (e.g. with `InteractionParameters` and `NoConvention`) actually correct?**
  _`GaussianParaxialLaser` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 23 inferred relationships involving `Bunch` (e.g. with `NoConvention` and `PhysicalMeaning`) actually correct?**
  _`Bunch` has 23 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `ComptonGUIApp` (e.g. with `InteractionParameters` and `Job`) actually correct?**
  _`ComptonGUIApp` has 4 INFERRED edges - model-reasoned connections that need verification._