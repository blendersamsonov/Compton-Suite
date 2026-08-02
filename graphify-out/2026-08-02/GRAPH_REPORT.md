# Graph Report - doc-cleanup  (2026-08-02)

## Corpus Check
- 73 files · ~77,441 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1076 nodes · 2205 edges · 73 communities (49 shown, 24 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 216 edges (avg confidence: 0.58)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `160c1995`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- IO Core + Validation
- Bunch Sampling
- GUI App Layout
- Laser Overlap
- XIGMA Deposition
- Electron Beam Model
- Physics Concepts
- Laser Specifications
- Photon Results
- Project Documentation
- Units & Conventions
- Interaction Parameters
- Canonical Sampling Tests
- GUI Conventions
- GUI Model Selection
- propagate
- cache.py
- fit_gaussian
- sample_gaussian_canonical
- test_bunch_improvements.py
- docs/gui/tasks.md — GUI task backlog
- XigmaAdapter
- core-simulation-api refactor (completed)
- Gaussian Paraxial Laser I/O Spec v0.1 (short)
- app.py
- TabulatedEngine
- GUI Calculations Section - Task Breakdown
- deposition.py (Stage 1 binning; occupancy_diagnostics, check_accumulation_precision)
- PhysicalQuantity
- physics_params module (proposed intermediate canonical layer)
- units.py
- api.py
- Electron Beam I/O Spec v0.1 (short)
- deposition (Stage 1)
- Bunch
- drift
- ValueError
- test_laser.py
- DirectAdapter
- spectrum4d.py
- xigma_i/params/spec.py — XIGMA_SPEC / XIGMA_DIAGNOSTIC_SPEC (model's own parameter contract)
- XIGMA-I code/method
- gui_adapter.py (bridge into compton_gui as pluggable ModelAdapter)
- .model_choices
- TestFitQuality
- .model_params
- OutputSpec
- build_params (CGS CollisionParams)
- Model Tasks (Xigma-i, Analytical model, All models)
- test_analytical.py
- Job dataclass
- test_constants.py
- opencode.json
- _const
- kascade run_simulation
- graphify.js
- isinstance trap (duck-typing boundary)
- GammaForge
- GammaForge planned rename
- pint unit registry
- gammaforge.io AGENTS.md
- KASCADE AGENTS.md
- Photons dataclass
- ModelCapabilities dataclass
- gammaforge
- a0 factorises out of the table (a0_shape)
- a0 is a trajectory average, not instantaneous sample
- Canonical variables with mass-shell enforcement
- Each model converts at its own boundary
- No derived-value properties on model Config
- No model-local particle sampling
- Single result contract (Photons)
- Twiss tilt emerges from waist sampling + drift

## God Nodes (most connected - your core abstractions)
1. `GaussianElectronBeam` - 69 edges
2. `ComptonGUIApp` - 51 edges
3. `GaussianParaxialLaser` - 50 edges
4. `Bunch` - 45 edges
5. `PhysicalQuantity` - 39 edges
6. `Job` - 39 edges
7. `Scenario` - 38 edges
8. `sample_gaussian_canonical()` - 35 edges
9. `OutputSpec` - 27 edges
10. `InteractionParameters` - 25 edges

## Surprising Connections (you probably didn't know these)
- `run()` --calls--> `check()`  [INFERRED]
  src/gammaforge/validation/tier0_wiring.py → scripts/headless_test.py
- `TestCanonicalSampling` --uses--> `Bunch`  [INFERRED]
  tests/test_bunch_improvements.py → src/gammaforge/io/bunch.py
- `TestChirpAndDispersion` --uses--> `Bunch`  [INFERRED]
  tests/test_bunch_improvements.py → src/gammaforge/io/bunch.py
- `TestDrift` --uses--> `Bunch`  [INFERRED]
  tests/test_bunch_improvements.py → src/gammaforge/io/bunch.py
- `TestEdgeCases` --uses--> `Bunch`  [INFERRED]
  tests/test_bunch_improvements.py → src/gammaforge/io/bunch.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **All ModelAdapter implementations** — models_kascade_kascadeadapter, models_xigma_i_xigmaadapter, models_xigma_i_directadapter, models_analytical_adapter, models_api_unavailableadapter [EXTRACTED 1.00]
- **XIGMA four-stage pipeline** — docs_models_xigma_i_push_and_sample, docs_models_xigma_i_deposition, docs_models_xigma_i_spectrum_kernel_4d, docs_models_xigma_i_spectrum4d_cpu, docs_models_xigma_i_reference [EXTRACTED 1.00]
- **Unified physics-parameter framework shared by the GUI and both models' own specs** — src_gammaforge_io_quantities, src_gammaforge_gui_physics_params, src_gammaforge_models_xigma_i_params_spec, src_gammaforge_models_kascade_params_spec [INFERRED 0.85]
- **gammaforge.io as the shared dependency layer for the GUI and every model** — src_gammaforge_io_init, src_gammaforge_gui_physics_params, src_gammaforge_models_xigma_i, src_gammaforge_models_kascade_kascade [INFERRED 0.85]
- **Duck-typing convention adopted to resolve the ModelAdapter isinstance bug** — src_gammaforge_gui_model_api, src_gammaforge_models_xigma_i_gui_adapter, docs_gui_agents_isinstance_bug [EXTRACTED 1.00]
- **Zero-correlation focus convention linking geometric convention, Twiss parameters, and angular divergence** — docs_io_specs_electron_beam_io_v0_1_full_geometric_convention, docs_io_specs_electron_beam_io_v0_1_full_twiss_parameters, docs_io_specs_electron_beam_io_v0_1_full_angular_profile [INFERRED 0.75]
- **Transverse and longitudinal sizes combine into the peak electron density formula** — docs_io_specs_electron_beam_io_v0_1_full_transverse_profile, docs_io_specs_electron_beam_io_v0_1_full_longitudinal_profile, docs_io_specs_electron_beam_io_v0_1_full_peak_density [EXTRACTED 1.00]
- **Input parameters flow through the 6D Gaussian model to produce derived output parameters** — docs_io_specs_electron_beam_io_v0_1_full_input_parameters, docs_io_specs_electron_beam_io_v0_1_full_6d_gaussian_distribution, docs_io_specs_electron_beam_io_v0_1_full_derived_output_parameters [INFERRED 0.85]
- **From electron-beam input parameters to derived-output quantities (energy/charge, bunch length/density, emittance/beta*)** — docs_io_specs_electron_beam_io_v0_1_short_input_parameters, docs_io_specs_electron_beam_io_v0_1_short_energy_charge_derivation, docs_io_specs_electron_beam_io_v0_1_short_bunch_length_density, docs_io_specs_electron_beam_io_v0_1_short_derived_output [INFERRED 0.75]
- **Xigma-i pending tasks (streaming GPU, crossing angle, rename, gamma-axis rescaling)** — docs_models_tasks_xigma_i_streaming_gpu, docs_models_tasks_xigma_i_crossing_angle, docs_models_tasks_xigma_i_rename, docs_models_tasks_xigma_i_gamma_rescaling [EXTRACTED 1.00]
- **Analytical model pending tasks (foci displacement, non-round beam yield, collimated spectrum)** — docs_models_tasks_analytical_foci_displacement, docs_models_tasks_analytical_nonround_beam, docs_models_tasks_analytical_collimated_spectrum [EXTRACTED 1.00]
- **Pulse energy/peak intensity jointly determined by transverse size, duration and energy input** — docs_io_specs_gaussian_paraxial_laser_io_v0_1_input_parameters, docs_io_specs_gaussian_paraxial_laser_io_v0_1_transverse_size_definition, docs_io_specs_gaussian_paraxial_laser_io_v0_1_duration_definition, docs_io_specs_gaussian_paraxial_laser_io_v0_1_energy_power_intensity [EXTRACTED 1.00]
- **Geometric focus offset -> Rayleigh length -> beam size/intensity at interaction point** — docs_io_specs_gaussian_paraxial_laser_io_v0_1_geometric_convention, docs_io_specs_gaussian_paraxial_laser_io_v0_1_rayleigh_length, docs_io_specs_gaussian_paraxial_laser_io_v0_1_beam_size_at_interaction [EXTRACTED 1.00]
- **Peak intensity (focus and interaction) chained into normalized amplitude a0** — docs_io_specs_gaussian_paraxial_laser_io_v0_1_energy_power_intensity, docs_io_specs_gaussian_paraxial_laser_io_v0_1_beam_size_at_interaction, docs_io_specs_gaussian_paraxial_laser_io_v0_1_a0_amplitude [EXTRACTED 1.00]
- **Input parameters -> intensity profile -> Rayleigh length -> a0 -> derived output pipeline** — gplaser_input_parameters, gplaser_intensity_profile, gplaser_rayleigh_length, gplaser_normalized_a0, gplaser_derived_output [EXTRACTED 1.00]
- **Spec completeness components: example, validation checks, and explicit out-of-scope boundary** — gplaser_example_input_yaml, gplaser_validation_checks, gplaser_out_of_scope [INFERRED 0.75]
- **xigma-i's gui_adapter.py as one of several pluggable ModelAdapters alongside kascade, delta, and analytical** — src_xigma_i_gui_adapter, models_kaskade, models_delta, models_analytical [EXTRACTED 1.00]
- **Documented gotchas specific to the spectrum_kernel_4d GPU kernel: a0 Jacobian factor, shared-memory aliasing, dr scope** — docs_models_xigma_a0_jacobian_trap, docs_models_xigma_shared_memory_aliasing_trap, docs_models_xigma_dr_scope_trap [INFERRED 0.85]
- **Documented deposition.py-level traps: sparse tables, float precision, CPU/GPU cell mismatch** — docs_models_xigma_table_too_sparse_trap, docs_models_xigma_float32_float64_accumulation_trap, docs_models_xigma_cpu_gpu_deposition_cells_trap [INFERRED 0.85]
- **Stage 0-1-2 tabulated pipeline data flow (particles -> deposition -> spectrum4d)** — models_xigma_i_particles, docs_models_xigma_i_deposition, models_xigma_i_spectrum4d [INFERRED 0.85]
- **reference.py cross-checks surface both known open discrepancies** — docs_models_xigma_i_reference, docs_models_xigma_passport_narrow_angle_noise, docs_models_xigma_passport_2pi_discrepancy [INFERRED 0.75]
- **XIGMA-I code ownership (author + organization)** — docs_models_xigma_passport_xigma_i, docs_models_xigma_passport_samsonov, docs_models_xigma_passport_ipfran [EXTRACTED 1.00]
- **Canonical variables, mass-shell constraint, and the waist-sampling algorithm that enforces them** — docs_refactor_better_gaussian_bunches_canonical_sampling_vars, docs_refactor_better_gaussian_bunches_mass_shell_constraint, docs_refactor_better_gaussian_bunches_waist_sampling_algorithm [EXTRACTED 1.00]
- **Structured fit model plus synthetic-baseline accuracy estimation via Mahalanobis distance** — docs_refactor_better_gaussian_bunches_structured_fit_model, docs_refactor_better_gaussian_bunches_fit_accuracy_baseline, docs_refactor_better_gaussian_bunches_mahalanobis_metric [EXTRACTED 1.00]
- **Waist sampling, vacuum drift, and structured fitting as the three core pipeline stages** — docs_refactor_better_gaussian_bunches_waist_sampling_algorithm, docs_refactor_better_gaussian_bunches_vacuum_drift, docs_refactor_better_gaussian_bunches_structured_fit_model [EXTRACTED 1.00]
- **kascade and xigma_i's independently-reimplemented Gaussian-pulse-envelope physics unified into io/laser_envelope.py's gaussian_pulse_envelope** — src_gammaforge_io_laser_envelope, src_gammaforge_models_kascade_kascade, src_gammaforge_models_xigma_i_particles [EXTRACTED 1.00]
- **The PhysicalQuantity/ModelSpec/adapt_to_model parameter-convention framework (io/quantities.py, io/schema.py, io/adapter.py) — fully built and demonstrated but not wired into any real GUI adapter as of this doc** — src_gammaforge_io_quantities, src_gammaforge_io_schema, src_gammaforge_io_adapter [EXTRACTED 1.00]
- **The original gammaforge.core package proposal and its two flagship abstractions (ModelProtocol/core-adapters, unified SimulationConfig/run_simulation) were all explicitly dropped in favor of io/ absorbing the shared-layer role directly** — docs_refactor_core_simulation_api_core_package, docs_refactor_core_simulation_api_modelprotocol_dropped, docs_refactor_core_simulation_api_simulationconfig_dropped [EXTRACTED 1.00]

## Communities (73 total, 24 thin omitted)

### Community 0 - "IO Core + Validation"
Cohesion: 0.14
Nodes (31): User-curated bank of additional cross-validation parameter sets.  Deliberately e, Cross-model validation suite -- entry point.  Runs kascade/xigma-i/delta/analyti, Tier 0 (config-only, no run needed) + Tier 1/2 (share one run of     each model), run(), _run_gated_scenario(), _kascade_electrons(), Run each model directly (bypassing the ModelAdapter/GUI string-field layer, same, Composite cache key that includes sampling parameters (n_mc, seed)     alongside (+23 more)

### Community 1 - "Bunch Sampling"
Cohesion: 0.06
Nodes (24): add_field_grid(), ComptonGUIApp, _float_or_none(), _native(), Return (total_flux, collimated_flux, cmask_or_None) [ph/s].          cmask is on, Place (label, default, key) triples in an n_cols-wide grid inside a     coloured, Load a 6-D electron bunch from an SDDS ``.ele`` file and turn the         Electr, Drop the loaded ``.ele`` bunch and restore the Electrons panel         to its no (+16 more)

### Community 2 - "GUI App Layout"
Cohesion: 0.07
Nodes (50): Bunch, Macroparticle electron bunch. SI units, flat arrays.      ``x``/``y``/``z`` are, Total number of physical electrons., Elegant / SDDS ``.ele`` file I/O for :class:`gammaforge.io.bunch.Bunch`.  Rel, Write a :class:`Bunch` in SDDS ASCII ``.ele`` format.      Mirrors :func:`load_e, save_elegant_ele(), KASCADE: Sequential multi-photon inverse-Compton Monte Carlo engine., _bunch_to_kascade_electrons() (+42 more)

### Community 3 - "Laser Overlap"
Cohesion: 0.10
Nodes (29): ballistic_position_z0_reference(), laser_overlap_time_window(), Straight-line position at time offset ``t``, given a per-particle     reference, Per-particle time window ``[t0, t1]`` (normalised length units,     e.g. ``k0_la, Photon-density envelope of a Gaussian laser pulse at an arbitrary         point, _bin_spatial(), _bin_temporal(), _get_numba_kernel() (+21 more)

### Community 4 - "XIGMA Deposition"
Cohesion: 0.09
Nodes (37): canonical variables (x, y, z, thx, thy, gamma), energy chirp (z-gamma correlation), dispersion (position-energy correlation), geometric emittance, KS test (fit quality), Mahalanobis distance (fit quality), mass-shell enforcement (pz derived, never sampled), Twiss parameters (alpha, beta, gamma) (+29 more)

### Community 5 - "Electron Beam Model"
Cohesion: 0.09
Nodes (8): GaussianElectronBeam, Quantity, The ``gaussian_6d_waist`` v0.1 I/O contract.      A 6D factorized Gaussian defin, Bunch population (charge / e). A pure count has no unit to be         agnostic a, Absolute RMS dispersion for longitudinal momentum (normalized to mc)., Correlation coefficient ρ_zγ = chirp_h · σ_z / σ_γ., Correlation coefficient ρ_xγ = D_x · σ_γ / σ_x., Correlation coefficient ρ_yγ = D_y · σ_γ / σ_y.

### Community 6 - "Physics Concepts"
Cohesion: 0.05
Nodes (47): Raised when a convention has no registered conversion in its family., UnknownConversionError, _array_module(), build_table(), build_table_streaming(), _cell_indices(), check_accumulation_precision(), _deposit() (+39 more)

### Community 7 - "Laser Specifications"
Cohesion: 0.29
Nodes (9): propagate(), Ballistically drift every macroparticle in ``bunch`` by a time     offset ``dt``, Yield a propagated :class:`Bunch` snapshot at each time in     ``t_grid`` (SI se, stream(), Cross-checks for propagation.py's ballistic drift primitives.  No cupy/GPU/tkint, test_propagate_per_particle_positions_match_hand_formula(), test_propagate_recovers_waist_after_drift(), test_propagate_round_trips() (+1 more)

### Community 8 - "Photon Results"
Cohesion: 0.10
Nodes (8): Quantity, Rayleigh range for the x waist: ``pi * w0^2 / lambda`` with         ``w0 = 2 * w, RMS intensity-profile width in x at absolute position ``z_m``         (SI metres, Angular frequency, ``2*pi*c/wavelength``., Peak power at the pulse's temporal center, at focus:         ``E / (sqrt(2*pi) *, On-axis, peak-in-time intensity at absolute position ``z_m`` (SI         metres), Normalized vector potential at absolute position ``z_m`` (SI         metres), fr, ``a0_at(z_m)**2`` -- the mean-square amplitude.          Consumers that only eve

### Community 9 - "Project Documentation"
Cohesion: 0.14
Nodes (17): BinnedAngularSpectrum, BinnedSpatialDistribution, detect_device(), get_xp(), Auto-detect which backend to use: a real CUDA GPU via cupy if     available, els, Array module for ``device`` ('gpu' -> cupy, 'cpu' -> numpy) -- not a     unit co, Bring an array back to host (numpy) if it's on-device (cupy);     a no-op for CP, to_host() (+9 more)

### Community 10 - "Units & Conventions"
Cohesion: 0.09
Nodes (22): AGENTS.md, All models, Analytical model, Architecture, Commands, Cross-repo gotchas (still apply post-merge), Dependency flow, Design decisions (not to be revisited without good reason) (+14 more)

### Community 11 - "Interaction Parameters"
Cohesion: 0.24
Nodes (11): _old_kascade_laser_density(), _old_xigma_n_ph_shape(), Cross-checks for GaussianParaxialLaser.pulse_envelope against the two independen, Verbatim transcription of the pre-refactor kascade.py laser_density     body (cr, Verbatim transcription of the pre-refactor xigma_i/particles.py     _push_and_sa, test_beta_ff_matches_xigma_flying_focus(), test_crossing_angle_matches_kascade_tilted_axis(), test_focus_offset_matches_kascade_delta() (+3 more)

### Community 12 - "Canonical Sampling Tests"
Cohesion: 0.11
Nodes (31): Shared comparison metric for the cross-model validation suite.  Promoted (moved, s, spec, spec_ref: 1D arrays of equal shape (same grid for both     spectra -- r, Linear-interpolate spec_src(s_src) onto s_ref, zero outside s_src's range., resample_to(), window_integrated_relative_error(), _rel(), _total_yield_kascade(), compton_edge_eV() (+23 more)

### Community 13 - "GUI Conventions"
Cohesion: 0.17
Nodes (21): Electron Beam I/O Specification v0.1, 6D Factorized Gaussian Distribution Model (gaussian_6d_waist, no cross-correlations), Angular Divergence Profile (sigma_x', sigma_y'), Charge, Electron Count, and Peak Current (Q, N_e, I_peak), Rationale for choosing pC over nC as the charge unit, Recommended Derived Output Parameters (num_electrons, gamma_mean, beta_star, emit_norm, peak_current_A, peak_density_cm3, ...), Rationale for naming bunch_duration_rms_ps instead of sigma_l_ps, Energy Spread (rel_energy_spread_rms, sigma_gamma, sigma_gamma/gamma0) (+13 more)

### Community 14 - "GUI Model Selection"
Cohesion: 0.16
Nodes (10): Adapter, angle_integrated_spectrum(), estimate_spectrum_width(), estimate_yield(), ndarray, Fast, closed-form Compton-source physics: total yield, angle-integrated spectrum, Fast closed-form model: total yield, angle-integrated spectrum, and     an estim, Cheap analytic total-photon-yield estimate from an overlap integral     between (+2 more)

### Community 15 - "propagate"
Cohesion: 0.16
Nodes (11): fit_gaussian(), Fit a :class:`GaussianElectronBeam` from raw macroparticles: a full     covarian, load_elegant_ele(), Parse a 6-D electron-bunch ``.ele`` file in SDDS ASCII format.      Required col, test_sdds_round_trip_preserves_bunch_statistics(), Test structured Gaussian fitting., Verify fitted parameters match input beam., Verify fitting works correctly with drifted beam. (+3 more)

### Community 16 - "cache.py"
Cohesion: 0.18
Nodes (20): cache_key(), clear_cache(), get_or_compute(), _git_head_hash(), _git_is_dirty(), list_cache_entries(), load(), _paths() (+12 more)

### Community 17 - "fit_gaussian"
Cohesion: 0.12
Nodes (14): Draw macroparticles from a :class:`GaussianElectronBeam` using     canonical var, sample_gaussian_canonical(), Test chirp and dispersion correlations in sampling., Verify chirp creates measurable z-γ correlation., Verify chirp sign is respected., Verify mass-shell holds with chirp., Verify x-γ dispersion creates measurable correlation., Verify dispersion sign is respected. (+6 more)

### Community 18 - "sample_gaussian_canonical"
Cohesion: 0.16
Nodes (12): Validate a :class:`GaussianElectronBeam`.      Raises ``ValueError`` on hard req, Draw macroparticles from ``beam``. Delegates to     :func:`sample_gaussian_canon, sample_gaussian_bunch(), validate(), Cross-checks for bunch.py's Bunch / GaussianElectronBeam / sample_gaussian_bunch, test_derived_quantities_are_sane(), test_fit_gaussian_recovers_waist_after_drift(), test_fit_gaussian_round_trips_at_the_waist() (+4 more)

### Community 19 - "test_bunch_improvements.py"
Cohesion: 0.14
Nodes (14): mildly_relativistic_beam(), _pq(), Tests for the improved electron bunch sampling, fitting, and evaluation.  This m, Build a PhysicalQuantity for test fixtures., Random number generator for reproducible tests., Ultra-relativistic beam (gamma >> 1) for testing.      Physical parameters: 500, Test edge cases and error handling., Test with zero momentum spread (delta function in gamma).          Both sigma_pz (+6 more)

### Community 20 - "docs/gui/tasks.md — GUI task backlog"
Cohesion: 0.07
Nodes (42): tests/ — test_constants.py, test_conversions.py, test_bunch.py, test_laser.py, test_io_formats.py for gammaforge.io, extra_params() mechanism: ModelAdapter.extra_params() lets a model declare extra numeric fields (e.g. xigma-i's beta_ff/phi_pol) rendered in app.py's grey MODEL PARAMETERS panel, isinstance-vs-duck-typing bug: checking both model_api dataclasses via isinstance silently misses xigma-i's structurally-identical-but-different-class results, breaking validate_results() and app.py render methods; fixed by hasattr duck typing, Known gaps: xigma_i spatial-distribution normalization self-consistently rescaled (not first-principles); no automated Tkinter render test; Conventions-and-units.md now implemented as gammaforge.io, docs/gui/AGENTS.md — Compton-GUIde (gammaforge.gui), Task: abandon angular-range tab, Task: dropdown per-input unit selection (cm, m, mm, um) with automatic PhysicalQuantity-conserving conversion, works on read-only fields too, Task: grey out inputs after simulation done (except charge; XIGMA keeps pulse energy/gamma active), add a "release" button (+34 more)

### Community 21 - "XigmaAdapter"
Cohesion: 0.20
Nodes (16): ~2π angular-spectrum residual (open), crossing angle support, Klein-Nishina cross section, GUI tasks (pending), spectrum_from_particles (production table-free paths), BinnedSpectrum (pre-binned results), SampledSpectrum (unbinned results), Analytical Adapter (+8 more)

### Community 22 - "core-simulation-api refactor (completed)"
Cohesion: 0.13
Nodes (20): a0 formula discrepancy (RESOLVED), CollisionParams dead-field pruning (Phase 1 completed), sigma0_x/sigma0_l wrinkle (not raw inputs), core-simulation-api refactor (completed), Still-open ~49% discrepancy between xigma_i.config.py's a0 formula and GaussianParaxialLaser.a0_focus, flagged in io/laser.py's a0_from_fields docstring (this doc marks it unresolved; note: repo AGENTS.md roadmap elsewhere records this a0/N_l discrepancy as later RESOLVED via a pass-through fix — possible timeline conflict), Proposed gammaforge.core package (core/protocol.py, core/collision.py, core/simulation.py, core/adapters/) between io/ and gui/models/ — never built; io/ absorbed this role instead, Two copies of validation/: stale top-level validation/ (runtime artifacts only) alongside live src/gammaforge/validation/ — safe to rm -rf the stale copy, Dropped from original plan: core/protocol.py's ModelProtocol abstraction and core/adapters/ — no second package layer was ever introduced (+12 more)

### Community 23 - "Gaussian Paraxial Laser I/O Spec v0.1 (short)"
Cohesion: 0.16
Nodes (27): Rayleigh range, Normalized amplitude a0 (a0_focus, a0_interaction), derived not input, Beam size and intensity at the interaction point (paraxial defocus formula, sigma_x(z=0), sigma_y(z=0)), Recommended derived output parameters (waist radii, FWHM, Rayleigh lengths, peak power/intensity, a0), Temporal intensity duration definition (sigma_t; FWHM relation), Pulse energy, peak power and peak intensity (E_L, P_peak, I0_focus, I0_interaction), Minimal example input YAML (laser: model gaussian_paraxial, version 0.1), GaussianParaxialLaser model (gaussian_paraxial, version 0.1) (+19 more)

### Community 24 - "app.py"
Cohesion: 0.20
Nodes (6): Verify pz > 1 for all particles (physical constraint)., Verify sampled statistics match input parameters., Verify gamma is correctly calculated from momenta., Test canonical sampling with mass-shell enforcement., Verify mass-shell constraint: gamma^2 = 1 + px^2 + py^2 + pz^2., TestCanonicalSampling

### Community 25 - "TabulatedEngine"
Cohesion: 0.17
Nodes (7): table.total_weight -- retarget_a0 preserves it exactly., (t_seconds, rate) bin-center arrays -- photon-emission rate vs         time. Non, (x_centers, y_centers, density) -- transverse areal density         [photons/cm^, d2N/(ds dOmega) grid -- spectrum4d.calculate_angular_spectrum_4d         on this, Drives Stage 0/1/2 of the new path for one (beam, laser) collision.     `table`/, Stage 0 (particles.push_and_sample) + Stage 1         (deposition.build_table, a, TabulatedEngine

### Community 26 - "GUI Calculations Section - Task Breakdown"
Cohesion: 0.06
Nodes (35): Acceptance Criteria, Acceptance Criteria, Acceptance Criteria, Acceptance Criteria, Acceptance Criteria, Acceptance Criteria, Description, Description (+27 more)

### Community 27 - "deposition.py (Stage 1 binning; occupancy_diagnostics, check_accumulation_precision)"
Cohesion: 0.18
Nodes (15): gammaforge.io.collision (detect_device backend selector), Trap: 1/(1+a0) Jacobian factor must be recomputed inside the per-a0-bin quadrature loop, not shared across bins, Conda GPU environment setup (conda create -n xigma python=3.12, cupy-cuda12x) -- required for cupyx.scatter_add GPU deposition, Trap: CPU vs GPU deposition -- totals/marginals match tightly but individual edge-case cells can differ, benign, Trap: dr variable scope across syncthreads() in the final evaluation loop, Environment: CuPy (cupyx.jit rawkernels) + numba fallback required, no repo-tracked pytest suite, Trap: float32 vs float64 accumulation -- deposition.py defaults float64 on CPU and GPU; verify overrides with check_accumulation_precision, No repo-tracked test suite -- validation lives in ad hoc scripts against reference.py/spectrum_from_particles.py/deposition.py (+7 more)

### Community 29 - "physics_params module (proposed intermediate canonical layer)"
Cohesion: 0.32
Nodes (7): Like run_xigma, but always computes fresh and returns     ``(Photons, XigmaAdapt, Like run_delta, but always computes fresh and returns     ``(Photons, DirectAdap, run_delta_live(), run_xigma_live(), Tier 3: angular shape canary (xigma-i vs delta), reported not gated.  xigma-i an, Always returns True -- canary tier, reports findings, does not gate     the suit, run()

### Community 30 - "units.py"
Cohesion: 0.08
Nodes (40): Enum, specs/electron_beam_io_v0.1_full.md / _short.md — gaussian_6d_waist spec, specs/gaussian_paraxial_laser_io_v0.1.md / _short.md — gaussian_paraxial spec, adapters/kascade_adapter.py — params_to_config reads gammaforge.io.constants directly, _pq(), Shortcut to build a PhysicalQuantity -- the GUI wraps its own raw     floats dir, _pq(), Electron-bunch representations.  * :class:`Bunch` -- raw, engine-agnostic macrop (+32 more)

### Community 31 - "api.py"
Cohesion: 0.25
Nodes (18): Protocol, AngularRangeSpectrumResult, BinnedTemporalEnvelope, PhotonMultiplicity, Photons, Output-side observable representations: the spectrum/angular-spectrum/ temporal-, What every model's ``run()`` must return (shape-compatibly).      Only ``model_n, SampledSpatialDistribution (+10 more)

### Community 32 - "Electron Beam I/O Spec v0.1 (short)"
Cohesion: 0.25
Nodes (14): Electron Beam I/O Spec v0.1 (short), Longitudinal rms length sigma_z and peak electron density n_0, Derived output fields (bunch_charge_C/nC, num_electrons, gamma_mean, sigma_z_rms_um, peak_current_A, peak_density_cm3, etc.), Transverse n(x,y), temporal n(t), and energy f_E(delta_E) Gaussian distributions, Geometric rms emittance and divergence at focus (sigma_x'=epsilon_x/sigma_x0), Mean gamma_0/beta_0, charge Q, electron count N_e, peak current I_peak, gaussian_6d_waist model (6D Gaussian bunch, focus at interaction point), Geometric convention: interaction point at origin, electrons along +z, alpha_x=alpha_y=0 at focus (+6 more)

### Community 33 - "deposition (Stage 1)"
Cohesion: 0.10
Nodes (27): CLAUDE.md (repo conventions, convergence-testing tools), gammaforge.io.bunch.sample_gaussian_bunch, deposition (Stage 1), reference.py (validation tool), Table (4D overlap table H), ~2π systematic discrepancy: direct_binning_spectrum vs angle_integrated_spectrum, Trajectory-averaged effective a0 (ahat), build_table_streaming (streaming table build for large bunches) (+19 more)

### Community 34 - "Bunch"
Cohesion: 0.25
Nodes (5): discover_models(), Populate the model registry with direct imports. kascade and     analytical have, register(), registered_models(), KascadeAdapter

### Community 35 - "drift"
Cohesion: 0.15
Nodes (11): drift(), _drift_gaussian_fit(), Analytically propagate a :class:`GaussianElectronBeam`'s Twiss tilt     through, Ballistically propagate a bunch by a longitudinal distance ``L`` (SI     metres,, Test vacuum propagation., Verify x -> x + x' * L., Verify z, thx, thy, gamma unchanged by drift., Verify Twiss alpha emerges from waist + drift. (+3 more)

### Community 36 - "ValueError"
Cohesion: 0.24
Nodes (8): External-format I/O for gammaforge.io's bunch/laser representations., load_electron_beam(), YAML I/O for the ``gaussian_6d_waist``/``gaussian_paraxial`` v0.1 formats define, save_electron_beam(), save_laser(), Cross-checks for io_formats/sdds.py and io_formats/yaml_spec.py.  No cupy/GPU/tk, test_spec_example_electron_beam_yaml_round_trips(), test_spec_example_laser_yaml_round_trips()

### Community 37 - "test_laser.py"
Cohesion: 0.10
Nodes (22): InteractionParameters, The shared "interaction parameters" bundle: one canonical (laser, electrons) pai, One canonical (laser, electrons) pair -- the physics-parameter     bundle every, load_laser(), GaussianParaxialLaser, Photon count in the pulse: N_L = pulse_energy / (hbar*omega0). A         pure co, Validate a :class:`GaussianParaxialLaser`.      Raises ``ValueError`` on hard re, Only a paraxial Gaussian laser pulse *for now*.      Every field is a :class:`Ph (+14 more)

### Community 38 - "DirectAdapter"
Cohesion: 0.33
Nodes (3): BinnedSpectrum, DirectAdapter, Evaluate direct_binning_spectrum at a grid of OBSERVATION angles         spannin

### Community 39 - "spectrum4d.py"
Cohesion: 0.24
Nodes (7): calculate_angular_spectrum_4d(), _get_numba(), get_spectrum_kernel_4d_cpu(), CPU/numba equivalent of spectrum4d.py's GPU rawkernel (spectrum_kernel_4d), used, Lazily compiles and caches spectrum_kernel_4d_cpu (see module docstring)., Stage 2: `spectrum_kernel_4d`, the GPU kernel that turns a Stage 1 4D overlap ta, Host-side driver for spectrum_kernel_4d, the Stage-2 analogue of     Compton.cal

### Community 41 - "XIGMA-I code/method"
Cohesion: 0.47
Nodes (4): main(), Compton-GUIde: Tkinter GUI for pluggable Compton-scattering physics models., GammaForge: unified package for inverse-Compton scattering simulation.  Subpac, run_gui()

### Community 42 - "gui_adapter.py (bridge into compton_gui as pluggable ModelAdapter)"
Cohesion: 0.29
Nodes (11): XIGMA four-stage pipeline (Stage 0-2 + validation), tabulated overlap table H[gamma, theta_x, theta_y, a0], GUI integration section: gui_adapter.py as the sole bridge from xigma_i into compton_gui, Lazy-import pattern: cupy/config/tabulated_engine imported only inside available()/run_simulation()/spectrum_in_angular_range(), never at module scope, so import degrades gracefully without cupy/CUDA, Pluggable ModelAdapter pattern (xigma-i is one of several: kascade, delta, analytical), XIGMA-I passport (v0.1), gui/ (Tkinter desktop GUI), models/analytical (analytical ModelAdapter) (+3 more)

### Community 44 - "TestFitQuality"
Cohesion: 0.22
Nodes (8): evaluate_fit_quality(), ndarray, Evaluate Gaussian fit quality with sampling-noise baseline.      Compares the re, Test fit quality evaluation metrics., Verify Gaussian data produces low KS excess., Verify non-Gaussian data produces high KS excess., Verify log-likelihood comparison between real and synthetic., TestFitQuality

### Community 46 - "OutputSpec"
Cohesion: 0.27
Nodes (9): check(), main(), Mirrors app.py's on_start(): the always-on analytical preview runs     alongside, test_model(), test_preview_alongside(), Any, Defensive duck-type check, run right after a model's ``run()``     returns., validate_results() (+1 more)

### Community 48 - "build_params (CGS CollisionParams)"
Cohesion: 0.16
Nodes (15): convergence testing (resolution/particle/quadrature scans), CUDA OOM with large electron bunches, a0_shape (trajectory-averaged a0), push_and_sample (Stage 0), PushDiagnostics, retarget_a0, spectrum4d_cpu (numba fallback), spectrum_kernel_4d (GPU kernel) (+7 more)

### Community 49 - "Model Tasks (Xigma-i, Analytical model, All models)"
Cohesion: 0.22
Nodes (9): Model Tasks (Xigma-i, Analytical model, All models), All models: implement jitter and averaging over shots, Analytical model: collimated spectrum from total yield, collimation angle, spectrum width (convolution of single-electron spectrum with energy distribution and possibly a0), Analytical model: include foci displacement, Analytical model: consider closed-form total yield for non-round beams, Xigma-i: implement crossing angle (changes only polarization factor; geometric overlap uses photon density), Xigma-i: consider gamma-axis rescaling analogous to a0 rescaling (change mean energy without recomputing stages 0-1), Xigma-i: drop '-i' suffix, rename to just XIGMA (+1 more)

### Community 53 - "Job dataclass"
Cohesion: 0.29
Nodes (7): Bunch (flat arrays), drift (ballistic propagation), InteractionParameters, recoil_parameter, Job dataclass, OutputSpec dataclass, run_cross_validation.py

### Community 58 - "opencode.json"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

### Community 59 - "_const"
Cohesion: 0.50
Nodes (3): _const(), Quantity, A CODATA constant from pint's own table, as a plain float in SI base     units -

### Community 61 - "kascade run_simulation"
Cohesion: 0.67
Nodes (3): kascade Results, run_multiphoton_chain, kascade run_simulation

## Ambiguous Edges - Review These
- `Geometric Convention (interaction point at origin, e- along +z, focus at IP, alpha=0)` → `Laser Pulse I/O Specification v0.1 (companion spec referenced by name, exact path inferred)`  [AMBIGUOUS]
  docs/io/specs/electron_beam_io_v0.1_full.md · relation: conceptually_related_to

## Knowledge Gaps
- **119 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `gammaforge`, `Overview`, `Description` (+114 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **24 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Geometric Convention (interaction point at origin, e- along +z, focus at IP, alpha=0)` and `Laser Pulse I/O Specification v0.1 (companion spec referenced by name, exact path inferred)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `GaussianElectronBeam` connect `Electron Beam Model` to `IO Core + Validation`, `Bunch Sampling`, `drift`, `ValueError`, `test_laser.py`, `TestFitQuality`, `GUI Model Selection`, `propagate`, `fit_gaussian`, `sample_gaussian_canonical`, `test_bunch_improvements.py`, `app.py`, `TabulatedEngine`, `units.py`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **Why does `GaussianParaxialLaser` connect `test_laser.py` to `IO Core + Validation`, `Bunch Sampling`, `GUI App Layout`, `Laser Overlap`, `ValueError`, `Photon Results`, `GUI Model Selection`, `TabulatedEngine`, `units.py`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Why does `XigmaAdapter` connect `XigmaAdapter` to `build_params (CGS CollisionParams)`, `Model Tasks (Xigma-i, Analytical model, All models)`, `OutputSpec`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `GaussianElectronBeam` (e.g. with `ComptonGUIApp` and `NoConvention`) actually correct?**
  _`GaussianElectronBeam` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `ComptonGUIApp` (e.g. with `GaussianElectronBeam` and `InteractionParameters`) actually correct?**
  _`ComptonGUIApp` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `GaussianParaxialLaser` (e.g. with `ComptonGUIApp` and `InteractionParameters`) actually correct?**
  _`GaussianParaxialLaser` has 13 INFERRED edges - model-reasoned connections that need verification._