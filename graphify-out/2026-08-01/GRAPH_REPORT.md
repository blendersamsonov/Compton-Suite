# Graph Report - ComptonSuite  (2026-08-01)

## Corpus Check
- 75 files · ~81,690 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1120 nodes · 2280 edges · 79 communities (59 shown, 20 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 220 edges (avg confidence: 0.58)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c318917d`
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
- run_delta_live
- spectrum4d.py
- xigma_i/params/spec.py — XIGMA_SPEC / XIGMA_DIAGNOSTIC_SPEC (model's own parameter contract)
- XIGMA-I code/method
- gui_adapter.py (bridge into compton_gui as pluggable ModelAdapter)
- parameter-framework-and-collision-params refactor
- TestFitQuality
- TestFitting
- OutputSpec
- spectrum_kernel_4d (GPU kernel)
- build_params (CGS CollisionParams)
- Model Tasks (Xigma-i, Analytical model, All models)
- build_params
- test_analytical.py
- _xp_for
- Job dataclass
- bunch.py
- models.py — discover_models() model registry setup (no tkinter/matplotlib import)
- test_constants.py
- TestIntegration
- opencode.json
- _const
- kascade run_simulation
- graphify.js
- isinstance trap (duck-typing boundary)
- ComptonSuite
- GammaForge planned rename
- pint unit registry
- compton_suite.io AGENTS.md
- KASCADE AGENTS.md
- Photons dataclass
- ModelCapabilities dataclass
- compton-suite
- a0 factorises out of the table (a0_shape)
- a0 is a trajectory average, not instantaneous sample
- Canonical variables with mass-shell enforcement
- Each model converts at its own boundary
- No derived-value properties on model Config
- No model-local particle sampling
- Single result contract (Photons)
- Twiss tilt emerges from waist sampling + drift

## God Nodes (most connected - your core abstractions)
1. `GaussianElectronBeam` - 71 edges
2. `GaussianParaxialLaser` - 52 edges
3. `ComptonGUIApp` - 51 edges
4. `Bunch` - 45 edges
5. `PhysicalQuantity` - 39 edges
6. `Job` - 39 edges
7. `Scenario` - 39 edges
8. `sample_gaussian_canonical()` - 35 edges
9. `OutputSpec` - 27 edges
10. `InteractionParameters` - 25 edges

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

## Hyperedges (group relationships)
- **All ModelAdapter implementations** — models_kascade_kascadeadapter, models_xigma_i_xigmaadapter, models_xigma_i_directadapter, models_analytical_adapter, models_api_unavailableadapter [EXTRACTED 1.00]
- **XIGMA four-stage pipeline** — docs_models_xigma_i_push_and_sample, docs_models_xigma_i_deposition, docs_models_xigma_i_spectrum_kernel_4d, docs_models_xigma_i_spectrum4d_cpu, docs_models_xigma_i_reference [EXTRACTED 1.00]
- **io/ shared layer (constants, framework, beam/laser)** — io_gaussian_electron_beam, io_gaussian_paraxial_laser, io_interaction_parameters, io_bunch, io_build_params, io_collision_params, io_convert_width, io_physical_quantity, io_model_spec, io_adapt_to_model, io_params_to_floats, io_canonical [EXTRACTED 1.00]
- **Unified physics-parameter framework shared by the GUI and both models' own specs** — src_compton_suite_io_quantities, src_compton_suite_gui_physics_params, src_compton_suite_models_xigma_i_params_spec, src_compton_suite_models_kascade_params_spec [INFERRED 0.85]
- **compton_suite.io as the shared dependency layer for the GUI and every model** — src_compton_suite_io_init, src_compton_suite_gui_physics_params, src_compton_suite_models_xigma_i, src_compton_suite_models_kascade_kascade [INFERRED 0.85]
- **Duck-typing convention adopted to resolve the ModelAdapter isinstance bug** — src_compton_suite_gui_model_api, src_compton_suite_models_xigma_i_gui_adapter, docs_gui_agents_isinstance_bug [EXTRACTED 1.00]
- **GUI input -> PhysicalQuantity -> canonical conversion -> per-model adapter data flow** — docs_gui_conventions_and_units_physicalquantity, docs_gui_conventions_and_units_canonical_conventions, docs_gui_conventions_and_units_adapt_to_model [EXTRACTED 1.00]
- **WidthConvention enumerates RMS, FWHM, and 1/e² width definitions** — docs_gui_conventions_and_units_widthconvention, docs_gui_conventions_and_units_rms, docs_gui_conventions_and_units_fwhm, docs_gui_conventions_and_units_w0_1e2 [EXTRACTED 1.00]
- **compton_suite.io as the shared framework layer underlying both models' schemas** — src_compton_suite_io, src_compton_suite_models_xigma_i_params_spec, src_compton_suite_models_kascade_params_spec [EXTRACTED 1.00]
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
- **kascade and xigma_i's independently-reimplemented Gaussian-pulse-envelope physics unified into io/laser_envelope.py's gaussian_pulse_envelope** — src_compton_suite_io_laser_envelope, src_compton_suite_models_kascade_kascade, src_compton_suite_models_xigma_i_particles [EXTRACTED 1.00]
- **The PhysicalQuantity/ModelSpec/adapt_to_model parameter-convention framework (io/quantities.py, io/schema.py, io/adapter.py) — fully built and demonstrated but not wired into any real GUI adapter as of this doc** — src_compton_suite_io_quantities, src_compton_suite_io_schema, src_compton_suite_io_adapter [EXTRACTED 1.00]
- **The original compton_suite.core package proposal and its two flagship abstractions (ModelProtocol/core-adapters, unified SimulationConfig/run_simulation) were all explicitly dropped in favor of io/ absorbing the shared-layer role directly** — docs_refactor_core_simulation_api_core_package, docs_refactor_core_simulation_api_modelprotocol_dropped, docs_refactor_core_simulation_api_simulationconfig_dropped [EXTRACTED 1.00]

## Communities (79 total, 20 thin omitted)

### Community 0 - "IO Core + Validation"
Cohesion: 0.07
Nodes (67): Shared comparison metric for the cross-model validation suite.  Promoted (moved, s, spec, spec_ref: 1D arrays of equal shape (same grid for both     spectra -- r, Linear-interpolate spec_src(s_src) onto s_ref, zero outside s_src's range., resample_to(), window_integrated_relative_error(), User-curated bank of additional cross-validation parameter sets.  Deliberately e, Cross-model validation suite -- entry point.  Runs kascade/xigma-i/delta/analyti, Tier 0 (config-only, no run needed) + Tier 1/2 (share one run of     each model) (+59 more)

### Community 1 - "Bunch Sampling"
Cohesion: 0.07
Nodes (17): add_field_grid(), ComptonGUIApp, main(), Return (total_flux, collimated_flux, cmask_or_None) [ph/s].          cmask is on, Place (label, default, key) triples in an n_cols-wide grid inside a     coloured, Load a 6-D electron bunch from an SDDS ``.ele`` file and turn the         Electr, Drop the loaded ``.ele`` bunch and restore the Electrons panel         to its no, Flip the Electron-panel entries to read-only and write the         parameters de (+9 more)

### Community 2 - "GUI App Layout"
Cohesion: 0.09
Nodes (41): KASCADE: Sequential multi-photon inverse-Compton Monte Carlo engine., beta_of(), build_lambda_grid(), Config, cos_collision(), doppler_D(), _eps_L(), invert_lambda() (+33 more)

### Community 3 - "Laser Overlap"
Cohesion: 0.10
Nodes (28): ballistic_position_z0_reference(), laser_overlap_time_window(), Straight-line position at time offset ``t``, given a per-particle     reference, Per-particle time window ``[t0, t1]`` (normalised length units,     e.g. ``k0_la, Photon-density envelope of a Gaussian laser pulse at an arbitrary         point, _bin_spatial(), _bin_temporal(), _get_numba_kernel() (+20 more)

### Community 4 - "XIGMA Deposition"
Cohesion: 0.09
Nodes (37): canonical variables (x, y, z, thx, thy, gamma), energy chirp (z-gamma correlation), dispersion (position-energy correlation), geometric emittance, KS test (fit quality), Mahalanobis distance (fit quality), mass-shell enforcement (pz derived, never sampled), Twiss parameters (alpha, beta, gamma) (+29 more)

### Community 5 - "Electron Beam Model"
Cohesion: 0.09
Nodes (8): GaussianElectronBeam, Quantity, The ``gaussian_6d_waist`` v0.1 I/O contract.      A 6D factorized Gaussian defin, Bunch population (charge / e). A pure count has no unit to be         agnostic a, Absolute RMS dispersion for longitudinal momentum (normalized to mc)., Correlation coefficient ρ_zγ = chirp_h · σ_z / σ_γ., Correlation coefficient ρ_xγ = D_x · σ_γ / σ_x., Correlation coefficient ρ_yγ = D_y · σ_γ / σ_y.

### Community 6 - "Physics Concepts"
Cohesion: 0.07
Nodes (37): _array_module(), build_table(), build_table_streaming(), _cell_indices(), check_accumulation_precision(), _deposit(), deposit_cic(), deposit_nearest() (+29 more)

### Community 7 - "Laser Specifications"
Cohesion: 0.19
Nodes (11): Validate a :class:`GaussianElectronBeam`.      Raises ``ValueError`` on hard req, Draw macroparticles from ``beam``. Delegates to     :func:`sample_gaussian_canon, sample_gaussian_bunch(), validate(), Cross-checks for bunch.py's Bunch / GaussianElectronBeam / sample_gaussian_bunch, test_derived_quantities_are_sane(), test_fit_gaussian_recovers_waist_after_drift(), test_fit_gaussian_round_trips_at_the_waist() (+3 more)

### Community 8 - "Photon Results"
Cohesion: 0.13
Nodes (11): GaussianParaxialLaser, Quantity, Rayleigh range for the x waist: ``pi * w0^2 / lambda`` with         ``w0 = 2 * w, RMS intensity-profile width in x at absolute position ``z_m``         (SI metres, Angular frequency, ``2*pi*c/wavelength``., Photon count in the pulse: N_L = pulse_energy / (hbar*omega0). A         pure co, Peak power at the pulse's temporal center, at focus:         ``E / (sqrt(2*pi) *, On-axis, peak-in-time intensity at absolute position ``z_m`` (SI         metres) (+3 more)

### Community 9 - "Project Documentation"
Cohesion: 0.15
Nodes (18): AngularRangeSpectrumResult, BinnedAngularSpectrum, BinnedSpatialDistribution, BinnedSpectrum, BinnedTemporalEnvelope, Photons, Output-side observable representations: the spectrum/angular-spectrum/ temporal-, What every model's ``run()`` must return (shape-compatibly).      Only ``model_n (+10 more)

### Community 10 - "Units & Conventions"
Cohesion: 0.08
Nodes (25): AGENTS.md, All models, Analytical model, Architecture, Commands, Correcting the wiring pass's architecture regressions (this session), Cross-repo gotchas (still apply post-merge), Dependency flow (+17 more)

### Community 11 - "Interaction Parameters"
Cohesion: 0.24
Nodes (11): _old_kascade_laser_density(), _old_xigma_n_ph_shape(), Cross-checks for GaussianParaxialLaser.pulse_envelope against the two independen, Verbatim transcription of the pre-refactor kascade.py laser_density     body (cr, Verbatim transcription of the pre-refactor xigma_i/particles.py     _push_and_sa, test_beta_ff_matches_xigma_flying_focus(), test_crossing_angle_matches_kascade_tilted_axis(), test_focus_offset_matches_kascade_delta() (+3 more)

### Community 12 - "Canonical Sampling Tests"
Cohesion: 0.35
Nodes (10): Why compton_suite.io exists: each consumer used to hand-maintain its own physical constants (~1.6e-8 disagreement in xigma_i's older-CODATA copy) and its own structurally-identical-but-not-same-class parameter-semantics framework, so a PhysicalQuantity built with one copy's enums failed validation against another copy's ModelSpec; this package gives every consumer exactly one shared copy, docs/refactor/parameter-framework-and-collision-params.md — sigma0_x/sigma0_l wrinkle note, kascade has no model-owned ModelSpec yet — KASCADE_SPEC still lives in gui/physics_params/schemas/kascade.py, unlike XIGMA_SPEC which already moved to models/xigma_i/params/; doc recommends moving it to models/kascade/params/spec.py before Phase 2 wiring, gui/physics_params/schemas/kascade.py — KASCADE_SPEC, GUI-owned (not model-owned, unlike XIGMA_SPEC which already moved into models/xigma_i/params/), physics_params/ — thin re-export of compton_suite.io's parameter-semantics framework, io/adapter.py — adapt_to_model, params_to_floats, io/canonical.py — one canonical convention+unit per PhysicalMeaning; to_canonical/from_canonical, io/enums.py — PhysicalMeaning, WidthConvention, TimeConvention, AmplitudeConvention (+2 more)

### Community 13 - "GUI Conventions"
Cohesion: 0.17
Nodes (21): Electron Beam I/O Specification v0.1, 6D Factorized Gaussian Distribution Model (gaussian_6d_waist, no cross-correlations), Angular Divergence Profile (sigma_x', sigma_y'), Charge, Electron Count, and Peak Current (Q, N_e, I_peak), Rationale for choosing pC over nC as the charge unit, Recommended Derived Output Parameters (num_electrons, gamma_mean, beta_star, emit_norm, peak_current_A, peak_density_cm3, ...), Rationale for naming bunch_duration_rms_ps instead of sigma_l_ps, Energy Spread (rel_energy_spread_rms, sigma_gamma, sigma_gamma/gamma0) (+13 more)

### Community 14 - "GUI Model Selection"
Cohesion: 0.16
Nodes (10): Adapter, angle_integrated_spectrum(), estimate_spectrum_width(), estimate_yield(), ndarray, Fast, closed-form Compton-source physics: total yield, angle-integrated spectrum, Fast closed-form model: total yield, angle-integrated spectrum, and     an estim, Cheap analytic total-photon-yield estimate from an overlap integral     between (+2 more)

### Community 15 - "propagate"
Cohesion: 0.25
Nodes (10): propagate(), Ballistically drift every macroparticle in ``bunch`` by a time     offset ``dt``, Yield a propagated :class:`Bunch` snapshot at each time in     ``t_grid`` (SI se, stream(), Cross-checks for propagation.py's ballistic drift primitives.  No cupy/GPU/tkint, test_ballistic_position_z0_reference_matches_hand_formula(), test_propagate_per_particle_positions_match_hand_formula(), test_propagate_recovers_waist_after_drift() (+2 more)

### Community 16 - "cache.py"
Cohesion: 0.18
Nodes (20): cache_key(), clear_cache(), get_or_compute(), _git_head_hash(), _git_is_dirty(), list_cache_entries(), load(), _paths() (+12 more)

### Community 17 - "fit_gaussian"
Cohesion: 0.14
Nodes (11): fit_gaussian(), Fit a :class:`GaussianElectronBeam` from raw macroparticles: a full     covarian, Test chirp and dispersion correlations in sampling., Verify chirp creates measurable z-γ correlation., Verify chirp sign is respected., Verify x-γ dispersion creates measurable correlation., Verify dispersion sign is respected., Verify chirp and dispersion work together. (+3 more)

### Community 18 - "sample_gaussian_canonical"
Cohesion: 0.13
Nodes (11): Draw macroparticles from a :class:`GaussianElectronBeam` using     canonical var, sample_gaussian_canonical(), Verify pz > 1 for all particles (physical constraint)., Verify sampled statistics match input parameters., Verify gamma is correctly calculated from momenta., Verify mass-shell holds with chirp., Verify mass-shell holds with dispersion., Verify error on too-strong chirp/dispersion. (+3 more)

### Community 19 - "test_bunch_improvements.py"
Cohesion: 0.14
Nodes (14): mildly_relativistic_beam(), _pq(), Tests for the improved electron bunch sampling, fitting, and evaluation.  This m, Build a PhysicalQuantity for test fixtures., Random number generator for reproducible tests., Ultra-relativistic beam (gamma >> 1) for testing.      Physical parameters: 500, Test edge cases and error handling., Test with zero momentum spread (delta function in gamma).          Both sigma_pz (+6 more)

### Community 20 - "docs/gui/tasks.md — GUI task backlog"
Cohesion: 0.14
Nodes (17): extra_params() mechanism: ModelAdapter.extra_params() lets a model declare extra numeric fields (e.g. xigma-i's beta_ff/phi_pol) rendered in app.py's grey MODEL PARAMETERS panel, isinstance-vs-duck-typing bug: checking both model_api dataclasses via isinstance silently misses xigma-i's structurally-identical-but-different-class results, breaking validate_results() and app.py render methods; fixed by hasattr duck typing, Task: abandon angular-range tab, Task: dropdown per-input unit selection (cm, m, mm, um) with automatic PhysicalQuantity-conserving conversion, works on read-only fields too, Task: grey out inputs after simulation done (except charge; XIGMA keeps pulse energy/gamma active), add a "release" button, Task: 2D/3D interaction geometry sketches — axes, 3D ellipses for e-/laser, polarization arrows, time-delay "ghosts", Task: parameter scans/ranges, Task: remove hard-coded seed and number of macroelectrons from compton photons tab into model-specific parameters (+9 more)

### Community 21 - "XigmaAdapter"
Cohesion: 0.26
Nodes (13): ~2π angular-spectrum residual (open), GUI tasks (pending), spectrum_from_particles (production table-free paths), BinnedSpectrum (pre-binned results), SampledSpectrum (unbinned results), Analytical Adapter, discover_models() registry function, ModelAdapter protocol (+5 more)

### Community 22 - "core-simulation-api refactor (completed)"
Cohesion: 0.23
Nodes (12): core-simulation-api refactor (completed), Still-open ~49% discrepancy between xigma_i.config.py's a0 formula and GaussianParaxialLaser.a0_focus, flagged in io/laser.py's a0_from_fields docstring (this doc marks it unresolved; note: repo AGENTS.md roadmap elsewhere records this a0/N_l discrepancy as later RESOLVED via a pass-through fix — possible timeline conflict), Proposed compton_suite.core package (core/protocol.py, core/collision.py, core/simulation.py, core/adapters/) between io/ and gui/models/ — never built; io/ absorbed this role instead, Two copies of validation/: stale top-level validation/ (runtime artifacts only) alongside live src/compton_suite/validation/ — safe to rm -rf the stale copy, Dropped from original plan: core/protocol.py's ModelProtocol abstraction and core/adapters/ — no second package layer was ever introduced, Dropped from original plan: SimulationConfig/run_simulation() as a single unified entry point — never built; each model still driven via its own ModelAdapter.run(), Phase 1 (complete): pruned 11 dead CollisionParams fields (emit_x/emit_y/sigma_ez/beta_x/beta_y/sigma_thx/sigma_thy/lambda_l/delta_x/delta_y/delta_z) — zero downstream reads found by exhaustive grep across src/tests/scripts; kept the geometry parameter to build_params for a future crossing-offset feature, io/ as single source of truth for constants and framework (+4 more)

### Community 23 - "Gaussian Paraxial Laser I/O Spec v0.1 (short)"
Cohesion: 0.16
Nodes (27): Rayleigh range, Normalized amplitude a0 (a0_focus, a0_interaction), derived not input, Beam size and intensity at the interaction point (paraxial defocus formula, sigma_x(z=0), sigma_y(z=0)), Recommended derived output parameters (waist radii, FWHM, Rayleigh lengths, peak power/intensity, a0), Temporal intensity duration definition (sigma_t; FWHM relation), Pulse energy, peak power and peak intensity (E_L, P_peak, I0_focus, I0_interaction), Minimal example input YAML (laser: model gaussian_paraxial, version 0.1), GaussianParaxialLaser model (gaussian_paraxial, version 0.1) (+19 more)

### Community 24 - "app.py"
Cohesion: 0.16
Nodes (16): _float_or_none(), _native(), _pq(), Refresh electron and laser values derived from the current fields., Compute (wavelength_m, sigma0_l_m, sigma_par_L_m, pulse_energy_J,         focus_, Compile the current Electrons + Laser panel fields into the         analytic bea, Shortcut to build a PhysicalQuantity -- the GUI wraps its own raw     floats dir, Convert a raw field value from the GUI's own display unit     (``native_unit``, (+8 more)

### Community 25 - "TabulatedEngine"
Cohesion: 0.14
Nodes (8): Engine on the tabulated-energy pipeline (particles.py/deposition.py/ spectrum4d., table.total_weight -- retarget_a0 preserves it exactly., (t_seconds, rate) bin-center arrays -- photon-emission rate vs         time. Non, (x_centers, y_centers, density) -- transverse areal density         [photons/cm^, d2N/(ds dOmega) grid -- spectrum4d.calculate_angular_spectrum_4d         on this, Drives Stage 0/1/2 of the new path for one `params` (CollisionParams)     config, Stage 0 (particles.push_and_sample) + Stage 1         (deposition.build_table, a, TabulatedEngine

### Community 26 - "GUI Calculations Section - Task Breakdown"
Cohesion: 0.06
Nodes (35): Acceptance Criteria, Acceptance Criteria, Acceptance Criteria, Acceptance Criteria, Acceptance Criteria, Acceptance Criteria, Description, Description (+27 more)

### Community 27 - "deposition.py (Stage 1 binning; occupancy_diagnostics, check_accumulation_precision)"
Cohesion: 0.18
Nodes (15): compton_suite.io.collision (detect_device backend selector), Trap: 1/(1+a0) Jacobian factor must be recomputed inside the per-a0-bin quadrature loop, not shared across bins, Conda GPU environment setup (conda create -n xigma python=3.12, cupy-cuda12x) -- required for cupyx.scatter_add GPU deposition, Trap: CPU vs GPU deposition -- totals/marginals match tightly but individual edge-case cells can differ, benign, Trap: dr variable scope across syncthreads() in the final evaluation loop, Environment: CuPy (cupyx.jit rawkernels) + numba fallback required, no repo-tracked pytest suite, Trap: float32 vs float64 accumulation -- deposition.py defaults float64 on CPU and GPU; verify overrides with check_accumulation_precision, No repo-tracked test suite -- validation lives in ad hoc scripts against reference.py/spectrum_from_particles.py/deposition.py (+7 more)

### Community 28 - "PhysicalQuantity"
Cohesion: 0.22
Nodes (9): adapt_to_model, AmplitudeConvention enum, canonical representation, convert_width / convert_time / convert_amplitude, params_to_floats, PhysicalMeaning enum, PhysicalQuantity, TimeConvention enum (+1 more)

### Community 29 - "physics_params module (proposed intermediate canonical layer)"
Cohesion: 0.17
Nodes (15): AmplitudeConvention enum (A0_PEAK, A0_RMS), CANONICAL_CONVENTIONS mapping (one canonical convention per PhysicalMeaning), convert_width(value, from_conv, to_conv) unified conversion interface, FWHM (full width at half maximum) convention, ParameterSpec dataclass (name, meaning, convention, unit, description), PhysicalMeaning enum (LASER_WIDTH, PULSE_DURATION, LASER_AMPLITUDE), PhysicalQuantity dataclass (value, unit, meaning, convention), physics_params module (proposed intermediate canonical layer) (+7 more)

### Community 30 - "units.py"
Cohesion: 0.11
Nodes (23): Enum, adapters/kascade_adapter.py — params_to_config reads compton_suite.io.constants directly, io/constants.py — physical constants derived from units.py's pint registry (SI block + CGS-Gaussian views for xigma_i), Shared physical constants, pint unit registry, parameter-semantics vocabulary, a, AmplitudeConvention, _convert(), convert_amplitude(), convert_time() (+15 more)

### Community 31 - "api.py"
Cohesion: 0.21
Nodes (9): Protocol, discover_models(), ModelAdapter, Model-agnostic contract between app.py and physics-engine adapters.  This module, Model-specific parameters as (label, default, key) triples.          Default can, Optional: return a dict mapping parameter keys to allowed string         values, Populate the model registry with direct imports. kascade and     analytical have, register() (+1 more)

### Community 32 - "Electron Beam I/O Spec v0.1 (short)"
Cohesion: 0.25
Nodes (14): Electron Beam I/O Spec v0.1 (short), Longitudinal rms length sigma_z and peak electron density n_0, Derived output fields (bunch_charge_C/nC, num_electrons, gamma_mean, sigma_z_rms_um, peak_current_A, peak_density_cm3, etc.), Transverse n(x,y), temporal n(t), and energy f_E(delta_E) Gaussian distributions, Geometric rms emittance and divergence at focus (sigma_x'=epsilon_x/sigma_x0), Mean gamma_0/beta_0, charge Q, electron count N_e, peak current I_peak, gaussian_6d_waist model (6D Gaussian bunch, focus at interaction point), Geometric convention: interaction point at origin, electrons along +z, alpha_x=alpha_y=0 at focus (+6 more)

### Community 33 - "deposition (Stage 1)"
Cohesion: 0.20
Nodes (15): compton_suite.io.bunch.sample_gaussian_bunch, deposition (Stage 1), reference.py (validation tool), Table (4D overlap table H), ~2π systematic discrepancy: direct_binning_spectrum vs angle_integrated_spectrum, Trajectory-averaged effective a0 (ahat), build_table_streaming (streaming table build for large bunches), 4D overlap table H[γ, θx, θy, a0] (+7 more)

### Community 34 - "Bunch"
Cohesion: 0.15
Nodes (15): Bunch, Macroparticle electron bunch. SI units, flat arrays.      ``x``/``y``/``z`` are, Total number of physical electrons., PhotonMultiplicity, SampledSpatialDistribution, SampledSpectrum, SampledTemporalEnvelope, Job (+7 more)

### Community 35 - "drift"
Cohesion: 0.15
Nodes (11): drift(), _drift_gaussian_fit(), Analytically propagate a :class:`GaussianElectronBeam`'s Twiss tilt     through, Ballistically propagate a bunch by a longitudinal distance ``L`` (SI     metres,, Test vacuum propagation., Verify x -> x + x' * L., Verify z, thx, thy, gamma unchanged by drift., Verify Twiss alpha emerges from waist + drift. (+3 more)

### Community 36 - "ValueError"
Cohesion: 0.15
Nodes (16): External-format I/O for compton_suite.io's bunch/laser representations., load_elegant_ele(), Elegant / SDDS ``.ele`` file I/O for :class:`compton_suite.io.bunch.Bunch`.  Rel, Write a :class:`Bunch` in SDDS ASCII ``.ele`` format.      Mirrors :func:`load_e, Parse a 6-D electron-bunch ``.ele`` file in SDDS ASCII format.      Required col, save_elegant_ele(), load_electron_beam(), load_laser() (+8 more)

### Community 37 - "test_laser.py"
Cohesion: 0.21
Nodes (7): Validate a :class:`GaussianParaxialLaser`.      Raises ``ValueError`` on hard re, validate(), Cross-checks for laser.py's GaussianParaxialLaser.  No cupy/GPU/tkinter needed., test_defocused_interaction_intensity_is_lower(), test_derived_quantities_are_sane(), test_validate_rejects_nonpositive_fields(), test_validate_warns_on_astigmatism()

### Community 38 - "run_delta_live"
Cohesion: 0.32
Nodes (7): Like run_xigma, but always computes fresh and returns     ``(Photons, XigmaAdapt, Like run_delta, but always computes fresh and returns     ``(Photons, DirectAdap, run_delta_live(), run_xigma_live(), Tier 3: angular shape canary (xigma-i vs delta), reported not gated.  xigma-i an, Always returns True -- canary tier, reports findings, does not gate     the suit, run()

### Community 39 - "spectrum4d.py"
Cohesion: 0.24
Nodes (7): calculate_angular_spectrum_4d(), _get_numba(), get_spectrum_kernel_4d_cpu(), CPU/numba equivalent of spectrum4d.py's GPU rawkernel (spectrum_kernel_4d), used, Lazily compiles and caches spectrum_kernel_4d_cpu (see module docstring)., Stage 2: `spectrum_kernel_4d`, the GPU kernel that turns a Stage 1 4D overlap ta, Host-side driver for spectrum_kernel_4d, the Stage-2 analogue of     Compton.cal

### Community 40 - "xigma_i/params/spec.py — XIGMA_SPEC / XIGMA_DIAGNOSTIC_SPEC (model's own parameter contract)"
Cohesion: 0.22
Nodes (11): AGENTS.md (root), Compton-GUIde AGENTS.md, Known gaps: xigma_i spatial-distribution normalization self-consistently rescaled (not first-principles); no automated Tkinter render test; Conventions-and-units.md now implemented as compton_suite.io, Conventions-and-units design doc, adapt_to_model(canonical_params, model_spec) adapter function, from_canonical(quantity, target_convention) function, MODEL_SPEC per-model schema, validate_against_spec(params, spec) function (+3 more)

### Community 41 - "XIGMA-I code/method"
Cohesion: 0.17
Nodes (12): CLAUDE.md (repo conventions, convergence-testing tools), Compton-XIGMA GitHub repository, Crossing angle not supported architecturally (head-on only), Single-electron resonance replaced by delta-function, dfe5_compton_mc (MC-Kost alternative engine in same GUI), ИПФ РАН (responsible organization), Самсонов А. С. (code owner), Trust level C (unified physics-model rating) (+4 more)

### Community 42 - "gui_adapter.py (bridge into compton_gui as pluggable ModelAdapter)"
Cohesion: 0.29
Nodes (11): XIGMA four-stage pipeline (Stage 0-2 + validation), tabulated overlap table H[gamma, theta_x, theta_y, a0], GUI integration section: gui_adapter.py as the sole bridge from xigma_i into compton_gui, Lazy-import pattern: cupy/config/tabulated_engine imported only inside available()/run_simulation()/spectrum_in_angular_range(), never at module scope, so import degrades gracefully without cupy/CUDA, Pluggable ModelAdapter pattern (xigma-i is one of several: kascade, delta, analytical), XIGMA-I passport (v0.1), gui/ (Tkinter desktop GUI), models/analytical (analytical ModelAdapter) (+3 more)

### Community 43 - "parameter-framework-and-collision-params refactor"
Cohesion: 0.29
Nodes (7): CollisionParams dead-field pruning (Phase 1 completed), sigma0_x/sigma0_l wrinkle (not raw inputs), XIGMA_SPEC, parameter-framework-and-collision-params refactor, CollisionParams, ModelSpec, ParameterSpec

### Community 44 - "TestFitQuality"
Cohesion: 0.22
Nodes (8): evaluate_fit_quality(), ndarray, Evaluate Gaussian fit quality with sampling-noise baseline.      Compares the re, Test fit quality evaluation metrics., Verify Gaussian data produces low KS excess., Verify non-Gaussian data produces high KS excess., Verify log-likelihood comparison between real and synthetic., TestFitQuality

### Community 45 - "TestFitting"
Cohesion: 0.25
Nodes (5): Test structured Gaussian fitting., Verify fitted parameters match input beam., Verify sigma_gamma is correctly calculated from fit., Verify chirp is computed and finite for uncorrelated sampling., TestFitting

### Community 46 - "OutputSpec"
Cohesion: 0.15
Nodes (15): check(), main(), Mirrors app.py's on_start(): the always-on analytical preview runs     alongside, test_model(), test_preview_alongside(), Build an OutputSpec from the GUI fields., Build the ``Job.extra`` dict for ``adapter`` from its own         ``model_params, InteractionParameters (+7 more)

### Community 47 - "spectrum_kernel_4d (GPU kernel)"
Cohesion: 0.25
Nodes (9): convergence testing (resolution/particle/quadrature scans), spectrum4d_cpu (numba fallback), spectrum_kernel_4d (GPU kernel), fit_gaussian, GaussianElectronBeam, MacroBunch, sample_gaussian_bunch, sample_gaussian_canonical (+1 more)

### Community 48 - "build_params (CGS CollisionParams)"
Cohesion: 0.24
Nodes (11): a0 formula discrepancy (RESOLVED), CUDA OOM with large electron bunches, a0_shape (trajectory-averaged a0), push_and_sample (Stage 0), PushDiagnostics, retarget_a0, TabulatedEngine, a0_from_fields (+3 more)

### Community 49 - "Model Tasks (Xigma-i, Analytical model, All models)"
Cohesion: 0.22
Nodes (9): Model Tasks (Xigma-i, Analytical model, All models), All models: implement jitter and averaging over shots, Analytical model: collimated spectrum from total yield, collimation angle, spectrum width (convolution of single-electron spectrum with energy distribution and possibly a0), Analytical model: include foci displacement, Analytical model: consider closed-form total yield for non-round beams, Xigma-i: implement crossing angle (changes only polarization factor; geometric overlap uses photon density), Xigma-i: consider gamma-axis rescaling analogous to a0 rescaling (change mean energy without recomputing stages 0-1), Xigma-i: drop '-i' suffix, rename to just XIGMA (+1 more)

### Community 50 - "build_params"
Cohesion: 0.19
Nodes (8): detect_device(), Auto-detect which backend to use: a real CUDA GPU via cupy if     available, els, build_params(), CollisionParams, The CGS "collision parameters" bundle for this package's tabulated- overlap-styl, CGS scalars for one laser-electron collision -- immutable, built     once by :fu, Derive this convention's CGS :class:`CollisionParams` from     ``compton_suite.i, Physics constants and GPU kernel sizing constants for this pipeline (particles.p

### Community 52 - "_xp_for"
Cohesion: 0.32
Nodes (7): angle_integrated_spectrum(), direct_binning_spectrum(), Table-free spectrum paths computed directly from Stage 0/1 macroparticles -- no, Resolves backend='numpy'|'cupy' to its array module. cupy is imported     lazily, dN/ds integrated over all emission solid angle, from real Stage 0/1     macropar, `delta`'s actual computation: for each real macroparticle,     compute the photo, _xp_for()

### Community 53 - "Job dataclass"
Cohesion: 0.20
Nodes (10): crossing angle support, Klein-Nishina cross section, Bunch (flat arrays), drift (ballistic propagation), InteractionParameters, recoil_parameter, Job dataclass, OutputSpec dataclass (+2 more)

### Community 54 - "bunch.py"
Cohesion: 0.12
Nodes (23): tests/ — test_constants.py, test_conversions.py, test_bunch.py, test_laser.py, test_io_formats.py for compton_suite.io, docs/io/AGENTS.md — compton_suite.io package, specs/electron_beam_io_v0.1_full.md / _short.md — gaussian_6d_waist spec, specs/gaussian_paraxial_laser_io_v0.1.md / _short.md — gaussian_paraxial spec, _pq(), Electron-bunch representations.  * :class:`Bunch` -- raw, engine-agnostic macrop, Shortcut to build a PhysicalQuantity., io/converters.py — pure scalar-factor conversions (FWHM<->sigma, etc.) (+15 more)

### Community 55 - "models.py — discover_models() model registry setup (no tkinter/matplotlib import)"
Cohesion: 0.40
Nodes (5): docs/gui/AGENTS.md — Compton-GUIde (compton_suite.gui), models.py — discover_models() model registry setup (no tkinter/matplotlib import), physics_constants.py — re-export of compton_suite.io.constants for GUI formula helpers, compton_suite.models.delta — brute-force per-particle binning model, same GPU availability story as xigma-i, compton_suite.models.xigma_i — GPU/cupy-only Compton class; greyed out without cupy/CUDA

### Community 57 - "TestIntegration"
Cohesion: 0.33
Nodes (4): End-to-end integration tests., Test complete pipeline: sample -> drift -> fit -> evaluate., Verify sample_gaussian_bunch delegates to canonical sampling., TestIntegration

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
- `io/quantities.py — PhysicalQuantity (value + unit + meaning + convention)` → `compton_suite.io module (actual implementation of the framework + physical constants)`  [AMBIGUOUS]
  docs/refactor/parameter-framework-and-collision-params.md · relation: conceptually_related_to
- `io/schema.py — ParameterSpec/ModelSpec types (each model owns its own ModelSpec instance)` → `compton_suite.io module (actual implementation of the framework + physical constants)`  [AMBIGUOUS]
  docs/refactor/parameter-framework-and-collision-params.md · relation: conceptually_related_to
- `io/adapter.py — adapt_to_model, params_to_floats` → `compton_suite.io module (actual implementation of the framework + physical constants)`  [AMBIGUOUS]
  docs/refactor/parameter-framework-and-collision-params.md · relation: conceptually_related_to
- `Geometric Convention (interaction point at origin, e- along +z, focus at IP, alpha=0)` → `Laser Pulse I/O Specification v0.1 (companion spec referenced by name, exact path inferred)`  [AMBIGUOUS]
  docs/io/specs/electron_beam_io_v0.1_full.md · relation: conceptually_related_to

## Knowledge Gaps
- **134 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `compton-suite`, `Overview`, `Description` (+129 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **20 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `io/quantities.py — PhysicalQuantity (value + unit + meaning + convention)` and `compton_suite.io module (actual implementation of the framework + physical constants)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `io/schema.py — ParameterSpec/ModelSpec types (each model owns its own ModelSpec instance)` and `compton_suite.io module (actual implementation of the framework + physical constants)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `io/adapter.py — adapt_to_model, params_to_floats` and `compton_suite.io module (actual implementation of the framework + physical constants)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Geometric Convention (interaction point at origin, e- along +z, focus at IP, alpha=0)` and `Laser Pulse I/O Specification v0.1 (companion spec referenced by name, exact path inferred)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `GaussianElectronBeam` connect `Electron Beam Model` to `IO Core + Validation`, `Bunch Sampling`, `drift`, `ValueError`, `Laser Specifications`, `TestFitQuality`, `TestFitting`, `GUI Model Selection`, `fit_gaussian`, `sample_gaussian_canonical`, `build_params`, `test_bunch_improvements.py`, `bunch.py`, `app.py`, `TestIntegration`, `units.py`?**
  _High betweenness centrality (0.106) - this node is a cross-community bridge._
- **Why does `GaussianParaxialLaser` connect `Photon Results` to `IO Core + Validation`, `Bunch Sampling`, `GUI App Layout`, `Laser Overlap`, `ValueError`, `test_laser.py`, `OutputSpec`, `GUI Model Selection`, `build_params`, `bunch.py`, `app.py`, `units.py`?**
  _High betweenness centrality (0.083) - this node is a cross-community bridge._
- **Why does `core-simulation-api refactor (completed)` connect `core-simulation-api refactor (completed)` to `build_params (CGS CollisionParams)`, `parameter-framework-and-collision-params refactor`, `Canonical Sampling Tests`, `bunch.py`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._