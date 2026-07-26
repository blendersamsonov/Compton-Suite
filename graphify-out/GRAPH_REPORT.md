# Graph Report - .  (2026-07-26)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 762 nodes · 1663 edges · 40 communities (31 shown, 9 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 207 edges (avg confidence: 0.56)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `54e4ef6a`
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
- kascade
- xigma-direct
- xigma-i

## God Nodes (most connected - your core abstractions)
1. `MacroBunch` - 47 edges
2. `Scenario` - 43 edges
3. `GaussianElectronBeam` - 41 edges
4. `GaussianParaxialLaser` - 40 edges
5. `ComptonGuideApp` - 38 edges
6. `CommonResults` - 33 edges
7. `ModelCapabilities` - 27 edges
8. `ModelAdapter` - 26 edges
9. `UnavailableAdapter` - 24 edges
10. `BinnedSpectrum` - 21 edges

## Surprising Connections (you probably didn't know these)
- `run()` --calls--> `check()`  [INFERRED]
  validation/tier0_wiring.py → GUIde/scripts/headless_test.py
- `ParamError` --uses--> `MacroBunch`  [INFERRED]
  GUIde/src/compton_guide/adapters/kascade_adapter.py → IO/src/compton_io/bunch.py
- `KascadeAdapter` --uses--> `MacroBunch`  [INFERRED]
  GUIde/src/compton_guide/adapters/kascade_adapter.py → IO/src/compton_io/bunch.py
- `ModelCapabilities` --uses--> `MacroBunch`  [INFERRED]
  GUIde/src/compton_guide/model_api.py → IO/src/compton_io/bunch.py
- `ModelCapabilities` --uses--> `AngularRangeSpectrumResult`  [INFERRED]
  GUIde/src/compton_guide/model_api.py → IO/src/compton_io/photons.py

## Import Cycles
- None detected.

## Communities (40 total, 9 thin omitted)

### Community 0 - "Scenario"
Cohesion: 0.07
Nodes (77): Shared comparison metric for the cross-model validation suite.  Promoted (moved, s, spec, spec_ref: 1D arrays of equal shape (same grid for both     spectra -- r, Linear-interpolate spec_src(s_src) onto s_ref, zero outside s_src's range., resample_to(), window_integrated_relative_error(), User-curated bank of additional cross-validation parameter sets.  Deliberately e, Cross-model validation suite -- entry point.  Runs kascade/xigma-i/xigma-i-direc, Tier 0 (config-only, no run needed) + Tier 1/2 (share one run of     each model) (+69 more)

### Community 1 - "compton_io/__init__.py"
Cohesion: 0.06
Nodes (56): Enum, Thin re-export of ``compton_io``'s parameter-semantics/unit normalisation framew, ModelSpec for kascade (formerly dfe5_compton_mc), at the boundary its own adapte, adapt_to_model(), params_to_floats(), Canonical params -> a specific model's own convention/units.  This is the last s, ``canonical_params`` may be in any convention/unit as long as the     meaning ma, Strip the semantic wrapper for the final call into model code, which     wants p (+48 more)

### Community 2 - "ComptonGuideApp"
Cohesion: 0.05
Nodes (29): add_field_grid(), ComptonGuideApp, _float_or_none(), laser_focal_radii(), main(), peak_a0(), Peak normalised vector potential a_0 from the laser fields (or None)., Transverse rms bunch size [m] from normalized emittance, beta [m] and gamma. (+21 more)

### Community 3 - "deposition.py"
Cohesion: 0.06
Nodes (37): _array_module(), build_table(), build_table_streaming(), _cell_indices(), check_accumulation_precision(), _deposit(), deposit_cic(), deposit_nearest() (+29 more)

### Community 4 - "particles.py"
Cohesion: 0.08
Nodes (33): laser_overlap_time_window(), Per-particle time window ``[t0, t1]`` (normalised length units,     e.g. ``k0_la, Physics constants and GPU kernel sizing constants for this pipeline (particles.p, _bin_spatial(), _bin_temporal(), _get_numba_kernel(), _normalise_bunch(), push_and_sample() (+25 more)

### Community 5 - "kaskade/kascade.py"
Cohesion: 0.10
Nodes (34): beta_of(), build_lambda_grid(), Config, cos_collision(), doppler_D(), invert_lambda(), kn_over_thomson(), kn_sigma_ratio() (+26 more)

### Community 6 - "CommonResults"
Cohesion: 0.18
Nodes (20): ModelAdapter, ModelCapabilities, Any, Return (True, "") if the model can actually be run right now, else         (Fals, Model-specific numeric fields with no shared-panel analogue         (kascade's E, ``electrons`` is required: electron sampling is the IO layer's         (caller's, BinnedAngularSpectrum, BinnedSpatialDistribution (+12 more)

### Community 7 - "XigmaDirectAdapter"
Cohesion: 0.10
Nodes (16): _attach_private_cache(), DirectConfig, _float(), ParamError, params_to_config(), Exception, ndarray, ModelAdapter for the brute-force particle-binning model.  Reuses ``xigma_i.parti (+8 more)

### Community 8 - "GaussianElectronBeam"
Cohesion: 0.09
Nodes (6): beta_star_from_sigma_emit(), divergence_from_sigma_emit(), GaussianElectronBeam, Beta function at the waist, ``sigma**2 / emit_geom`` -- same units in     as out, RMS divergence at the waist, ``emit_geom / sigma`` -- a dimensionless     angle,, The ``gaussian_6d_waist`` v0.1 I/O contract.      A 6D factorized Gaussian defin

### Community 9 - "xigma_i/gui_adapter.py"
Cohesion: 0.10
Nodes (17): capabilities(), extra_params(), _float(), ParamError, params_to_config(), Exception, GUI-facing adapter exposing the xigma_i pipeline through a dfe5-shaped Config/ru, Model-specific numeric fields with no dfe5 analogue, for the GUI to     render i (+9 more)

### Community 10 - "MacroBunch"
Cohesion: 0.14
Nodes (11): Raise NotImplementedError if capabilities().supports_ele_file_io is False., fit_gaussian(), MacroBunch, Fit a :class:`GaussianElectronBeam` (at its waist) from raw macroparticles., Raw, engine-agnostic macroparticle representation. SI units.      ``x``/``y``/``, load_elegant_ele(), Elegant / SDDS ``.ele`` file I/O for :class:`compton_io.bunch.MacroBunch`.  Relo, Write a :class:`MacroBunch` in SDDS ASCII ``.ele`` format.      Mirrors :func:`l (+3 more)

### Community 11 - "cache.py"
Cohesion: 0.18
Nodes (20): cache_key(), clear_cache(), get_or_compute(), _git_head_hash(), _git_is_dirty(), list_cache_entries(), load(), _paths() (+12 more)

### Community 12 - "headless_test.py"
Cohesion: 0.17
Nodes (15): check(), FakeVar, main(), make_fields(), Same mechanism app.py's on_start() uses: whichever model is     'selected', the, Stand-in for tk.StringVar -- params_to_config only ever calls .get()., sample_electrons_for(), test_model() (+7 more)

### Community 13 - "GaussianParaxialLaser"
Cohesion: 0.15
Nodes (5): GaussianParaxialLaser, ``a0_at(z_m)**2`` -- the mean-square amplitude.          Consumers that only eve, A paraxial Gaussian laser pulse. SI units.      ``waist_rms_x_m``/``waist_rms_y_, Photon count in the pulse: N_L = pulse_energy_J / (hbar*omega0)., Normalized vector potential at ``z_m``, from the exact SI         plane-wave rel

### Community 14 - "propagation.py"
Cohesion: 0.18
Nodes (15): ballistic_position_simultaneous(), ballistic_position_z0_reference(), propagate(), Ballistic (straight-line, no-acceleration) electron-beam propagation, and a time, Ballistically drift every macroparticle in ``bunch`` by a time     offset ``dt``, Yield a propagated :class:`MacroBunch` snapshot at each time in     ``t_grid`` (, Straight-line position at time offset ``dt`` later, given a     per-particle ref, Straight-line position at time offset ``t``, given a per-particle     reference (+7 more)

### Community 15 - "KascadeAdapter"
Cohesion: 0.14
Nodes (8): _float(), KascadeAdapter, _macrobunch_to_kascade_electrons(), ParamError, Exception, ModelAdapter wrapping the existing kascade engine.  This is the "control" adapte, Convert a MacroBunch into the plain dict kascade.run_simulation's     ``electron, Convert the GUI fields into a physics Config plus a driver dict.          Ported

### Community 16 - "InteractionGeometry"
Cohesion: 0.17
Nodes (13): build_params(), Derive this convention's CGS :class:`CollisionParams` from     ``compton_io``'s, InteractionGeometry, Collision geometry: foci displacement and crossing angle. SI units., _attach_private_cache(), Config, ndarray, Stash this adapter's own private recompute cache on a     ``compton_io.results.C (+5 more)

### Community 17 - "AnalyticalConfig"
Cohesion: 0.14
Nodes (4): AnalyticalConfig, _float(), ModelAdapter for the fast, always-on analytical model.  Duck-typed against ``com, Wraps beam/pulse, but ALSO exposes the flat SI fields every other     model's Co

### Community 18 - "BinnedSpectrum"
Cohesion: 0.15
Nodes (11): Mask the cached per-photon lab angles against the picked range and         histo, Optional capability -- check         capabilities().supports_angular_range_spect, AngularRangeSpectrumResult, BinnedSpectrum, Binned dN/dE spectrum (semi-analytic/tabulated models)., ParamError, Exception, Evaluate direct_binning_spectrum at a grid of OBSERVATION angles     spanning th (+3 more)

### Community 19 - "sample_gaussian_bunch"
Cohesion: 0.21
Nodes (13): Validate a :class:`GaussianElectronBeam` per spec Sec. 16.      Raises ``ValueEr, Draw macroparticles from a :class:`GaussianElectronBeam`.      Independent facto, sample_gaussian_bunch(), validate(), Cross-checks for bunch.py's MacroBunch / GaussianElectronBeam / sample_gaussian_, test_derived_quantities_are_sane(), test_fit_gaussian_recovers_waist_after_drift(), test_fit_gaussian_round_trips_at_the_waist() (+5 more)

### Community 20 - "laser.py"
Cohesion: 0.20
Nodes (10): External-format I/O for compton_io's bunch/laser representations., load_electron_beam(), load_laser(), YAML I/O for the ``gaussian_6d_waist``/``gaussian_paraxial`` v0.1 formats define, save_electron_beam(), save_laser(), Laser-pulse representation: the ``gaussian_paraxial`` v0.1 I/O contract (``specs, Cross-checks for io_formats/sdds.py and io_formats/yaml_spec.py.  No cupy/GPU/tk (+2 more)

### Community 21 - "test_laser.py"
Cohesion: 0.18
Nodes (10): laser_from_shared_fields(), Validate a :class:`GaussianParaxialLaser` per spec Sec. 12.      Raises ``ValueE, Build a :class:`GaussianParaxialLaser` from the flat SI field set every     mode, validate(), Cross-checks for laser.py's GaussianParaxialLaser.  No cupy/GPU/tkinter needed., test_defocused_interaction_intensity_is_lower(), test_derived_quantities_are_sane(), test_laser_from_shared_fields_round_trips() (+2 more)

### Community 22 - "bunch.py"
Cohesion: 0.20
Nodes (6): Electron-bunch representations.  Two distinct types, matching the distinction th, CollisionParams, The CGS "collision parameters" bundle for tabulated-overlap-style GPU/CPU pipeli, CGS scalars for one laser-electron collision -- immutable, built     once by :fu, Physical constants, derived from ``units.py``'s pint registry rather than hand-t, The shared "interaction parameters" bundle: an electron beam, a laser pulse, and

### Community 24 - "_interp4d"
Cohesion: 0.27
Nodes (8): _interp4d(), Cross-validation-only tooling: a brute-force, non-GPU-kernel way to turn a Stage, Brute-force quadrature of dN/(ds dOmega) at a single observation point     (x0,, Resolves backend='numpy'|'cupy' to its array module. cupy is imported     lazily, Array-module-agnostic core of interp4d: takes H already converted to     xp's mo, Quadrilinear interpolation of table.H at query points (arrays of equal     shape, spectrum_from_table(), _xp_for()

### Community 25 - "analytical.py"
Cohesion: 0.22
Nodes (8): angle_integrated_spectrum(), estimate_spectrum_width(), estimate_yield(), ndarray, Fast, closed-form Compton-source physics: total yield, angle-integrated spectrum, Cheap analytic total-photon-yield estimate from an overlap integral     between, Cheap analytic estimate of the collimated-spectrum FWHM (in units of     the Com, dN/ds integrated over all emission solid angle, from the standard     angle-inde

### Community 26 - "compton_suite/__init__.py"
Cohesion: 0.32
Nodes (5): ``comptonsuite-gui`` console-script entry point.  Thin wrapper launching the sam, run_gui(), ComptonSuite: the umbrella package tying together the toolkit's pieces (all now, run_gui(), Headless-safe model discovery for scripting -- no tkinter/matplotlib import, sam

### Community 27 - "detect_device"
Cohesion: 0.29
Nodes (7): detect_device(), Auto-detect which backend to use: a real CUDA GPU via cupy if     available, els, available(), available(), _backend_note(), True if either supported backend can actually run: a real CUDA GPU     (cupy + a, Which backend run_simulation will actually use, for display/logging     -- not p

### Community 32 - "InteractionParameters"
Cohesion: 0.50
Nodes (3): InteractionParameters, One canonical (beam, pulse, geometry) triple -- the physics-parameter     subset, This scenario's (beam, pulse, geometry) as one shared         compton_io.interac

### Community 33 - "compton-io"
Cohesion: 0.50
Nodes (4): compton-analytical, compton-guide, compton-io, compton-suite

## Knowledge Gaps
- **5 isolated node(s):** `compton-guide`, `compton-analytical`, `kascade`, `xigma-i`, `xigma-direct`
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MacroBunch` connect `MacroBunch` to `kaskade/kascade.py`, `CommonResults`, `XigmaDirectAdapter`, `xigma_i/gui_adapter.py`, `propagation.py`, `KascadeAdapter`, `InteractionGeometry`, `AnalyticalConfig`, `BinnedSpectrum`, `sample_gaussian_bunch`, `bunch.py`, `UnavailableAdapter`, `AnalyticalAdapter`, `XigmaAdapter`?**
  _High betweenness centrality (0.136) - this node is a cross-community bridge._
- **Why does `ComptonGuideApp` connect `ComptonGuideApp` to `CommonResults`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Why does `sample_gaussian_bunch()` connect `sample_gaussian_bunch` to `Scenario`, `ComptonGuideApp`, `deposition.py`, `GaussianElectronBeam`, `MacroBunch`, `headless_test.py`, `propagation.py`, `bunch.py`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `MacroBunch` (e.g. with `KascadeAdapter` and `ParamError`) actually correct?**
  _`MacroBunch` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `Scenario` (e.g. with `GaussianElectronBeam` and `InteractionGeometry`) actually correct?**
  _`Scenario` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `GaussianElectronBeam` (e.g. with `CollisionParams` and `InteractionGeometry`) actually correct?**
  _`GaussianElectronBeam` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `GaussianParaxialLaser` (e.g. with `CollisionParams` and `InteractionGeometry`) actually correct?**
  _`GaussianParaxialLaser` has 9 INFERRED edges - model-reasoned connections that need verification._