# CLAUDE.md

## What this is

CuPy/CUDA code for computing inverse Compton scattering spectra from an
electron bunch colliding with a laser pulse, in `src/xigma_i/`: total
photon yield, angle-integrated spectrum (dN/dE), and angular spectrum
(d²N/dE dΩ). A tabulated-energy pipeline: rather than assuming the
electron energy distribution factorises out of phase space and is
Gaussian, it builds a 4D overlap table `H[gamma, theta_x, theta_y, a0]` by
particle deposition, carrying arbitrary correlations (chirp,
divergence-energy, focusing) directly in the data. Falls back to a
CPU/numba implementation automatically when no CUDA GPU is available.

Physics reference is the accompanying paper. The single-electron resonance
function is replaced by a delta function, collapsing the energy integral
analytically to a 4D overlap table (`H`) plus a 3D quadrature
(`theta_x, theta_y, a0`) per output point.

## Current state

Stage 0 (particle source + ballistic pusher) and Stage 1 (nearest + CIC
deposition, CPU and GPU) are done. Stage 2 (`spectrum_kernel_4d`) is done
and its `coef` normalisation (see `reference.py` below) is validated: at a
typical bunch config (`compare_direct_vs_table.py --grid-integrate`), all
three of `spectrum_from_table`/`direct_binning_spectrum`/
`spectrum_kernel_4d` agree within ~15% (clustered around a still-open,
deliberately-deferred `~2*pi` residual against `angle_integrated_spectrum`
-- see `reference.py`'s module docstring). `spectrum_kernel_4d` has a
CPU/numba fallback (`spectrum4d_cpu.py`), validated against the real GPU
kernel (see "Architecture" below). The GUI integration (`gui_adapter.py`)
is fully wired onto this pipeline -- total yield, angle-integrated
spectrum, angular spectrum, temporal envelope, and spatial distribution
all come from it; nothing in this repo runs any other compute path.

One open caveat: in narrow-angle/sparse-table configs, `spectrum_kernel_4d`
alone shows large, unstable variance (overshoots the other two reference
paths by anywhere from ~3x to >30x depending on particle count/table
resolution, sometimes with huge run-to-run std), while
`spectrum_from_table`/`direct_binning_spectrum` still agree tightly with
each other in the same configs. Reads as heavy-tailed importance-sampling
noise from sparse/zero `H` cells in the kernel's own quadrature (see
"Table too sparse" trap), not a normalisation error -- flagged, not
chased further.

`push_and_sample`'s `a0` output is an `a0`-independent shape factor
(`a0_shape`) rather than the physical `ahat`, with
`deposition.retarget_a0(table, a0, a0_max=...)` converting an
`a0_kind='shape'` table into a physical, spectrum-ready `a0_kind='ahat'`
table for one chosen `a0` by a cheap, exact rebin -- see "a0 factorises
out of the table" below. `a0_max` is a fixed *model* parameter (the a0
range the weakly-nonlinear approximation is meant to be valid over), not
derived per-collision from `compton.a0` -- current default guidance is
`a0_max=0.5`. `compare_direct_vs_table.py`, `deposition.build_table_streaming`,
and `tabulated_engine.py` all use this convention; the rest of
`validation/*.py` (`refs.py`'s `make_samples`/table builders,
`fig_gridres.py`'s a0-axis bin-count scan in particular) was **not**
updated -- those scripts still treat `push_and_sample`'s 4th output column
as physical `ahat` directly, so re-running them now would silently produce
wrong a0-axis physics rather than erroring. Flagged, not fixed, pending a
scoped follow-up (`fig_gridres.py`'s a0-scan in particular needs a design
call, not a mechanical find/replace -- see that script's own module
docstring for why).

**Not done**: systematic resolution/convergence scans (tooling exists, see
"Convergence testing" below -- no results recorded yet); chasing the
narrow-angle `spectrum_kernel_4d` sampling-noise caveat above further.

## Architecture

Four stages, plus shared config and validation/GUI layers.

### Stage 0 -- `particles.py`

`sample_bunch(compton, n_particles, gamma0, sigma_gamma0, chirp=,
angle_energy_corr=, rng=)` draws a macroparticle bunch with real
per-particle `(x0, y0, z0, gamma, theta_x, theta_y)` from their *true*
(untruncated) distributions. `push_and_sample(compton, bunch, n_steps=,
backend=)` ballistically pushes each particle through the pulse and emits
**one** `(gamma, theta_x, theta_y, a0, weight)` sample **per particle**
(not per timestep -- see "a0 is a trajectory average" below).
`gamma`/`theta_x`/`theta_y` are constant per particle (straight-line, no
pusher acceleration); `weight` is the luminosity functional `L(zeta) = sum
over the trajectory of v_rel * n_ph_shape * dt * weight_macro`
(Paper/xigma.tex eq. "lumfun"); `a0` is the trajectory-averaged effective
intensity, see below. `n_steps` sets the resolution of the internal
per-timestep integrals that produce `weight` and `a0`, not the length of
`push_and_sample`'s output (always `n_particles`).

`backend`: `'numpy'` (default, single-threaded vectorised), `'numba'`
(CPU multithreading, `numba.prange` over particles, avoids materialising
the full `(n_particles, n_steps)` intermediate arrays), or `'cupy'` (GPU
offload, same vectorised form as `'numpy'` run with cupy arrays -- output
stays on-device, ready to feed `deposition.build_table` without a host
round-trip).

**a0 is a trajectory average, not an instantaneous sample.** `a0` (`H`'s
4th axis) is `ahat(zeta) = (TrXi/2) * integral[a0_local(t)^2]^2 dt /
integral[a0_local(t)^2] dt` (Paper/xigma.tex eq. "ahattraj"), where
`a0_local(t) = compton.a0 * sqrt(local intensity / peak intensity)` is the
*instantaneous* local field amplitude computed internally at each
timestep, and `TrXi/2 = (1 + compton.ellipticity**2) / 2` (eq. "Xi";
`ellipticity=0` linear, `+-1` circular -- `Compton.ellipticity`, set via
`set_laser_parameters`). This is a genuine, previously-made mistake, not a
hypothetical one: an earlier version of this code deposited `a0_local(t)`
itself into `H` once per timestep, i.e. treated `a0` the same way as
`gamma`/`theta_x`/`theta_y` -- one distribution smeared over each
particle's whole trajectory. That's only valid in the synchrotron/wiggler
regime, where the photon formation length is about one laser cycle and
the trajectory can be split into independently-radiating segments. This
codebase is in the opposite, weakly-nonlinear regime (`a0 <~ 1`; see the
paper's regime-validity discussion), where the formation length spans the
*whole* trajectory: an electron radiates one line, shaped by the single
effective intensity value it experienced over its entire passage, not a
sequence of per-instant emissions. **Do not reintroduce per-timestep a0
deposition.**

**a0 factorises out of the table -- `a0_shape` and `deposition.retarget_a0`.**
`ahat(zeta)` above splits *exactly* as `ahat(zeta) = compton.a0**2 *
a0_shape(zeta)`, because `a0_local(t)**2 = compton.a0**2 * ratio(t)` with
`ratio(t) = local intensity / peak intensity` depending only on the
particle's (ballistic, `a0`-independent) trajectory -- never on
`compton.a0` itself. `push_and_sample` computes and returns `a0_shape`
(built without referencing `compton.a0` at all), not the physical `ahat`,
so one Stage 0/1 run's table can be re-targeted to *any* actual `a0` (any
pulse energy) after the fact, without rerunning particle push or
deposition. `deposition.Table.a0_kind` (`'shape'` or `'ahat'`, default
`'ahat'`) records which one a table's 4th axis holds;
`deposition.retarget_a0(table, a0, n_bins=, a0_min=, a0_max=)` converts a
`'shape'` table into a spectrum-ready `'ahat'` table for one specific `a0`
via a conservative (mass-preserving) 1D histogram regrid of the
(already-built, can be finer-resolution) `a0_shape` axis onto a small
fixed-size target grid spanning `[a0_min, a0_max]` (the *model's* valid
`a0` range, not whatever range the current `a0_shape` table happens to
span) -- source-bin mass below `a0_min`/above `a0_max` folds fully into
the corresponding edge target bin rather than being lost, so a small
actual `a0` collapses most/all of the table into one low bin (cheap,
effectively-linear-Compton case) and a large one spreads mass across the
full target range. `spectrum_kernel_4d`/`spectrum_from_table` require
`a0_kind='ahat'` and assert it -- passing a `'shape'` table straight to
either is a real trap (silently wrong resonance condition, since the axis
values wouldn't be physical `ahat`), guarded against explicitly.

**Temporal envelope / spatial distribution (`PushDiagnostics`).**
`push_and_sample`'s optional `n_time_bins`/`t_edges`/`n_spatial_bins`/
`spatial_edges` bin the exact same per-`(particle, timestep)`
`contribution` array that already gets summed into `weight` (L), during
the same trajectory-integration loop -- not a second pass, and not
per-timestep `a0` sampling (the warning above is specifically about `a0`;
time and space are orthogonal axes with no such regime-validity caveat).
Nearest-cell histogrammed by absolute time `t` (-> a photon-emission-rate
`time_envelope` vs `t_edges`, seconds) and by position `(x, y)` (-> an
areal-density `spatial_envelope` vs `spatial_x_edges`/`spatial_y_edges`,
cm). Needs no post-hoc rescale to reproduce `total_yield`: both histograms
bin the already-correctly-normalised `contribution` array directly, so
summing either over all bins reproduces `sum(L)` exactly by construction
(verified numerically: exact for the time histogram, since every
particle's own sampled `t` values are provably a subset of the
auto-derived aggregate window; the spatial histogram can lose a small
tail fraction -- measured ~0.1% in one test config -- since its
auto-derived window is a fixed formula, not the true per-particle range).
Backward compatible by construction: `push_and_sample` returns its
original 5-tuple unchanged unless one of these is passed, in which case a
6th value (a `PushDiagnostics`) is appended. Not implemented for
`backend='numba'` (raises `NotImplementedError` if requested there) -- no
current caller uses that backend. `TabulatedEngine.temporal_envelope`/
`.spatial_distribution` (below) wrap this for GUI use.

### Stage 1 -- `deposition.py`

`Grid4D.from_samples` derives axis extents from the sample data plus a
margin. `deposit_nearest`/`deposit_cic` bin `(gamma, theta_x, theta_y, a0,
weight)` samples into `H`; both are **array-module-agnostic** (`xp = numpy`
or `cupy`, auto-detected via `_array_module` or forced) -- there is no
CUDA-specific kernel here, because deposition is an independent-per-sample
scatter-add and there's nothing for a custom kernel to do beyond
`xp.ravel_multi_index` + scatter-add. Scatter-add is `np.bincount` on CPU
and `cupyx.scatter_add` on GPU (see `_scatter_add`'s docstring).
`build_table(gamma, theta_x, theta_y, a0, weight, scheme='nearest'|'cic',
device=None|'cpu'|'gpu', n_bins=, batch_size=, a0_kind=, ...)` orchestrates
grid derivation + deposition + diagnostics into a `Table`; `device=None`
auto-detects from the input arrays' type, `batch_size` streams host arrays
through GPU deposition in bounded chunks so a Stage-0 sample set larger
than GPU memory still works. `build_table_streaming` combines Stage 0+1
into one chunked pipeline for particle counts too large to draw and push
in one call. `Table.save`/`.load` round-trip through `.npz`, always as
host (numpy) arrays regardless of which device built them.
`occupancy_diagnostics(table)` gives the empty-cell fraction and per-cell
count histogram ("Table too sparse" in Traps); `gamma_bracket(H, grid, q=)`
computes the quantile-based `(gamma_lo, gamma_hi)` Stage 2 needs (also
computed automatically inside `build_table`, stored as
`table.gamma_bracket`).

### Stage 2 -- `spectrum4d.py`

`spectrum_kernel_4d` turns a Stage 1 table into a spectrum via ring/arc
annulus geometry, inverse-CDF importance-sampled phi/theta sampling, and
quadrilinear interpolation of `H`, with a plain midpoint quadrature over
`a0`'s (few, by design) bins nested inside the phi/subsampling loop -- see
the module's own docstring for the full per-block algorithm. `coef = 1.5`,
a pure numerical constant from eq. "main"/"Fmatrix" in the accompanying
paper. `calculate_angular_spectrum_4d(table, s, theta_x, theta_y, phi_pol,
samples_per_point=, device=None)` is the host driver; `device=None|'cpu'|
'gpu'` auto-detects via `config._detect_device()`; callers pass
`theta_x`/`theta_y`/`s` as `cp`/`np` arrays matching the chosen device
(caller converts via `xp.asarray`, not the function itself -- see
`gui_adapter.py`'s calls for the pattern).

**CPU/numba fallback (`spectrum4d_cpu.py`).** `spectrum4d.py`'s
`cupy`/`cupyx` import is a `try/except` at module scope (own `_HAS_CUPY`),
and `spectrum_kernel_4d` is only decorated/compiled (`jit.rawkernel()`)
when cupy imports successfully (else `None`).
`spectrum4d_cpu.get_spectrum_kernel_4d_cpu()` lazily compiles and caches
`spectrum_kernel_4d_cpu`, a literal `numba.prange`-parallel-over-`out_idx`
transliteration of `spectrum_kernel_4d` (same ring/arc annulus geometry,
same inverse-CDF phi sampling, GPU-only thread-bookkeeping dropped since
it doesn't affect the numeric result), including quadrilinear (not
bilinear) interpolation of `H` and the a0-quadrature loop inside the final
evaluation with `g`/`prefac` recomputed per a0 bin (the `1/(1+a0)`
Jacobian factor this codebase has gotten wrong once already -- see
"Traps"). The two zero-weight guards (`inv_cdf` falling back to a cell's
left edge, `sample_area` contributing zero) carry over unchanged, since
they exist because of `H`'s sparsity. Runs in float64 throughout
(deliberate -- no shared-memory/bandwidth pressure on CPU motivating
single precision, unlike the GPU kernel's `CP_FLOAT` convention).
**Validated numerically against the real GPU kernel** on a CUDA machine,
fed the identical `H`/`H_marginal`/grid-scalar arguments: correlation
0.9992, total integrated flux within ~2%. Also cross-checked directly
against `reference.spectrum_from_table` (brute-force grid quadrature, no
kernel at all) on the CPU path alone: agrees within the same ~10-30% band
documented above for `spectrum_kernel_4d` vs `spectrum_from_table` (the
GPU kernel's own known importance-sampling noise, not something the CPU
port adds).

### Validation tools -- `reference.py`

Three independent, non-GPU-kernel ways to compute a spectrum from Stage
0/1 output, used to validate Stage 2 without trusting it:

- `angle_integrated_spectrum(gamma, particle_weight, s)`: dN/ds integrated
  over all emission solid angle, from the standard angle-independent
  Compton edge shape alone (no table, no coef, no theta quadrature).
- `spectrum_from_table(table, x0, y0, s, phi_pol)`: brute-force grid
  quadrature over `H`, no importance sampling. `coef = 1.5`, the same
  pure numerical constant as `spectrum_kernel_4d`'s. `compton` is not
  part of this function's signature (it was only ever needed for
  `compton.Wph`). **Validated** by grid-integrated cross-check against
  `direct_binning_spectrum` (agrees to <5% for a typical bunch).
- `direct_binning_spectrum(gamma, theta_x, theta_y, particle_weight, a0,
  x0, y0, s_edges, phi_pol)`: per-real-macroparticle resonance binning, no
  table, no quadrature at all. Uses the single-electron prefactor from eq.
  "xsec" (`g**2`, not the ensemble-collapsed `g**5` of eq. "Fmatrix"),
  pure numerical coefficient `3` (no `Wph`/`pi**4`), no extra `1/s**2` --
  see `reference.py`'s module docstring for the full derivation. Intended
  as the assumption-free correctness test for correlated bunches and a
  permanent debug tool. A small, deliberately-deferred `~2*pi` residual
  remains against `angle_integrated_spectrum` in grid-integrated
  comparisons.

### Shared config -- `config.py`

Physical constants (`hbar`, `me`, `c`, `el`, `rel`, `sigma_T`, `alpha`,
`PHI`), the GPU kernel sizing constants `spectrum_kernel_4d` needs
(`X_THREADS`, `MAX_RINGS`, `MAX_ARCS`, `PHI_EDGES`, `CDF_PHI_RESOLUTION`,
...; see "Sizing constants" in Conventions), `_detect_device()`, and the
`Compton` collision-configuration class.

`Compton` holds a laser-electron collision's physical parameters
(`set_electron_parameters`/`set_laser_parameters`/`set_foci_displacement`)
and the quantities derived from them (`k0_las`, `Wph`, `a0`, `N_e`, `N_l`,
`sigma_thx`/`sigma_thy`, ...); it runs no computation itself --
`particles.sample_bunch`/`push_and_sample` take an instance of it purely
as their parameter source. `Compton(device=None)` auto-detects a backend
via `_detect_device()`: a real CUDA GPU via cupy if
`cp.cuda.runtime.getDeviceCount() > 0`, else CPU (requires `numba`), else
raises -- there is no third backend. `.xp` (`cp` or `np`) and
`.asnumpy(x)` (`.get()` on GPU, no-op on CPU) are a thin convenience for
host orchestration code (`gui_adapter.py`) that builds arrays on the
chosen device and needs to bring results back to host afterwards.
`estimate_yield`/`estimate_spectrum_width` are cheap analytic sanity-check
estimates, not used by the real computation.

### GUI-facing engine -- `tabulated_engine.py`

`TabulatedEngine` wraps a `config.Compton` instance purely for its
config-bag properties and drives Stages 0/1/2 for one collision config:
`.run(n_particles, gamma_0, sigma_gamma0, n_steps=, n_bins=, scheme=,
backend=, a0_max=0.5, n_time_bins=, n_spatial_bins=, ...)` samples a
bunch, pushes and samples it (`a0_shape` output), builds an
`a0_kind='shape'` table, and retargets it to this run's `compton.a0` in
one call, producing a physical, spectrum-ready table.
`.total_yield`/`.spectrum(s)`/`.angular_spectrum(s, theta_x, theta_y,
phi_pol, device=)`/`.temporal_envelope`/`.spatial_distribution` wrap
`table.total_weight`/`reference.angle_integrated_spectrum`/
`spectrum4d.calculate_angular_spectrum_4d`/`PushDiagnostics` respectively
(the angular-spectrum call auto-selects the GPU or CPU kernel unless
`device` is given explicitly; the temporal/spatial properties are `None`
unless `.run()` was called with `n_time_bins`/`n_spatial_bins`).

Unit conversion: both `reference.angle_integrated_spectrum` and
`spectrum4d.calculate_angular_spectrum_4d` return **dN/ds** (dimensionless
`s`), not dN/dE. `E = 4*Wph*s` converts (`dN/dE_MeV = dN/ds /
(4*compton.Wph)`) -- getting this backwards silently produces a spectrum
off by a factor of `s` (~`gamma0**2`, i.e. wrong by many orders of
magnitude), easy to miss if you're not looking at absolute scale.

## Conventions

- **Units are CGS**, constants at the top of `config.py`. Lengths and
  times inside `spectrum_kernel_4d` are normalised to the laser wavenumber
  `k0_las`: positions are `k0_las * x`, times are `k0_las * c * t`.
  Particle coordinates produced by `particles.py` already use this
  convention.
- **`theta_x`, `theta_y` are momentum angles** `p_{x,y}/gamma`, never
  positions. The tabulation half-widths `dx`, `dy` (derived from
  `table.grid` in `calculate_angular_spectrum_4d`) are angular
  half-widths, not to be confused with grid spacing (`grid.widths`).
- **Single precision by default** in `spectrum_kernel_4d`'s arithmetic.
  `SINGLE_PRECISION = True` (config.py) sets `CP_FLOAT` etc. Inside the
  kernel, wrap literals as `CP_FLOAT(...)` and use
  `CP_ONE`/`CP_ZERO`/`CP_TWO_PI` rather than bare Python floats -- mixing
  promotes to float64 silently and costs performance. `deposition.py`'s
  CPU/GPU deposition defaults to **float64** accumulation on both devices
  instead (`cupyx.scatter_add` supports it) -- a deliberate difference
  from the kernel's convention, not an oversight.
- **Shared memory is aliased and reused** in `spectrum_kernel_4d`.
  `TMP_FLOAT_ARRAY = inv_cdf`, and `rings`/`phi_cur` are both views into
  it at different offsets (`RINGS_SIZE` separates them). This is
  deliberate but fragile. Any new shared allocation must respect the
  existing offsets, and any change to `MAX_RINGS`, `PHI_EDGES`, or
  `CDF_PHI_RESOLUTION` must be checked against total shared-memory use.
- **Sizing constants** at the top of `config.py` are interdependent:
  `MAX_ARCS = 4 * MAX_RINGS`, `CDF_SIZE`, `THREAD_STRIDE` all derive from
  `MAX_RINGS`/`PHI_EDGES`/`SAMPLES_TOTAL`/`X_THREADS`. Change the
  primitives, not the derived values.
- **Radial sampling is uniform in `theta**2`, not `theta`** -- required by
  the polar measure. Easy to "fix" incorrectly; leave it.
- `spectrum_kernel_4d` has an *active* debug path (`debug_arr`,
  `debug_idx`, writes `x, y, f` per sample) plus a separate
  `dbg_scalars` output (per-output-point `skip`/`rmin`/`rmax`/`n_arcs`/
  `total_weight`) -- cheap, left in, useful for debugging a specific
  output point.
- **`Table` (deposition.py) always holds host/numpy arrays**, regardless
  of which device built it (`build_table` calls `.get()` before
  returning). Don't assume `table.H` needs `.get()` again, and don't pass
  cupy arrays into `Table(...)` directly.

## Convergence testing

Tooling for resolution/deposition-scheme scans exists; no scan has been run
and recorded yet. Pattern:

    import numpy as np
    from xigma_i import particles, deposition, spectrum4d

    bunch = particles.sample_bunch(compton, n_particles, gamma0, sigma_gamma0)
    gamma, tx, ty, a0, w = particles.push_and_sample(compton, bunch, n_steps=200)

    results = {}
    for n_bins in [(32, 32, 32, 8), (64, 64, 64, 16), (128, 128, 128, 32)]:
        table = deposition.build_table(gamma, tx, ty, a0, w, n_bins=n_bins,
                                        scheme='nearest', device='gpu')
        spec, _, _ = spectrum4d.calculate_angular_spectrum_4d(
            table, s_array, theta_x_array, theta_y_array, phi_pol=0.0)
        results[n_bins] = spec

    # compare consecutive resolutions -- the difference should shrink as
    # resolution increases, not just change

What to vary, one at a time (hold the rest at a value already "clearly
enough"):

- **Grid resolution** (gamma/theta_x/theta_y/a0 bins): `n_bins=` on
  `build_table`. Gamma resolution is the physically constrained axis
  (`δω/ω ≈ 2δγ/γ` against the reporting resolution you actually need); the
  other three are a memory/accuracy tradeoff (a 128×128×128×32 float32
  table is ~270 MB).
- **Particle statistics**: `n_particles` (`sample_bunch`) sets how many
  `(gamma, theta_x, theta_y, a0, weight)` rows land in the table --
  `push_and_sample` emits exactly one per particle. `n_steps`
  (`push_and_sample`) does *not* affect that count; it's purely the
  trajectory-integration resolution feeding each particle's `weight`/`a0`
  (see "a0 is a trajectory average" above) -- too coarse and those two
  integrals are inaccurate, but it never changes how many rows Stage 1
  sees.
- **Quadrature resolution**: `samples_per_point=` on
  `calculate_angular_spectrum_4d`.
- **Deposition scheme**: rerun the same scan with `scheme='nearest'` vs
  `scheme='cic'` at fixed particle count. CIC should be smoother -- fewer
  near-zero-occupancy cells, check via `deposition.occupancy_diagnostics`
  -- without shifting peak position or integrated flux (integrate
  `spectrum_from_table` or `calculate_angular_spectrum_4d`'s output over
  theta/s).

Statistical and discretisation error don't disentangle automatically: at
fixed `n_bins`, increasing particle count converges the estimate toward
the true density; at fixed particle count, increasing `n_bins` past
roughly `sqrt(particles per cell)` starts amplifying per-cell shot noise
instead of reducing bias. Check `occupancy_diagnostics`' per-cell count
histogram, not just whether the spectrum shape looks smoother.

`deposition.check_accumulation_precision(H_f64, H_f32)` compares two
`build_table` runs with different `accumulate_dtype` directly, for the
"does float32 lose anything here" question specifically -- run both on
`device='cpu'` (float64 accumulation isn't optional on GPU here; both
devices default to float64, see Conventions).

## Traps

- **`cupyx.scatter_add`'s dtype restriction.** It's backed by `cupy.add.at`,
  which only supports `int32, float16, float32, float64, uint32, uint64` --
  notably not `int64`. `deposition.py`'s occupancy counting used to
  accumulate in `int64` and hit `TypeError: cupy.add.at only supports
  int32, float16, float32, float64, uint32, uint64` on some cupy/driver
  combinations but not others (nothing to do with problem size or GPU
  model) -- fixed by switching occupancy to `int32` (plenty for realistic
  per-cell counts). Any new GPU scatter-add target must stay off `int64`.
- **`emulate_nonlinearity`** is accepted by `gui_adapter.Config` for
  interface stability but has no effect: `a0` is a real table axis, not a
  phenomenological correction, so applying a ponderomotive shift on top
  would double-count it.
- **`Table.a0_kind` mismatch.** A `'shape'` table (built from
  `push_and_sample`'s `a0_shape`) is not spectrum-ready -- its 4th axis
  isn't a physical `ahat`, so feeding it straight to `spectrum_kernel_4d`/
  `spectrum_from_table` would silently evaluate the resonance condition at
  the wrong `a0` values. Both assert `a0_kind='ahat'` and raise instead of
  computing garbage; always route a `'shape'` table through
  `deposition.retarget_a0(table, a0)` first.
- **Depositing `a0` per timestep instead of trajectory-averaged.** A real
  mistake made and fixed on this codebase, not hypothetical -- see
  "Stage 0"'s "a0 is a trajectory average" section. If you're about to
  make `push_and_sample` emit more than one row per particle, or bin
  `a0_local(t)` directly into `H`, stop and re-read that section first.
- **The `1/(1+a0)` Jacobian factor.** `spectrum_kernel_4d`'s resonance
  condition and evaluation prefactor both depend on `a0` (eq. "Gamma",
  "Fmatrix" in the accompanying paper) -- `g` and `prefac` must be
  recomputed *inside* the per-a0-bin quadrature loop, not shared across
  bins. An earlier version of this kernel (and `reference.spectrum_from_table`)
  got this wrong, using a single a0-independent `g` and missing the
  `1/(1+a0)` factor entirely -- a real, previously-made mistake.
- **Shared-memory aliasing.** Adding a shared array without accounting for
  `TMP_FLOAT_ARRAY`/`rings`/`phi_cur` overlap corrupts the arc geometry in
  ways that produce plausible-looking output.
- **`dr` scope.** In the final evaluation loop `dr` is read from a
  variable set inside the earlier `if not skip:` block, across
  `syncthreads()`. Verify it is still in scope and correct after any
  restructuring -- cupyx.jit scoping is easy to get subtly wrong.
- **Table too sparse.** Check `deposition.occupancy_diagnostics`'
  per-cell occupancy histogram before blaming the quadrature for noise.
- **`theta_x`/`theta_y` as positions** rather than momentum angles.
- **Float32 vs float64 accumulation.** `deposition.py` defaults to
  float64 on both CPU and GPU (see Conventions) -- if you override to
  float32 for memory, verify with
  `deposition.check_accumulation_precision` rather than assuming it's
  fine.
- **CPU vs GPU deposition, individual cells.** Total weight and marginals
  match tightly between `device='cpu'` and `device='gpu'` on the same
  samples, but a small fraction of *individual* cells can differ (a
  sample whose bin coordinate sits extremely close to an edge can round
  to different neighbouring cells under different floating-point paths).
  Benign, documented in `_deposit_gpu`'s docstring; don't expect
  bit-identical per-cell tables between devices.

## Environment

CuPy with `cupyx.jit` rawkernels for the GPU path; `numba` for the CPU
fallback (either is sufficient, see `config._detect_device`). `scipy` is
used on the host for `erfcx`. No build system beyond `pyproject.toml`
(setuptools), no repo-tracked test suite at present (validation lives in
ad hoc scripts run against `reference.py`/`deposition.py`'s functions, not
a `pytest` tree).

A working GPU environment was set up as a conda env (`conda create -n
xigma python=3.12`, then `pip install numpy scipy pytest cupy-cuda12x
tomli`, matched to the local driver's CUDA version). Recreate similarly if
starting fresh; `cupyx.scatter_add` needs a real cupy install (not just a
CUDA driver) for `deposition.py`'s GPU deposition path specifically --
`deposition.py` itself doesn't *require* cupy to import or to run its CPU
path.

## GUI integration (`gui_adapter.py`)

`src/xigma_i/gui_adapter.py` is the bridge that plugs this package into
the sibling `compton-gui` project's Tkinter desktop GUI as one of two
pluggable `ModelAdapter`s (the other being `dfe5_compton_mc`, checked out
at `../MC-Kost`). See `passport.md` for the full physics "passport" this
adapter reports through its `capabilities()`.

- Never imports `cupy`/`config`/`tabulated_engine` at module scope -- only
  inside `available()`/`run_simulation()`/`spectrum_in_angular_range()` --
  so `import xigma_i.gui_adapter` degrades gracefully when cupy/CUDA isn't
  installed (the GUI wraps that import in `try/except` and shows the model
  greyed-out instead of crashing).
- Defines its **own local** `BinnedSpectrum`/`BinnedAngularSpectrum`/
  `BinnedTemporalEnvelope`/`BinnedSpatialDistribution`/
  `AngularRangeSpectrumResult` dataclasses -- deliberately *not* imported
  from `compton_gui.model_api`, so this package doesn't have to depend on
  the GUI project. They're structurally identical, duck-type compatible,
  but **not the same Python class** -- this tripped up an
  `isinstance`-based check on the GUI side once; don't "fix" it by
  importing `compton_gui` here, the decoupling is intentional.
- `Config` mirrors `dfe5_compton_mc.Config`'s field names/SI units where a
  physical mapping exists (so the GUI's model-agnostic spread-estimate
  formula keeps working regardless of active model), but converts to CGS
  at the `set_*_parameters` call boundary. `crossing_angle` must be `0.0`
  (head-on only -- this pipeline has no crossing-angle support at all).
  `quantum` is accepted for interface symmetry but has no effect. `beta_ff`/
  `phi_pol`, plus a block of pipeline numerical/resolution knobs
  (`n_particles_01`, `n_steps_0`, `n_bins_gamma`/`n_bins_theta_x`/
  `n_bins_theta_y`/`n_bins_a0`, `a0_max`, `samples_per_point_2`,
  `n_time_bins`, `n_spatial_bins_x`/`n_spatial_bins_y`), are xigma-i-only
  extras with no dfe5 analogue -- surfaced in the GUI through
  `XigmaAdapter.extra_params()` (a small `(label, default, key)` spec list,
  mirrored in `compton_gui.model_api.ModelAdapter.extra_params`) rather
  than the shared Electrons/Laser/Compton field panels, since those are
  common to every model. The numerical knobs have no physical meaning --
  they're `run_simulation`'s former hardcoded module constants
  (`_N_PARTICLES_NEW_*`/`_N_STEPS_NEW`/`_N_BINS_NEW`/
  `_SAMPLES_PER_POINT_NEW`/`_N_TIME_BINS_NEW`/`_N_SPATIAL_BINS_NEW`),
  moved onto `Config` (the only object `run_simulation` receives) so the
  GUI can tune Stage 0/1/2 cost/accuracy per run instead of it being fixed
  at import time; the shared "Number of macroelectrons" field (`n_mc`) is
  parsed but ignored by xigma-i now that `n_particles_01` is the real
  Stage 0/1 particle-count knob (`params_to_config` raises a warning
  saying so). `run_simulation` still clamps `n_particles_01` against a
  `_N_PARTICLES_SANITY_MAX` ceiling (2,000,000) as a guard against a
  fat-fingered GUI value hanging the GPU/CPU; every numerical field is
  otherwise floored at 1 in `params_to_config`, not re-validated later.
  `emulate_nonlinearity` is still accepted/parsed (interface stability)
  but is inert (see Traps).
- `XigmaAdapter` caches `self._last_results` (which itself carries private
  `_compton`/`_gamma_0`/`_sigma_gamma_0`/`_engine`/`_device`, set by
  `run_simulation`) so `spectrum_in_angular_range()` can reuse the cached
  `TabulatedEngine`'s table for a fresh on-demand
  `calculate_angular_spectrum_4d` call over a user-picked window, without
  re-running the whole simulation or rebuilding the table. `_compton` is
  `TabulatedEngine`'s config source (see "Shared config" above).
- `run_simulation` builds one `Compton` config, wraps it in a
  `TabulatedEngine`, and calls `.run(..., n_time_bins=, n_spatial_bins=)`
  once to get every observable the GUI needs (total yield, angle-integrated
  spectrum, angular spectrum, temporal envelope, spatial distribution) in
  a single Stage 0/1/2 pass. `n_particles`/`n_steps`/`n_bins`/`a0_max`/
  `samples_per_point`/`n_time_bins`/`n_spatial_bins` for this call come
  from `cfg`'s numerical-control fields (see above); their *defaults*
  (`Config`'s field defaults, echoed in `extra_params()`) are a first-cut,
  **not profiled against a real interactive GUI session** -- smoke-tested
  only (see below). One full `run()` takes ~1s on a GTX 1660 Ti GPU / ~8s
  on CPU/numba at those defaults (`n_particles_01=60_000`, `n_steps_0=64`,
  `n_bins=(48,48,48,12)`, `a0_max=0.5`, `samples_per_point_2=32`,
  `n_time_bins=128`, `n_spatial_bins=(64,64)`) -- dialing any of them up
  from the GUI increases cost roughly as documented in "Convergence
  testing" above (particles/steps ~linear in Stage 0/1 cost, quadrature
  samples independent of particle count in Stage 2).

### GUI-side testing

No unit tests in this repo. Validated via the sibling `compton-gui` repo's
`scripts/headless_test.py` (calls `params_to_config -> run ->
validate_results` plus the temporal/spatial/angular fields through
`XigmaAdapter`), and via standalone smoke scripts (no `compton-gui`
checkout needed) exercising `XigmaAdapter.available`/`params_to_config`/
`run`/`spectrum_in_angular_range` end-to-end on three configurations: the
real GPU backend, a forced-CPU backend (numba, cupy still importable), and
a forced-CPU backend with cupy import actually blocked -- all three
produce finite, sane `total_yield`/`spectrum`/`angular_spectrum`/
`temporal_envelope`/`spatial_distribution`, and GPU/CPU `total_yield`
agree to <0.01%.

On this dev machine: system Python has no pip/cupy/matplotlib. Use the
`miniforge3` conda env named `core` (has cupy 14.0.1, numpy, matplotlib,
tkinter, all working against the local GTX 1660 Ti):

```bash
conda run -n core --no-capture-output python3 /path/to/compton-gui/scripts/headless_test.py
```

`conda run` silently swallows stdout unless you pass `--no-capture-output`.
`cupyx.jit.rawkernel` needs to introspect real source -- write test scripts
to an actual `.py` file and run that; it can't compile a kernel defined via
`python3 -c "..."`/stdin (`RuntimeError: JIT needs access to the Python
source code ... cannot be retrieved within the Python interactive
interpreter`).

### Relationship to sibling repos

- `../MC-Kost` (or wherever it's checked out) -- `dfe5_compton_mc`, the
  other physics engine plugged into the same GUI. No dependency either
  direction.
- `../compton-gui` -- the shared Tkinter GUI. Depends on this repo only
  through `gui_adapter.py`'s contract (never touches `deposition.py`/etc.
  directly). `bootstrap.py` there assumes this repo's `src/` sits at a
  sibling path unless overridden via `COMPTON_GUI_XIGMA_SRC`.

### pyproject.toml note

Currently pins `requires-python = ">=3.14"` and `cupy>=13.6`. Prebuilt
cupy wheels can lag brand-new CPython releases; if `pip install
cupy-cudaXXx` doesn't find a matching wheel, a conda/mamba env
(conda-forge) is the more reliable install path -- confirmed working on
this machine via the `core` env above.
