# Implementation plan: 4D geometric-factor table

## Goal

Replace the analytic, separable energy distribution in the spectrum kernel with a
tabulated 4D array built by depositing macroparticles. This removes the
factorisation assumption entirely: arbitrary correlations between energy,
divergence, position and field amplitude are carried by the table.

The existing annulus reduction and Fibonacci/QMC quadrature are kept unchanged in
structure. Only the integrand lookup changes.

## What is being computed

Stage 1 builds

    H[gamma, theta_y, theta_z, a0]
      = integral over (t, r) of  v_rel * n_ph(t,r) * f_e(t,r,theta,gamma) * dt d3r

i.e. the electron distribution in (gamma, theta_y, theta_z, a0), weighted by local
laser photon density. Space and time are consumed during deposition and do not
appear as axes.

Stage 2 evaluates, per output point (n_y, n_z, omega):

    dN/(dw dOmega) = C / omega^2 *
      integral F(Gamma, theta) * H_kin * |dGamma/dw| *
               H[Gamma(omega,theta,a0), theta_y, theta_z, a0]
      d theta_y d theta_z d a0

where Gamma(omega, theta, a0) is the resonant energy from inverting the resonance
condition, and H_kin is the kinematic Heaviside cutoff.

## Stage 0: particle source

CORRECTED (was wrong in the original version of this plan -- see below): load
macroparticles and produce, for each particle, ONE tuple (not one per
trajectory time sample):

    (gamma, theta_y, theta_z, a0, weight)

where:
  - gamma        Lorentz factor
  - theta_y      p_y / gamma      MOMENTUM ANGLE, not position
  - theta_z      p_z / gamma      MOMENTUM ANGLE, not position
  - a0           trajectory-averaged effective intensity hat-a(zeta),
                 Paper/xigma.tex eq. "ahattraj":
                     ahat(zeta) = (TrXi/2) * int[a^2(t)]^2 dt / int a^2(t) dt
                 with a^2(t;zeta) the instantaneous squared potential at the
                 particle's position, and TrXi/2 = (1+ellipticity^2)/2
                 (eq. "Xi"; ellipticity=0 linear, +-1 circular).
  - weight       L(zeta) = integral over the trajectory of
                 v_rel * n_ph(t, r) * dt * macroparticle_weight  (eq. "lumfun")

This was originally specified (wrongly) as "a0 = local normalised field
amplitude at (t, r)", sampled once per trajectory timestep like the other
tuple elements. That is only valid in the synchrotron/wiggler regime, where
the photon formation length is ~one laser cycle and the trajectory can be
split into independently-radiating segments. This code is in the opposite,
weakly-nonlinear regime (a0 <~ 1), where the formation length spans the
*whole* trajectory: the electron radiates a single line shaped by one
effective intensity for its entire passage, not a sequence of per-instant
emissions. Hence a0 must be a single trajectory-averaged scalar per particle.
See CLAUDE.md for the full explanation; do not reintroduce per-timestep a0
sampling.

Notes:
  - v_rel is the relative velocity factor from the collision formalism; for
    near-backscattering it is ~2c. Keep it explicit rather than folded into a
    constant, so head-on vs crossing-angle geometries stay correct.
  - n_ph(t, r) is the laser photon number density from the pulse model.
  - dt is the trajectory sampling interval; n_steps (the number of samples)
    sets the resolution of the L/ahat integrals only, not the number of
    output tuples, which is always one per particle.

Assume that the input file provides only initial conditions, so add a ballistic pusher here.

## Stage 1: deposition

### Grid

Four axes, uniform, with configurable extents and counts. Suggested defaults:

    gamma    128 bins   spanning the populated energy range with margin
    theta_y  128 bins
    theta_z  128 bins
    a0        32 bins   from 0 to max encountered

Memory at these sizes is ~270 MB in float32. Make dtype configurable; float32 is
adequate for the table itself but see the accumulation note below.

Grid extents should be derived from the particle data on a first pass (min/max
with margin), not hard-coded, and then stored alongside the table.

### Deposition scheme

Implement both, selected by a flag, defaulting to nearest-cell for the first
working version:

  - `nearest`  — nearest-cell binning. Simple, use to get the pipeline correct.
  - `cic`      — cloud-in-cell. Distribute each deposit over the 16 neighbouring
                 cells of the 4D grid with weights that are the product of the
                 1D linear weights along each axis. Handle edge cells by
                 clamping or by discarding out-of-range deposits; be explicit
                 about which and keep a counter of discards.

CIC matters here. With realistic particle counts the table has only a few
deposits per cell, and nearest-cell binning produces graininess that the
quadrature interpolates straight into the spectrum. Build the pipeline with
`nearest`, validate, then switch to `cic` and confirm the spectrum is smoother
without shifting.

Structure the deposition so the two schemes share everything except the
weight-distribution step.

### Parallelisation

Trivially parallel over particles. Both a CPU and a GPU path are wanted.

  - CPU: parallel over particle batches with per-thread partial tables reduced at
    the end, or atomic adds if memory pressure forbids replication.
  - GPU: atomic adds into the table. Deposits are scattered, so contention is
    low in practice; do not attempt sorting/binning optimisations until profiling
    shows they are needed.

Batch particles so that memory use is bounded and configurable. The table stays
resident; particles stream through.

### Accumulation precision

Many small deposits into the same cell. Accumulate in float32 initially, but add
a check: if the ratio of max to min per-cell contribution is large, or if
float64 accumulation changes the result by more than a small tolerance, switch
the accumulator to float64 and downcast when writing. Do not silently assume
float32 is fine.

### Diagnostics to emit

  - Total deposited weight, compared against the analytic expectation for the
    integrated luminosity if available.
  - Fraction of cells with zero deposits, and the histogram of per-cell deposit
    counts. This is the direct measure of whether the table is adequately
    populated.
  - Count of deposits falling outside the grid.
  - Marginal distributions along each axis, for eyeballing against the input
    beam.

### Output

Write the table plus its grid metadata (axis extents, bin counts, deposition
scheme, particle count, total weight) to a file. Stage 2 must be runnable from
this file alone.

## Stage 2: modifications to the existing kernel

Three changes. Everything else — annulus bracketing, ring/rectangle arc
geometry, coarse proposal, inverse-CDF sampling, Fibonacci sequences, sampling
uniform in theta^2 — is unchanged.

### 1. Lookup becomes 4D

The existing 3D interpolation of G over (theta_y, theta_z, a0) becomes
quadrilinear interpolation of H over (gamma, theta_y, theta_z, a0), evaluated at
gamma = Gamma(omega, theta, a0).

Gamma is already computed per sample in the current kernel. Feed it into the
lookup as the fourth coordinate.

The coarse proposal evaluation currently uses nearest-neighbour lookup — keep
that, now 4D nearest.

### 2. Remove f_a(a0)

`f_a` must no longer appear anywhere in the kernel. Depositing each particle's
trajectory-averaged a0 (ahat, see Stage 0 above) IS the intensity distribution
across the focal volume, sampled with its correlations to energy and position
intact. Applying `f_a` as well would double-count.

Grep for it and remove every occurrence. This is the single most likely source
of a silent factor error.

### 3. Annulus brackets from the table

The radial brackets currently come from inverting the resonance condition at
gamma_0 +/- 3 sigma of an assumed Gaussian. Replace with brackets derived from
the actual populated extent of the gamma axis of H.

Compute once, at load time, the lowest and highest gamma at which the table has
non-negligible content — a small quantile of the total weight, e.g. 1e-4 and
1 - 1e-4, rather than the raw min/max, so that isolated stray particles do not
inflate the domain. Store alongside the table and use for bracketing.

The brackets should be conservative rather than tight. Excess area costs nothing
because the importance sampling removes it; a bracket that is too narrow
silently truncates signal.

## Validation

In order. Do not proceed past a failing step.

1. **Uncorrelated reference.** Construct a synthetic particle set with no
   correlations — Gaussian in energy, independent Gaussian divergence, uniform
   a0 — and confirm the tabulated pipeline reproduces the existing analytic
   result within quadrature error. This tests deposition, lookup and
   normalisation together, and is the check that catches an `f_a` double-count.

2. **Direct binning cross-check.** Implement the trivial reference path:
   iterate particles, compute each one's resonance frequency for a given
   observation direction, bin with its weight, repeat over directions. No
   tables, no quadrature, roughly twenty lines. Agreement with the table
   pipeline on a correlated particle set is the primary correctness test, and it
   is assumption-free on both sides. Keep this path permanently as a debug tool.

3. **Correlated case.** Run a chirped bunch and a divergence-correlated bunch.
   Confirm agreement with the direct binning path. Confirm the spectrum differs
   from what the uncorrelated model would predict — if it does not, the
   correlation is not reaching the table.

4. **Deposition scheme.** Compare `nearest` against `cic` at fixed particle
   count. CIC should be smoother without shifting peak positions or changing
   integrated flux.

5. **Resolution scans.** Vary independently: number of gamma bins, number of a0
   bins, particle count, QMC samples per output point. Each should show
   convergence. Record the settings at which the result is stable, for the
   convergence study in the paper.

## Ordering

    Stage 0 loader + weight computation
    Stage 1 nearest-cell deposition, single-threaded CPU
    Validation 1 and 2
    Stage 2 kernel modifications
    Validation 3
    CIC deposition
    GPU deposition path
    Validation 4 and 5

Get correctness on the slow path first. The deposition is embarrassingly
parallel and will not be the thing that is subtly wrong; the normalisation and
the lookup coordinates will be.

## Things likely to go wrong

  - `f_a` left in the kernel — silent factor error, will look like a
    normalisation problem.
  - theta_y, theta_z taken as positions rather than momentum angles.
  - Gamma passed to the lookup in the wrong units or before the kinematic cutoff
    is applied.
  - Grid extents derived from one particle set and then reused with another.
  - Table too sparse: check the per-cell occupancy histogram before blaming the
    quadrature for noise.
  - Edge handling in CIC: deposits at the grid boundary losing part of their
    weight, showing up as a deficit in the total.

---

# Phase 2: retire core.py from the GUI path; make the new path the only path

**Status of the above (Phase 1) as of this writing**: essentially done and
validated per CLAUDE.md's "Current state" section. Stage 0
(`particles.sample_bunch`/`push_and_sample`), Stage 1
(`deposition.deposit_nearest`/`deposit_cic`/`build_table`, CPU+GPU), and
Stage 2 (`spectrum4d.spectrum_kernel_4d`/`calculate_angular_spectrum_4d`) are
implemented; `reference.py`'s three independent validation paths
(`angle_integrated_spectrum`, `spectrum_from_table`, `direct_binning_spectrum`)
now agree with each other and with the table pipeline to ~15% (see CLAUDE.md
"Current state" for the exact figures and the one open caveat: narrow-angle/
sparse-table configs show large run-to-run variance in `spectrum_kernel_4d`
alone, believed to be importance-sampling noise, not a normalisation bug).
Validation steps 4 (nearest vs. cic) and 5 (resolution scans) from Phase 1's
list above are **not done yet** — do them (or at least steps 4/5 for the
specific configs the GUI will actually drive) before or during this phase,
since they're the last correctness gap on the new path itself, independent
of the GUI-migration work below.

Decision (per conversation, 2026-07-24): stop maintaining `core.py` as the
GUI's compute engine. The new tabulated path is far enough along that the
GUI should be rebuilt on top of it directly, instead of validating the new
path against `core.py` and then still shipping `core.py` to users. This
phase is about making that swap for real, not just adding a CPU fallback to
the old kernels (which is what the prior PR — `feat/gui-cpu-fallback` — did;
see "Relationship to the CPU-fallback PR" below for what happens to it).

## Gap analysis: what gui_adapter.py currently gets from core.Compton

`gui_adapter.run_simulation`/`spectrum_in_angular_range` currently call, and
the new path's equivalent (or lack of one):

| Needed by GUI | `core.Compton` today | New-path equivalent |
|---|---|---|
| Total yield | `calculate_total()` | `Table.total_weight` — **already there**, no new work (deposition.py's docstring/CLAUDE.md: Stage 1 validated to 1-3% against `calculate_total`) |
| Angle-integrated spectrum (dN/dE) | `calculate_spectrum()` | `reference.angle_integrated_spectrum(gamma, particle_weight, s)` — **already there**, but takes raw per-particle `(gamma, weight)` from Stage 0, not the table; decide whether the GUI keeps Stage 0's arrays around for this or whether it should be re-derived from `H`'s gamma marginal instead (cheaper, avoids keeping the full particle set resident) |
| Angular spectrum (d2N/dE dOmega grid) | `calculate_angular_spectrum()` | `spectrum4d.calculate_angular_spectrum_4d(table, s, theta_x, theta_y, phi_pol, samples_per_point)` — **already there**, GPU-only (see Stage A below) |
| On-demand angular-range spectrum | `spectrum_in_angular_range` (reslices cached `Compton`) | Same call as above with a different `theta_x`/`theta_y` window, reusing the cached `Table` — **trivial, no new work**, mirrors the existing function almost line for line |
| Temporal envelope (rate vs. time) | `Compton.time_envelope`/`env_ts`, populated inside `particle_kernel` | **does not exist in the new path at all** — `particles.py`/`deposition.py` never track a time axis (Stage 0 collapses each particle's trajectory into one `(gamma, theta_x, theta_y, a0, weight)` tuple, by design — see CLAUDE.md's "a0 is a trajectory average" section). This is new development, not wiring. See Stage C below |
| Spatial distribution ((x,y) areal density) | `Compton.spatial_envelope`/`spatial_x_edges`/`spatial_y_edges`, populated inside `particle_kernel` | **does not exist in the new path at all** — same root cause, Stage 0 doesn't carry (x, y) either. New development. See Stage C below |
| `Compton`-as-config-bag (`k0_las`, `Wph`, `N_l`, `a0`, `beta_ff`, `ellipticity`, `sigma_ex`/`sigma_ey`/`emit_x`/`emit_y`, ...) | `set_electron_parameters`/`set_laser_parameters`/`set_foci_displacement` — pure Python/numpy, no cupy/GPU touched | `particles.sample_bunch(compton, ...)`/`push_and_sample(compton, ...)` **already take a `compton` object as their parameter source** — so this part of `Compton` is reused as-is by the new path today; see "Open decision: what happens to the Compton class" below for whether to keep constructing `core.Compton` just for this or extract a dependency-free config class |

Bottom line: the *spectrum-shaped* observables (total yield, angle-integrated
spectrum, angular spectrum, angular-range spectrum) are essentially wiring
work — the new path already computes all of them. The *temporal* and
*spatial* observables are not wiring work — they don't exist yet and need
new deposition logic. Scope/estimate that honestly before committing to a
full swap in one PR; consider shipping the spectrum-shaped swap first with
temporal/spatial still reading off a `core.Compton` call running alongside
(both engines running only until the new path grows those two outputs), if
a two-step migration is preferred over a single big-bang one.

## Stage A: CPU/numba fallback for spectrum_kernel_4d

`spectrum4d.py`'s `spectrum_kernel_4d` is GPU-only today (`cupyx.jit.
rawkernel()`, module-level `import cupy as cp`/`from cupyx import jit`,
unconditional). Give it the same treatment `feat/gui-cpu-fallback` gave
`core.py`'s two kernels:

  - Make `spectrum4d.py`'s cupy import lazy/optional (same `try/except`
    pattern as `core.py`'s current top-of-file guard).
  - Write `spectrum_kernel_4d_cpu` in a new `spectrum4d_cpu.py` (mirroring
    `core_cpu.py`): same literal-transliteration approach — same annulus/
    ring/arc geometry (shared with the legacy kernel, per spectrum4d.py's
    own docstring: "reuses spectrum_kernel's ring/rectangle annulus
    geometry... unchanged in structure"), same inverse-CDF phi sampling,
    same drop-the-thread-bookkeeping simplification `core_cpu.py` already
    uses (parallelize over `out_idx` via `numba.prange`, loop directly over
    (arc, sample, subsample) triples instead of GPU round-robin scheduling).
    Two things are genuinely new relative to `core_cpu.py`'s port and need
    care, not just copying:
      1. Quadrilinear (not bilinear) interpolation of `H` over
         `(gamma, theta_x, theta_y)`, using `grid.gamma_edges`/`widths`/etc.
         for the coordinate mapping (see `calculate_angular_spectrum_4d`'s
         host-side unpacking of `grid` at spectrum4d.py:474-483 for exactly
         what scalars the kernel needs).
      2. The nested a0-quadrature loop inside the final evaluation
         (spectrum4d.py's module docstring, point 4: `g`/`prefac` computed
         *inside* the a0 loop, with the `1/(1+a0)` Jacobian factor — this is
         exactly the bug CLAUDE.md's "Known bugs" flags as previously
         gotten wrong once; don't re-derive it from physical intuition,
         transliterate the GPU kernel's actual loop structure line by line
         like `core_cpu.py` did).
      3. The two GPU-only zero-weight guards already present in
         `spectrum_kernel_4d` (inv_cdf falls back to a cell's left edge on
         a flat CDF; `sample_area` contributes zero instead of `x/0`) need
         to carry over into the CPU port too — `core_cpu.py`'s port of the
         *legacy* kernel didn't need these (that kernel's `collision` array
         is smooth and analytic), but `H` is a sparse finite-particle
         deposition and can have exact-zero cells, so skipping these guards
         will produce NaNs on realistic tables.
  - **Validate the same way `feat/gui-cpu-fallback` validated `core_cpu.py`**:
    on a machine with a real CUDA GPU (this one has one — see CLAUDE.md
    "Environment", the `xigma` conda env), feed the *same* `H`/`H_marginal`/
    grid-scalar arguments into both the GPU kernel and the CPU port directly
    and compare correlation + total-flux agreement, not just "runs without
    crashing." Don't trust a transliteration on code this dense without
    numeric cross-checking against the real kernel — that's what caught
    nothing wrong in the `core_cpu.py` port, but it's what would have caught
    it if there had been a bug.
  - `Table.build_table`'s `device=None|'cpu'|'gpu'` auto-detect already
    exists for Stage 1; extend the same `_detect_device()`-style auto-detect
    (reuse `core._detect_device` directly, or copy its ~10 lines into a
    dependency-free helper if `core.py` itself is meant to go away later —
    see the open decision below) to decide which `spectrum_kernel_4d*` to
    call from the new host-side driver.

## Stage B: new GUI-facing engine, replacing gui_adapter.py's core.Compton calls

Build the spectrum-shaped observables (total yield, angle-integrated
spectrum, angular spectrum, angular-range spectrum — the "already there"
rows in the gap table above) on top of `particles.py`/`deposition.py`/
`spectrum4d.py`/`reference.py`, with the same external call shape
`gui_adapter.py` already expects, so the diff in `gui_adapter.py` itself
stays small:

  - A new class (name TBD, e.g. `TabulatedCompton` or similar — avoid
    calling it `Compton` again, to not collide with/shadow `core.Compton`
    while both exist during migration) that:
      - Reuses the existing `set_electron_parameters`/`set_laser_parameters`/
        `set_foci_displacement` field-setting logic (see "Open decision:
        what happens to the Compton class" below for exactly how) so
        `particles.sample_bunch`/`push_and_sample`'s `compton` argument
        keeps working unmodified.
      - `run(n_particles, n_steps, n_bins, scheme, device, samples_per_point,
        ...)` (or split into stages, matching `gui_adapter.run_simulation`'s
        existing structure) that: samples a bunch, pushes and samples it,
        builds the table (`device=None` auto-detect, same convention as
        `deposition.build_table`), and stashes the table + Stage-0 arrays
        for later on-demand angular-range recompute (mirroring
        `XigmaResults._compton`/`_gamma_0`/`_sigma_gamma_0` caching today).
      - Exposes `total_yield` (= `table.total_weight`), `spectrum()` (wraps
        `reference.angle_integrated_spectrum`), `angular_spectrum()` (wraps
        `spectrum4d.calculate_angular_spectrum_4d`, auto-selecting the GPU
        or CPU kernel per Stage A).
  - Rewrite `gui_adapter.run_simulation`/`spectrum_in_angular_range` to call
    this instead of `core.Compton`. `available()`/`capabilities()` change
    accordingly — `_detect_device()` no longer needs to come from `core.py`
    at all if `core.py` is being retired (see open decision).
  - Decide `n_particles`/`n_bins`/`samples_per_point` defaults for GUI use
    the same way `gui_adapter.py` currently clamps `particles_amount` for
    `core.Compton` (comment at the current `run_simulation`'s
    `particles_amount = int(np.clip(...))` line) — the new path's cost model
    is different (particle count is now real per-particle Stage 0/1 work,
    not `core.py`'s grid-cell-weighted importance sampling), so the old
    clamp range is not necessarily still the right one; profile rather than
    guess.

## Stage C: temporal envelope and spatial distribution (new development)

Neither exists in the new path. Two independent small additions, either of
which could be skipped/deferred if the GUI can live without one of them
during an interim period:

  - **Temporal envelope**: needs a per-timestep binned accumulation
    somewhere in the Stage 0/1 pipeline, analogous to `core.py`'s
    `particle_kernel` depositing into `envelope`/`env_ts` every timestep
    (see CLAUDE.md's legacy-path Stage 1 description). This does *not*
    mean reintroducing per-timestep `a0` sampling into `H` (CLAUDE.md's
    "Traps" section is explicit that this is a real, previously-made
    mistake, not hypothetical) — the time axis is orthogonal to `a0`; this
    would be a *new*, separate per-timestep deposition (e.g. into a
    1D `t` histogram) alongside, not instead of, the existing one-tuple-
    per-particle `(gamma, theta_x, theta_y, a0, weight)` output.
  - **Spatial distribution**: same shape of problem, a per-(x, y) 2D
    histogram deposited during the same trajectory integration
    `push_and_sample` already does, analogous to `core.py`'s `spatial`
    bilinear deposition.
  - Both should reuse `push_and_sample`'s existing per-timestep loop (the
    `backend='numpy'`/`'cupy'` vectorized form and the `backend='numba'`
    per-particle form both already iterate every timestep internally —
    see `particles.py`'s `_push_and_sample_vectorized`/
    `_push_and_sample_numba`) rather than adding a second trajectory
    integration pass; the trajectory data needed is already being computed
    there and currently discarded after producing `L`/`ahat`.

## Open decisions (need a call before/while implementing, not something to guess)

  - **What happens to `core.py`/`core_cpu.py`** once the new path is fully
    wired into the GUI and validated: delete outright, or keep
    unreferenced-by-the-GUI as a standalone validation/reference tool (this
    is literally the unanswered `passport.md` question: "Is the code
    primary, auxiliary, or reference code?"). If kept, say so explicitly in
    CLAUDE.md and `passport.md` this time, so it doesn't stay unrecorded.
  - **What happens to `feat/gui-cpu-fallback`** (the CPU/numba fallback PR
    for `core.Compton`, opened during this same conversation before the
    decision to migrate off `core.py` was made): it still isolates and
    validates a real, working GPU-detection + numba-parallel-CPU pattern
    (the transliteration approach, the `numba.get_thread_id()`-indexed
    private-accumulator trick for the particle kernel, the numeric
    cross-validation-against-real-GPU-kernel approach) that Stage A above
    is meant to reuse directly for `spectrum_kernel_4d`. Whether to merge
    it as an interim safety net before this migration lands, or close it
    unmerged in favour of going straight to Stage A/B/C, is a call to make
    up front — either way, read its diff/description (PR #7 on
    `blendersamsonov/Compton-XIGMA`) before starting Stage A, since most of
    the CPU-port mechanics transfer directly.
  - **Whether `Compton`'s parameter-setting logic should be extracted** out
    of `core.py` into a new, `core.py`-independent config dataclass that
    both the old and new paths (and Stage B's new engine class) can share
    without importing anything cupy-adjacent, or whether Stage B's new
    engine class should just keep constructing/reusing `core.Compton`
    purely for its setters (which, per the gap-analysis table above, never
    touch cupy/GPU regardless of `core.Compton`'s other methods). The
    former is cleaner long-term if `core.py` is going away; the latter is
    less churn if `core.py` is being kept as a reference tool anyway.
  - **Interim/two-step vs. big-bang migration**: given Stage C is new
    development (not wiring) and could slip, decide up front whether
    landing Stage A+B alone (spectrum-shaped observables only, temporal/
    spatial still served by `core.Compton` in parallel) is an acceptable
    intermediate state, or whether the GUI must not lose those two
    observables even temporarily.

## Suggested ordering

    Finish Phase 1 validation steps 4/5 (nearest vs cic, resolution scans)
      for GUI-realistic configs, if not already covered by ad hoc runs
    Resolve the open decisions above (at least the core.py-retirement and
      feat/gui-cpu-fallback ones -- both block how Stage A is scoped)
    Stage A: spectrum_kernel_4d CPU port + GPU cross-validation
    Stage B: new engine class + gui_adapter.py rewire (spectrum-shaped
      observables only)
    Smoke-test against compton-gui's headless_test.py / a manual GUI run,
      side by side against core.Compton on the same config, before removing
      the old call path -- same principle as feat/gui-cpu-fallback's
      GPU-vs-CPU comparison, just old-engine-vs-new-engine this time
    Stage C: temporal envelope + spatial distribution
    Decide and execute on core.py/core_cpu.py's fate
