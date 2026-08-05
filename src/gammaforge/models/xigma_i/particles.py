"""Stage 0: macroparticle source and ballistic pusher.

Produces, for a bunch of macroparticles pushed ballistically through the
laser pulse, per-particle samples (gamma, theta_x, theta_y, a0, weight)
consumed by Stage 1 deposition (see deposition.py) to build the 4D overlap
table H[gamma, theta_x, theta_y, a0].

This model holds no complex internal representation of the electron bunch:
``push_and_sample`` takes a ``gammaforge.io.bunch.MacroBunch`` (SI units,
sampled/loaded by the caller via ``gammaforge.io.bunch``) directly and
converts to this pipeline's CGS/k0_las-normalised convention inline, at
the top of the function -- there is no separate, persistent "Bunch" class
duplicating what ``MacroBunch`` already provides.

z is the beam axis, theta_x/theta_y are the transverse momentum angles
p_{x,y}/gamma (same convention throughout this package).

Every quantity -- position, angle, energy -- is drawn directly from its
true (untruncated) distribution, rather than an efficiency-motivated
importance-sampled/truncated domain. That trades sampling efficiency for a
normalisation that requires no correction factors (no cell-weighting, no
domain-truncation trims), the right trade for a Monte Carlo path meant to
be simple to reason about and validate independently.
"""
import numpy as np
from dataclasses import dataclass

from gammaforge.io.laser import GaussianParaxialLaser
from gammaforge.io.bunch import ballistic_position_z0_reference, laser_overlap_time_window
from gammaforge.misc import available_ram_bytes, available_vram_bytes

from .config import GAUSS_WIDTH, LORENTZ_WIDTH

V_REL = 2.0  # relative-velocity factor for near-backscattering geometry
_M_TO_CM = 100.0

# push_and_sample(..., chunk=...)'s OOM-retry bound: halve the remaining
# chunk size on a cupy OutOfMemoryError, up to this many times, before
# giving up and re-raising -- robust to the static byte estimate below
# being off (other GPU processes, driver/cupy-version drift).
_MAX_OOM_HALVINGS = 6

# Calibrated on a GTX 1660 Ti (6GB, cupy 14.0.1): push_and_sample(backend=
# 'cupy') at n_steps=64 succeeded at 400,000 particles, OOM'd at 800,000
# ("Out of memory allocating 409,600,000 bytes" for n=800_000,
# n_steps=64), i.e. ~113 bytes per (particle x step) -- consistent with
# roughly 14 concurrently-live (n, n_steps) float64 temporaries in
# _integrate_trajectory_core. Rounded up here for headroom against other
# GPU processes / cupy-version drift.
_BYTES_PER_PARTICLE_STEP = 200


@dataclass
class PushDiagnostics:
    """Optional per-timestep binned outputs from push_and_sample, riding
    along the same trajectory-integration loop that produces L/a0_shape --
    see push_and_sample's n_time_bins/n_spatial_bins. Fields stay None for
    whichever wasn't requested.

    time_envelope is a photon-emission RATE (photons/s) vs t_edges
    (seconds, N+1 edges for N bins); spatial_envelope is an areal DENSITY
    (photons/cm^2) vs spatial_x_edges/spatial_y_edges (cm). Nearest-cell
    binned (matching deposition.py's 'nearest' scheme).

    Needs no post-hoc rescale to reproduce total_yield: both histograms
    bin the exact same per-timestep `contribution` array that L already
    sums over time (no angular-grid/truncated-domain normalisation baked
    in to begin with, see push_and_sample's docstring), so summing either
    histogram over all bins reproduces sum(L) exactly by construction.
    """
    t_edges: object = None
    time_envelope: object = None
    spatial_x_edges: object = None
    spatial_y_edges: object = None
    spatial_envelope: object = None


def _weighted_bincount(idx, val, n, xp):
    """Scatter-add val into a zero-initialised length-n array at integer
    idx (already in [0, n)) -- the same tool deposition.py's _scatter_add
    uses, reimplemented locally (a few lines) rather than imported, since
    deposition.py already imports particles.py and importing back would be
    circular."""
    out = xp.zeros(n, dtype=xp.float64)
    if xp is np:
        out += np.bincount(idx, weights=val, minlength=n)[:n]
    else:
        import cupyx
        cupyx.scatter_add(out, idx, val)
    return out


def _resolve_time_range(t0_local, t1_local, t_edges, n_time_bins, xp):
    """(t_lo, t_hi, n_time_bins) in k0_las*c*t units -- t_edges verbatim if
    given, else the bunch-wide window (min of every particle's own t0_local,
    max of every particle's own t1_local)."""
    if t_edges is not None:
        return float(t_edges[0]), float(t_edges[-1]), len(t_edges) - 1
    t_lo = float(xp.min(t0_local))
    t_hi = float(xp.max(t1_local))
    return t_lo, t_hi, n_time_bins


def _resolve_spatial_range(k0_las, sigma_ex, sigma_ey, sigma_lr0, spatial_edges, n_spatial_bins):
    """(sx_lo, sx_hi, sy_lo, sy_hi, nsx, nsy) in k0_las-normalised units --
    spatial_edges=(x_edges, y_edges) verbatim if given, else a "few sigma
    of whichever of the electron beam / laser waist is larger" window."""
    if spatial_edges is not None:
        x_edges, y_edges = spatial_edges
        return (float(x_edges[0]), float(x_edges[-1]),
                float(y_edges[0]), float(y_edges[-1]),
                len(x_edges) - 1, len(y_edges) - 1)
    nsx, nsy = (n_spatial_bins, n_spatial_bins) if np.isscalar(n_spatial_bins) else n_spatial_bins
    sx_half = GAUSS_WIDTH * k0_las * max(sigma_ex, sigma_lr0)
    sy_half = GAUSS_WIDTH * k0_las * max(sigma_ey, sigma_lr0)
    return -sx_half, sx_half, -sy_half, sy_half, nsx, nsy


def _bin_temporal(contribution, t, t0_local, t1_local, t_edges, n_time_bins, omega_las, xp):
    """Nearest-cell time histogram of `contribution` (shape (n, n_steps),
    same array push_and_sample sums into L) -> a photon-emission-rate
    PushDiagnostics.t_edges/time_envelope pair, in seconds."""
    t_lo, t_hi, n_time_bins = _resolve_time_range(t0_local, t1_local, t_edges, n_time_bins, xp)
    span = t_hi - t_lo
    dt_bin = span / n_time_bins if n_time_bins > 0 and span > 0 else 1.0

    t_idx = xp.floor((t - t_lo) / dt_bin).astype(xp.int64)
    in_range = (t_idx >= 0) & (t_idx < n_time_bins)
    idx_flat, val_flat = t_idx.ravel()[in_range.ravel()], contribution.ravel()[in_range.ravel()]

    hist = _weighted_bincount(idx_flat, val_flat, n_time_bins, xp)
    t_edges_out = xp.linspace(t_lo, t_hi, n_time_bins + 1) / omega_las
    dt_sec = dt_bin / omega_las
    time_envelope = hist / dt_sec if span > 0 else hist
    return t_edges_out, time_envelope


def _bin_spatial(contribution, x, y, spatial_edges, n_spatial_bins, k0_las, sigma_ex, sigma_ey, sigma_lr0, xp):
    """Nearest-cell (x, y) histogram of `contribution` -> an areal-density
    PushDiagnostics.spatial_x_edges/spatial_y_edges/spatial_envelope
    triple, in cm / photons/cm^2."""
    sx_lo, sx_hi, sy_lo, sy_hi, nsx, nsy = _resolve_spatial_range(
        k0_las, sigma_ex, sigma_ey, sigma_lr0, spatial_edges, n_spatial_bins)
    dx_bin = (sx_hi - sx_lo) / nsx if nsx > 0 else 1.0
    dy_bin = (sy_hi - sy_lo) / nsy if nsy > 0 else 1.0

    xi = xp.floor((x - sx_lo) / dx_bin).astype(xp.int64)
    yi = xp.floor((y - sy_lo) / dy_bin).astype(xp.int64)
    in_range = (xi >= 0) & (xi < nsx) & (yi >= 0) & (yi < nsy)
    flat_idx = (xi * nsy + yi).ravel()[in_range.ravel()]
    val_flat = contribution.ravel()[in_range.ravel()]

    hist = _weighted_bincount(flat_idx, val_flat, nsx * nsy, xp).reshape(nsx, nsy)
    sx_edges_out = xp.linspace(sx_lo, sx_hi, nsx + 1) / k0_las
    sy_edges_out = xp.linspace(sy_lo, sy_hi, nsy + 1) / k0_las
    dx_cm, dy_cm = dx_bin / k0_las, dy_bin / k0_las
    bin_area = dx_cm * dy_cm
    spatial_envelope = hist / bin_area if bin_area > 0 else hist
    return sx_edges_out, sy_edges_out, spatial_envelope


def _normalise_bunch(k0_las, N_e, macrobunch, xp):
    """Convert a ``gammaforge.io.bunch.MacroBunch`` (SI, external/engine-
    agnostic) into this pipeline's CGS, ``k0_las``-normalised position
    arrays. ``gamma``/``theta_x``/``theta_y`` pass through unchanged --
    both kascade and xigma_i use the same position-at-a-reference-slice-
    plus-ballistic-angle convention, so no angle/energy conversion is
    needed, only the position normalisation below (``k0_las * x[cm]``).

    ``weight`` is deliberately NOT taken from ``macrobunch.weight`` --
    recomputed as ``N_e / macrobunch.n_particles`` instead, so the caller's
    own ``GaussianElectronBeam.N_e`` stays authoritative, matching how a
    loaded ``.ele`` file carries no charge information of its own (see
    ``gammaforge.io.io_formats.sdds``'s module docstring) and how
    kascade's own ``run_simulation`` already ignores a loaded bunch's
    weight the same way.

    Returns (x0, y0, z0, gamma, theta_x, theta_y, weight) as ``xp`` arrays.
    """
    x0 = k0_las * (xp.asarray(macrobunch.x, dtype=float) * _M_TO_CM)
    y0 = k0_las * (xp.asarray(macrobunch.y, dtype=float) * _M_TO_CM)
    z0 = k0_las * (xp.asarray(macrobunch.z, dtype=float) * _M_TO_CM)
    gamma = xp.asarray(macrobunch.gamma, dtype=float)
    theta_x = xp.asarray(macrobunch.thx, dtype=float)
    theta_y = xp.asarray(macrobunch.thy, dtype=float)
    weight = N_e / macrobunch.n_particles
    return x0, y0, z0, gamma, theta_x, theta_y, weight


def estimate_chunk_size(n_particles, n_steps, backend, safety_frac=None):
    """Auto-sized ``chunk`` for ``push_and_sample``, from currently free
    memory on whichever backend is in play: VRAM (``backend='cupy'``, via
    ``gammaforge.misc.available_vram_bytes``) or system RAM
    (``backend='numpy'``, via ``gammaforge.misc.available_ram_bytes``) --
    the "query VRAM, cap allocation at ~70%" backlog item
    (docs/models/tasks.md), extended to numpy's own memory budget.

    ``backend='numba'`` is the one case that's never chunked: its own
    per-particle compiled loop already avoids the O(n*n_steps) blowup this
    function exists to bound, so there's nothing to chunk. ``'numpy'`` used
    to get the same "don't chunk" treatment on the assumption that plain
    numpy is bounded by system RAM, not the much tighter VRAM budget this
    estimates -- wrong at large enough ``n_particles*n_steps`` (a user
    report: 5,000,000 particles blew a 128GB machine's RAM in a related
    unchunked broadcast, see spectrum_from_particles._estimate_s_chunk),
    so ``'numpy'`` is now chunked against ``available_ram_bytes()`` the
    same way ``'cupy'`` is chunked against ``available_vram_bytes()``.

    Returns ``n_particles`` unchanged (i.e. "don't chunk") only if the
    relevant free-memory query itself fails (no GPU/cupy usable for
    ``'cupy'``; unsupported platform for ``'numpy'``) -- the caller can
    still pass an explicit ``chunk`` in that case.

    ``safety_frac``: fraction of free memory to budget for. Defaults to
    0.7 for ``'cupy'`` (an OutOfMemoryError there is a contained,
    catchable failure) and a more conservative 0.5 for ``'numpy'`` (an
    oversized host allocation can trigger the kernel OOM-killer or heavy
    swapping instead of a catchable exception -- a much larger blast
    radius to get wrong).
    """
    if backend == 'numba':
        return n_particles
    if backend == 'cupy':
        free_bytes = available_vram_bytes()
        frac = 0.7 if safety_frac is None else safety_frac
    elif backend == 'numpy':
        free_bytes = available_ram_bytes()
        frac = 0.5 if safety_frac is None else safety_frac
    else:
        raise ValueError(f"backend must be 'numpy', 'numba', or 'cupy', got {backend!r}")
    if free_bytes is None:
        return n_particles
    budget = free_bytes * frac
    chunk = int(budget / (_BYTES_PER_PARTICLE_STEP * n_steps))
    return max(1, min(chunk, n_particles))


def push_and_sample(macrobunch, *, k0_las, beta_ff, sigma_lr0, sigma_lz, omega_las,
                     N_l, ellipticity, N_e, sigma_ex, sigma_ey,
                     n_steps=200, backend='numpy',
                     n_time_bins=None, t_edges=None,
                     n_spatial_bins=None, spatial_edges=None,
                     chunk=None):
    """Ballistically push each macroparticle and emit one sample per particle.

    Every physics scalar (``k0_las``, ``beta_ff``, ``sigma_lr0``,
    ``sigma_lz``, ``omega_las``, ``N_l``, ``ellipticity``, ``N_e``,
    ``sigma_ex``, ``sigma_ey``) is a plain float, CGS/``k0_las``-normalised,
    converted by the caller from the shared ``GaussianElectronBeam``/
    ``GaussianParaxialLaser`` at its own call site -- no shared parameter
    bundle here; each value is exactly what this function (or its
    ``backend='numba'`` kernel) needs, extracted once in Python before
    entering the numpy/numba/cupy dispatch below (numba's ``nopython`` mode
    cannot accept a ``GaussianElectronBeam``/pint ``Quantity`` object at
    all, only these already-extracted primitives).

    Returns arrays (gamma, theta_x, theta_y, a0_shape, weight) of length
    n_particles, ready for Stage 1 deposition. gamma/theta_x/theta_y are
    constant per particle (no pusher acceleration -- straight-line
    trajectories).

    a0_shape is NOT the physical trajectory-averaged effective intensity
    ahat -- it is ahat's a0-independent *shape factor* (ahat = a0**2 *
    a0_shape), computed without reference to the actual a0 at all, so one
    push_and_sample run can be re-targeted to any actual a0 after the fact
    without rerunning Stage 0/1 (see ``deposition.retarget_a0``). A single
    scalar per particle, not a per-timestep distribution, because in this
    weakly-nonlinear regime (a0 <~ 1) the photon formation length spans the
    whole trajectory -- unlike synchrotron radiation, splitting the
    trajectory into segments and radiating each independently is not valid
    here. See CLAUDE.md "Known bugs"/"Traps"; do not go back to
    per-timestep a0 deposition.

    IMPORTANT: a0_shape is not directly usable as H's 4th axis for
    spectrum4d/reference's table consumers -- those need the physical ahat.
    Build the table with a0_shape as the 4th axis
    (deposition.build_table(..., a0_kind='shape')), then call
    deposition.retarget_a0(table, a0) for a spectrum-ready table at a
    specific a0. Table.a0_kind guards against passing a 'shape' table
    straight to spectrum4d/reference.

    backend: 'numpy' (default, single-threaded broadcast), 'numba' (CPU,
    per-particle @njit(parallel=True) loop -- lower memory at large
    n_particles*n_steps), 'cupy' (GPU, same broadcast form, stays on-device).

    n_time_bins/t_edges, n_spatial_bins/spatial_edges: opt-in
    temporal-envelope/spatial-distribution diagnostics binned during the
    same trajectory-integration loop (see PushDiagnostics). If neither is
    given, the return value is the plain 5-tuple every caller already
    unpacks; if either is given, a 6th value (PushDiagnostics) is appended.
    Only supported for backend='numpy'/'cupy' -- see
    _push_and_sample_numba's docstring for why numba doesn't implement this.

    chunk: if given (and less than n_particles), the O(n_particles *
    n_steps) trajectory-integration step runs on `chunk`-sized slices of
    the bunch instead of all particles at once, concatenating the small
    per-particle outputs and summing the (shared-bin) diagnostics
    histograms across chunks -- since each particle's trajectory is
    independent (the final reductions are all axis=1, i.e. per-particle),
    this is an exact repartition, not an approximation. See
    estimate_chunk_size for auto-sizing this from free VRAM. Ignored for
    backend='numba' (that backend's own per-particle compiled loop already
    avoids the O(n*n_steps) blowup this chunks around, see its docstring).
    """
    if backend == 'numpy':
        return _push_and_sample_vectorized(
            k0_las, beta_ff, sigma_lr0, sigma_lz, omega_las, N_l, ellipticity, N_e,
            sigma_ex, sigma_ey, macrobunch, n_steps, np,
            n_time_bins, t_edges, n_spatial_bins, spatial_edges, chunk)
    if backend == 'cupy':
        import cupy as cp
        return _push_and_sample_vectorized(
            k0_las, beta_ff, sigma_lr0, sigma_lz, omega_las, N_l, ellipticity, N_e,
            sigma_ex, sigma_ey, macrobunch, n_steps, cp,
            n_time_bins, t_edges, n_spatial_bins, spatial_edges, chunk)
    if backend == 'numba':
        if n_time_bins is not None or t_edges is not None or n_spatial_bins is not None or spatial_edges is not None:
            raise NotImplementedError(
                "push_and_sample(backend='numba', n_time_bins=..., n_spatial_bins=...): "
                "Stage C diagnostics aren't implemented for the numba backend -- no current "
                "caller needs backend='numba' (the GUI/TabulatedEngine use 'numpy'/'cupy'), "
                "so this was scoped out rather than adding a second compiled kernel variant "
                "for an unused path. Use backend='numpy' or 'cupy' if you need these.")
        return _push_and_sample_numba(
            k0_las, beta_ff, sigma_lr0, sigma_lz, N_l, ellipticity, N_e, macrobunch, n_steps)
    raise ValueError(f"backend must be 'numpy', 'numba', or 'cupy', got {backend!r}")


def _integrate_trajectory_core(x0, y0, z0, theta_x, theta_y, weight, t0_local, t1_local, n_steps,
                                w0, zT, z_rayleigh, beta_ff, N_l, ellipticity, sigma_T, k0, xp):
    """The O(n_particles * n_steps) trajectory-integration math shared by
    both the unchunked and chunked paths of _push_and_sample_vectorized --
    every array here scales with whatever per-particle slice it's given
    (x0/y0/z0/theta_x/theta_y/t0_local/t1_local), with no cross-particle
    coupling (the reductions below are all axis=1, i.e. per-particle, over
    the timestep axis). `weight` is the scalar N_e/n_particles_total from
    _normalise_bunch, NOT sliced -- it's already the correct per-particle
    weight for the true full population regardless of which/how-many
    particles this call integrates. Calling this once on the full bunch,
    or many times on chunks of it, is therefore numerically identical.

    Returns (L, a0_shape, contribution, t, x, y) -- contribution/t/x/y are
    only needed by the diagnostics binners (_bin_temporal/_bin_spatial);
    callers that don't want diagnostics can discard them.
    """
    span = xp.maximum(0.0, t1_local - t0_local)
    dt = span / n_steps

    step = (xp.arange(n_steps) + 0.5) / n_steps  # midpoint rule, shape (n_steps,)
    t = t0_local[:, None] + step[None, :] * span[:, None]  # (n, n_steps)

    x, y, z = ballistic_position_z0_reference(
        x0[:, None], y0[:, None], z0[:, None], theta_x[:, None], theta_y[:, None], t)

    # n_ph_shape: this pipeline's own CGS/k0_las-normalised call into the
    # shared spatiotemporal envelope (GaussianParaxialLaser.pulse_envelope)
    # -- axis/focus left at their defaults (head-on, no offset), matching
    # this pipeline's own convention (only kascade has a crossing-angle/
    # foci-offset knob; xigma_i/delta are head-on-only, see
    # io/interaction.py's module docstring).
    n_ph_shape = GaussianParaxialLaser.pulse_envelope(
        x, y, z, t, sigma0=w0, rayleigh_range=z_rayleigh, sigma_ct=zT,
        beta_ff=beta_ff, xp=xp,
    )

    peak_shape = 1.0 / (2 * np.pi * w0 * w0) / (np.sqrt(2 * np.pi) * zT)
    # ratio = (a0_local/a0)**2 -- deliberately built without a0 at all, see
    # "a0_shape decouples from a0" below.
    ratio = xp.clip(n_ph_shape / peak_shape, 0.0, None)

    contribution = V_REL * n_ph_shape * dt[:, None] * weight * sigma_T * k0**2 * N_l

    L = contribution.sum(axis=1)  # eq. "lumfun", per-particle deposited weight

    denom = ratio.sum(axis=1)
    F_pol = (1.0 + ellipticity**2) / 2.0  # TrXi/2, eq. "Xi"
    # xp.where instead of np.divide(..., where=) -- cupy's ufunc `where=`
    # kwarg support is version-dependent; xp.where is safe on both.
    a0_shape = xp.where(denom > 0, F_pol * (ratio**2).sum(axis=1) / xp.maximum(denom, 1e-300), 0.0)

    return L, a0_shape, contribution, t, x, y


def _push_and_sample_vectorized(k0_las, beta_ff, sigma_lr0, sigma_lz, omega_las, N_l, ellipticity, N_e,
                                 sigma_ex, sigma_ey, macrobunch, n_steps, xp,
                                 n_time_bins=None, t_edges=None,
                                 n_spatial_bins=None, spatial_edges=None,
                                 chunk=None):
    """MUST USE io/bunch.py stream function!!
    """
    from .config import sigma_T

    k0 = k0_las
    w0 = k0 * sigma_lr0
    zT = k0 * sigma_lz
    z_rayleigh = 2 * w0 * w0 * (1.0 + beta_ff)

    x0, y0, z0, gamma, theta_x, theta_y, weight = _normalise_bunch(k0_las, N_e, macrobunch, xp)

    t0_local, t1_local = laser_overlap_time_window(
        z0, k0_las=k0, sigma_lz=sigma_lz, sigma_lr0=sigma_lr0,
        beta_ff=beta_ff, gauss_width=GAUSS_WIDTH, lorentz_width=LORENTZ_WIDTH, xp=xp)

    want_time = n_time_bins is not None or t_edges is not None
    want_spatial = n_spatial_bins is not None or spatial_edges is not None
    n = int(x0.shape[0])

    if chunk is None or chunk >= n:
        L, a0_shape, contribution, t, x, y = _integrate_trajectory_core(
            x0, y0, z0, theta_x, theta_y, weight, t0_local, t1_local, n_steps,
            w0, zT, z_rayleigh, beta_ff, N_l, ellipticity, sigma_T, k0, xp)
        if not want_time and not want_spatial:
            return gamma, theta_x, theta_y, a0_shape, L
        diagnostics = PushDiagnostics()
        if want_time:
            diagnostics.t_edges, diagnostics.time_envelope = _bin_temporal(
                contribution, t, t0_local, t1_local, t_edges, n_time_bins, omega_las, xp)
        if want_spatial:
            (diagnostics.spatial_x_edges, diagnostics.spatial_y_edges,
             diagnostics.spatial_envelope) = _bin_spatial(
                contribution, x, y, spatial_edges, n_spatial_bins, k0, sigma_ex, sigma_ey, sigma_lr0, xp)
        return gamma, theta_x, theta_y, a0_shape, L, diagnostics

    # Chunked path: resolve shared diagnostics bin edges once from the
    # FULL bunch's t0_local/t1_local (not per-chunk -- _resolve_time_range/
    # _resolve_spatial_range auto-derive a window from whatever slice
    # they're given, which would differ chunk to chunk and make the
    # per-chunk histograms unsummable). Every chunk below then bins into
    # these identical edges and the resulting envelopes (already rate/
    # density, not raw counts -- see _bin_temporal/_bin_spatial) are
    # directly summable since dt_sec/bin_area depend only on the shared
    # edges, not on which chunk produced them.
    if want_time and t_edges is None:
        t_lo, t_hi, n_tb = _resolve_time_range(t0_local, t1_local, None, n_time_bins, xp)
        t_edges = xp.linspace(t_lo, t_hi, n_tb + 1)
    if want_spatial and spatial_edges is None:
        sx_lo, sx_hi, sy_lo, sy_hi, nsx, nsy = _resolve_spatial_range(
            k0, sigma_ex, sigma_ey, sigma_lr0, None, n_spatial_bins)
        spatial_edges = (xp.linspace(sx_lo, sx_hi, nsx + 1), xp.linspace(sy_lo, sy_hi, nsy + 1))

    is_cupy = xp is not np
    # MemoryError on the numpy path: same halve-and-retry safety net as
    # cupy's OutOfMemoryError, secondary to estimate_chunk_size's proactive
    # sizing above (Linux overcommit means a too-large host allocation can
    # be killed by the OOM-killer before ever raising MemoryError).
    oom_exc = xp.cuda.memory.OutOfMemoryError if is_cupy else MemoryError

    a0_parts, L_parts = [], []
    time_envelope_total = spatial_envelope_total = None
    t_edges_out = sx_edges_out = sy_edges_out = None

    start = 0
    cur_chunk = int(chunk)
    halvings = 0
    while start < n:
        end = min(start + cur_chunk, n)
        sl = slice(start, end)
        try:
            L_c, a0_c, contribution_c, t_c, x_c, y_c = _integrate_trajectory_core(
                x0[sl], y0[sl], z0[sl], theta_x[sl], theta_y[sl], weight,
                t0_local[sl], t1_local[sl], n_steps,
                w0, zT, z_rayleigh, beta_ff, N_l, ellipticity, sigma_T, k0, xp)
            # Binning runs inside the SAME try block as the integration step,
            # not after it: _bin_temporal/_bin_spatial allocate their own
            # (chunk, n_steps)-sized temporaries on top of contribution_c/
            # t_c/x_c/y_c (still alive at this point), comparable in size to
            # the integration step's own -- and both adapters always request
            # diagnostics (XigmaAdapter/DirectAdapter default n_time_bins/
            # n_spatial_bins to non-None), so this is the codepath real runs
            # actually take. An OOM here needs the same halve-and-retry
            # coverage as the integration step, or the retry loop protects a
            # path nothing calls in practice.
            time_env_c = spat_env_c = None
            if want_time:
                t_edges_out, time_env_c = _bin_temporal(
                    contribution_c, t_c, t0_local[sl], t1_local[sl], t_edges, n_time_bins, omega_las, xp)
            if want_spatial:
                sx_edges_out, sy_edges_out, spat_env_c = _bin_spatial(
                    contribution_c, x_c, y_c, spatial_edges, n_spatial_bins, k0, sigma_ex, sigma_ey, sigma_lr0, xp)
        except oom_exc:
            if halvings >= _MAX_OOM_HALVINGS or cur_chunk <= 1:
                raise
            cur_chunk = max(1, cur_chunk // 2)
            halvings += 1
            xp.get_default_memory_pool().free_all_blocks()
            continue

        a0_parts.append(a0_c)
        L_parts.append(L_c)
        if want_time:
            time_envelope_total = time_env_c if time_envelope_total is None else time_envelope_total + time_env_c
        if want_spatial:
            spatial_envelope_total = spat_env_c if spatial_envelope_total is None else spatial_envelope_total + spat_env_c

        if is_cupy:
            # Same reasoning as deposition.py's chunked branches (_deposit,
            # build_table_streaming): a memory-constrained GPU has no
            # headroom for several stale chunks' temporaries plus the
            # running accumulators -- drop the pool's hold explicitly
            # rather than trusting GC timing before the next iteration.
            del L_c, a0_c, contribution_c, t_c, x_c, y_c
            xp.get_default_memory_pool().free_all_blocks()
        start = end

    a0_shape = xp.concatenate(a0_parts)
    L = xp.concatenate(L_parts)

    if not want_time and not want_spatial:
        return gamma, theta_x, theta_y, a0_shape, L

    diagnostics = PushDiagnostics()
    if want_time:
        diagnostics.t_edges, diagnostics.time_envelope = t_edges_out, time_envelope_total
    if want_spatial:
        diagnostics.spatial_x_edges, diagnostics.spatial_y_edges = sx_edges_out, sy_edges_out
        diagnostics.spatial_envelope = spatial_envelope_total
    return gamma, theta_x, theta_y, a0_shape, L, diagnostics


_numba_kernel_cache = None


def _get_numba_kernel():
    """Lazily compiles and caches the numba kernel so importing particles.py
    doesn't require numba to be installed unless backend='numba' is used.
    """
    global _numba_kernel_cache
    if _numba_kernel_cache is not None:
        return _numba_kernel_cache
    try:
        import numba
    except ImportError as e:
        raise ImportError("backend='numba' requires the numba package (pip install numba)") from e

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def kernel(x0, y0, z0, vx, vy, vz, t0_local, t1_local, n_steps,
               beta_ff, w0, zT, z_rayleigh, particle_weight, v_rel, sigma_T_,
               k0_sq, N_l, F_pol):
        n = x0.shape[0]
        L = np.empty(n, dtype=np.float64)
        a0_shape = np.empty(n, dtype=np.float64)
        two_pi = 2.0 * np.pi
        sqrt_two_pi = np.sqrt(two_pi)
        peak_shape = 1.0 / (two_pi * w0 * w0) / (sqrt_two_pi * zT)

        for i in numba.prange(n):
            span = t1_local[i] - t0_local[i]
            if span < 0.0:
                span = 0.0
            dt = span / n_steps
            dt0 = z0[i] / vz[i]

            contribution_sum = 0.0
            ratio_sum = 0.0
            ratio_sq_sum = 0.0
            for j in range(n_steps):
                step = (j + 0.5) / n_steps
                t = t0_local[i] + step * span

                x = x0[i] + vx[i] * (t + dt0)
                y = y0[i] + vy[i] * (t + dt0)
                z = z0[i] + vz[i] * t

                # Scalar, numba-compatible hand-inlining of
                # GaussianParaxialLaser.pulse_envelope (head-on,
                # focus=(0,0,0) case: u=-z, perp2=x^2+y^2, u_spot=-z+beta_ff*t
                # -- see that method's docstring for the general form).
                # numba's nopython mode can't accept that function's `xp`
                # parameter (not a numba-representable type), so this stays a
                # separate copy from _push_and_sample_vectorized's call into
                # the shared function -- if the shared formula's math ever
                # changes, this block must be updated by hand in lockstep;
                # there is no automated sync.
                zr_term = z - beta_ff * t
                sigma_l_sq = w0 * w0 * (1.0 + zr_term * zr_term / (z_rayleigh * z_rayleigh))
                env = np.exp(-((z + t) / zT) ** 2 / 2.0) / sqrt_two_pi / zT
                n_ph_shape = np.exp(-(x * x + y * y) / sigma_l_sq / 2.0) / two_pi / sigma_l_sq * env

                # ratio = (a0_local/a0)**2 -- built without a0, see
                # _push_and_sample_vectorized / "a0_shape decouples from a0"
                # in push_and_sample's docstring.
                ratio = n_ph_shape / peak_shape
                if ratio < 0.0:
                    ratio = 0.0

                contribution_sum += v_rel * n_ph_shape * dt * particle_weight * sigma_T_ * k0_sq * N_l

                ratio_sum += ratio
                ratio_sq_sum += ratio * ratio

            L[i] = contribution_sum
            a0_shape[i] = F_pol * ratio_sq_sum / ratio_sum if ratio_sum > 0.0 else 0.0

        return L, a0_shape

    _numba_kernel_cache = kernel
    return kernel


def _push_and_sample_numba(k0_las, beta_ff, sigma_lr0, sigma_lz, N_l, ellipticity, N_e, macrobunch, n_steps):
    """Per-particle @numba.njit(parallel=True) form of push_and_sample: same
    physics as _push_and_sample_vectorized, but integrated with an explicit
    inner loop over n_steps instead of a materialised (n_particles, n_steps)
    array, parallelised across particles (numba.prange) instead of relying
    on numpy's (single-threaded, for elementwise ops) vectorisation. Wins
    both wall-clock (multiple CPU cores) and peak memory (no O(n_particles *
    n_steps) temporaries) at large problem sizes.
    """
    from .config import sigma_T

    k0 = k0_las
    w0 = k0 * sigma_lr0
    zT = k0 * sigma_lz
    z_rayleigh = 2 * w0 * w0 * (1.0 + beta_ff)

    x0, y0, z0, gamma, theta_x, theta_y, weight = _normalise_bunch(k0_las, N_e, macrobunch, np)

    vz = np.sqrt(np.maximum(0.0, 1.0 - theta_x**2 - theta_y**2))
    t0_local, t1_local = laser_overlap_time_window(
        z0, k0_las=k0, sigma_lz=sigma_lz, sigma_lr0=sigma_lr0,
        beta_ff=beta_ff, gauss_width=GAUSS_WIDTH, lorentz_width=LORENTZ_WIDTH, xp=np)

    kernel = _get_numba_kernel()
    L, a0_shape = kernel(
        np.ascontiguousarray(x0), np.ascontiguousarray(y0),
        np.ascontiguousarray(z0), np.ascontiguousarray(theta_x),
        np.ascontiguousarray(theta_y), vz, t0_local, t1_local, n_steps,
        beta_ff, w0, zT, z_rayleigh, weight, V_REL, sigma_T, k0**2,
        N_l, (1.0 + ellipticity**2) / 2.0,
    )

    return gamma, theta_x, theta_y, a0_shape, L
