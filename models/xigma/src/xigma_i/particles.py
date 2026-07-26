"""Stage 0: macroparticle source and ballistic pusher.

Produces, for a bunch of macroparticles pushed ballistically through the
laser pulse, per-particle samples (gamma, theta_x, theta_y, a0, weight)
consumed by Stage 1 deposition (see deposition.py) to build the 4D overlap
table H[gamma, theta_x, theta_y, a0].

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

from .config import GAUSS_WIDTH, LORENTZ_WIDTH

V_REL = 2.0  # relative-velocity factor for near-backscattering geometry


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


def _resolve_spatial_range(compton, spatial_edges, n_spatial_bins):
    """(sx_lo, sx_hi, sy_lo, sy_hi, nsx, nsy) in k0_las-normalised units --
    spatial_edges=(x_edges, y_edges) verbatim if given, else a "few sigma
    of whichever of the electron beam / laser waist is larger" window."""
    if spatial_edges is not None:
        x_edges, y_edges = spatial_edges
        return (float(x_edges[0]), float(x_edges[-1]),
                float(y_edges[0]), float(y_edges[-1]),
                len(x_edges) - 1, len(y_edges) - 1)
    nsx, nsy = (n_spatial_bins, n_spatial_bins) if np.isscalar(n_spatial_bins) else n_spatial_bins
    sx_half = GAUSS_WIDTH * compton.k0_las * max(compton.sigma_ex, compton.sigma_lr0)
    sy_half = GAUSS_WIDTH * compton.k0_las * max(compton.sigma_ey, compton.sigma_lr0)
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


def _bin_spatial(contribution, x, y, spatial_edges, n_spatial_bins, k0_las, compton, xp):
    """Nearest-cell (x, y) histogram of `contribution` -> an areal-density
    PushDiagnostics.spatial_x_edges/spatial_y_edges/spatial_envelope
    triple, in cm / photons/cm^2."""
    sx_lo, sx_hi, sy_lo, sy_hi, nsx, nsy = _resolve_spatial_range(compton, spatial_edges, n_spatial_bins)
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


@dataclass
class Bunch:
    """Macroparticles with real per-particle energy and momentum angles.

    x0, y0, z0 are k0_las-normalised positions. gamma, theta_x, theta_y are
    true per-particle values -- not grid-supplied. weight is the number of
    physical electrons represented by each macroparticle (uniform across
    the bunch).
    """
    x0: np.ndarray
    y0: np.ndarray
    z0: np.ndarray
    gamma: np.ndarray
    theta_x: np.ndarray
    theta_y: np.ndarray
    weight: float

    @property
    def n_particles(self):
        return self.x0.shape[0]


_M_TO_CM = 100.0


def bunch_from_macrobunch(macrobunch, compton) -> "Bunch":
    """Convert a ``compton_io.bunch.MacroBunch`` (SI, external/engine-
    agnostic representation) into this module's ``Bunch`` (CGS,
    ``k0_las``-normalised positions). ``gamma``/``theta_x``/``theta_y``
    pass through unchanged -- both kascade and xigma_i use the same
    position-at-a-reference-slice-plus-ballistic-angle convention, so no
    angle/energy conversion is needed, only the position normalisation
    below (``k0_las * x[cm]``).

    ``weight`` is deliberately NOT taken from ``macrobunch.weight`` --
    recomputed as ``compton.N_e / macrobunch.n_particles`` instead, so the
    GUI's charge/N_e field (which sets ``compton.N_e`` via
    ``set_electron_parameters``) stays authoritative, matching how a
    loaded ``.ele`` file carries no charge information of its own (see
    ``compton_io.io_formats.sdds``'s module docstring) and how kascade's
    own ``run_simulation`` already ignores a loaded bunch's weight the
    same way.
    """
    k0 = compton.k0_las
    x0 = k0 * (np.asarray(macrobunch.x, dtype=float) * _M_TO_CM)
    y0 = k0 * (np.asarray(macrobunch.y, dtype=float) * _M_TO_CM)
    z0 = k0 * (np.asarray(macrobunch.z, dtype=float) * _M_TO_CM)
    gamma = np.asarray(macrobunch.gamma, dtype=float)
    theta_x = np.asarray(macrobunch.thx, dtype=float)
    theta_y = np.asarray(macrobunch.thy, dtype=float)
    weight = compton.N_e / macrobunch.n_particles
    return Bunch(x0=x0, y0=y0, z0=z0, gamma=gamma, theta_x=theta_x, theta_y=theta_y, weight=weight)


def _time_window(compton, z0, xp=np):
    """Per-particle time window [t0, t1] (k0_las*c*t units) bounding where the
    particle is within ~2 Rayleigh ranges transversely and ~1 Gauss-width
    temporally of the pulse. Same bound as calculate_intersection's p_t0/p_t1,
    ported to plain numpy and evaluated per-particle rather than per-batch.

    xp: array module z0 belongs to (np or cp) -- array-module-agnostic so the
    same function serves both the numpy and cupy push_and_sample backends.
    """
    beta_ff = compton.beta_ff
    zT = compton.k0_las * compton.sigma_lz
    zR = (compton.k0_las * compton.sigma_lr0)**2 * (1.0 + beta_ff) * 2.0

    sigma_tau = GAUSS_WIDTH * zT
    sigma_raileigh = LORENTZ_WIDTH * zR

    t0 = (xp.maximum(-sigma_tau, (-z0 * (1 + beta_ff) - 2 * sigma_raileigh) / (1 - beta_ff)) - z0) / 2
    t1 = (xp.minimum(sigma_tau, (-z0 * (1 + beta_ff) + 2 * sigma_raileigh) / (1 - beta_ff)) - z0) / 2
    return t0, t1


def push_and_sample(compton, bunch, n_steps=200, backend='numpy', *,
                     n_time_bins=None, t_edges=None,
                     n_spatial_bins=None, spatial_edges=None):
    """Ballistically push each macroparticle and emit one sample per particle.

    backend: 'numpy' (default) -- the original vectorised (n_particles,
        n_steps) broadcast, single-threaded. 'numba' -- CPU multithreading:
        a per-particle @numba.njit(parallel=True) loop (numpy.prange) that
        integrates each particle's trajectory without materialising the full
        (n_particles, n_steps) intermediate arrays, so it also uses far less
        memory at large n_particles*n_steps. Requires the numba package.
        'cupy' -- GPU offload: the same broadcast form as 'numpy', run with
        cupy arrays (array-module-agnostic, same pattern as deposition.py).
        Output arrays stay on-device (cupy), ready to feed straight into
        deposition.build_table without a host round-trip. Requires cupy and
        a CUDA device.

    Returns arrays (gamma, theta_x, theta_y, a0_shape, weight) of length
    n_particles, ready for Stage 1 deposition. gamma/theta_x/theta_y are
    constant per particle (no pusher acceleration -- straight-line
    trajectories).

    a0_shape here is NOT the instantaneous local field amplitude, and (new)
    NOT the physical trajectory-averaged effective intensity ahat either --
    it is ahat's a0-independent *shape factor*. The physical quantity,
    Paper/xigma.tex eq. "ahattraj":

        ahat(zeta) = (TrXi/2) * integral[a^2(t)]^2 dt / integral a^2(t) dt

    with a^2(t;zeta) = compton.a0**2 * ratio(t;zeta) (ratio = local/peak
    photon density, what this function computes internally), factorises
    *exactly* as ahat(zeta) = compton.a0**2 * a0_shape(zeta), because
    ratio(t;zeta) depends only on the particle's (ballistic, a0-independent)
    trajectory through the pulse envelope, never on compton.a0 itself:

        a0_shape(zeta) = (TrXi/2) * integral[ratio(t)]^2 dt / integral ratio(t) dt

    So a0_shape is computed here *without reference to compton.a0 at all*
    (TrXi/2 = (1 + ellipticity**2)/2, eq. "Xi", generalised from linear
    polarisation to Compton.ellipticity -- ellipticity is a laser-polarisation
    property, not an intensity/a0 one, so it stays baked in). This means one
    push_and_sample run's output can be re-targeted to *any* actual a0 (any
    pulse energy) after the fact, without rerunning Stage 0/1 -- see
    deposition.retarget_a0. A single scalar per particle, not a distribution
    sampled along its own trajectory, because in this weakly-nonlinear regime
    (a0 <~ 1) the photon formation length spans the *whole* trajectory --
    unlike the synchrotron regime, splitting the trajectory into short
    segments and radiating each independently is not valid here. See
    CLAUDE.md "Known bugs" / "Traps" for the full explanation; do not go back
    to per-timestep a0 deposition.

    IMPORTANT: a0_shape is not directly usable as H's 4th axis for
    spectrum4d/reference's table consumers -- those need the *physical* ahat
    (a0_shape scaled by an actual a0**2). Build the table with a0_shape as
    the 4th axis (deposition.build_table(..., a0_kind='shape')), then call
    deposition.retarget_a0(table, a0) to get a physical, spectrum-ready
    table for a specific a0. Passing a 'shape' table straight to
    spectrum4d/reference is a normalisation error they now guard against
    (see Table.a0_kind).

    weight[i] = sum over the particle's timesteps of
                v_rel * n_ph_shape(t, r) * dt * weight_macro * sigma_T *
                k0_las**2 * N_l
    i.e. the luminosity functional L(zeta) (Paper/xigma.tex eq. "lumfun"),
    already fully CGS-normalised -- no angular-grid normalisation
    (2*pi*sigma_thx*sigma_thy) or position-truncation correction needed,
    since positions/angles are drawn from their true distributions rather
    than an importance-sampled truncated domain.

    n_steps sets the trajectory-integration resolution for L and a0_shape
    (not the output array length, which is always n_particles).

    n_time_bins/t_edges, n_spatial_bins/spatial_edges: opt-in
    temporal-envelope / spatial-distribution diagnostics, binned during
    this same trajectory-integration loop (see
    PushDiagnostics). Backward compatible by construction: if neither is
    given (the default), the return value is unchanged, the plain
    (gamma, theta_x, theta_y, a0_shape, weight) 5-tuple every existing
    caller already unpacks. If either is given, a 6th value -- a
    PushDiagnostics instance -- is appended; callers that want these
    diagnostics must opt in and unpack 6 values, not 5. n_time_bins/
    n_spatial_bins are bin counts (n_spatial_bins may be an (nsx, nsy)
    pair); t_edges/spatial_edges=(x_edges, y_edges) override the
    auto-derived bunch-wide window with explicit edges (and imply their
    own bin count). Only supported for backend='numpy'/'cupy' -- see
    _push_and_sample_numba's docstring for why the numba backend doesn't
    implement this.
    """
    if backend == 'numpy':
        return _push_and_sample_vectorized(compton, bunch, n_steps, np,
                                            n_time_bins, t_edges, n_spatial_bins, spatial_edges)
    if backend == 'cupy':
        import cupy as cp
        return _push_and_sample_vectorized(compton, bunch, n_steps, cp,
                                            n_time_bins, t_edges, n_spatial_bins, spatial_edges)
    if backend == 'numba':
        if n_time_bins is not None or t_edges is not None or n_spatial_bins is not None or spatial_edges is not None:
            raise NotImplementedError(
                "push_and_sample(backend='numba', n_time_bins=..., n_spatial_bins=...): "
                "Stage C diagnostics aren't implemented for the numba backend -- no current "
                "caller needs backend='numba' (the GUI/TabulatedEngine use 'numpy'/'cupy'), "
                "so this was scoped out rather than adding a second compiled kernel variant "
                "for an unused path. Use backend='numpy' or 'cupy' if you need these.")
        return _push_and_sample_numba(compton, bunch, n_steps)
    raise ValueError(f"backend must be 'numpy', 'numba', or 'cupy', got {backend!r}")


def _push_and_sample_vectorized(compton, bunch, n_steps, xp,
                                 n_time_bins=None, t_edges=None,
                                 n_spatial_bins=None, spatial_edges=None):
    """The (n_particles, n_steps) broadcast form of push_and_sample, shared
    by the 'numpy' and 'cupy' backends -- array-module-agnostic like
    deposition.py's deposit_nearest/deposit_cic, since every operation here
    is elementwise or a reduction along the n_steps axis (nothing that needs
    a hand-written kernel). For xp=cp, bunch's (host numpy) fields are
    transferred once at the top and results stay on-device.

    n_time_bins/t_edges/n_spatial_bins/spatial_edges: see push_and_sample's
    docstring -- binned here, after `contribution` (the same (n, n_steps)
    array L sums over time) is computed, since that's
    the exact quantity being partitioned differently rather than collapsed.
    """
    from .config import sigma_T

    k0 = compton.k0_las
    beta_ff = compton.beta_ff
    w0 = k0 * compton.sigma_lr0
    zT = k0 * compton.sigma_lz
    z_rayleigh = 2 * w0 * w0 * (1.0 + beta_ff)

    x0, y0, z0, gamma, theta_x, theta_y = (
        xp.asarray(a) for a in
        (bunch.x0, bunch.y0, bunch.z0, bunch.gamma, bunch.theta_x, bunch.theta_y))

    vx, vy = theta_x, theta_y
    vz = xp.sqrt(xp.maximum(0.0, 1.0 - vx**2 - vy**2))
    dt0 = z0 / vz

    t0_local, t1_local = _time_window(compton, z0, xp)
    span = xp.maximum(0.0, t1_local - t0_local)
    dt = span / n_steps

    step = (xp.arange(n_steps) + 0.5) / n_steps  # midpoint rule, shape (n_steps,)
    t = t0_local[:, None] + step[None, :] * span[:, None]  # (n, n_steps)

    x = x0[:, None] + vx[:, None] * (t + dt0[:, None])
    y = y0[:, None] + vy[:, None] * (t + dt0[:, None])
    z = z0[:, None] + vz[:, None] * t

    sigma_l_sq = w0 * w0 * (1.0 + (z - beta_ff * t)**2 / z_rayleigh**2)
    env = xp.exp(-((z + t) / zT)**2 / 2) / xp.sqrt(2 * np.pi) / zT
    n_ph_shape = xp.exp(-(x**2 + y**2) / sigma_l_sq / 2) / (2 * np.pi) / sigma_l_sq * env

    peak_shape = 1.0 / (2 * np.pi * w0 * w0) / (np.sqrt(2 * np.pi) * zT)
    # ratio = (a0_local/compton.a0)**2 -- deliberately built without compton.a0
    # at all, see "a0_shape decouples from compton.a0" below.
    ratio = xp.clip(n_ph_shape / peak_shape, 0.0, None)

    contribution = V_REL * n_ph_shape * dt[:, None] * bunch.weight * sigma_T * k0**2 * compton.N_l

    L = contribution.sum(axis=1)  # eq. "lumfun", per-particle deposited weight

    denom = ratio.sum(axis=1)
    F_pol = (1.0 + compton.ellipticity**2) / 2.0  # TrXi/2, eq. "Xi"
    # xp.where instead of np.divide(..., where=) -- cupy's ufunc `where=`
    # kwarg support is version-dependent; xp.where is safe on both.
    a0_shape = xp.where(denom > 0, F_pol * (ratio**2).sum(axis=1) / xp.maximum(denom, 1e-300), 0.0)

    want_time = n_time_bins is not None or t_edges is not None
    want_spatial = n_spatial_bins is not None or spatial_edges is not None
    if not want_time and not want_spatial:
        return gamma, theta_x, theta_y, a0_shape, L

    diagnostics = PushDiagnostics()
    if want_time:
        diagnostics.t_edges, diagnostics.time_envelope = _bin_temporal(
            contribution, t, t0_local, t1_local, t_edges, n_time_bins, compton.omega_las, xp)
    if want_spatial:
        (diagnostics.spatial_x_edges, diagnostics.spatial_y_edges,
         diagnostics.spatial_envelope) = _bin_spatial(
            contribution, x, y, spatial_edges, n_spatial_bins, k0, compton, xp)
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

                zr_term = z - beta_ff * t
                sigma_l_sq = w0 * w0 * (1.0 + zr_term * zr_term / (z_rayleigh * z_rayleigh))
                env = np.exp(-((z + t) / zT) ** 2 / 2.0) / sqrt_two_pi / zT
                n_ph_shape = np.exp(-(x * x + y * y) / sigma_l_sq / 2.0) / two_pi / sigma_l_sq * env

                # ratio = (a0_local/compton.a0)**2 -- built without compton.a0,
                # see _push_and_sample_vectorized / "a0_shape decouples from
                # compton.a0" in push_and_sample's docstring.
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


def _push_and_sample_numba(compton, bunch, n_steps):
    """Per-particle @numba.njit(parallel=True) form of push_and_sample: same
    physics as _push_and_sample_vectorized, but integrated with an explicit
    inner loop over n_steps instead of a materialised (n_particles, n_steps)
    array, parallelised across particles (numba.prange) instead of relying
    on numpy's (single-threaded, for elementwise ops) vectorisation. Wins
    both wall-clock (multiple CPU cores) and peak memory (no O(n_particles *
    n_steps) temporaries) at large problem sizes.
    """
    from .config import sigma_T

    k0 = compton.k0_las
    beta_ff = compton.beta_ff
    w0 = k0 * compton.sigma_lr0
    zT = k0 * compton.sigma_lz
    z_rayleigh = 2 * w0 * w0 * (1.0 + beta_ff)

    vz = np.sqrt(np.maximum(0.0, 1.0 - bunch.theta_x**2 - bunch.theta_y**2))
    t0_local, t1_local = _time_window(compton, bunch.z0, np)

    kernel = _get_numba_kernel()
    L, a0_shape = kernel(
        np.ascontiguousarray(bunch.x0), np.ascontiguousarray(bunch.y0),
        np.ascontiguousarray(bunch.z0), np.ascontiguousarray(bunch.theta_x),
        np.ascontiguousarray(bunch.theta_y), vz, t0_local, t1_local, n_steps,
        beta_ff, w0, zT, z_rayleigh, bunch.weight, V_REL, sigma_T, k0**2,
        compton.N_l, (1.0 + compton.ellipticity**2) / 2.0,
    )

    return bunch.gamma, bunch.theta_x, bunch.theta_y, a0_shape, L
