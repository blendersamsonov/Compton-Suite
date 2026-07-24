"""CPU/numba equivalents of core.py's two GPU rawkernels (particle_kernel,
spectrum_kernel), used as the fallback compute backend when no CUDA GPU is
available (see Compton's ``device``/``backend`` handling in core.py).

This module never imports cupy -- only numpy and (lazily) numba, so it stays
importable on a machine with no CUDA/cupy at all.

Both kernels here are deliberately literal, mechanical transliterations of
their GPU counterparts: same index arithmetic, same branch structure, same
edge-case behavior (including latent GPU quirks that look like no-ops -- see
inline notes in ``spectrum_kernel_cpu``), just restructured from CUDA's
grid/block/shared-memory/syncthreads model into plain serial-per-work-item
code parallelized with ``numba.prange`` over the axis that has no
cross-work-item data dependency. The goal is numerical parity with the GPU
kernels for the same inputs, not an independent reimplementation -- validate
any change here against ``core.py``'s actual GPU kernels on a CUDA machine,
not just against physical intuition.

Two deliberate, CPU-only additions (both pure safety nets, not present on
GPU and not changing normal-case output): defensive clamps on a few
+1-neighbor array indices that are exact-boundary edge cases the GPU
silently tolerates (undefined-but-harmless out-of-bounds behavior) but numba
will raise on, since numba bounds-checks CPU array access.
"""

from __future__ import annotations

import math

import numpy as np

from .core import (
    MAX_RINGS, MAX_ARCS, N_RINGS_MIN, PHI_EDGES, PHI_CELLS,
    CDF_PHI_RESOLUTION, SAMPLES_TOTAL, R_MAX_NUDGE, PHI,
)

_numba_module = None


def _get_numba():
    global _numba_module
    if _numba_module is None:
        try:
            import numba
        except ImportError as e:
            raise ImportError(
                "the CPU backend requires the numba package (pip install numba)"
            ) from e
        _numba_module = numba
    return _numba_module


_particle_kernel_cache = None
_spectrum_kernel_cache = None


def get_particle_kernel_cpu():
    """Lazily compiles and caches particle_kernel_cpu (see module docstring)."""
    global _particle_kernel_cache
    if _particle_kernel_cache is not None:
        return _particle_kernel_cache
    numba = _get_numba()

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def particle_kernel_cpu(particles, THX, THY, f_th, w0, beta_ff, sigma_lz,
                             t0, t1, dvx, dvy, sx0, sx1, sy0, sy1,
                             n_steps, n_spatial):
        n_cells = THX.shape[0]
        n_particles_total = particles.shape[0]
        intersect = np.zeros(n_cells)

        n_threads = numba.get_num_threads()
        envelope_local = np.zeros((n_threads, n_steps))
        spatial_local = np.zeros((n_threads, n_spatial, n_spatial))

        two_pi = 2.0 * np.pi
        z_rayleigh = 2.0 * w0 * w0 * (1.0 + beta_ff)

        for xy_idx in numba.prange(n_cells):
            tid = numba.get_thread_id()
            weight = f_th[xy_idx]
            n_particles = int(math.ceil(weight * n_particles_total))
            vx0 = THX[xy_idx]
            vy0 = THY[xy_idx]
            local_sum = 0.0

            for p_idx in range(n_particles):
                x0 = particles[p_idx, 0]
                y0 = particles[p_idx, 1]
                z0 = particles[p_idx, 2]
                t_start = particles[p_idx, 3]
                t_end = particles[p_idx, 4]

                reg = p_idx / n_particles
                fib = (0.5 + p_idx / PHI) % 1.0
                vx = vx0 + (2.0 * reg - 1.0) * dvx / 2.0
                vy = vy0 + (2.0 * fib - 1.0) * dvy / 2.0
                vz = np.sqrt(max(0.0, 1.0 - vx * vx - vy * vy))
                dt0 = z0 / vz
                dt = (t_end - t_start) / n_steps

                for step_idx in range(n_steps):
                    t = step_idx * dt + t_start
                    x = x0 + vx * (t + dt0)
                    y = y0 + vy * (t + dt0)
                    z = z0 + vz * t

                    env = np.exp(-((z + t) / sigma_lz) ** 2 / 2.0) / np.sqrt(two_pi) / sigma_lz
                    zr_term = z - beta_ff * t
                    sigma_l_sq = w0 * w0 * (1.0 + zr_term * zr_term / (z_rayleigh * z_rayleigh))
                    f_cur = np.exp(-(x * x + y * y) / sigma_l_sq / 2.0) / two_pi / sigma_l_sq * env
                    f_cur *= dt * weight / n_particles

                    local_sum += f_cur

                    if t >= t0 and t < t1:
                        env_fac = n_steps * (t - t0) / (t1 - t0)
                        env_idx = int(np.floor(env_fac))
                        env_frac = env_fac - env_idx
                        if env_idx < n_steps:
                            envelope_local[tid, env_idx] += f_cur * (1.0 - env_frac)
                        # CPU-only defensive bound (GPU relies on undefined-
                        # but-harmless OOB tolerance at the exact t->t1 edge;
                        # numba bounds-checks, so guard instead of crashing).
                        if env_idx + 1 < n_steps:
                            envelope_local[tid, env_idx + 1] += f_cur * env_frac

                    if x >= sx0 and x < sx1 and y >= sy0 and y < sy1:
                        sx_fac = n_spatial * (x - sx0) / (sx1 - sx0)
                        sy_fac = n_spatial * (y - sy0) / (sy1 - sy0)
                        sx_idx = int(np.floor(sx_fac))
                        sy_idx = int(np.floor(sy_fac))
                        sx_frac = sx_fac - sx_idx
                        sy_frac = sy_fac - sy_idx
                        if sx_idx < n_spatial and sy_idx < n_spatial:
                            spatial_local[tid, sx_idx, sy_idx] += (
                                f_cur * (1.0 - sx_frac) * (1.0 - sy_frac))
                        if sx_idx + 1 < n_spatial and sy_idx < n_spatial:
                            spatial_local[tid, sx_idx + 1, sy_idx] += (
                                f_cur * sx_frac * (1.0 - sy_frac))
                        if sx_idx < n_spatial and sy_idx + 1 < n_spatial:
                            spatial_local[tid, sx_idx, sy_idx + 1] += (
                                f_cur * (1.0 - sx_frac) * sy_frac)
                        if sx_idx + 1 < n_spatial and sy_idx + 1 < n_spatial:
                            spatial_local[tid, sx_idx + 1, sy_idx + 1] += (
                                f_cur * sx_frac * sy_frac)

            intersect[xy_idx] = local_sum

        envelope = envelope_local.sum(axis=0)
        spatial = spatial_local.sum(axis=0)
        return intersect, envelope, spatial

    _particle_kernel_cache = particle_kernel_cpu
    return particle_kernel_cpu


def get_spectrum_kernel_cpu():
    """Lazily compiles and caches spectrum_kernel_cpu (see module docstring)."""
    global _spectrum_kernel_cache
    if _spectrum_kernel_cache is not None:
        return _spectrum_kernel_cache
    numba = _get_numba()

    INVAL = 9999.0

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def spectrum_kernel_cpu(params_arr, collision, sigma_g, gamma0, dx, dy,
                             phi_pol, subsampling):
        n_out = params_arr.shape[0]
        output = np.zeros(n_out)
        two_pi = 2.0 * np.pi
        nx_col = collision.shape[0]
        ny_col = collision.shape[1]

        for out_idx in numba.prange(n_out):
            x0 = params_arr[out_idx, 0]
            y0 = params_arr[out_idx, 1]
            s = params_arr[out_idx, 2]

            rmin_g = np.sqrt(max(0.0, 1.0 / s - 1.0 / (gamma0 - 3.0 * sigma_g) ** 2))
            rmax_g = np.sqrt(max(0.0, 1.0 / s - 1.0 / (gamma0 + 3.0 * sigma_g) ** 2))
            rmin_r = np.sqrt(max(abs(x0) - dx, 0.0) ** 2 + max(abs(y0) - dy, 0.0) ** 2)
            diam = 2.0 * np.sqrt(dx ** 2 + dy ** 2)
            xm = dx + abs(x0)
            ym = dy + abs(y0)
            rmax_r = np.sqrt(xm ** 2 + ym ** 2) - diam / R_MAX_NUDGE

            rmin = max(rmin_g, rmin_r)
            rmax = min(rmax_g, rmax_r)

            if rmin >= rmax:
                output[out_idx] = 0.0
                continue

            r_inside = max(0.0, min(dx - abs(x0), dy - abs(y0)))
            n_rings = max(N_RINGS_MIN, int(MAX_RINGS * (rmax - rmin) / diam))
            if n_rings > MAX_RINGS:
                n_rings = MAX_RINGS
            dr = (rmax - rmin) / n_rings

            # --- Phase 1: per-ring quadrant-intersection geometry --------
            # (mirrors the GPU's per-thread-per-ring computation; each r_idx
            # is fully independent of every other, so a plain serial loop
            # over r_idx is exactly equivalent to the GPU's parallel
            # per-thread version followed by a syncthreads barrier.)
            rings_n = np.zeros(MAX_RINGS, dtype=np.int64)
            rings_phimin = np.zeros(MAX_RINGS * 4)
            rings_phimax = np.zeros(MAX_RINGS * 4)

            for r_idx in range(n_rings):
                r = rmin + dr * (r_idx + 0.5)
                n_arcs_ring = 0
                if r < r_inside:
                    rings_phimin[r_idx * 4 + 0] = 0.0
                    rings_phimax[r_idx * 4 + 0] = two_pi
                    n_arcs_ring = 1
                else:
                    phi_cur0 = -INVAL
                    phi_cur1 = INVAL
                    for q_idx in range(4):
                        sin_pos = q_idx // 2
                        cos_pos = ((q_idx + 1) // 2) % 2
                        sin_sign = 2 * sin_pos - 1
                        cos_sign = 2 * cos_pos - 1

                        cos_0 = (dx - x0) / r
                        sin_0 = np.sqrt(max(0.0, 1.0 - cos_0 * cos_0))
                        cos_1 = (-dx - x0) / r
                        sin_1 = np.sqrt(max(0.0, 1.0 - cos_1 * cos_1))
                        sin_2 = (dy - y0) / r
                        cos_2 = np.sqrt(max(0.0, 1.0 - sin_2 * sin_2))
                        sin_3 = (-dy - y0) / r
                        cos_3 = np.sqrt(max(0.0, 1.0 - sin_3 * sin_3))

                        if cos_sign * cos_0 > 0 and abs(y0 + r * sin_0 * sin_sign) < dy:
                            val = math.atan2(sin_0 * sin_sign, cos_0)
                            if (1 - sin_pos) == 0:
                                phi_cur0 = val
                            else:
                                phi_cur1 = val

                        if cos_sign * cos_1 > 0 and abs(y0 + r * sin_1 * sin_sign) < dy:
                            val = math.atan2(sin_1 * sin_sign, cos_1)
                            if sin_pos == 0:
                                phi_cur0 = val
                            else:
                                phi_cur1 = val

                        if sin_sign * sin_2 > 0 and abs(x0 + r * cos_2 * cos_sign) < dx:
                            val = math.atan2(sin_2, cos_2 * cos_sign)
                            if cos_pos == 0:
                                phi_cur0 = val
                            else:
                                phi_cur1 = val

                        if sin_sign * sin_3 > 0 and abs(x0 + r * cos_3 * cos_sign) < dx:
                            val = math.atan2(sin_3, cos_3 * cos_sign)
                            if (1 - cos_pos) == 0:
                                phi_cur0 = val
                            else:
                                phi_cur1 = val

                        if phi_cur1 < 1000.0:
                            rings_phimin[r_idx * 4 + n_arcs_ring] = phi_cur0
                            rings_phimax[r_idx * 4 + n_arcs_ring] = phi_cur1
                            phi_cur0 = -INVAL
                            phi_cur1 = INVAL
                            n_arcs_ring += 1

                    # 4th<->1st quadrant merge -- literal port of the GPU
                    # check (including its GPU-side quirk: this only ever
                    # fires when n_arcs_ring is still 0, in which case it
                    # patches a phi_min that's never subsequently read
                    # since rings_n[r_idx] stays 0 below -- kept faithfully
                    # rather than "fixed", to preserve GPU/CPU parity).
                    if phi_cur0 > -1000.0 and rings_phimin[r_idx * 4 + 0] < -1000.0:
                        rings_phimin[r_idx * 4 + 0] = phi_cur0 - two_pi

                rings_n[r_idx] = n_arcs_ring

            # --- Phase 2: serialize rings -> arcs -------------------------
            arcs_r = np.zeros(MAX_ARCS)
            arcs_phimin = np.zeros(MAX_ARCS)
            arcs_phimax = np.zeros(MAX_ARCS)
            n_arcs = 0
            for i in range(n_rings):
                r_i = rmin + dr * (i + 0.5)
                for j in range(rings_n[i]):
                    if n_arcs < MAX_ARCS:
                        arcs_r[n_arcs] = r_i
                        arcs_phimin[n_arcs] = rings_phimin[i * 4 + j]
                        arcs_phimax[n_arcs] = rings_phimax[i * 4 + j]
                        n_arcs += 1

            if n_arcs == 0:
                output[out_idx] = 0.0
                continue

            # --- Phase 3: coarse per-cell importance-sampling weights ----
            cum_w = np.zeros(MAX_ARCS * PHI_EDGES)
            for arc_idx in range(n_arcs):
                r = arcs_r[arc_idx]
                phi_min = arcs_phimin[arc_idx]
                phi_max = arcs_phimax[arc_idx]

                g = np.sqrt(1.0 / (1.0 / s - r * r))
                fr = np.exp(-(g - gamma0) ** 2 / 2.0 / sigma_g ** 2)

                for phi_idx in range(PHI_CELLS):
                    phi = phi_min + ((phi_idx + 0.5) / PHI_CELLS) * (phi_max - phi_min)
                    x = x0 + r * np.cos(phi)
                    y = y0 + r * np.sin(phi)

                    if -dx < x < dx and -dy < y < dy:
                        xi = int(np.floor((nx_col - 1) * (x + dx) / dx / 2.0))
                        yj = int(np.floor((ny_col - 1) * (y + dy) / dy / 2.0))
                        xi = min(max(xi, 0), nx_col - 1)
                        yj = min(max(yj, 0), ny_col - 1)
                        w = collision[xi, yj]
                    else:
                        w = 0.0
                    w *= fr
                    cum_w[arc_idx * PHI_EDGES + phi_idx] = w * (phi_max - phi_min) * r

                # exclusive prefix sum in place (cum_w[...+PHI_EDGES-1] ends
                # up holding the arc's total weight -- see module docstring)
                total = 0.0
                for k in range(PHI_EDGES):
                    tmp = cum_w[arc_idx * PHI_EDGES + k]
                    cum_w[arc_idx * PHI_EDGES + k] = total
                    total += tmp

            total_weight = 0.0
            for k in range(n_arcs):
                total_weight += cum_w[k * PHI_EDGES + (PHI_EDGES - 1)]

            # --- Phase 4: inverse-CDF tabulation per arc ------------------
            inv_cdf = np.zeros(MAX_ARCS * CDF_PHI_RESOLUTION)
            for arc_idx in range(n_arcs):
                phi_min = arcs_phimin[arc_idx]
                phi_max = arcs_phimax[arc_idx]
                dphi = (phi_max - phi_min) / PHI_CELLS
                arc_total = cum_w[arc_idx * PHI_EDGES + (PHI_EDGES - 1)]
                for r_idx in range(CDF_PHI_RESOLUTION):
                    target = arc_total * r_idx / (CDF_PHI_RESOLUTION - 1)
                    left = 0
                    right = PHI_EDGES - 1
                    while right - left > 1:
                        mid = (left + right) // 2
                        if cum_w[arc_idx * PHI_EDGES + mid] <= target:
                            left = mid
                        else:
                            right = mid
                    cdf_i = cum_w[arc_idx * PHI_EDGES + left]
                    cdf_ip1 = cum_w[arc_idx * PHI_EDGES + left + 1]
                    fac = (target - cdf_i) / (cdf_ip1 - cdf_i)
                    inv_cdf[arc_idx * CDF_PHI_RESOLUTION + r_idx] = (
                        phi_min + (left + fac) * dphi)

            # --- Phase 5: evaluation ---------------------------------------
            f_tot = 0.0
            for arc_idx in range(n_arcs):
                r = arcs_r[arc_idx]
                phi_min = arcs_phimin[arc_idx]
                phi_max = arcs_phimax[arc_idx]
                arc_total_weight = cum_w[arc_idx * PHI_EDGES + (PHI_EDGES - 1)]
                arc_area = (phi_max - phi_min) * r * dr

                if total_weight <= 0.0:
                    continue
                n_arc_samples = int(np.floor(SAMPLES_TOTAL * arc_total_weight / total_weight))
                if n_arc_samples <= 0:
                    continue

                theta_min = r - dr / 2.0
                theta_max = theta_min + dr

                for arc_sample_idx in range(n_arc_samples):
                    for di in range(subsampling):
                        subsample_idx = arc_sample_idx * subsampling + di

                        reg = (subsample_idx + 0.5) / n_arc_samples / subsampling
                        fib = (subsample_idx * PHI) % 1.0

                        theta_sq = theta_min ** 2 + fib * (theta_max ** 2 - theta_min ** 2)
                        theta = np.sqrt(theta_sq)

                        il = int(np.floor(reg * (CDF_PHI_RESOLUTION - 1)))
                        fac = reg * (CDF_PHI_RESOLUTION - 1) - il
                        phi = (inv_cdf[arc_idx * CDF_PHI_RESOLUTION + il] * (1.0 - fac)
                               + inv_cdf[arc_idx * CDF_PHI_RESOLUTION + il + 1] * fac)

                        phi_idx = min(PHI_CELLS - 1,
                                      int(PHI_CELLS * (phi - phi_min) / (phi_max - phi_min)))
                        cell_weight = (cum_w[arc_idx * PHI_EDGES + phi_idx + 1]
                                       - cum_w[arc_idx * PHI_EDGES + phi_idx])
                        if cell_weight == 0.0:
                            continue
                        sample_area = (arc_area / n_arc_samples / subsampling
                                        * arc_total_weight / cell_weight)

                        x = x0 + theta * np.cos(phi)
                        y = y0 + theta * np.sin(phi)
                        g_sq = 1.0 / (1.0 / s - theta_sq)
                        if g_sq >= 0.0:
                            Xi = (nx_col - 1) * (x + dx) / dx / 2.0
                            Yj = (ny_col - 1) * (y + dy) / dy / 2.0
                            xi = int(np.floor(Xi))
                            yj = int(np.floor(Yj))
                            Xi_frac = Xi - xi
                            Yj_frac = Yj - yj
                            xi = min(max(xi, 0), nx_col - 2)
                            yj = min(max(yj, 0), ny_col - 2)

                            if -dx < x < dx and -dy < y < dy:
                                col = (collision[xi, yj] * (1.0 - Yj_frac) * (1.0 - Xi_frac)
                                       + collision[xi + 1, yj] * (1.0 - Yj_frac) * Xi_frac
                                       + collision[xi, yj + 1] * Yj_frac * (1.0 - Xi_frac)
                                       + collision[xi + 1, yj + 1] * Yj_frac * Xi_frac)
                            else:
                                col = 0.0

                            g = np.sqrt(g_sq)
                            ffac = np.exp(-(g - gamma0) ** 2 / 2.0 / sigma_g ** 2) / np.sqrt(two_pi * sigma_g ** 2)
                            gth_sq_inv = 1.0 / (1.0 + theta_sq * g_sq) ** 2
                            cos_pol = np.cos(phi_pol - phi) ** 2
                            a_fac = 1.0 - 4.0 * cos_pol * theta_sq * g_sq * gth_sq_inv
                            f = ffac * a_fac * col * g ** 5 * gth_sq_inv
                            f_tot += f * sample_area

            output[out_idx] = f_tot / s ** 2

        return output

    _spectrum_kernel_cache = spectrum_kernel_cpu
    return spectrum_kernel_cpu
