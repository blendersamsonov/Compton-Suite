"""CPU/numba equivalent of spectrum4d.py's GPU rawkernel (spectrum_kernel_4d),
used as the fallback compute backend when no CUDA GPU is available (see
calculate_angular_spectrum_4d's `device` handling in spectrum4d.py).

This module never imports cupy -- only numpy and (lazily) numba, so it stays
importable on a machine with no CUDA/cupy at all.

A deliberate, literal transliteration of spectrum_kernel_4d: same annulus/
ring/arc geometry, same inverse-CDF phi sampling, restructured from CUDA's
grid/block/shared-memory/syncthreads model into plain serial-per-work-item
code parallelised with numba.prange over out_idx (each GPU block computes
one, fully independent, output point, so a serial loop over out_idx is
exactly equivalent to the GPU's parallel-per-block version). The GPU
kernel's `thread_samples`/round-robin sample-to-thread scheduling is
dropped entirely -- that existed purely to load-balance across 128 GPU
threads and doesn't affect the numeric result (a straight sum over all
(arc, sample, subsample) triples), so this port just loops directly over
them. Quadrilinear (not bilinear) interpolation of H over (gamma, theta_x,
theta_y), and the nested a0-quadrature loop inside the final evaluation
(g/prefac recomputed *inside* the a0 loop, not shared across bins -- see
spectrum4d.py's module docstring for why that matters) are ported
line-for-line from the GPU kernel's actual loop structure, not re-derived
from physical intuition.

The two zero-weight guards spectrum4d.py's docstring calls out (inv_cdf
falling back to a cell's left edge on a flat CDF; sample_area contributing
zero instead of x/0) are carried over unchanged -- H is a sparse
finite-particle deposition and can have exact-zero cells.

Goal is numerical parity with the real GPU kernel for the same inputs --
validate any change here against spectrum4d.py's actual GPU kernel on a CUDA
machine, not just against physical intuition (see this repo's CLAUDE.md,
"GUI-side testing").
"""

from __future__ import annotations

import numpy as np

from .config import (
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


_spectrum_kernel_4d_cache = None


def get_spectrum_kernel_4d_cpu():
    """Lazily compiles and caches spectrum_kernel_4d_cpu (see module docstring)."""
    global _spectrum_kernel_4d_cache
    if _spectrum_kernel_4d_cache is not None:
        return _spectrum_kernel_4d_cache
    numba = _get_numba()

    INVAL = 9999.0

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def spectrum_kernel_4d_cpu(params_arr, H, H_marginal,
                                gamma_min, gamma_width, n_gamma,
                                theta_x_min, theta_x_width, n_theta_x,
                                theta_y_min, theta_y_width, n_theta_y,
                                a0_min, a0_width, n_a0,
                                gamma_lo, gamma_hi, dx, dy,
                                phi_pol, subsampling):
        n_out = params_arr.shape[0]
        output = np.zeros(n_out)
        two_pi = 2.0 * np.pi

        for out_idx in numba.prange(n_out):
            x0 = params_arr[out_idx, 0]
            y0 = params_arr[out_idx, 1]
            s = params_arr[out_idx, 2]

            a0_max = a0_min + a0_width * n_a0
            rmin_g = np.sqrt(max(0.0, 1.0 / s - (1.0 + a0_max) / gamma_lo ** 2))
            rmax_g = np.sqrt(max(0.0, 1.0 / s - (1.0 + a0_min) / gamma_hi ** 2))
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
                            val = np.arctan2(sin_0 * sin_sign, cos_0)
                            if (1 - sin_pos) == 0:
                                phi_cur0 = val
                            else:
                                phi_cur1 = val

                        if cos_sign * cos_1 > 0 and abs(y0 + r * sin_1 * sin_sign) < dy:
                            val = np.arctan2(sin_1 * sin_sign, cos_1)
                            if sin_pos == 0:
                                phi_cur0 = val
                            else:
                                phi_cur1 = val

                        if sin_sign * sin_2 > 0 and abs(x0 + r * cos_2 * cos_sign) < dx:
                            val = np.arctan2(sin_2, cos_2 * cos_sign)
                            if cos_pos == 0:
                                phi_cur0 = val
                            else:
                                phi_cur1 = val

                        if sin_sign * sin_3 > 0 and abs(x0 + r * cos_3 * cos_sign) < dx:
                            val = np.arctan2(sin_3, cos_3 * cos_sign)
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

                    # 4th<->1st quadrant merge -- a GPU-side quirk kept
                    # faithfully for parity: this only ever fires when
                    # n_arcs_ring is still 0, in which case it patches a
                    # phi_min that's never read since rings_n[r_idx] stays
                    # 0 below.
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
            # Proposal density: H_marginal, the (theta_x, theta_y) marginal
            # of H summed over gamma and a0 (see spectrum4d.py's module
            # docstring) -- no resonance condition here, nearest-cell lookup.
            cum_w = np.zeros(MAX_ARCS * PHI_EDGES)
            for arc_idx in range(n_arcs):
                r = arcs_r[arc_idx]
                phi_min = arcs_phimin[arc_idx]
                phi_max = arcs_phimax[arc_idx]
                dphi_cell = (phi_max - phi_min) / PHI_CELLS

                for phi_idx in range(PHI_CELLS):
                    phi = phi_min + ((phi_idx + 0.5) / PHI_CELLS) * (phi_max - phi_min)
                    x = x0 + r * np.cos(phi)
                    y = y0 + r * np.sin(phi)

                    w = 0.0
                    if (theta_x_min < x < theta_x_min + theta_x_width * n_theta_x and
                            theta_y_min < y < theta_y_min + theta_y_width * n_theta_y):
                        xi = int(np.floor((x - theta_x_min) / theta_x_width))
                        yj = int(np.floor((y - theta_y_min) / theta_y_width))
                        xi = min(max(xi, 0), n_theta_x - 1)
                        yj = min(max(yj, 0), n_theta_y - 1)
                        w = H_marginal[xi, yj]

                    cum_w[arc_idx * PHI_EDGES + phi_idx] = w * dphi_cell * r

                # exclusive prefix sum in place
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
                    cdf_span = cdf_ip1 - cdf_i
                    # H can have exact-zero cells; a run of zero-weight cells
                    # makes cdf_span 0 at the boundary sample. Fall back to
                    # the cell's left edge instead of dividing by zero -- see
                    # spectrum4d.py's module docstring.
                    fac = 0.0
                    if cdf_span > 0.0:
                        fac = (target - cdf_i) / cdf_span
                    inv_cdf[arc_idx * CDF_PHI_RESOLUTION + r_idx] = (
                        phi_min + (left + fac) * dphi)

            # --- Phase 5: evaluation ---------------------------------------
            f_tot = 0.0
            for arc_idx in range(n_arcs):
                r = arcs_r[arc_idx]
                phi_min = arcs_phimin[arc_idx]
                phi_max = arcs_phimax[arc_idx]
                arc_total_weight = cum_w[arc_idx * PHI_EDGES + (PHI_EDGES - 1)]
                dphi_cell = (phi_max - phi_min) / PHI_CELLS
                arc_area = dphi_cell * r * dr  # dphi_cell, not (phi_max-phi_min) -- see spectrum4d.py's module docstring

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
                        # A sample can land in a zero-weight cell (see inv_cdf
                        # comment above); contributes nothing rather than x/0.
                        if cell_weight <= 0.0:
                            continue
                        sample_area = (arc_area / n_arc_samples / subsampling
                                        * arc_total_weight / cell_weight)

                        x = x0 + theta * np.cos(phi)
                        y = y0 + theta * np.sin(phi)

                        if not (theta_x_min < x < theta_x_min + theta_x_width * n_theta_x and
                                theta_y_min < y < theta_y_min + theta_y_width * n_theta_y):
                            continue

                        cos_pol = np.cos(phi_pol - phi) ** 2

                        Xf = (x - theta_x_min) / theta_x_width - 0.5
                        Yf = (y - theta_y_min) / theta_y_width - 0.5
                        xi2 = int(np.floor(Xf))
                        yj2 = int(np.floor(Yf))
                        xw = Xf - xi2
                        yw = Yf - yj2
                        xi2 = min(max(xi2, 0), n_theta_x - 2)
                        yj2 = min(max(yj2, 0), n_theta_y - 2)

                        # Each a0 bin resonates at its own gamma (eq. "Gamma"),
                        # with its own Jacobian factor 1/(1+a0) in the
                        # prefactor (eq. "Fmatrix") -- both g and the
                        # gamma-axis interpolation are recomputed per a0 bin,
                        # not shared across the quadrature the way
                        # x/y/theta_sq are. See spectrum4d.py's module
                        # docstring, point 4.
                        inv_base = 1.0 / s - theta_sq
                        h_sum = 0.0
                        if inv_base > 0.0:
                            for ai2 in range(n_a0):
                                a0_val = a0_min + (ai2 + 0.5) * a0_width
                                g_sq = (1.0 + a0_val) / inv_base
                                g = np.sqrt(g_sq)

                                if gamma_min < g < gamma_min + gamma_width * n_gamma:
                                    gth_sq_inv = 1.0 / (1.0 + theta_sq * g_sq) ** 2
                                    a_fac = 1.0 - 4.0 * cos_pol * theta_sq * g_sq * gth_sq_inv
                                    prefac = a_fac * g ** 5 * gth_sq_inv / (1.0 + a0_val)

                                    Gf = (g - gamma_min) / gamma_width - 0.5
                                    gi2 = int(np.floor(Gf))
                                    gw = Gf - gi2
                                    gi2 = min(max(gi2, 0), n_gamma - 2)

                                    h000 = H[gi2, xi2, yj2, ai2]
                                    h100 = H[gi2 + 1, xi2, yj2, ai2]
                                    h010 = H[gi2, xi2 + 1, yj2, ai2]
                                    h110 = H[gi2 + 1, xi2 + 1, yj2, ai2]
                                    h001 = H[gi2, xi2, yj2 + 1, ai2]
                                    h101 = H[gi2 + 1, xi2, yj2 + 1, ai2]
                                    h011 = H[gi2, xi2 + 1, yj2 + 1, ai2]
                                    h111 = H[gi2 + 1, xi2 + 1, yj2 + 1, ai2]

                                    h_yj = (h000 * (1.0 - xw) + h010 * xw) * (1.0 - yw) \
                                         + (h001 * (1.0 - xw) + h011 * xw) * yw
                                    h_yj1 = (h100 * (1.0 - xw) + h110 * xw) * (1.0 - yw) \
                                          + (h101 * (1.0 - xw) + h111 * xw) * yw
                                    h_val = h_yj * (1.0 - gw) + h_yj1 * gw

                                    h_sum += h_val * a0_width * prefac

                        f_tot += h_sum * sample_area

            output[out_idx] = f_tot / s ** 2

        return output

    _spectrum_kernel_4d_cache = spectrum_kernel_4d_cpu
    return spectrum_kernel_4d_cpu
