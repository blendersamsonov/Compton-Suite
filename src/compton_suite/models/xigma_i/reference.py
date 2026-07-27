"""Cross-validation-only tooling: a brute-force, non-GPU-kernel way to turn
a Stage 1 `H` table into a spectrum, used to validate
`spectrum4d.spectrum_kernel_4d` without trusting it.

**Not production code, and not imported by any production adapter.**
`angle_integrated_spectrum`/`direct_binning_spectrum` used to live in this
module too; they were extracted to `spectrum_from_particles.py` because
they turned out to be load-bearing for real model output (`xigma-i`'s
`TabulatedEngine.spectrum(s)`, `delta`'s total_yield/spectrum/
angular_spectrum) rather than pure validation, and this module is expected
to eventually leave this repo entirely (folded into a separate,
standalone cross-validation project) -- kept here for now for ad hoc Stage
2 cross-checks, not wired into anything that runs during a normal
simulation.

  - spectrum_from_table: brute-force grid quadrature over the H table (no
    annulus/arc/inverse-CDF importance sampling), for the per-solid-angle
    (theta_x, theta_y, s) spectrum. `coef = 1.5`, a pure numerical constant
    derived directly from eq. "main"/"Fmatrix" (in the accompanying paper)
    -- no pi, no Wph, no PHI_CELLS belongs here, since this is a plain grid
    quadrature with no phi cells and H's weights are already correctly
    CGS-normalised coming out of push_and_sample.

VALIDATED: agrees with Stage 0/1's own total weight (via
spectrum_from_particles.angle_integrated_spectrum) to 1-3%, and with
spectrum_from_particles.direct_binning_spectrum to <5% for a typical bunch.
a0/ahat resonance term is included (s_res = g**2/(1+a0+g**2*r_sq)), with no
extra Jacobian in the prefactor -- see spectrum_from_table's own docstring
for why that differs from spectrum_from_particles.direct_binning_spectrum.
"""
import numpy as np


def _xp_for(backend):
    """Resolves backend='numpy'|'cupy' to its array module. cupy is imported
    lazily so this module still imports fine without it installed unless
    backend='cupy' is actually requested -- same convention as
    particles.push_and_sample.
    """
    if backend == 'numpy':
        return np
    if backend == 'cupy':
        import cupy as cp
        return cp
    raise ValueError(f"backend must be 'numpy' or 'cupy', got {backend!r}")


def _interp4d(H, grid, gamma, theta_x, theta_y, a0, xp):
    """Array-module-agnostic core of interp4d: takes H already converted to
    xp's module (so callers looping over many query batches against the same
    table -- e.g. spectrum_from_table's loop over s -- transfer H once
    instead of once per call).
    """
    axes_edges = (grid.gamma_edges, grid.theta_x_edges, grid.theta_y_edges, grid.a0_edges)
    coords = [xp.asarray(c, dtype=xp.float64) for c in (gamma, theta_x, theta_y, a0)]
    shape = H.shape

    out_shape = np.broadcast_shapes(*(c.shape for c in coords))
    coords = [xp.broadcast_to(c, out_shape) for c in coords]

    f = xp.zeros(out_shape, dtype=xp.float64)
    in_range = xp.ones(out_shape, dtype=bool)

    i0s, ws = [], []
    for edges, x, n in zip(axes_edges, coords, shape):
        edges = xp.asarray(edges)
        width = edges[1] - edges[0]
        centers0 = edges[0] + 0.5 * width  # centre of first cell
        f_idx = (x - centers0) / width
        i0 = xp.floor(f_idx).astype(xp.int64)
        w = f_idx - i0
        in_range &= (i0 >= -1) & (i0 < n)  # allow i0==-1/n-1 edge cases below via clipping + zero-weight
        i0s.append(i0)
        ws.append(w)

    for dg in (0, 1):
        for dtx in (0, 1):
            for dty in (0, 1):
                for da in (0, 1):
                    idxs = []
                    corner_w = xp.ones(out_shape, dtype=xp.float64)
                    valid = xp.ones(out_shape, dtype=bool)
                    for (i0, w, d, n) in zip(i0s, ws, (dg, dtx, dty, da), shape):
                        ci = i0 + d
                        valid &= (ci >= 0) & (ci < n)
                        corner_w = corner_w * (w if d else (1 - w))
                        idxs.append(xp.clip(ci, 0, n - 1))
                    take = valid & in_range
                    if bool(xp.any(take)):
                        vals = H[idxs[0][take], idxs[1][take], idxs[2][take], idxs[3][take]]
                        f[take] += vals * corner_w[take]

    f[~in_range] = 0.0
    return f


def interp4d(table, gamma, theta_x, theta_y, a0, backend='numpy'):
    """Quadrilinear interpolation of table.H at query points (arrays of equal
    shape). Points outside the tabulated extent return 0.

    backend: 'numpy' (default) or 'cupy' -- table.H is always host (numpy),
    per deposition.Table's invariant, and is transferred to the target
    module here. For repeated calls against the same table (e.g. inside a
    loop), prefer calling _interp4d directly with a pre-transferred H
    instead of re-transferring on every call -- see spectrum_from_table.
    """
    xp = _xp_for(backend)
    H = xp.asarray(table.H)
    return _interp4d(H, table.grid, gamma, theta_x, theta_y, a0, xp)


def spectrum_from_table(table, x0, y0, s, phi_pol, backend='numpy'):
    """Brute-force quadrature of dN/(ds dOmega) at a single observation point
    (x0, y0) over a grid of frequencies s, integrating the table over its
    full (theta_x, theta_y, a0) extent at each s.

    x0, y0, s: floats / 1D array for s. Returns array matching s's shape.

    Resonance condition includes a0 (Paper/xigma.tex eq. "Gamma", section
    "Reduction to three dimensions"): g**2 = (1+a0) / (1/s - r_sq), each a0
    bin resonating at its own gamma, with a Jacobian factor 1/(1+a0) in the
    prefactor (eq. "jacobian", "Fmatrix"). An earlier version of this
    function used g**2 = 1/(1/s - r_sq) (a0-independent) with no 1/(1+a0)
    factor -- the same gap spectrum_kernel_4d had; see CLAUDE.md "Known
    bugs"/"Traps". Fixed alongside that kernel.

    coef = 3/2, a pure numerical constant from eq. "main"/"Fmatrix"'s own
    normalisation -- no pi, no Wph, no PHI_CELLS. `compton` is not part of
    this function's signature: it was only ever needed for `compton.Wph`.

    backend: 'numpy' (default) or 'cupy'. table.H is transferred to the
    target module once up front (not once per s, and not once per interp4d
    call inside the loop) -- see _interp4d's docstring for why that matters.
    The loop over s stays a host Python loop (s is typically a small grid);
    each iteration's (theta_x, theta_y, a0) quadrature -- the actual
    O(grid_size) work -- runs on the target device.
    """
    if table.a0_kind != 'ahat':
        raise ValueError(
            f"spectrum_from_table requires a physical-ahat table "
            f"(a0_kind='ahat'), got a0_kind={table.a0_kind!r} -- pass it "
            f"through deposition.retarget_a0(table, a0) for a specific a0 "
            f"first, see that function's docstring.")

    xp = _xp_for(backend)
    H = xp.asarray(table.H)

    theta_x_c, theta_y_c, a0_c = (xp.asarray(c) for c in
                                   (table.grid.centers[1], table.grid.centers[2], table.grid.centers[3]))
    TX, TY, A0 = xp.meshgrid(theta_x_c, theta_y_c, a0_c, indexing='ij')
    r_sq = (TX - x0)**2 + (TY - y0)**2

    dtx, dty, da = table.grid.widths[1], table.grid.widths[2], table.grid.widths[3]
    cell_vol = dtx * dty * da

    cos_pol = xp.cos(phi_pol - xp.arctan2(TY - y0, TX - x0))**2

    coef = 1.5  # pure numerical constant, eq. "main"/"Fmatrix" -- see docstring above

    s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
    out = np.zeros_like(s_arr)
    for k, sk in enumerate(s_arr):
        inv_base = 1.0 / sk - r_sq
        g_sq = xp.where(inv_base > 0, (1.0 + A0) / xp.where(inv_base > 0, inv_base, 1.0), -1.0)
        valid = g_sq >= 0
        if not bool(xp.any(valid)):
            continue
        g = xp.sqrt(xp.where(valid, g_sq, 0.0))
        gth_sq_inv = 1.0 / (1.0 + r_sq * g_sq)**2
        a_fac = 1.0 - 4.0 * cos_pol * r_sq * g_sq * gth_sq_inv
        prefac = a_fac * g**5 * gth_sq_inv / (1.0 + A0)

        H_val = _interp4d(H, table.grid, g, TX, TY, A0, xp)

        f = xp.where(valid, H_val * prefac, 0.0)
        out[k] = float(coef * f.sum() * cell_vol / sk**2)

    return out if np.ndim(s) else out[0]
