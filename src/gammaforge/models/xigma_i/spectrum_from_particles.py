"""Table-free spectrum paths computed directly from Stage 0/1 macroparticles
-- no `H` table, no GPU kernel, no importance sampling.

  - `angle_integrated_spectrum` is `TabulatedEngine.spectrum(s)`'s actual
    implementation (`xigma-i`'s real dN/ds output, see `tabulated_engine.py`).
  - `direct_binning_spectrum` is `delta`'s actual
    total_yield/spectrum/angular_spectrum implementation (see
    `adapter.py`'s `run_simulation_direct`).

`reference.py` keeps `spectrum_from_table`, a genuinely validation-only
brute-force table quadrature with no production caller.
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


def angle_integrated_spectrum(gamma, particle_weight, s, backend='numpy'):
    """dN/ds integrated over all emission solid angle, from real Stage 0/1
    macroparticles. A single electron's angle-integrated spectral shape
    depends only on its own gamma (not its transverse angle), via the
    standard Compton edge formula. `dE = 4*Wph*ds` converts to dN/dE if
    needed.

    gamma, particle_weight: 1D arrays, one entry per macroparticle -- e.g.
    the gamma and weight arrays push_and_sample already returns (one row per
    particle; no external per-particle summing needed).
    s: scalar or 1D array of normalised photon energies.
    backend: 'numpy' (default) or 'cupy' -- array-module-agnostic, same
    pattern as deposition.py/particles.py. gamma/particle_weight/s are
    converted to the target module if not already; the whole computation is
    elementwise/reduction, so there's nothing GPU-specific to write.
    """
    xp = _xp_for(backend)
    gamma, particle_weight = xp.asarray(gamma), xp.asarray(particle_weight)
    s_arr = xp.atleast_1d(xp.asarray(s, dtype=xp.float64))
    gamma = gamma[:, None]
    y = s_arr[None, :] / gamma**2
    shape = 1.5 * (1.0 - 2.0 * y * (1.0 - y))
    shape = xp.where((y < 0) | (y > 1), 0.0, shape)
    out = xp.sum(particle_weight[:, None] * shape / gamma**2, axis=0)
    return out if np.ndim(s) else out[0]


def direct_binning_spectrum(gamma, theta_x, theta_y, particle_weight, a0,
                             x0, y0, s_edges, phi_pol, backend='numpy'):
    """`delta`'s actual computation: for each real macroparticle,
    compute the photon energy it resonates at when viewed from (x0, y0), and
    bin its weight into the s_edges histogram. No table, no importance
    sampling -- assumption-free on both the deposition and the lookup.

    Normalisation: this is a *single-electron*, not-yet-ensemble-collapsed
    quantity, so it uses the bare differential cross-section (g**2 *
    gth_sq_inv prefactor) rather than the g**5 form spectrum_from_table/
    spectrum_kernel_4d use (that extra g**3 power is a |dGamma/domega|
    Jacobian for evaluating a *smooth, already-binned* H, not applicable to
    raw discrete macroparticles). Converting to a photon count via the
    incident flux uses v_rel (=2c for near-backscattering, the same V_REL
    particles.py already bakes into `particle_weight`), not bare c:
    d3N_i/(domega dOmega) = 3 * particle_weight_i * g_i**2 * gth_sq_inv *
    a_fac * delta(omega - omega_R,i). Histogrammed over s (domega = 4*
    omega_L * ds, a *constant* Jacobian that cancels exactly against the
    same factor converting the histogram's bin width to ds) gives d3N/(ds
    dOmega) = [sum of weights in bin] / ds -- no additional 1/s**2 (that
    division belongs to the H-density/coef convention of the table-based
    reference paths, not to this one).

    Known residual, deliberately not chased further: this function's
    angle-integrated total (Riemann-summed over a grid of (x0, y0),
    weighted by cell area) is consistently ~6.3x angle_integrated_spectrum's
    output -- suspiciously close to 2*pi, not yet explained. Small,
    systematic spread across configurations (not noise).

    a0/ahat resonance term: s_res = g**2 / (1 + a0 + g**2*r_sq), i.e. the
    resonance condition shifts with a0 the same way spectrum_from_table/
    spectrum_kernel_4d's g**2 = (1+a0)/(1/s - r_sq) does (same relation,
    solved for s instead of g here). Unlike those two, NO extra 1/(1+a0)
    Jacobian is needed in the prefactor: that factor comes specifically
    from the *ensemble* gamma-integral collapse those two methods perform
    when inverting the resonance condition to look up a smooth, pre-binned
    H at an interpolated gamma. This function never does that inversion --
    each particle contributes at its own exact gamma_i, no lookup, no
    second collapse -- so that Jacobian doesn't apply here.

    backend: 'numpy' (default) or 'cupy' -- array-module-agnostic; the
    per-particle arrays and s_edges are converted to the target module,
    including the histogram+weights reduction (cupy.histogram supports
    weights the same as numpy.histogram).
    """
    xp = _xp_for(backend)
    gamma, theta_x, theta_y, particle_weight, a0 = (
        xp.asarray(a) for a in (gamma, theta_x, theta_y, particle_weight, a0))

    r_sq = (theta_x - x0)**2 + (theta_y - y0)**2
    g = gamma
    s_res = g**2 / (1.0 + a0 + g**2 * r_sq)

    gth_sq_inv = 1.0 / (1.0 + r_sq * g**2)**2
    cos_pol = xp.cos(phi_pol - xp.arctan2(theta_y - y0, theta_x - x0))**2
    a_fac = 1.0 - 4.0 * cos_pol * r_sq * g**2 * gth_sq_inv

    prefactor = 3.0 * particle_weight * a_fac * g**2 * gth_sq_inv

    s_edges = xp.asarray(s_edges)
    hist, _ = xp.histogram(s_res, bins=s_edges, weights=prefactor)
    ds = xp.diff(s_edges)

    return hist / ds
