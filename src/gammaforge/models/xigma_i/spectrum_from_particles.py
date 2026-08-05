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

from gammaforge.misc import available_ram_bytes, available_vram_bytes

# angle_integrated_spectrum's OOM-retry bound, mirroring particles.py's
# push_and_sample chunking: halve the remaining s-chunk on an
# OutOfMemoryError (cupy) or MemoryError (numpy), up to this many times,
# before giving up and re-raising.
_MAX_OOM_HALVINGS = 6

# Calibration for _estimate_s_chunk: the (n_particles, chunk)-shaped
# broadcast in angle_integrated_spectrum (y, shape, and the final
# particle_weight*shape/gamma**2 product) keeps ~3 float64 temporaries of
# that shape concurrently live; rounded up for headroom against other GPU
# processes / cupy-version drift, same reasoning as particles.py's
# _BYTES_PER_PARTICLE_STEP. Used for both backends -- the broadcast shape
# (and so the temporaries' byte cost) doesn't depend on which array module
# holds it.
_BYTES_PER_PARTICLE_S_PAIR = 60


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


def _estimate_s_chunk(n_particles, n_s, backend, safety_frac=None):
    """Auto-sized s-axis chunk for angle_integrated_spectrum, from currently
    free memory on whichever backend is in play -- VRAM (cupy, via
    gammaforge.misc.available_vram_bytes) or system RAM (numpy, via
    gammaforge.misc.available_ram_bytes). The particle axis (n_particles,
    can be several million) is never chunked (gamma/particle_weight are
    cheap, O(n) 1D arrays); only the (n_particles, chunk)-shaped broadcast
    temporaries need bounding.

    Both backends are chunked, not just cupy: "plain numpy is bounded by
    system RAM, so it doesn't need chunking" is only true if that RAM
    budget is actually respected -- a 5,000,000-particle x 1,024-point
    broadcast is over 100GB with this function's handful of live
    temporaries, enough to exhaust RAM (and, via the Linux OOM-killer or
    heavy swapping, crash the machine outright -- not a catchable
    exception the way a cupy OutOfMemoryError is) on a well-provisioned
    workstation. See available_ram_bytes' docstring.

    Returns n_s unchanged (don't chunk) only if the relevant free-memory
    query itself fails (no GPU/cupy usable for 'cupy'; unsupported
    platform for 'numpy').

    safety_frac: fraction of free memory to budget for. Defaults to 0.7 for
    'cupy' (an OutOfMemoryError there is a contained, catchable failure)
    and a more conservative 0.5 for 'numpy' (getting this wrong risks the
    OOM-killer/swapping failure mode above, a much larger blast radius).
    """
    if backend == 'cupy':
        free_bytes = available_vram_bytes()
        frac = 0.7 if safety_frac is None else safety_frac
    elif backend == 'numpy':
        free_bytes = available_ram_bytes()
        frac = 0.5 if safety_frac is None else safety_frac
    else:
        raise ValueError(f"backend must be 'numpy' or 'cupy', got {backend!r}")
    if free_bytes is None:
        return n_s
    budget = free_bytes * frac
    chunk = int(budget / (_BYTES_PER_PARTICLE_S_PAIR * max(n_particles, 1)))
    return max(1, min(chunk, n_s))


def angle_integrated_spectrum(gamma, particle_weight, s, backend='numpy', chunk=None):
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
    chunk: if given (and less than len(s)), the (n_particles, len(s))
    broadcast runs on `chunk`-sized slices of `s` instead of all at once,
    writing into the matching slice of the output -- each s value's result
    depends on no other s value, so this is an exact repartition, not an
    approximation. None (default) auto-sizes from free VRAM (cupy) or free
    system RAM (numpy) via _estimate_s_chunk -- both backends are chunked,
    see that function's docstring for why 'numpy' is not exempt. Needed
    because unlike push_and_sample's Stage 0 trajectory integration, this
    reduction has no chunking of its own: n_particles can be several
    million while s is only a few hundred points, so the naive
    (n_particles, len(s)) broadcast can be tens to hundreds of GB in one
    allocation (see CLAUDE.md CUDA/RAM-OOM notes).
    """
    xp = _xp_for(backend)
    gamma, particle_weight = xp.asarray(gamma), xp.asarray(particle_weight)
    s_arr = xp.atleast_1d(xp.asarray(s, dtype=xp.float64))
    gamma2 = (gamma ** 2)[:, None]

    n_particles = gamma.shape[0]
    n_s = s_arr.shape[0]
    if chunk is None:
        chunk = _estimate_s_chunk(n_particles, n_s, backend)
    chunk = max(1, min(int(chunk), n_s))

    is_cupy = xp is not np
    # MemoryError on the numpy path is the same halve-and-retry safety net
    # as cupy's OutOfMemoryError -- a secondary defense (Linux overcommit
    # means a too-large host allocation can be killed by the OOM-killer
    # before ever raising MemoryError, so _estimate_s_chunk's proactive
    # sizing above is the primary one, not this).
    oom_exc = xp.cuda.memory.OutOfMemoryError if is_cupy else MemoryError

    out = xp.empty(n_s, dtype=xp.float64)
    start = 0
    cur_chunk = chunk
    halvings = 0
    while start < n_s:
        end = min(start + cur_chunk, n_s)
        try:
            y = s_arr[None, start:end] / gamma2
            shape = 1.5 * (1.0 - 2.0 * y * (1.0 - y))
            shape = xp.where((y < 0) | (y > 1), 0.0, shape)
            out[start:end] = xp.sum(particle_weight[:, None] * shape / gamma2, axis=0)
        except oom_exc:
            if halvings >= _MAX_OOM_HALVINGS or cur_chunk <= 1:
                raise
            cur_chunk = max(1, cur_chunk // 2)
            halvings += 1
            xp.get_default_memory_pool().free_all_blocks()
            continue
        if is_cupy:
            del y, shape
            xp.get_default_memory_pool().free_all_blocks()
        start = end

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
