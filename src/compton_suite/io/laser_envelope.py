"""Shared spatiotemporal Gaussian-pulse envelope -- the piece
``compton_suite.io.laser.GaussianParaxialLaser`` deliberately doesn't provide.

``laser.py``'s module docstring is explicit that v0.1's
``GaussianParaxialLaser`` is on-axis-peak-only: head-on, no transverse
profile, no time dependence, no crossing angle, no flying focus. Two
models each independently reimplemented the full spatiotemporal envelope
before this module existed -- ``kascade.laser_density``/``laser_a0sq`` (SI,
arbitrary crossing angle via ``laser_axis(cfg)``, arbitrary focus offset)
and ``xigma_i.particles``' inline ``n_ph_shape``/``sigma_l_sq``/``env``
construction (CGS, ``k0_las``-normalised, head-on, flying-focus
``beta_ff``). Verified algebraically to be the same formula at
``axis=(0,0,-1), focus=(0,0,0), beta_ff=0`` (see
``tests/io_tests/test_laser_envelope.py``), so this is the one shared
evaluator both now call into -- the same kind of move
``propagation.laser_overlap_time_window`` already made (that function was
originally ``xigma_i.particles._time_window``).
"""

from __future__ import annotations

import numpy as np

__all__ = ["gaussian_pulse_envelope"]


def gaussian_pulse_envelope(x, y, z, ct, *, sigma0, rayleigh_range, sigma_ct,
                             axis=(0.0, 0.0, -1.0), focus=(0.0, 0.0, 0.0),
                             beta_ff=0.0, xp=np):
    """Photon-density envelope of a Gaussian laser pulse at an arbitrary
    point in space and time, propagating along an arbitrary axis, with an
    optional flying-focus shift. Normalised so the density integrates to 1
    over all space (x, y, z) at any FIXED ``ct`` (photon number is
    conserved as the pulse translates through space -- integrating over
    ``ct`` too, on top of that, would double-count and diverge):

        u = (r - focus) . axis                # position along axis, from focus
        perp2 = |r - focus|^2 - u^2            # squared transverse offset from axis
        u_spot = u + beta_ff * ct              # flying-focus-shifted spot-size position
        sp2 = sigma0^2 * (1 + (u_spot / rayleigh_range)^2)
        norm = 1 / ((2*pi)^1.5 * sp2 * sigma_ct)
        density = norm * exp(-perp2/(2*sp2) - (u - ct)^2/(2*sigma_ct^2))

    x, y, z, ct: position and light-travel-time expressed as a LENGTH
        (``C_LIGHT * t`` in SI -- kascade's convention; or the already
        ``k0_las*c``-normalised time xigma_i's callers already carry, same
        convention ``propagation.laser_overlap_time_window`` documents).
        Any consistent length unit works (SI metres for kascade,
        ``k0_las``-normalised cm for xigma_i) -- this function has no
        embedded unit system. Passing a bare ``t`` in seconds here is a
        caller bug, not something this function can detect.
    sigma0: RMS transverse (photon-density) width at the waist, same
        length unit as x/y/z/ct. Round beam only (matches every current
        consumer's own round-beam convention, e.g.
        ``compton_suite.io.collision.build_params``'s ``sigma_lr0``) -- kascade's
        ``cfg.sigma0_l``, xigma_i's ``k0_las * params.sigma_lr0``.
    rayleigh_range: Rayleigh-range-like transverse-spreading scale, same
        length unit. Plain Rayleigh range at ``beta_ff=0`` (kascade's
        ``cfg.R_sf`` == ``GaussianParaxialLaser.rayleigh_x_m``); a caller
        using ``beta_ff != 0`` is responsible for pre-scaling this by its
        own flying-focus convention (xigma_i passes
        ``2 * w0**2 * (1 + beta_ff)``) -- this function applies no such
        scaling itself, since it's specific to xigma_i's flying-focus
        formalism, not part of the generic geometry here.
    sigma_ct: RMS pulse duration, expressed as a length (``C_LIGHT *
        duration_rms_s`` in SI; xigma_i's ``k0_las * params.sigma_lz``).
    axis: unit vector the pulse propagates along, default ``(0, 0, -1)``
        (head-on, counter-propagating against a ``+z`` electron beam --
        xigma_i's implicit convention, so its call site never needs to
        pass this). kascade passes ``laser_axis(cfg)`` =
        ``(sin(crossing_angle), 0, -cos(crossing_angle))``.
    focus: pulse-focus position ``(x, y, z)``, default ``(0, 0, 0)``.
        kascade passes ``(cfg.delta_x, cfg.delta_y, cfg.delta_z)``.
        xigma_i's ``CollisionParams.delta_x/y/z`` are NOT threaded through
        to this call -- nothing in ``xigma_i.particles``/``deposition``/
        ``config`` reads them back today, so leaving this at the default
        preserves xigma_i's actual current behavior rather than silently
        changing it as a side effect of this move; wiring them in is a
        separate, deliberate follow-up if ever wanted.
    beta_ff: flying-focus factor, default 0 (static focus). Enters ONLY
        the spot-size term (``u_spot``), not the longitudinal envelope
        (``(u - ct)^2`` stays beta_ff-independent) -- extracted verbatim
        from xigma_i's own ``zr_term = z - beta_ff*t`` construction
        (head-on: ``u = -z``, so ``u_spot = -z + beta_ff*ct = -(z -
        beta_ff*ct)``, squared -> matches exactly). An engine-specific
        extra (see ``compton_suite.io.laser``'s and ``compton_suite.io.collision``'s
        module docstrings on why it isn't part of ``GaussianParaxialLaser``
        itself), exposed here as a plain generic optional scalar -- same
        treatment ``propagation.laser_overlap_time_window`` already gives
        it. kascade's ``Config`` has no ``beta_ff`` field and never passes
        one, so its call is a strict no-op with respect to this parameter.
    xp: array module (``numpy`` or ``cupy``) -- needed for ``exp``/
        ``maximum``, not expressible as bare operators (same justification
        as ``propagation.laser_overlap_time_window``'s ``xp`` parameter).
        Uses ``xp.pi`` rather than a hardcoded constant so the function has
        no hard numpy dependency in its compute path.
    """
    fx, fy, fz = focus
    ax, ay, az = axis
    rx, ry, rz = x - fx, y - fy, z - fz
    u = rx * ax + ry * ay + rz * az
    perp2 = xp.maximum(rx * rx + ry * ry + rz * rz - u * u, 0.0)
    u_spot = u + beta_ff * ct
    sp2 = sigma0 ** 2 * (1.0 + (u_spot / rayleigh_range) ** 2)
    norm = 1.0 / ((2.0 * xp.pi) ** 1.5 * sp2 * sigma_ct)
    arg = -perp2 / (2.0 * sp2) - (u - ct) ** 2 / (2.0 * sigma_ct ** 2)
    return norm * xp.exp(arg)
