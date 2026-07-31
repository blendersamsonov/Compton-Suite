"""The CGS "collision parameters" bundle for this package's tabulated-
overlap-style GPU/CPU pipelines -- :class:`CollisionParams` (immutable,
CGS, ``k0_las``-normalised scalars) plus :func:`build_params`, the one
function that derives it from ``compton_suite.io``'s own SI beam/laser
description.

Lives here (not in ``compton_suite.io``) because it's this package's own
CGS/``k0_las`` convention, not a cross-model shared representation --
kascade works directly in SI and has no use for it (see
``compton_suite.io.interaction``'s module docstring: every model decides
its own unit system/convention at its own boundary).

``beta_ff``/``ellipticity`` are laser extras specific to this CGS/``k0_las``
convention (flying-focus factor, polarization ellipticity), not currently
consumed by any other model -- kept here as plain scalar fields on
``CollisionParams``, not on ``compton_suite.io.laser.GaussianParaxialLaser``
(which deliberately excludes them, see that module's own docstring), since
:func:`build_params` is their only constructor.

``a0``'s formula here has a known, unresolved ~49% relative discrepancy
against ``GaussianParaxialLaser.a0_focus`` (see ``validation/
tier0_wiring.py``'s ``check_a0_formula_agreement``) -- xigma's own
convention, not yet reconciled with ``compton_suite.io.laser``'s.

Runs no compute of its own: ``.xp``/``.asnumpy`` are a thin numpy/cupy-
selection convenience for host orchestration code that builds arrays on
the chosen device and needs to bring results back to host afterwards --
cupy stays optional here (imported lazily/guarded, exactly like every
other optional-GPU spot in this codebase), so importing this module never
requires cupy to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False

from compton_suite.io.bunch import GaussianElectronBeam
from compton_suite.io.laser import GaussianParaxialLaser
from compton_suite.io.units import ALPHA, C_CM_S, HBAR_ERG_S, M_TO_CM, R_E_CM, ureg
from compton_suite.misc import detect_device

_Q = ureg.Quantity

__all__ = ["CollisionParams", "build_params", "detect_device"]


@dataclass(frozen=True)
class CollisionParams:
    """CGS scalars for one laser-electron collision -- immutable, built
    once by :func:`build_params`. Not a stateful object; runs no compute of
    its own."""

    # Electron parameters
    N_e: float
    sigma_ex: float
    sigma_ey: float

    # Laser parameters
    sigma_lr0: float
    sigma_lz: float
    omega_las: float
    k0_las: float
    Wph: float
    N_l: float
    a0: float
    beta_ff: float
    ellipticity: float

    device: str = 'cpu'

    @property
    def xp(self):
        return cp if self.device == 'gpu' else np

    def asnumpy(self, x):
        return x.get() if self.device == 'gpu' else x


def build_params(beam: GaussianElectronBeam, laser: GaussianParaxialLaser, *,
                  beta_ff: float = 0.0, ellipticity: float = 0.0,
                  device: str | None = None) -> CollisionParams:
    """Derive this convention's CGS :class:`CollisionParams` from
    ``compton_suite.io``'s SI beam/laser description -- the pipeline's only
    "interaction" step: one pure function call, not three ``set_*``
    mutations on a persistent object. ``beta_ff``/``ellipticity`` are
    extras specific to this convention, with no shared-representation
    analogue (see ``compton_suite.io.laser``'s module docstring), passed as
    plain scalars.
    """
    device = device or detect_device()
    if device == 'gpu' and not _HAS_CUPY:
        raise RuntimeError("build_params(device='gpu') requested but cupy is not importable")
    if device not in ('gpu', 'cpu'):
        raise ValueError(f"device must be 'gpu', 'cpu', or None, got {device!r}")

    # Extract raw SI floats from PhysicalQuantity-based beam/laser.
    sx_m = beam._sx_m
    sy_m = beam._sy_m
    wl_m = laser._wl_m
    wx_m = laser._wx_m
    dur_s = laser._dur_s

    sigma_ex = sx_m * M_TO_CM
    sigma_ey = sy_m * M_TO_CM

    lambda_l = wl_m * M_TO_CM
    # sigma_lr0: RMS radius of the *photon density* distribution (round-
    # beam approximation -- waist_rms_y_m is not separately modelled here,
    # matching every current consumer's own round-beam convention).
    sigma_lr0 = wx_m * M_TO_CM
    sigma_lz = dur_s * C_CM_S
    omega_las = 2 * np.pi * C_CM_S / lambda_l
    k0_las = omega_las / C_CM_S
    Wph_erg = HBAR_ERG_S * omega_las  # photon energy, erg
    Wph = _Q(Wph_erg, "erg").to("MeV").magnitude  # photon energy, MeV
    # N_l/a0 are model-agnostic and come directly from the shared
    # GaussianParaxialLaser input -- not re-derived here in CGS. This is
    # the single a0/N_l formula (GaussianParaxialLaser.a0_focus/n_photons,
    # SI plane-wave derivation); no independent CGS formula exists to
    # disagree with it anymore.
    N_l = laser.n_photons
    a0 = laser.a0_focus

    return CollisionParams(
        N_e=beam.N_e, sigma_ex=sigma_ex, sigma_ey=sigma_ey,
        sigma_lr0=sigma_lr0, sigma_lz=sigma_lz,
        omega_las=omega_las, k0_las=k0_las, Wph=Wph, N_l=N_l, a0=a0,
        beta_ff=beta_ff, ellipticity=ellipticity,
        device=device,
    )
