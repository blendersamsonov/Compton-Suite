"""The CGS "collision parameters" bundle for tabulated-overlap-style
GPU/CPU pipelines (``xigma_i``, ``delta``) -- :class:`CollisionParams`
(immutable, CGS, ``k0_las``-normalised scalars) plus :func:`build_params`,
the one function that derives it from ``compton_suite.io``'s own SI beam/laser/
geometry description.

Moved here from ``xigma_i/config.py`` (formerly the only consumer) so any
model built on this same CGS/``k0_las`` convention can reuse it without
re-deriving the same formulas -- the electron-side derivation (``beta_x``/
``beta_y``/``sigma_thx``/``sigma_thy``) already went through this move
first (see ``bunch.beta_star_from_sigma_emit``/``divergence_from_sigma_emit``);
this finishes the rest of the bundle.

``beta_ff``/``ellipticity`` are laser extras specific to this CGS/``k0_las``
convention (flying-focus factor, polarization ellipticity), not currently
consumed by any other model -- kept here as plain scalar fields on
``CollisionParams``, not on ``compton_suite.io.laser.GaussianParaxialLaser``
(which deliberately excludes them, see that module's own docstring), since
:func:`build_params` is their only constructor.

``a0``'s formula here has a known, unresolved ~49% relative discrepancy
against ``GaussianParaxialLaser.a0_focus`` (see ``validation/
tier0_wiring.py``'s ``check_a0_formula_agreement``) -- carried over
unreconciled: this is xigma's own convention, not yet reconciled with
``compton_suite.io.laser``'s. Not something to silently pick a side on via this
move.

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

from . import constants as _constants
from .bunch import GaussianElectronBeam
from .interaction import InteractionGeometry
from .laser import GaussianParaxialLaser
from .units import ureg

_Q = ureg.Quantity
_M_TO_CM = _constants.M_TO_CM
_C_CM_S = _constants.C_CM_S
_HBAR_ERG_S = _constants.HBAR_ERG_S
_R_E_CM = _constants.R_E_CM
_ALPHA = _constants.ALPHA

__all__ = ["CollisionParams", "build_params", "detect_device"]


def detect_device() -> str:
    """Auto-detect which backend to use: a real CUDA GPU via cupy if
    available, else CPU (requires numba). Raises if neither works -- there
    is no third backend."""
    if _HAS_CUPY:
        try:
            if cp.cuda.runtime.getDeviceCount() > 0:
                return 'gpu'
        except Exception:
            pass
    try:
        import numba  # noqa: F401
    except ImportError:
        pass
    else:
        return 'cpu'
    raise RuntimeError(
        "compton_suite.io.collision: no usable backend -- no CUDA-capable GPU "
        "detected (or cupy isn't installed), and numba isn't installed for "
        "the CPU fallback. Install numba (pip install numba) for CPU-only "
        "use, or a working cupy+CUDA setup for GPU use.")


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


def build_params(beam: GaussianElectronBeam, laser: GaussianParaxialLaser,
                  geometry: InteractionGeometry | None = None, *,
                  beta_ff: float = 0.0, ellipticity: float = 0.0,
                  device: str | None = None) -> CollisionParams:
    """Derive this convention's CGS :class:`CollisionParams` from
    ``compton_suite.io``'s SI beam/laser/geometry description -- the pipeline's
    only "interaction" step: one pure function call, not three ``set_*``
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
    geometry = geometry if geometry is not None else InteractionGeometry()

    # Extract raw SI floats from PhysicalQuantity-based beam/laser.
    sx_m = beam._sx_m
    sy_m = beam._sy_m
    wl_m = laser._wl_m
    wx_m = laser._wx_m
    dur_s = laser._dur_s
    energy_J = laser._E_J

    sigma_ex = sx_m * _M_TO_CM
    sigma_ey = sy_m * _M_TO_CM

    lambda_l = wl_m * _M_TO_CM
    # sigma_lr0: RMS radius of the *photon density* distribution (round-
    # beam approximation -- waist_rms_y_m is not separately modelled here,
    # matching every current consumer's own round-beam convention).
    sigma_lr0 = wx_m * _M_TO_CM
    sigma_lz = dur_s * _C_CM_S
    WL = _Q(energy_J, "joule").to("erg").magnitude  # pulse energy, erg
    omega_las = 2 * np.pi * _C_CM_S / lambda_l
    k0_las = omega_las / _C_CM_S
    Wph_erg = _HBAR_ERG_S * omega_las  # photon energy, erg
    Wph = _Q(Wph_erg, "erg").to("MeV").magnitude  # photon energy, MeV
    N_l = WL / Wph_erg
    a0 = (2 * _R_E_CM**2 * lambda_l / _ALPHA * N_l
          / (np.power(np.pi, 3 / 2) * sigma_lr0**2 * sigma_lz))

    return CollisionParams(
        N_e=beam.N_e, sigma_ex=sigma_ex, sigma_ey=sigma_ey,
        sigma_lr0=sigma_lr0, sigma_lz=sigma_lz,
        omega_las=omega_las, k0_las=k0_las, Wph=Wph, N_l=N_l, a0=a0,
        beta_ff=beta_ff, ellipticity=ellipticity,
        device=device,
    )
