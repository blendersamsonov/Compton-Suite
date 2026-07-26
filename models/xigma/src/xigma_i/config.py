"""Shared physics constants, GPU kernel sizing constants, and
`CollisionParams` -- the CGS scalars every stage of the pipeline
(particles.py, deposition.py, spectrum4d.py/spectrum4d_cpu.py, reference.py,
tabulated_engine.py, gui_adapter.py) needs for one laser-electron collision.

This model owns no persistent representation of "the interaction" --
`CollisionParams` is a plain, immutable dataclass built in one call by
`build_params()` from `compton_io`'s own beam/laser/geometry description
(`compton_io.bunch.GaussianElectronBeam`, `compton_io.laser.
GaussianParaxialLaser`, `compton_io.interaction.InteractionGeometry`), not
a stateful object accumulated via `set_*` calls. `particles.push_and_sample`
takes a `CollisionParams` instance as its parameter source; `CollisionParams`
itself computes nothing beyond what `build_params` derives once, and runs
no GPU kernels -- `TabulatedEngine` (tabulated_engine.py) and the functions
in `reference.py`/`spectrum4d.py` do that.

FUTURE: `build_params`'s electron-side derivation (`beta_x`/`beta_y`/
`sigma_thx`/`sigma_thy`) is arithmetically identical to
`GaussianElectronBeam.beta_star_x_m`/`beta_star_y_m`/`divergence_x_rad`/
`divergence_y_rad` (already CGS-convertible, just needs `*100`) -- and the
whole function's *shape* (SI beam/laser -> a pipeline's own unit-converted
scalar bundle) has no xigma-specific reasoning beyond the CGS/`k0_las`
convention and the `a0` formula (which has a known, separately-flagged
~49% discrepancy against `GaussianParaxialLaser.a0_focus`, see
`validation/tier0_wiring.py` -- not something to silently reconcile here).
This whole function belongs in `compton_io` as a model-agnostic helper
once that's sorted out, so other models can reuse it too -- noted, not
done in this pass.

Units are CGS throughout; lengths and times inside `spectrum_kernel_4d`
(spectrum4d.py) are normalised to the laser wavenumber `k0_las`: positions
are `k0_las * x`, times are `k0_las * c * t`. `CollisionParams.device`/`.xp`/
`.asnumpy` are a thin `numpy`/`cupy`-selection convenience for host
orchestration code (gui_adapter.py) that builds `theta_x`/`theta_y`/`s`
arrays on the chosen device and needs to bring results back to host
afterwards -- `CollisionParams` runs no GPU kernels itself, so `cupy` is
only needed here for `.xp`/`.asnumpy` to work when `device='gpu'`, and is
therefore imported lazily/optionally like everywhere else in this package.

`hbar`/`me`/`c`/`el`/`elC` below come from `compton_io.constants` rather
than local literals -- the single shared source of truth also used by
`compton_guide`/`kascade`, see `compton_io`'s own CLAUDE.md.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.special import erfcx

try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False

from compton_io import constants as _io_constants
from compton_io.bunch import GaussianElectronBeam
from compton_io.interaction import InteractionGeometry
from compton_io.laser import GaussianParaxialLaser

hbar = _io_constants.HBAR_ERG_S      # Planck's constant, erg*s
me = _io_constants.ME_G              # electron mass, g
c = _io_constants.C_CM_S             # speed of light, cm/s
el = _io_constants.EL_STATC          # electron charge, statcoulombs
elC = _io_constants.E_CHARGE         # electron charge, Coulombs
rel = el ** 2 / (me * c ** 2)        # classical electron radius
sigma_T = 8.0 * np.pi / 3.0 * rel**2 # Thomson cross section
alpha = el ** 2 / (hbar * c)         # fine structure constant
PHI = 1.618033988749894848           # golden ratio

_M_TO_CM = 100.0

SINGLE_PRECISION = True

# np.* rather than cp.* -- cupy reuses numpy's dtype scalars, so this is
# behavior-neutral and avoids depending on cupy for these sizing/dtype
# constants specifically (spectrum4d_cpu.py's CPU kernel must not require
# cupy to import).
CP_FLOAT  = np.float32 if SINGLE_PRECISION else np.float64
CP_UINT   = np.uint32  if SINGLE_PRECISION else np.uint64
CP_INT    = np.int32   if SINGLE_PRECISION else np.int64
CP_PI     = CP_FLOAT(np.pi)
CP_TWO_PI = CP_FLOAT(2.0 * np.pi)
CP_ONE    = CP_FLOAT(1.0)
CP_ZERO   = CP_FLOAT(0.0)

# spectrum_kernel_4d (spectrum4d.py) launch geometry and shared-memory
# sizing -- interdependent, see spectrum4d.py's/spectrum4d_cpu.py's module
# docstrings ("Sizing constants" convention): MAX_ARCS derives from
# MAX_RINGS, WEIGHTS_SIZE/CDF_SIZE/THREAD_STRIDE from MAX_RINGS/PHI_EDGES/
# SAMPLES_TOTAL/X_THREADS. Change the primitives, not the derived values.
X_THREADS = 128

N_RINGS_MIN = 32
MAX_RINGS = 32
MAX_ARCS = 4 * MAX_RINGS
ARC_STRIDE = 3
RING_STRIDE = 9
RINGS_SIZE = CP_UINT(RING_STRIDE * MAX_RINGS)
INVAL = CP_FLOAT(9999.)

PHI_EDGES = 32
PHI_CELLS = PHI_EDGES - 1
CUM_WEIGHTS_SIZE = MAX_ARCS * PHI_EDGES

CDF_PHI_RESOLUTION = 32
CDF_PHI_REPEAT = ( CDF_PHI_RESOLUTION + X_THREADS - 1 ) // X_THREADS
CDF_SIZE = CDF_PHI_RESOLUTION * MAX_ARCS

SAMPLES_TOTAL = 256
SAMPLES_REPEAT = ( SAMPLES_TOTAL + X_THREADS - 1) // X_THREADS
THREAD_STRIDE = 3 * SAMPLES_REPEAT + 1

R_MAX_NUDGE = 128

# particles.py's push_and_sample: how many pulse-duration Gaussian widths /
# Rayleigh-range Lorentzian widths out a particle's trajectory is
# considered "possibly inside the pulse" (compton_io.propagation.
# laser_overlap_time_window's gauss_width/lorentz_width).
GAUSS_WIDTH   = CP_FLOAT(3)
LORENTZ_WIDTH = CP_FLOAT(8)


def _detect_device():
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
        "xigma_i: no usable backend -- no CUDA-capable GPU detected (or "
        "cupy isn't installed), and numba isn't installed for the CPU "
        "fallback. Install numba (pip install numba) for CPU-only use, or "
        "a working cupy+CUDA setup for GPU use.")


@dataclass(frozen=True)
class CollisionParams:
    """CGS scalars for one laser-electron collision -- immutable, built
    once by :func:`build_params`. Not a stateful object; runs no compute
    of its own beyond the two cheap analytic estimates below."""

    # Electron parameters
    N_e: float
    emit_x: float
    emit_y: float
    sigma_ex: float
    sigma_ey: float
    sigma_ez: float
    beta_x: float
    beta_y: float
    sigma_thx: float
    sigma_thy: float

    # Laser parameters
    lambda_l: float
    sigma_lr0: float
    sigma_lz: float
    omega_las: float
    k0_las: float
    Wph: float
    N_l: float
    a0: float
    beta_ff: float
    ellipticity: float

    # Foci displacement
    delta_x: float = 0.0
    delta_y: float = 0.0
    delta_z: float = 0.0

    device: str = 'cpu'

    @property
    def xp(self):
        return cp if self.device == 'gpu' else np

    def asnumpy(self, x):
        return x.get() if self.device == 'gpu' else x

    def estimate_yield(self):
        """Cheap analytic estimate, for sanity-checking against the real
        (Stage 0/1/2) computation -- not used by it."""
        sb_av = np.sqrt(self.sigma_ex * self.sigma_ey / self.beta_x / self.beta_y)
        sigma0 = np.sqrt(self.sigma_ex**2 + self.sigma_lr0**2)
        nu = np.sqrt(2) * sigma0 / np.sqrt(self.sigma_ez**2 + self.sigma_lz**2) / np.sqrt(sb_av**2 + self.lambda_l**2 / np.pi**2 / self.sigma_lr0**2)
        return self.N_e * self.N_l * sigma_T / 2 / np.sqrt(np.pi) / sigma0**2 * nu * erfcx(nu)

    def estimate_spectrum_width(self, gamma0, sigma_gamma, theta_col):
        """Cheap analytic estimate, for sanity-checking -- not used by the
        real computation."""
        emit_width = np.sqrt(self.sigma_thx*self.sigma_thy)
        return 0.5*2.355*np.sqrt((gamma0 * theta_col)**4 + (gamma0 * emit_width)**4 + (sigma_gamma / gamma0)**2 + (0.5*self.a0**2)**2)


def build_params(beam: GaussianElectronBeam, laser: GaussianParaxialLaser,
                  geometry: InteractionGeometry | None = None, *,
                  beta_ff: float = 0.0, ellipticity: float = 0.0,
                  device: str | None = None) -> CollisionParams:
    """Derive this pipeline's CGS :class:`CollisionParams` from
    ``compton_io``'s SI beam/laser/geometry description -- the model's only
    "interaction" step: one pure function call, not three ``set_*``
    mutations on a persistent object. ``beta_ff``/``ellipticity`` are
    xigma_i-only laser extras with no shared-representation analogue (see
    ``compton_io.laser``'s module docstring), passed as plain scalars.
    """
    device = device or _detect_device()
    if device == 'gpu' and not _HAS_CUPY:
        raise RuntimeError("build_params(device='gpu') requested but cupy is not importable")
    if device not in ('gpu', 'cpu'):
        raise ValueError(f"device must be 'gpu', 'cpu', or None, got {device!r}")
    geometry = geometry if geometry is not None else InteractionGeometry()

    emit_x = beam.emit_geom_x_m * _M_TO_CM
    emit_y = beam.emit_geom_y_m * _M_TO_CM
    sigma_ex = beam.sigma_x_m * _M_TO_CM
    sigma_ey = beam.sigma_y_m * _M_TO_CM
    sigma_ez = beam.sigma_z_m * _M_TO_CM
    beta_x = sigma_ex**2 / emit_x
    beta_y = sigma_ey**2 / emit_y
    sigma_thx = emit_x / sigma_ex
    sigma_thy = emit_y / sigma_ey

    lambda_l = laser.wavelength_m * _M_TO_CM
    # sigma_lr0: RMS radius of the *photon density* distribution (round-
    # beam approximation -- waist_rms_y_m is not separately modelled here,
    # matching every current consumer's own round-beam convention).
    sigma_lr0 = laser.waist_rms_x_m * _M_TO_CM
    sigma_lz = laser.duration_rms_s * _io_constants.C_LIGHT * _M_TO_CM
    WL = laser.pulse_energy_J * 1e7  # pulse energy, erg
    omega_las = 2 * np.pi * c / lambda_l
    k0_las = omega_las / c
    Wph_erg = hbar * omega_las  # photon energy, erg
    Wph = Wph_erg * 1e-6 / (elC * 1e7)  # photon energy, MeV
    N_l = WL / Wph_erg
    a0 = 4 * rel**2 * lambda_l / alpha * N_l / (np.power(np.pi, 3 / 2) * sigma_lr0**2 * sigma_lz)

    return CollisionParams(
        N_e=beam.N_e, emit_x=emit_x, emit_y=emit_y,
        sigma_ex=sigma_ex, sigma_ey=sigma_ey, sigma_ez=sigma_ez,
        beta_x=beta_x, beta_y=beta_y, sigma_thx=sigma_thx, sigma_thy=sigma_thy,
        lambda_l=lambda_l, sigma_lr0=sigma_lr0, sigma_lz=sigma_lz,
        omega_las=omega_las, k0_las=k0_las, Wph=Wph, N_l=N_l, a0=a0,
        beta_ff=beta_ff, ellipticity=ellipticity,
        delta_x=geometry.delta_x_m * _M_TO_CM, delta_y=geometry.delta_y_m * _M_TO_CM,
        delta_z=geometry.delta_z_m * _M_TO_CM,
        device=device,
    )
