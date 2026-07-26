"""Shared physics constants, GPU kernel sizing constants, and the `Compton`
collision-configuration class -- used by every stage of the pipeline
(particles.py, deposition.py, spectrum4d.py/spectrum4d_cpu.py, reference.py,
tabulated_engine.py, gui_adapter.py).

`Compton` holds a laser-electron collision's physical parameters
(`set_electron_parameters`/`set_laser_parameters`/`set_foci_displacement`)
and the quantities derived from them (`k0_las`, `Wph`, `a0`, `N_e`, `N_l`,
...); `particles.push_and_sample` takes an instance of it as its parameter
source. It is a plain config object -- it does not compute
a spectrum itself; `TabulatedEngine` (tabulated_engine.py) and the
functions in `reference.py`/`spectrum4d.py` do that.

Units are CGS throughout; lengths and times inside `spectrum_kernel_4d`
(spectrum4d.py) are normalised to the laser wavenumber `k0_las`: positions
are `k0_las * x`, times are `k0_las * c * t`. `Compton.device`/`.xp`/
`.asnumpy` are a thin `numpy`/`cupy`-selection convenience for host
orchestration code (gui_adapter.py) that builds `theta_x`/`theta_y`/`s`
arrays on the chosen device and needs to bring results back to host
afterwards -- `Compton` itself runs no GPU kernels, so `cupy` is only
needed here for `.xp`/`.asnumpy` to work when `device='gpu'`, and is
therefore imported lazily/optionally like everywhere else in this package.

`hbar`/`me`/`c`/`el`/`elC` below come from `compton_io.constants` rather
than local literals -- the single shared source of truth also used by
`compton_guide`/`kascade`, see `compton_io`'s own CLAUDE.md.
"""
import numpy as np
from scipy.special import erfcx

try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False

from compton_io import constants as _io_constants

hbar = _io_constants.HBAR_ERG_S      # Planck's constant, erg*s
me = _io_constants.ME_G              # electron mass, g
c = _io_constants.C_CM_S             # speed of light, cm/s
el = _io_constants.EL_STATC          # electron charge, statcoulombs
elC = _io_constants.E_CHARGE         # electron charge, Coulombs
rel = el ** 2 / (me * c ** 2)        # classical electron radius
sigma_T = 8.0 * np.pi / 3.0 * rel**2 # Thomson cross section
alpha = el ** 2 / (hbar * c)         # fine structure constant
PHI = 1.618033988749894848           # golden ratio

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

# particles.py's _time_window: how many pulse-duration Gaussian widths /
# Rayleigh-range Lorentzian widths out a particle's trajectory is
# considered "possibly inside the pulse".
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


class Compton:
    # Electron parameters
    chargeNC = None
    emit_x = None
    emit_y = None
    sigma_ex = None
    sigma_ey = None
    sigma_ez = None
    N_e = None

    # Laser parameters
    WL = None
    lambda_l = None
    sigma_lr0 = None
    sigma_lz = None
    omega_las = None
    Wph = None

    # Foci displacement
    delta_x = 0.0
    delta_y = 0.0
    delta_z = 0.0

    device = None

    def __init__(self, device=None):
        """device: 'gpu' or 'cpu', or None to auto-detect (real CUDA GPU
        via cupy if available, else CPU/numba -- see _detect_device)."""
        self.device = device or _detect_device()
        if self.device == 'gpu' and not _HAS_CUPY:
            raise RuntimeError("Compton(device='gpu') requested but cupy is not importable")
        if self.device not in ('gpu', 'cpu'):
            raise ValueError(f"device must be 'gpu', 'cpu', or None, got {device!r}")

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

    def set_electron_parameters(self, chargeNC, emit_x, emit_y, sigma_ex, sigma_ey, sigma_ez):
        self.chargeNC = chargeNC
        self.emit_x = emit_x
        self.emit_y = emit_y
        self.sigma_ex = sigma_ex
        self.sigma_ey = sigma_ey
        self.sigma_ez = sigma_ez
        self.N_e = self.chargeNC * 1e-9 / elC

        self.beta_x = self.sigma_ex**2 / self.emit_x
        self.beta_y = self.sigma_ey**2 / self.emit_y
        self.sigma_thx = self.emit_x / self.sigma_ex
        self.sigma_thy = self.emit_y / self.sigma_ey

    def set_laser_parameters(self, WL, lambda_l, sigma_lr0, sigma_lz, beta_ff = 0.0, ellipticity = 0.0):
        self.WL = WL * 1e7  # pulse energy, erg
        self.lambda_l = lambda_l
        self.beta_ff = beta_ff
        self.ellipticity = ellipticity  # laser polarisation ellipticity; 0 = linear, +-1 = circular. Used by particles.push_and_sample's TrXi/2 = (1+ellipticity**2)/2 (see CLAUDE.md).
        self.sigma_lr0 = sigma_lr0  # NOTE: this is the RMS radius of the *photon density* distribution. The corresponding Rayleigh range is 2 * sigma_lr0**2 * omega (compare to sigma**2 * omega / 2 for sigma at which the field amplitude is e times smaller than at the maximum)
        self.sigma_lz = sigma_lz
        self.omega_las = 2 * np.pi*c / self.lambda_l
        self.k0_las = self.omega_las / c
        Wph = hbar * self.omega_las  # photon energy, erg
        self.Wph = Wph * 1e-6 / ( elC * 1e7 )  # photon energy, MeV
        self.N_l = self.WL / Wph
        self.a0 = 4 * rel**2 * lambda_l / alpha * self.N_l / (np.power(np.pi, 3/2) * sigma_lr0**2 * sigma_lz)

    def set_foci_displacement(self, delta_x, delta_y, delta_z):
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_z = delta_z
