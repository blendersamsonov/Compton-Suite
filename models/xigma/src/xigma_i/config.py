"""Physics constants and GPU kernel sizing constants for this pipeline
(particles.py, deposition.py, spectrum4d.py/spectrum4d_cpu.py, reference.py,
tabulated_engine.py, gui_adapter.py).

The CGS "collision parameters" bundle this pipeline builds one of per run
(``CollisionParams``/``build_params``) has moved to
``compton_io.collision`` -- it turned out to have nothing xigma-specific
about it beyond the CGS/``k0_las`` convention and the ``a0`` formula (see
that module's own docstring for the still-unreconciled ~49% discrepancy
against ``GaussianParaxialLaser.a0_focus``), so any model built on the same
convention (currently ``xigma_direct``) can now reuse it directly instead
of importing it through this package. ``detect_device`` moved there too
(``CollisionParams.xp``/``.asnumpy`` need it); ``_detect_device`` below is
a thin re-export so this package's own internal callers
(spectrum4d.py, gui_adapter.py) don't need to change their import path.

`hbar`/`me`/`c`/`el` below come from `compton_io.constants` rather than
local literals -- the single shared source of truth also used by
`compton_guide`/`kascade`/`compton_io.collision`, see `compton_io`'s own
CLAUDE.md. Only `sigma_T` (Thomson cross section, used by `particles.py`)
still needs deriving locally in CGS; classical-electron-radius and
fine-structure-constant now come from `compton_io.constants` directly
(`R_E_CM`/`ALPHA`) wherever something outside this module needs them
(`compton_io.collision`'s `a0` formula).
"""
from dataclasses import dataclass, field

import numpy as np

from compton_io import constants as _io_constants
from compton_io.collision import detect_device as _detect_device  # noqa: F401 (re-exported for spectrum4d.py/gui_adapter.py)

me = _io_constants.ME_G              # electron mass, g
c = _io_constants.C_CM_S             # speed of light, cm/s
el = _io_constants.EL_STATC          # electron charge, statcoulombs
rel = el ** 2 / (me * c ** 2)        # classical electron radius, cm
sigma_T = 8.0 * np.pi / 3.0 * rel**2 # Thomson cross section, cm^2
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

# particles.py's push_and_sample: how many pulse-duration Gaussian widths /
# Rayleigh-range Lorentzian widths out a particle's trajectory is
# considered "possibly inside the pulse" (compton_io.propagation.
# laser_overlap_time_window's gauss_width/lorentz_width).
GAUSS_WIDTH   = CP_FLOAT(3)
LORENTZ_WIDTH = CP_FLOAT(8)
