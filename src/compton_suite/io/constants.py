"""Physical constants, derived from ``units.py``'s pint registry rather than
hand-typed literals.

pint ships its own CODATA physical-constants table internally; deriving
every value here from ``ureg`` instead of retyping digits means this module
is self-verifying (no risk of a transcription typo silently drifting from
the real CODATA value) and means "physical constants" and "the shared pint
registry" are one mechanism, not two independently-maintained ones.

**SI values are canonical.** ``compton_suite.gui`` and ``kascade`` are SI
throughout and can use the top block directly. ``xigma_i`` is CGS-Gaussian
throughout (see its own ``CLAUDE.md``, "Units are CGS") and needs the
second block -- CGS-Gaussian isn't a pure unit re-scaling of SI for
electromagnetic quantities (charge isn't dimensionally the same in the two
systems: SI treats current/charge as its own base dimension, Gaussian
folds it into mass/length/time via Coulomb's-law-constant=1), so pint's
registry can't convert electromagnetic quantities between them directly
-- ``EL_STATC`` below applies the textbook exact conversion
(``1 C = c[cm/s] / 10 statC``) by hand instead.

This module replaces the physical-constants blocks previously duplicated
(with a real, ~1.6e-8 relative numeric disagreement on ``hbar``/electron
mass between ``xigma_i``'s older-CODATA-vintage copy and the
``compton_suite.gui``/``kascade`` copies, which already agreed with each other
and with pint's built-in values) in ``xigma_i/config.py``,
``xigma_i/gui_adapter.py``, ``compton_suite.gui/physics_constants.py``, and
``kascade.py``.
"""

from __future__ import annotations

from .units import ureg

_Q = ureg.Quantity

# ---------------------------------------------------------------------------
# SI (canonical) -- for compton_suite.gui and kascade
# ---------------------------------------------------------------------------
C_LIGHT = _Q(1, "speed_of_light").to("meter / second").magnitude
E_CHARGE = _Q(1, "elementary_charge").to("coulomb").magnitude
HBAR = _Q(1, "hbar").to("joule * second").magnitude
ME = _Q(1, "electron_mass").to("kilogram").magnitude
MEC2_EV = (_Q(1, "electron_mass") * _Q(1, "speed_of_light") ** 2).to("electron_volt").magnitude
EPS0 = _Q(1, "vacuum_permittivity").to("farad / meter").magnitude
ALPHA = _Q(1, "fine_structure_constant").to("dimensionless").magnitude
R_E_M = _Q(1, "classical_electron_radius").to("meter").magnitude
SIGMA_T_M2 = _Q(1, "thomson_cross_section").to("meter ** 2").magnitude

# ---------------------------------------------------------------------------
# CGS-Gaussian views -- for xigma_i's config.py and compton_suite.io.collision
# (the CGS CollisionParams/build_params convention those pipelines share).
# Only the primitives actually needed as literals move here.
# ---------------------------------------------------------------------------
HBAR_ERG_S = _Q(1, "hbar").to("erg * second").magnitude
ME_G = _Q(1, "electron_mass").to("gram").magnitude
C_CM_S = _Q(1, "speed_of_light").to("centimeter / second").magnitude

# Charge: 1 coulomb = c[cm/s] / 10 statcoulombs (exact, textbook
# CGS-Gaussian <-> SI conversion for electric charge). Cross-checked in
# tests/test_constants.py against xigma_i's own previously-correct `el`
# literal (charge has no CODATA-vintage mismatch between repos -- only
# hbar/electron mass did).
EL_STATC = E_CHARGE * C_CM_S / 10.0

# Classical electron radius, CGS (cm) -- numerically agrees with
# el**2/(me*c**2) computed from EL_STATC/ME_G/C_CM_S above to ~1.3e-10
# relative (verified), but sourced directly from pint's own CODATA value
# instead of re-deriving the formula, same rationale as R_E_M above.
R_E_CM = _Q(1, "classical_electron_radius").to("centimeter").magnitude

# Meter <-> centimeter, derived rather than a hand-typed 100.0 -- used by
# every CGS-unit consumer (compton_suite.io.collision, xigma_i/delta's
# gui_adapter.py, particles.py) instead of each keeping its own local
# `_M_TO_CM = 100.0`/`1.0e2` literal.
M_TO_CM = _Q(1, "meter").to("centimeter").magnitude
