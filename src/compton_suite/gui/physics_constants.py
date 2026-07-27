"""CODATA-style physical constants used by the GUI's local formula helpers
(peak_a0, sigma_e, the recoil-parameter/electron-energy stats in
_update_outputs, ...).

Re-exported from ``compton_io.constants`` (the shared, pint-derived
source of truth also used by ``xigma_i``/``kascade``) rather than defined
here.
"""

from compton_suite.io import constants as _io_constants

C_LIGHT = _io_constants.C_LIGHT           # [m/s]
E_CHARGE = _io_constants.E_CHARGE         # [C]
HBAR = _io_constants.HBAR                 # [J*s]
EPS0 = _io_constants.EPS0                 # [F/m]
MEC2_EV = _io_constants.MEC2_EV           # electron rest energy [eV]
MEC2_J = MEC2_EV * E_CHARGE               # electron rest energy [J]
