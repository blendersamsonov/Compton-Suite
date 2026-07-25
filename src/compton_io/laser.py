"""Laser-pulse representation: the ``gaussian_paraxial`` v0.1 I/O contract
(``specs/gaussian_paraxial_laser_io_v0.1.md``).

Head-on only in v0.1 -- the laser propagates along ``-z``, the electron
beam along ``+z`` (see ``bunch.py``'s module docstring for the shared
geometric convention). No crossing angle, no polarization/ellipticity, no
flying focus in this shared representation -- those stay engine-specific
extras (e.g. ``xigma_i``'s ``beta_ff``/``ellipticity``, surfaced through
its own ``extra_params()``), not part of this shared subset.

KNOWN FUTURE GAP: ``a0_at()``'s energy-to-intensity-to-a0 chain assumes
linear polarization. Adding elliptical/circular polarization support
later must revisit that formula, not just add an ``ellipticity`` field --
see ``a0_at()``'s docstring for the specific correction this needs
(xigma_i's own ``TrXi/2 = (1+ellipticity**2)/2`` factor is the relevant
precedent).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .constants import C_LIGHT, E_CHARGE, EPS0, HBAR, ME

__all__ = ["GaussianParaxialLaser", "validate"]


@dataclass(frozen=True)
class GaussianParaxialLaser:
    """A paraxial Gaussian laser pulse. SI units.

    ``waist_rms_x_m``/``waist_rms_y_m``/``duration_rms_s`` are RMS sizes of
    the *intensity* profile at focus (not the field, and not the 1/e^2
    radius -- see ``sigma_x_at``/``a0_focus`` for the relevant conversions).
    ``a0`` is not a settable field: it is derived from pulse energy, waist,
    duration and wavelength via the standard plane-wave relation (spec
    Sec. 9), so it can never disagree with the other fields by
    construction.
    """

    pulse_energy_J: float
    wavelength_m: float
    waist_rms_x_m: float
    waist_rms_y_m: float
    duration_rms_s: float
    focus_z_m: float = 0.0

    @property
    def rayleigh_x_m(self) -> float:
        return 4.0 * np.pi * self.waist_rms_x_m**2 / self.wavelength_m

    @property
    def rayleigh_y_m(self) -> float:
        return 4.0 * np.pi * self.waist_rms_y_m**2 / self.wavelength_m

    def sigma_x_at(self, z_m: float) -> float:
        return self.waist_rms_x_m * (1.0 + ((z_m - self.focus_z_m) / self.rayleigh_x_m) ** 2) ** 0.5

    def sigma_y_at(self, z_m: float) -> float:
        return self.waist_rms_y_m * (1.0 + ((z_m - self.focus_z_m) / self.rayleigh_y_m) ** 2) ** 0.5

    @property
    def n_photons(self) -> float:
        """Photon count in the pulse: N_L = pulse_energy_J / (hbar*omega0)."""
        omega0 = 2.0 * np.pi * C_LIGHT / self.wavelength_m
        return self.pulse_energy_J / (HBAR * omega0)

    @property
    def peak_intensity_focus_W_m2(self) -> float:
        return self.pulse_energy_J / (
            (2.0 * np.pi) ** 1.5 * self.waist_rms_x_m * self.waist_rms_y_m * self.duration_rms_s
        )

    def peak_intensity_at(self, z_m: float) -> float:
        return self.peak_intensity_focus_W_m2 * (
            (self.waist_rms_x_m * self.waist_rms_y_m) / (self.sigma_x_at(z_m) * self.sigma_y_at(z_m))
        )

    @property
    def peak_power_W(self) -> float:
        return self.pulse_energy_J / ((2.0 * np.pi) ** 0.5 * self.duration_rms_s)

    def a0_at(self, z_m: float) -> float:
        """Normalized vector potential at ``z_m``, from the exact SI
        plane-wave relation (spec Sec. 9): ``a0 = (e/(me*c*omega0)) *
        sqrt(2*I0/(eps0*c))``, ``omega0 = 2*pi*c/wavelength``. Uses the
        exact formula (not the spec's ``a0 ~= 0.855 * lambda_um *
        sqrt(I0/1e18 W/cm^2)`` practical shortcut), since compton_io
        already carries precise, pint-derived constants.

        LINEAR POLARIZATION ONLY. This intensity<->E0 relation
        (``I0 = eps0*c*E0^2/2``) assumes a single oscillating field
        component; for elliptical/circular polarization the same peak
        intensity corresponds to a different a0 (xigma_i's
        ``config.py`` already carries this as its ``TrXi/2 =
        (1+ellipticity**2)/2`` factor on the trajectory-averaged
        intensity, applied downstream of this a0). GaussianParaxialLaser
        has no ellipticity field in v0.1 (deliberately excluded, see the
        module docstring) -- when that's added, this formula needs an
        ellipticity-dependent correction here, not just a new stored
        field, or a0_focus/a0_interaction will silently be wrong for
        anything but linear polarization. Flagged now so it isn't
        forgotten when that generalization happens."""
        omega0 = 2.0 * np.pi * C_LIGHT / self.wavelength_m
        I0 = self.peak_intensity_at(z_m)
        E0 = (2.0 * I0 / (EPS0 * C_LIGHT)) ** 0.5
        return (E_CHARGE / (ME * C_LIGHT * omega0)) * E0

    @property
    def a0_focus(self) -> float:
        return self.a0_at(self.focus_z_m)

    @property
    def a0_interaction(self) -> float:
        return self.a0_at(0.0)


def validate(pulse: GaussianParaxialLaser) -> list[str]:
    """Validate a :class:`GaussianParaxialLaser` per spec Sec. 12.

    Raises ``ValueError`` on the spec's hard requirements; returns a list
    of warning strings for the spec's soft checks (defocused interaction
    point, transverse astigmatism note).
    """
    if pulse.pulse_energy_J <= 0:
        raise ValueError("GaussianParaxialLaser: pulse_energy_J must be > 0")
    if pulse.wavelength_m <= 0:
        raise ValueError("GaussianParaxialLaser: wavelength_m must be > 0")
    if pulse.waist_rms_x_m <= 0 or pulse.waist_rms_y_m <= 0:
        raise ValueError("GaussianParaxialLaser: waist_rms_x_m/waist_rms_y_m must be > 0")
    if pulse.duration_rms_s <= 0:
        raise ValueError("GaussianParaxialLaser: duration_rms_s must be > 0")
    if not isfinite(pulse.focus_z_m):
        raise ValueError("GaussianParaxialLaser: focus_z_m must be finite")

    warnings: list[str] = []
    if abs(pulse.focus_z_m) > pulse.rayleigh_x_m or abs(pulse.focus_z_m) > pulse.rayleigh_y_m:
        warnings.append(
            "Focus far from the interaction point; the interaction-point "
            "intensity may be substantially below the focus's peak intensity."
        )
    if pulse.waist_rms_x_m != pulse.waist_rms_y_m:
        warnings.append(
            "Elliptical beam at focus. Allowed in v0.1, but astigmatism as "
            "different x/y focus positions is not supported."
        )
    return warnings
