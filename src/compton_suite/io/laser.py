"""Laser-pulse representation: the ``gaussian_paraxial`` v0.1 I/O contract
(``specs/gaussian_paraxial_laser_io_v0.1.md``).

Every physical parameter travels as a :class:`PhysicalQuantity` (never a bare
float) -- models extract the value in whatever unit they need internally.

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
from .enums import NoConvention, PhysicalMeaning, WidthConvention, TimeConvention
from .quantities import PhysicalQuantity

__all__ = ["GaussianParaxialLaser", "validate", "laser_from_shared_fields",
           "a0_from_fields", "focal_radii_m"]


@dataclass(frozen=True)
class GaussianParaxialLaser:
    """A paraxial Gaussian laser pulse.

    Every field is a :class:`PhysicalQuantity` carrying its own unit,
    physical meaning and (where relevant) width/duration convention.
    ``pulse_energy_J``, ``wavelength_m`` and ``focus_z_m`` have no convention
    ambiguity (they are always the energy, the central wavelength, and the
    focus position in z) and carry ``NoConvention.PLAIN``.

    ``waist_rms_x_m`` / ``waist_rms_y_m`` / ``duration_rms_s`` are RMS sizes
    of the *intensity* profile at focus (not the field, and not the 1/e^2
    radius -- see ``sigma_x_at`` / ``a0_focus`` for the relevant conversions).

    ``a0`` is not a settable field: it is derived from pulse energy, waist,
    duration and wavelength via the standard plane-wave relation (spec
    Sec. 9), so it can never disagree with the other fields by construction.
    """

    pulse_energy_J: PhysicalQuantity
    wavelength_m: PhysicalQuantity
    waist_rms_x_m: PhysicalQuantity
    waist_rms_y_m: PhysicalQuantity
    duration_rms_s: PhysicalQuantity
    focus_z_m: PhysicalQuantity = PhysicalQuantity(0.0, "meter",
                                                   PhysicalMeaning.DISPLACEMENT,
                                                   NoConvention.PLAIN)

    # -- SI convenience helpers used by every property below ----------------
    @property
    def _E_J(self) -> float:
        return self.pulse_energy_J.to_unit("joule").magnitude

    @property
    def _wl_m(self) -> float:
        return self.wavelength_m.to_unit("meter").magnitude

    @property
    def _wx_m(self) -> float:
        return self.waist_rms_x_m.to_unit("meter").magnitude

    @property
    def _wy_m(self) -> float:
        return self.waist_rms_y_m.to_unit("meter").magnitude

    @property
    def _dur_s(self) -> float:
        return self.duration_rms_s.to_unit("second").magnitude

    @property
    def _fz_m(self) -> float:
        return self.focus_z_m.to_unit("meter").magnitude

    # -- derived properties ------------------------------------------------
    @property
    def rayleigh_x_m(self) -> float:
        return 4.0 * np.pi * self._wx_m**2 / self._wl_m

    @property
    def rayleigh_y_m(self) -> float:
        return 4.0 * np.pi * self._wy_m**2 / self._wl_m

    def sigma_x_at(self, z_m: float) -> float:
        return self._wx_m * (1.0 + ((z_m - self._fz_m) / self.rayleigh_x_m) ** 2) ** 0.5

    def sigma_y_at(self, z_m: float) -> float:
        return self._wy_m * (1.0 + ((z_m - self._fz_m) / self.rayleigh_y_m) ** 2) ** 0.5

    @property
    def n_photons(self) -> float:
        """Photon count in the pulse: N_L = pulse_energy / (hbar*omega0)."""
        omega0 = 2.0 * np.pi * C_LIGHT / self._wl_m
        return self._E_J / (HBAR * omega0)

    @property
    def peak_intensity_focus_W_m2(self) -> float:
        return self._E_J / (
            (2.0 * np.pi) ** 1.5 * self._wx_m * self._wy_m * self._dur_s
        )

    def peak_intensity_at(self, z_m: float) -> float:
        return self.peak_intensity_focus_W_m2 * (
            (self._wx_m * self._wy_m) / (self.sigma_x_at(z_m) * self.sigma_y_at(z_m))
        )

    @property
    def peak_power_W(self) -> float:
        return self._E_J / ((2.0 * np.pi) ** 0.5 * self._dur_s)

    def a0_at(self, z_m: float) -> float:
        """Normalized vector potential at ``z_m``, from the exact SI
        plane-wave relation (spec Sec. 9): ``a0 = (e/(me*c*omega0)) *
        sqrt(2*I0/(eps0*c))``, ``omega0 = 2*pi*c/wavelength``. Uses the
        exact formula (not the spec's ``a0 ~= 0.855 * lambda_um *
        sqrt(I0/1e18 W/cm^2)`` practical shortcut), since compton_suite.io
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
        omega0 = 2.0 * np.pi * C_LIGHT / self._wl_m
        I0 = self.peak_intensity_at(z_m)
        E0 = (2.0 * I0 / (EPS0 * C_LIGHT)) ** 0.5
        return (E_CHARGE / (ME * C_LIGHT * omega0)) * E0

    @property
    def a0_focus(self) -> float:
        return self.a0_at(self._fz_m)

    @property
    def a0_interaction(self) -> float:
        return self.a0_at(0.0)

    def a0sq_at(self, z_m: float) -> float:
        """``a0_at(z_m)**2`` -- the mean-square amplitude.

        Consumers that only ever work with intensity/photon density (not a
        field amplitude) -- e.g. ``xigma_i``, whose weakly-nonlinear-regime
        formalism only ever needs ``a0**2`` (its own docs call this
        quantity ``a0`` by convention, but it is never a linear field
        amplitude anywhere in that codebase) -- should read this directly
        rather than squaring ``a0_at()`` themselves at each call site.
        """
        return self.a0_at(z_m) ** 2

    @property
    def a0sq_focus(self) -> float:
        return self.a0sq_at(self._fz_m)

    @property
    def a0sq_interaction(self) -> float:
        return self.a0sq_at(0.0)


# ---------------------------------------------------------------------------
# Constructor helpers
# ---------------------------------------------------------------------------

def _pq(value: float, unit: str, meaning: PhysicalMeaning, convention=None) -> PhysicalQuantity:
    """Shortcut to build a PhysicalQuantity."""
    return PhysicalQuantity(value, unit, meaning, convention)


# Default conventions for laser parameters that have width/duration ambiguity.
_LASER_WIDTH_CONV = WidthConvention.SIGMA_INTENSITY_RMS
_LASER_DURATION_CONV = TimeConvention.SIGMA_INTENSITY_RMS


def laser_from_shared_fields(*, lambda_L: float | PhysicalQuantity,
                             sigma0_l: float | PhysicalQuantity,
                             sigma_par_L: float | PhysicalQuantity,
                             pulse_energy_J: float | PhysicalQuantity,
                             focus_z_m: float | PhysicalQuantity = 0.0) -> GaussianParaxialLaser:
    """Build a :class:`GaussianParaxialLaser` from the flat SI field set every
    model's own ``Config`` already derives and agrees on (``lambda_L``,
    ``sigma0_l``, ``sigma_par_L``, ``pulse_energy_J`` -- see
    ``ComptonSuite/validation/scenarios.py``'s ``scenario_to_shared_fields``,
    this function's exact inverse). Mirrors :func:`compton_suite.io.bunch.
    beam_from_shared_fields`'s role for the electron beam.

    Accepts either plain floats (interpreted as SI) or ``PhysicalQuantity``
    objects.  When a float is passed it is wrapped in a ``PhysicalQuantity``
    with the appropriate SI unit and the canonical convention.

    ``sigma0_l`` is a single round-beam transverse width, applied to both
    ``waist_rms_x_m`` and ``waist_rms_y_m`` -- every current consumer
    (kascade, xigma_i, delta) models the laser spot as round, never
    elliptical, at this shared-field boundary. ``sigma_par_L`` is a
    *length* (``duration_rms_s * C_LIGHT``, the same ``c*t`` convention
    ``compton_suite.io.bunch``'s ``sigma_par_e`` uses for the electron bunch),
    converted back to a duration here.
    """
    if isinstance(lambda_L, PhysicalQuantity):
        wl_q = lambda_L
    else:
        wl_q = _pq(lambda_L, "meter", PhysicalMeaning.WAVELENGTH)

    if isinstance(sigma0_l, PhysicalQuantity):
        w_q = sigma0_l
    else:
        w_q = _pq(sigma0_l, "meter", PhysicalMeaning.LASER_WIDTH, _LASER_WIDTH_CONV)

    if isinstance(sigma_par_L, PhysicalQuantity):
        length_q = sigma_par_L
    else:
        length_q = _pq(sigma_par_L, "meter", PhysicalMeaning.PULSE_DURATION, _LASER_DURATION_CONV)

    if isinstance(pulse_energy_J, PhysicalQuantity):
        e_q = pulse_energy_J
    else:
        e_q = _pq(pulse_energy_J, "joule", PhysicalMeaning.PULSE_ENERGY)

    if isinstance(focus_z_m, PhysicalQuantity):
        fz_q = focus_z_m
    else:
        fz_q = _pq(float(focus_z_m), "meter", PhysicalMeaning.DISPLACEMENT)

    # sigma_par_L is a length (c * duration) -- convert to a duration.
    dur_s = length_q.to_unit("meter").magnitude / C_LIGHT
    wl_m = wl_q.to_unit("meter").magnitude
    w_m = w_q.to_unit("meter").magnitude
    e_j = e_q.to_unit("joule").magnitude
    fz_m = fz_q.to_unit("meter").magnitude

    return GaussianParaxialLaser(
        pulse_energy_J=_pq(e_j, "joule", PhysicalMeaning.PULSE_ENERGY),
        wavelength_m=_pq(wl_m, "meter", PhysicalMeaning.WAVELENGTH),
        waist_rms_x_m=_pq(w_m, "meter", PhysicalMeaning.LASER_WIDTH, _LASER_WIDTH_CONV),
        waist_rms_y_m=_pq(w_m, "meter", PhysicalMeaning.LASER_WIDTH, _LASER_WIDTH_CONV),
        duration_rms_s=_pq(dur_s, "second", PhysicalMeaning.PULSE_DURATION, _LASER_DURATION_CONV),
        focus_z_m=_pq(fz_m, "meter", PhysicalMeaning.DISPLACEMENT),
    )


def a0_from_fields(*, wavelength_m: float | PhysicalQuantity,
                   pulse_energy_J: float | PhysicalQuantity,
                   waist_rms_m: float | PhysicalQuantity,
                   duration_rms_s: float | PhysicalQuantity) -> float:
    """Peak normalised vector potential a_0 from raw SI laser fields.

    This is the **single source of truth** for a0 computation across the
    suite.  Every consumer (GUI live preview, analytical model, validation
    cross-check) should call this rather than re-implementing the formula.

    Accepts either plain floats (SI) or ``PhysicalQuantity`` objects.

    Constructs a temporary :class:`GaussianParaxialLaser` at focus (z=0)
    and returns its ``a0_focus`` property.  The formula is the standard
    SI plane-wave relation (spec Sec. 9, ``laser.py``'s ``a0_at``
    docstring for the linear-polarization caveat)::

        a0 = (e / (me * c * omega0)) * sqrt(2 * I0 / (eps0 * c))

    where ``I0 = pulse_energy / ((2*pi)^1.5 * waist^2 * duration)``.

    .. note::

       The CGS equivalent in ``collision.py``'s ``build_params`` has a
       known ~49% discrepancy against this SI formula
       (``validation/tier0_wiring.py``'s ``check_a0_formula_agreement``).
       This function is the authoritative SI implementation; the CGS
       formula needs human verification and should not be treated as
       equivalent until reconciled.
    """
    def _to_si(v, unit):
        return v.to_unit(unit).magnitude if isinstance(v, PhysicalQuantity) else float(v)

    wl = _to_si(wavelength_m, "meter")
    ej = _to_si(pulse_energy_J, "joule")
    wr = _to_si(waist_rms_m, "meter")
    dr = _to_si(duration_rms_s, "second")

    laser = GaussianParaxialLaser(
        pulse_energy_J=_pq(ej, "joule", PhysicalMeaning.PULSE_ENERGY),
        wavelength_m=_pq(wl, "meter", PhysicalMeaning.WAVELENGTH),
        waist_rms_x_m=_pq(wr, "meter", PhysicalMeaning.LASER_WIDTH, _LASER_WIDTH_CONV),
        waist_rms_y_m=_pq(wr, "meter", PhysicalMeaning.LASER_WIDTH, _LASER_WIDTH_CONV),
        duration_rms_s=_pq(dr, "second", PhysicalMeaning.PULSE_DURATION, _LASER_DURATION_CONV),
        focus_z_m=_pq(0.0, "meter", PhysicalMeaning.DISPLACEMENT),
    )
    return laser.a0_focus


def focal_radii_m(*, wavelength_m: float | PhysicalQuantity,
                  rayleigh_length_m: float | PhysicalQuantity) -> tuple[float, float, float]:
    """Return ``(rms, fwhm, e^{-1/2})`` focal radii [m] for a round
    Gaussian beam given its wavelength and Rayleigh length.

    Accepts either plain floats (SI) or ``PhysicalQuantity`` objects.

    The three definitions commonly used in the literature:

    * **RMS** (``sigma = w0 / 2``): the RMS of the intensity profile,
      used by ``GaussianParaxialLaser``'s ``waist_rms_*_m`` fields.
    * **FWHM** (``sigma * 2*sqrt(2*ln2)``): full width at half maximum
      of the intensity profile.
    * **e^{-1/2}** (``sigma * sqrt(2)``): the 1/e^2 intensity radius
      (sometimes written w0 in laser-physics notation), which is
      ``sqrt(Rayleigh * lambda / pi)``.
    """
    def _to_si(v, unit):
        return v.to_unit(unit).magnitude if isinstance(v, PhysicalQuantity) else float(v)

    wl = _to_si(wavelength_m, "meter")
    rl = _to_si(rayleigh_length_m, "meter")

    # waist_1e2 = 1/e^2 intensity radius = sqrt(R * lambda / pi)
    waist_1e2 = np.sqrt(rl * wl / np.pi)
    rms = waist_1e2 / 2.0
    fwhm = waist_1e2 * np.sqrt(np.log(2.0) / 2.0)
    e_inv_sqrt = waist_1e2 / (2.0 * np.sqrt(2.0))
    return (float(rms), float(fwhm), float(e_inv_sqrt))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(pulse: GaussianParaxialLaser) -> list[str]:
    """Validate a :class:`GaussianParaxialLaser` per spec Sec. 12.

    Raises ``ValueError`` on the spec's hard requirements; returns a list
    of warning strings for the spec's soft checks (defocused interaction
    point, transverse astigmatism note).
    """
    _e = pulse._E_J
    _wl = pulse._wl_m
    _wx = pulse._wx_m
    _wy = pulse._wy_m
    _dur = pulse._dur_s
    _fz = pulse._fz_m

    if _e <= 0:
        raise ValueError("GaussianParaxialLaser: pulse_energy must be > 0")
    if _wl <= 0:
        raise ValueError("GaussianParaxialLaser: wavelength must be > 0")
    if _wx <= 0 or _wy <= 0:
        raise ValueError("GaussianParaxialLaser: waist_rms_x/y must be > 0")
    if _dur <= 0:
        raise ValueError("GaussianParaxialLaser: duration_rms_s must be > 0")
    if not isfinite(_fz):
        raise ValueError("GaussianParaxialLaser: focus_z must be finite")

    warnings: list[str] = []
    if abs(_fz) > pulse.rayleigh_x_m or abs(_fz) > pulse.rayleigh_y_m:
        warnings.append(
            "Focus far from the interaction point; the interaction-point "
            "intensity may be substantially below the focus's peak intensity."
        )
    if _wx != _wy:
        warnings.append(
            "Elliptical beam at focus. Allowed in v0.1, but astigmatism as "
            "different x/y focus positions is not supported."
        )
    return warnings
