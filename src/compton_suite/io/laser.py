"""Laser-pulse representation: the ``gaussian_paraxial`` v0.1 I/O contract.

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

from .units import (
    C_LIGHT,
    E_CHARGE,
    EPS0,
    HBAR,
    ME,
    NoConvention,
    PhysicalMeaning,
    PhysicalQuantity,
    TimeConvention,
    WidthConvention,
    sigma_intensity_to_fwhm,
    sigma_intensity_to_w0,
)

__all__ = [
    "GaussianParaxialLaser", "validate", "laser_from_shared_fields",
    "a0_from_fields", "focal_radii_m", "gaussian_pulse_envelope",
]


def _pq(value: float, unit: str, meaning: PhysicalMeaning, convention=None) -> PhysicalQuantity:
    """Shortcut to build a PhysicalQuantity."""
    return PhysicalQuantity(value, unit, meaning, convention)


# Default conventions for laser parameters that have width/duration ambiguity.
_LASER_WIDTH_CONV = WidthConvention.SIGMA_INTENSITY_RMS
_LASER_DURATION_CONV = TimeConvention.SIGMA_INTENSITY_RMS

_ZERO_DISPLACEMENT = _pq(0.0, "meter", PhysicalMeaning.DISPLACEMENT, NoConvention.PLAIN)


@dataclass(frozen=True)
class GaussianParaxialLaser:
    """Only a paraxial Gaussian laser pulse *for now*.

    Every field is a :class:`PhysicalQuantity` carrying its own unit,
    physical meaning and (where relevant) width/duration convention.
    ``pulse_energy_J``, ``wavelength_m`` and ``focus_z_m`` have no convention
    ambiguity (they are always the energy, the central wavelength, and the
    focus position in z) and carry ``NoConvention.PLAIN``.

    ``waist_rms_x_m`` / ``waist_rms_y_m`` / ``duration_rms_s`` are RMS sizes
    of the *intensity* profile at focus (not the field, and not the 1/e^2
    radius -- see ``sigma_x_at`` / ``a0_focus`` for the relevant conversions).

    ``a0`` is not a settable field: it is derived from pulse energy, waist,
    duration and wavelength via the standard plane-wave relation, so it can
    never disagree with the other fields by construction.
    """

    pulse_energy_J: PhysicalQuantity
    wavelength_m: PhysicalQuantity
    waist_rms_x_m: PhysicalQuantity
    waist_rms_y_m: PhysicalQuantity
    duration_rms_s: PhysicalQuantity
    focus_z_m: PhysicalQuantity = _ZERO_DISPLACEMENT

    # -- SI convenience helpers (used by every derived property) ------------
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

    # -- Gaussian-beam propagation ------------------------------------------
    @property
    def rayleigh_x_m(self) -> float:
        """Rayleigh range for the x waist: ``pi * w0^2 / lambda`` with
        ``w0 = 2 * waist_rms_x_m`` (the 1/e^2 intensity radius, twice the
        intensity-profile RMS -- see ``units.w0_to_sigma_intensity``)."""
        return 4.0 * np.pi * self._wx_m ** 2 / self._wl_m

    @property
    def rayleigh_y_m(self) -> float:
        return 4.0 * np.pi * self._wy_m ** 2 / self._wl_m

    def sigma_x_at(self, z_m: float) -> float:
        """RMS intensity-profile width in x at absolute position ``z_m``,
        from standard Gaussian-beam expansion about the focus."""
        return self._wx_m * (1.0 + ((z_m - self._fz_m) / self.rayleigh_x_m) ** 2) ** 0.5

    def sigma_y_at(self, z_m: float) -> float:
        return self._wy_m * (1.0 + ((z_m - self._fz_m) / self.rayleigh_y_m) ** 2) ** 0.5

    @property
    def omega0(self) -> float:
        """Angular frequency, ``2*pi*c/wavelength`` [rad/s]."""
        return 2.0 * np.pi * C_LIGHT / self._wl_m

    @property
    def n_photons(self) -> float:
        """Photon count in the pulse: N_L = pulse_energy / (hbar*omega0)."""
        return self._E_J / (HBAR * self.omega0)

    @property
    def peak_power_W(self) -> float:
        """Peak power at the pulse's temporal center, at focus:
        ``E / (sqrt(2*pi) * duration_rms_s)`` (Gaussian temporal profile)."""
        return self._E_J / ((2.0 * np.pi) ** 0.5 * self._dur_s)

    def peak_intensity_at(self, z_m: float) -> float:
        """On-axis, peak-in-time intensity [W/m^2] at absolute position
        ``z_m``, from the pulse energy spread over the (z-dependent)
        transverse Gaussian spot and the (z-independent) temporal Gaussian
        profile: ``I0(z) = E / ((2*pi)^1.5 * sigma_x(z) * sigma_y(z) *
        duration_rms_s)``."""
        sx = self.sigma_x_at(z_m)
        sy = self.sigma_y_at(z_m)
        return self._E_J / ((2.0 * np.pi) ** 1.5 * sx * sy * self._dur_s)

    @property
    def peak_intensity_focus_W_m2(self) -> float:
        return self.peak_intensity_at(self._fz_m)

    def a0_at(self, z_m: float) -> float:
        """Normalized vector potential at ``z_m``, from the exact SI
        plane-wave relation: ``a0 = (e/(me*c*omega0)) * sqrt(2*I0/(eps0*c))``,
        ``omega0 = 2*pi*c/wavelength``.

        LINEAR POLARIZATION ONLY. This intensity<->E0 relation
        (``I0 = eps0*c*E0^2/2``) assumes a single oscillating field
        component; for elliptical/circular polarization the same peak
        intensity corresponds to a different a0 (xigma_i's own
        ``TrXi/2 = (1+ellipticity**2)/2`` factor is the relevant
        precedent). ``GaussianParaxialLaser`` has no ellipticity field in
        v0.1 (deliberately excluded, see the module docstring) -- when
        that's added, this formula needs an ellipticity-dependent
        correction here, not just a new stored field."""
        I0 = self.peak_intensity_at(z_m)
        E0 = (2.0 * I0 / (EPS0 * C_LIGHT)) ** 0.5
        return (E_CHARGE / (ME * C_LIGHT * self.omega0)) * E0

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
        formalism only ever needs ``a0**2`` -- should read this directly
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
# Validation
# ---------------------------------------------------------------------------
def validate(pulse: GaussianParaxialLaser) -> list[str]:
    """Validate a :class:`GaussianParaxialLaser`.

    Raises ``ValueError`` on hard requirements; returns a list of warning
    strings for soft checks (defocused interaction point, transverse
    astigmatism note).
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


# ---------------------------------------------------------------------------
# Constructor helpers
# ---------------------------------------------------------------------------
def laser_from_shared_fields(*, lambda_L: float, sigma0_l: float, sigma_par_L: float,
                              pulse_energy_J: float, focus_z_m: float = 0.0) -> GaussianParaxialLaser:
    """Build a :class:`GaussianParaxialLaser` from the flat SI fields a
    model's own ``Config``/``params_to_config`` already computes: a central
    wavelength, a round-beam transverse RMS width, a pulse RMS length
    (``= c * duration``, matching kascade/xigma_i's own convention -- see
    ``units.py``'s light-time context), and a pulse energy -- all plain SI
    floats, not yet wrapped as :class:`PhysicalQuantity`. Round beam only
    (``sigma0_l`` is used for both x and y)."""
    return GaussianParaxialLaser(
        pulse_energy_J=_pq(pulse_energy_J, "joule", PhysicalMeaning.PULSE_ENERGY, NoConvention.PLAIN),
        wavelength_m=_pq(lambda_L, "meter", PhysicalMeaning.WAVELENGTH, NoConvention.PLAIN),
        waist_rms_x_m=_pq(sigma0_l, "meter", PhysicalMeaning.LASER_WIDTH, _LASER_WIDTH_CONV),
        waist_rms_y_m=_pq(sigma0_l, "meter", PhysicalMeaning.LASER_WIDTH, _LASER_WIDTH_CONV),
        duration_rms_s=_pq(sigma_par_L / C_LIGHT, "second", PhysicalMeaning.PULSE_DURATION, _LASER_DURATION_CONV),
        focus_z_m=_pq(focus_z_m, "meter", PhysicalMeaning.DISPLACEMENT, NoConvention.PLAIN),
    )


def a0_from_fields(*, pulse_energy_J: float, wavelength_m: float, waist_rms_x_m: float,
                    waist_rms_y_m: float, duration_rms_s: float, focus_z_m: float = 0.0) -> float:
    """Peak a0 at the interaction point (z=0), from raw SI laser fields --
    the single source of truth every model/GUI call site should use instead
    of re-deriving the plane-wave a0 formula itself."""
    pulse = GaussianParaxialLaser(
        pulse_energy_J=_pq(pulse_energy_J, "joule", PhysicalMeaning.PULSE_ENERGY, NoConvention.PLAIN),
        wavelength_m=_pq(wavelength_m, "meter", PhysicalMeaning.WAVELENGTH, NoConvention.PLAIN),
        waist_rms_x_m=_pq(waist_rms_x_m, "meter", PhysicalMeaning.LASER_WIDTH, _LASER_WIDTH_CONV),
        waist_rms_y_m=_pq(waist_rms_y_m, "meter", PhysicalMeaning.LASER_WIDTH, _LASER_WIDTH_CONV),
        duration_rms_s=_pq(duration_rms_s, "second", PhysicalMeaning.PULSE_DURATION, _LASER_DURATION_CONV),
        focus_z_m=_pq(focus_z_m, "meter", PhysicalMeaning.DISPLACEMENT, NoConvention.PLAIN),
    )
    return pulse.a0_interaction


def focal_radii_m(waist_rms_m: float) -> dict[str, float]:
    """RMS / FWHM / 1/e^2 ("w0") focal radii for a Gaussian transverse
    intensity profile of RMS width ``waist_rms_m`` -- the three conventions
    a GUI status line typically wants to display side by side."""
    return {
        "rms": waist_rms_m,
        "fwhm": sigma_intensity_to_fwhm(waist_rms_m),
        "w0_1e2": sigma_intensity_to_w0(waist_rms_m),
    }


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
        ``k0_las*c``-normalised time xigma_i's callers already carry). Any
        consistent length unit works (SI metres for kascade, ``k0_las``-
        normalised cm for xigma_i) -- this function has no embedded unit
        system. Passing a bare ``t`` in seconds here is a caller bug, not
        something this function can detect.
    sigma0: RMS transverse (photon-density) width at the waist, same
        length unit as x/y/z/ct. Round beam only -- kascade's
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
    beta_ff: flying-focus factor, default 0 (static focus). Enters ONLY
        the spot-size term (``u_spot``), not the longitudinal envelope
        (``(u - ct)^2`` stays beta_ff-independent) -- extracted verbatim
        from xigma_i's own ``zr_term = z - beta_ff*t`` construction
        (head-on: ``u = -z``, so ``u_spot = -z + beta_ff*ct = -(z -
        beta_ff*ct)``, squared -> matches exactly). kascade's ``Config``
        has no ``beta_ff`` field and never passes one, so its call is a
        strict no-op with respect to this parameter.
    xp: array module (``numpy`` or ``cupy``) -- needed for ``exp``/
        ``maximum``, not expressible as bare operators. Uses ``xp.pi``
        rather than a hardcoded constant so the function has no hard numpy
        dependency in its compute path.
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
