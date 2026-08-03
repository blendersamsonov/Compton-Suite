"""Laser-pulse representation: the ``gaussian_paraxial`` v0.1 I/O contract.

Every physical parameter travels as a :class:`PhysicalQuantity` (never a bare
float) -- models extract the value in whatever unit they need internally.
``beta_ff``/``phi_pol``/``ellipticity`` are plain floats -- laser-pulse
properties (flying-focus factor, polarization angle, polarization
ellipticity), not model-specific parameters, so they live here rather than
on any individual model's config -- any model that cares reads them
straight off the shared laser.

Head-on by default -- the laser propagates along ``-z``, the electron beam
along ``+z`` (see ``bunch.py``'s module docstring for the shared geometric
convention); ``crossing_angle`` tilts the laser away from this. Not every
model supports a nonzero value yet -- kascade does, xigma-i/delta remain
head-on-only and warn-and-ignore a nonzero ``crossing_angle`` (see
`docs/models/tasks.md`'s tracked "crossing angle support" item for xigma-i).

KNOWN FUTURE GAP: ``a0_at()``'s energy-to-intensity-to-a0 chain still
assumes linear polarization and does not yet use ``ellipticity`` -- adding
the field doesn't by itself fix the formula; see ``a0_at()``'s docstring
for the specific correction this still needs (xigma_i's own ``TrXi/2 =
(1+ellipticity**2)/2`` factor is the relevant precedent). `docs/io/specs/
gaussian_paraxial_laser_io_v0.1.md` predates these three fields and stays
as a historical snapshot of the narrower v0.1 contract -- not updated to
match.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .units import (
    C_LIGHT_Q,
    E_CHARGE_Q,
    EPS0_Q,
    HBAR_Q,
    ME_Q,
    NoConvention,
    PhysicalMeaning,
    PhysicalQuantity,
    Quantity,
    TimeConvention,
    WidthConvention,
)

__all__ = [
    "GaussianParaxialLaser", "validate",
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

    ``crossing_angle`` (radians) tilts the laser's propagation direction
    away from head-on counter-propagation against the fixed +z electron
    beam axis -- a property of the laser's own geometry, not a model-owned
    knob. Not every model supports a nonzero value yet (xigma-i/delta are
    head-on-only); those adapters warn and ignore it rather than silently
    treating it as zero.
    """

    pulse_energy_J: PhysicalQuantity
    wavelength_m: PhysicalQuantity
    waist_rms_x_m: PhysicalQuantity
    waist_rms_y_m: PhysicalQuantity
    duration_rms_s: PhysicalQuantity
    focus_z_m: PhysicalQuantity = _ZERO_DISPLACEMENT
    beta_ff: float = 0.0
    phi_pol: float = 0.0
    ellipticity: float = 0.0
    crossing_angle: float = 0.0

    # -- Gaussian-beam propagation -------------------------------------------
    # Same two-category split as bunch.py's GaussianElectronBeam: genuinely
    # dimensional results (a length, a frequency, a power, an intensity)
    # return a plain pint ``Quantity`` -- never a private "force it into
    # meters/seconds first" shortcut -- so each model extracts
    # `.to(its_own_unit).magnitude` at its own boundary; genuinely
    # dimensionless results (a photon count, a0) stay bare float since
    # there's no unit choice to be agnostic about.
    @property
    def rayleigh_x(self) -> Quantity:
        """Rayleigh range for the x waist: ``pi * w0^2 / lambda`` with
        ``w0 = 2 * waist_rms_x_m`` (the 1/e^2 intensity radius, twice the
        intensity-profile RMS -- see ``units.w0_to_sigma_intensity``)."""
        return (4.0 * np.pi * self.waist_rms_x_m.quantity ** 2 / self.wavelength_m.quantity).to("meter")

    @property
    def rayleigh_y(self) -> Quantity:
        return (4.0 * np.pi * self.waist_rms_y_m.quantity ** 2 / self.wavelength_m.quantity).to("meter")

    def sigma_x_at(self, z_m: float) -> Quantity:
        """RMS intensity-profile width in x at absolute position ``z_m``
        (SI metres), from standard Gaussian-beam expansion about the
        focus."""
        fz = self.focus_z_m.to_unit("meter").magnitude
        rx = self.rayleigh_x.to("meter").magnitude
        return self.waist_rms_x_m.quantity * (1.0 + ((z_m - fz) / rx) ** 2) ** 0.5

    def sigma_y_at(self, z_m: float) -> Quantity:
        fz = self.focus_z_m.to_unit("meter").magnitude
        ry = self.rayleigh_y.to("meter").magnitude
        return self.waist_rms_y_m.quantity * (1.0 + ((z_m - fz) / ry) ** 2) ** 0.5

    @property
    def omega0(self) -> Quantity:
        """Angular frequency, ``2*pi*c/wavelength``."""
        return (2.0 * np.pi * C_LIGHT_Q / self.wavelength_m.quantity).to("1 / second")

    @property
    def n_photons(self) -> float:
        """Photon count in the pulse: N_L = pulse_energy / (hbar*omega0). A
        pure count has no unit to be agnostic about, so this stays a bare
        float."""
        return (self.pulse_energy_J.quantity / (HBAR_Q * self.omega0)).to("dimensionless").magnitude

    @property
    def peak_power(self) -> Quantity:
        """Peak power at the pulse's temporal center, at focus:
        ``E / (sqrt(2*pi) * duration_rms_s)`` (Gaussian temporal profile)."""
        return (self.pulse_energy_J.quantity / ((2.0 * np.pi) ** 0.5 * self.duration_rms_s.quantity)).to("watt")

    def peak_intensity_at(self, z_m: float) -> Quantity:
        """On-axis, peak-in-time intensity at absolute position ``z_m`` (SI
        metres), from the pulse energy spread over the (z-dependent)
        transverse Gaussian spot and the (z-independent) temporal Gaussian
        profile: ``I0(z) = E / ((2*pi)^1.5 * sigma_x(z) * sigma_y(z) *
        duration_rms_s)``."""
        sx = self.sigma_x_at(z_m)
        sy = self.sigma_y_at(z_m)
        return (self.pulse_energy_J.quantity
                / ((2.0 * np.pi) ** 1.5 * sx * sy * self.duration_rms_s.quantity)).to("watt / meter ** 2")

    @property
    def peak_intensity_focus(self) -> Quantity:
        fz = self.focus_z_m.to_unit("meter").magnitude
        return self.peak_intensity_at(fz)

    def a0_at(self, z_m: float) -> float:
        """Normalized vector potential at absolute position ``z_m`` (SI
        metres), from the exact SI plane-wave relation:
        ``a0 = (e/(me*c*omega0)) * sqrt(2*I0/(eps0*c))``,
        ``omega0 = 2*pi*c/wavelength``. Dimensionless by definition, so this
        stays a bare float.

        LINEAR POLARIZATION ONLY. This intensity<->E0 relation
        (``I0 = eps0*c*E0^2/2``) assumes a single oscillating field
        component; for elliptical/circular polarization the same peak
        intensity corresponds to a different a0 (xigma_i's own
        ``TrXi/2 = (1+ellipticity**2)/2`` factor is the relevant
        precedent). ``ellipticity`` is a real field on
        ``GaussianParaxialLaser`` but this formula does not yet use it --
        adding the field doesn't by itself fix the formula; that's a
        separate, deliberately deferred correction (see the module
        docstring's KNOWN FUTURE GAP)."""
        I0 = self.peak_intensity_at(z_m)
        E0 = (2.0 * I0 / (EPS0_Q * C_LIGHT_Q)) ** 0.5
        a0 = (E_CHARGE_Q / (ME_Q * C_LIGHT_Q * self.omega0)) * E0
        return a0.to("dimensionless").magnitude

    @property
    def a0_focus(self) -> float:
        fz = self.focus_z_m.to_unit("meter").magnitude
        return self.a0_at(fz)

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
        fz = self.focus_z_m.to_unit("meter").magnitude
        return self.a0sq_at(fz)

    @property
    def a0sq_interaction(self) -> float:
        return self.a0sq_at(0.0)

    @staticmethod
    def pulse_envelope(x, y, z, ct, *, sigma0, rayleigh_range, sigma_ct,
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

        A ``@staticmethod`` rather than an instance method reading ``self``:
        real callers pass their own already-derived/collapsed plain-float
        values (kascade's round-beam-collapsed ``_sigma0_l`` plus its own
        ``Config``-owned crossing-angle axis/foci; xigma_i's own CGS/
        ``k0_las``-normalised scalars, extracted from ``GaussianElectronBeam``/
        ``GaussianParaxialLaser`` by the caller, with no laser *instance* in
        that code path at all) -- forcing this onto ``self`` would require
        restructuring one of those two callers for no benefit.

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
            ``cfg.R_sf`` == ``GaussianParaxialLaser.rayleigh_x``); a caller
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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(pulse: GaussianParaxialLaser) -> list[str]:
    """Validate a :class:`GaussianParaxialLaser`.

    Raises ``ValueError`` on hard requirements; returns a list of warning
    strings for soft checks (defocused interaction point, transverse
    astigmatism note).
    """
    _e = pulse.pulse_energy_J.to_unit("joule").magnitude
    _wl = pulse.wavelength_m.to_unit("meter").magnitude
    _wx = pulse.waist_rms_x_m.to_unit("meter").magnitude
    _wy = pulse.waist_rms_y_m.to_unit("meter").magnitude
    _dur = pulse.duration_rms_s.to_unit("second").magnitude
    _fz = pulse.focus_z_m.to_unit("meter").magnitude

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
    if abs(_fz) > pulse.rayleigh_x.to("meter").magnitude or abs(_fz) > pulse.rayleigh_y.to("meter").magnitude:
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
