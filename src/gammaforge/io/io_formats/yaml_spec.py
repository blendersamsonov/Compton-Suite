"""YAML I/O for the ``gaussian_6d_waist``/``gaussian_paraxial`` v0.1
formats defined in ``specs/electron_beam_io_v0.1_full.md`` and
``specs/gaussian_paraxial_laser_io_v0.1.md``.

Reads/writes exactly the shape shown in each spec's minimal example
(``electron_beam: {model: gaussian_6d_waist, version: "0.1", ...}`` /
``laser: {model: gaussian_paraxial, version: "0.1", ...}``), including the
spec's own unit choices at the file boundary. Conversion to/from SI happens
only here; the in-memory dataclasses (:class:`~gammaforge.io.bunch.
GaussianElectronBeam`, :class:`~gammaforge.io.laser.GaussianParaxialLaser`)
always stay SI.
"""

from __future__ import annotations

import yaml

from ..bunch import GaussianElectronBeam
from ..laser import GaussianParaxialLaser
from ..units import (
    NoConvention,
    PhysicalMeaning,
    PhysicalQuantity,
    TimeConvention,
    WidthConvention,
)

__all__ = ["load_electron_beam", "save_electron_beam", "load_laser", "save_laser"]

_ELECTRON_BEAM_MODEL = "gaussian_6d_waist"
_LASER_MODEL = "gaussian_paraxial"
_SPEC_VERSION = "0.1"


def load_electron_beam(path: str) -> GaussianElectronBeam:
    with open(path, "r") as fh:
        doc = yaml.safe_load(fh)
    block = doc["electron_beam"]
    if block.get("model") != _ELECTRON_BEAM_MODEL:
        raise ValueError(
            f"{path}: unsupported electron_beam model {block.get('model')!r}, "
            f"expected {_ELECTRON_BEAM_MODEL!r}"
        )
    if block.get("propagation_direction", "+z") != "+z":
        raise ValueError(f"{path}: propagation_direction must be '+z' in v0.1")

    q_C = float(block["bunch_charge_pC"]) * 1e-12
    ke_eV = float(block["kinetic_energy_MeV"]) * 1e6
    sx_m = float(block["sigma_x_rms_um"]) * 1e-6
    sy_m = float(block["sigma_y_rms_um"]) * 1e-6
    ex_m = float(block["emit_geom_x_um"]) * 1e-6
    ey_m = float(block["emit_geom_y_um"]) * 1e-6
    st_s = float(block["bunch_duration_rms_ps"]) * 1e-12
    return GaussianElectronBeam(
        bunch_charge_C=PhysicalQuantity(q_C, "coulomb", PhysicalMeaning.BUNCH_CHARGE),
        kinetic_energy_eV=PhysicalQuantity(ke_eV, "electron_volt", PhysicalMeaning.BEAM_ENERGY),
        rel_energy_spread_rms=float(block["rel_energy_spread_rms"]),
        sigma_x_m=PhysicalQuantity(sx_m, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE, WidthConvention.SIGMA_INTENSITY_RMS),
        sigma_y_m=PhysicalQuantity(sy_m, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE, WidthConvention.SIGMA_INTENSITY_RMS),
        emit_geom_x_m=PhysicalQuantity(ex_m, "meter", PhysicalMeaning.EMITTANCE),
        emit_geom_y_m=PhysicalQuantity(ey_m, "meter", PhysicalMeaning.EMITTANCE),
        sigma_t_s=PhysicalQuantity(st_s, "second", PhysicalMeaning.BUNCH_LENGTH, TimeConvention.SIGMA_INTENSITY_RMS),
        sigma_pz=float(block.get("sigma_pz", 0.0)),
    )


def save_electron_beam(beam: GaussianElectronBeam, path: str) -> None:
    doc = {
        "electron_beam": {
            "model": _ELECTRON_BEAM_MODEL,
            "version": _SPEC_VERSION,
            "bunch_charge_pC": beam.bunch_charge_C.to_unit("picocoulomb").magnitude,
            "kinetic_energy_MeV": beam.kinetic_energy_eV.to_unit("megaelectron_volt").magnitude,
            "rel_energy_spread_rms": beam.rel_energy_spread_rms,
            "sigma_x_rms_um": beam.sigma_x_m.to_unit("micrometer").magnitude,
            "sigma_y_rms_um": beam.sigma_y_m.to_unit("micrometer").magnitude,
            "emit_geom_x_um": beam.emit_geom_x_m.to_unit("micrometer").magnitude,
            "emit_geom_y_um": beam.emit_geom_y_m.to_unit("micrometer").magnitude,
            "bunch_duration_rms_ps": beam.sigma_t_s.to_unit("picosecond").magnitude,
            "sigma_pz": beam.sigma_pz,
            "propagation_direction": "+z",
        }
    }
    with open(path, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)


def load_laser(path: str) -> GaussianParaxialLaser:
    with open(path, "r") as fh:
        doc = yaml.safe_load(fh)
    block = doc["laser"]
    if block.get("model") != _LASER_MODEL:
        raise ValueError(
            f"{path}: unsupported laser model {block.get('model')!r}, "
            f"expected {_LASER_MODEL!r}"
        )
    if block.get("propagation_direction", "-z") != "-z":
        raise ValueError(f"{path}: propagation_direction must be '-z' in v0.1")

    e_j = float(block["pulse_energy_J"])
    wl_m = float(block["wavelength_um"]) * 1e-6
    wx_m = float(block["waist_rms_x_um"]) * 1e-6
    wy_m = float(block["waist_rms_y_um"]) * 1e-6
    dur_s = float(block["duration_rms_fs"]) * 1e-15
    fz_m = float(block.get("focus_z_um", 0.0)) * 1e-6
    return GaussianParaxialLaser(
        pulse_energy_J=PhysicalQuantity(e_j, "joule", PhysicalMeaning.PULSE_ENERGY, NoConvention.PLAIN),
        wavelength_m=PhysicalQuantity(wl_m, "meter", PhysicalMeaning.WAVELENGTH, NoConvention.PLAIN),
        waist_rms_x_m=PhysicalQuantity(wx_m, "meter", PhysicalMeaning.LASER_WIDTH, WidthConvention.SIGMA_INTENSITY_RMS),
        waist_rms_y_m=PhysicalQuantity(wy_m, "meter", PhysicalMeaning.LASER_WIDTH, WidthConvention.SIGMA_INTENSITY_RMS),
        duration_rms_s=PhysicalQuantity(dur_s, "second", PhysicalMeaning.PULSE_DURATION, TimeConvention.SIGMA_INTENSITY_RMS),
        focus_z_m=PhysicalQuantity(fz_m, "meter", PhysicalMeaning.DISPLACEMENT, NoConvention.PLAIN),
    )


def save_laser(pulse: GaussianParaxialLaser, path: str) -> None:
    doc = {
        "laser": {
            "model": _LASER_MODEL,
            "version": _SPEC_VERSION,
            "pulse_energy_J": pulse.pulse_energy_J.to_unit("joule").magnitude,
            "wavelength_um": pulse.wavelength_m.to_unit("micrometer").magnitude,
            "waist_rms_x_um": pulse.waist_rms_x_m.to_unit("micrometer").magnitude,
            "waist_rms_y_um": pulse.waist_rms_y_m.to_unit("micrometer").magnitude,
            "duration_rms_fs": pulse.duration_rms_s.to_unit("femtosecond").magnitude,
            "focus_z_um": pulse.focus_z_m.to_unit("micrometer").magnitude,
            "propagation_direction": "-z",
        }
    }
    with open(path, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)
