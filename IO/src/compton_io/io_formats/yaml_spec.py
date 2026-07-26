"""YAML I/O for the ``gaussian_6d_waist``/``gaussian_paraxial`` v0.1
formats defined in ``specs/electron_beam_io_v0.1_full.md`` and
``specs/gaussian_paraxial_laser_io_v0.1.md``.

Reads/writes exactly the shape shown in each spec's minimal example
(``electron_beam: {model: gaussian_6d_waist, version: "0.1", ...}`` /
``laser: {model: gaussian_paraxial, version: "0.1", ...}``), including the
spec's own unit choices at the file boundary. Conversion to/from SI happens
only here; the in-memory dataclasses (:class:`~compton_io.bunch.
GaussianElectronBeam`, :class:`~compton_io.laser.GaussianParaxialLaser`)
always stay SI.
"""

from __future__ import annotations

import yaml

from ..bunch import GaussianElectronBeam
from ..laser import GaussianParaxialLaser

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

    return GaussianElectronBeam(
        bunch_charge_C=float(block["bunch_charge_pC"]) * 1e-12,
        kinetic_energy_eV=float(block["kinetic_energy_MeV"]) * 1e6,
        rel_energy_spread_rms=float(block["rel_energy_spread_rms"]),
        sigma_x_m=float(block["sigma_x_rms_um"]) * 1e-6,
        sigma_y_m=float(block["sigma_y_rms_um"]) * 1e-6,
        emit_geom_x_m=float(block["emit_geom_x_um"]) * 1e-6,
        emit_geom_y_m=float(block["emit_geom_y_um"]) * 1e-6,
        sigma_t_s=float(block["bunch_duration_rms_ps"]) * 1e-12,
    )


def save_electron_beam(beam: GaussianElectronBeam, path: str) -> None:
    doc = {
        "electron_beam": {
            "model": _ELECTRON_BEAM_MODEL,
            "version": _SPEC_VERSION,
            "bunch_charge_pC": beam.bunch_charge_C * 1e12,
            "kinetic_energy_MeV": beam.kinetic_energy_eV * 1e-6,
            "rel_energy_spread_rms": beam.rel_energy_spread_rms,
            "sigma_x_rms_um": beam.sigma_x_m * 1e6,
            "sigma_y_rms_um": beam.sigma_y_m * 1e6,
            "emit_geom_x_um": beam.emit_geom_x_m * 1e6,
            "emit_geom_y_um": beam.emit_geom_y_m * 1e6,
            "bunch_duration_rms_ps": beam.sigma_t_s * 1e12,
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

    return GaussianParaxialLaser(
        pulse_energy_J=float(block["pulse_energy_J"]),
        wavelength_m=float(block["wavelength_um"]) * 1e-6,
        waist_rms_x_m=float(block["waist_rms_x_um"]) * 1e-6,
        waist_rms_y_m=float(block["waist_rms_y_um"]) * 1e-6,
        duration_rms_s=float(block["duration_rms_fs"]) * 1e-15,
        focus_z_m=float(block.get("focus_z_um", 0.0)) * 1e-6,
    )


def save_laser(pulse: GaussianParaxialLaser, path: str) -> None:
    doc = {
        "laser": {
            "model": _LASER_MODEL,
            "version": _SPEC_VERSION,
            "pulse_energy_J": pulse.pulse_energy_J,
            "wavelength_um": pulse.wavelength_m * 1e6,
            "waist_rms_x_um": pulse.waist_rms_x_m * 1e6,
            "waist_rms_y_um": pulse.waist_rms_y_m * 1e6,
            "duration_rms_fs": pulse.duration_rms_s * 1e15,
            "focus_z_um": pulse.focus_z_m * 1e6,
            "propagation_direction": "-z",
        }
    }
    with open(path, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)
