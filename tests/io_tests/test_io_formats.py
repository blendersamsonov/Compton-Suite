"""Cross-checks for io_formats/sdds.py and io_formats/yaml_spec.py.

No cupy/GPU/tkinter needed. Run with `python3 -m pytest tests/` or
`python3 tests/test_io_formats.py` directly (plain asserts).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from gammaforge.io.bunch import GaussianElectronBeam, fit_gaussian, sample_gaussian_bunch  # noqa: E402
from gammaforge.io.io_formats.sdds import load_elegant_ele, save_elegant_ele  # noqa: E402
from gammaforge.io.io_formats.yaml_spec import (  # noqa: E402
    load_electron_beam,
    load_laser,
    save_electron_beam,
    save_laser,
)
from gammaforge.io.laser import GaussianParaxialLaser  # noqa: E402
from gammaforge.io.units import (  # noqa: E402
    PhysicalMeaning,
    PhysicalQuantity,
    TimeConvention,
    WidthConvention,
)

_EXAMPLE_BEAM = GaussianElectronBeam(
    bunch_charge_C=PhysicalQuantity(100e-12, "coulomb", PhysicalMeaning.BUNCH_CHARGE),
    kinetic_energy_eV=PhysicalQuantity(200e6, "electron_volt", PhysicalMeaning.BEAM_ENERGY),
    rel_energy_spread_rms=0.001,
    sigma_x_m=PhysicalQuantity(10e-6, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE, WidthConvention.SIGMA_INTENSITY_RMS),
    sigma_y_m=PhysicalQuantity(10e-6, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE, WidthConvention.SIGMA_INTENSITY_RMS),
    emit_geom_x_m=PhysicalQuantity(0.05e-6, "meter", PhysicalMeaning.EMITTANCE),
    emit_geom_y_m=PhysicalQuantity(0.05e-6, "meter", PhysicalMeaning.EMITTANCE),
    sigma_t_s=PhysicalQuantity(1e-12, "second", PhysicalMeaning.BUNCH_LENGTH, TimeConvention.SIGMA_INTENSITY_RMS),
    sigma_pz=0.001,
)


def test_sdds_round_trip_preserves_bunch_statistics():
    rng = np.random.default_rng(3)
    bunch = sample_gaussian_bunch(_EXAMPLE_BEAM, 50_000, rng=rng)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "bunch.ele")
        save_elegant_ele(bunch, path)
        loaded = load_elegant_ele(path)

    assert loaded.n_particles == bunch.n_particles
    assert abs(np.std(loaded.x) / np.std(bunch.x) - 1.0) < 1e-3
    assert abs(np.mean(loaded.gamma) / np.mean(bunch.gamma) - 1.0) < 1e-3

    # weight is documented as non-authoritative for a loaded .ele file --
    # fit_gaussian should still recover the beam's SHAPE parameters fine.
    fit = fit_gaussian(loaded)
    assert abs(fit.sigma_x_m.to_unit("meter").magnitude
               / _EXAMPLE_BEAM.sigma_x_m.to_unit("meter").magnitude - 1.0) < 0.05
    assert abs(fit.emit_geom_x_m.to_unit("meter").magnitude
               / _EXAMPLE_BEAM.emit_geom_x_m.to_unit("meter").magnitude - 1.0) < 0.08


def test_spec_example_electron_beam_yaml_round_trips():
    spec_path = (
        Path(__file__).resolve().parent.parent.parent / "docs" / "io" / "specs" / "electron_beam_io_v0.1_full.md"
    )
    example_yaml = """
electron_beam:
  model: gaussian_6d_waist
  version: "0.1"
  bunch_charge_pC: 100.0
  kinetic_energy_MeV: 200.0
  rel_energy_spread_rms: 0.001
  sigma_x_rms_um: 10.0
  sigma_y_rms_um: 10.0
  emit_geom_x_um: 0.05
  emit_geom_y_um: 0.05
  bunch_duration_rms_ps: 1.0
  propagation_direction: "+z"
"""
    assert spec_path.exists(), "spec doc should still exist alongside this test"
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "beam.yaml")
        Path(path).write_text(example_yaml)
        beam = load_electron_beam(path)

    assert abs(beam.bunch_charge_C.to_unit("coulomb").magnitude - 100.0e-12) < 1e-20
    assert abs(beam.kinetic_energy_eV.to_unit("electron_volt").magnitude - 200.0e6) < 1e-3
    assert abs(beam.sigma_x_m.to_unit("meter").magnitude - 10.0e-6) < 1e-15

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "beam_out.yaml")
        save_electron_beam(beam, path)
        reloaded = load_electron_beam(path)
    assert abs(reloaded.bunch_charge_C.to_unit("coulomb").magnitude
               / beam.bunch_charge_C.to_unit("coulomb").magnitude - 1.0) < 1e-9


def test_spec_example_laser_yaml_round_trips():
    example_yaml = """
laser:
  model: gaussian_paraxial
  version: "0.1"
  pulse_energy_J: 0.05
  wavelength_um: 0.8
  waist_rms_x_um: 2.5
  waist_rms_y_um: 2.5
  duration_rms_fs: 12.74
  focus_z_um: 0.0
  propagation_direction: "-z"
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "laser.yaml")
        Path(path).write_text(example_yaml)
        pulse = load_laser(path)

    assert abs(pulse.pulse_energy_J.to_unit("joule").magnitude - 0.05) < 1e-12
    assert abs(pulse.wavelength_m.to_unit("meter").magnitude - 0.8e-6) < 1e-15

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "laser_out.yaml")
        save_laser(pulse, path)
        reloaded = load_laser(path)
    assert abs(reloaded.a0_focus / pulse.a0_focus - 1.0) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
    print("\nAll io_formats tests passed.")
