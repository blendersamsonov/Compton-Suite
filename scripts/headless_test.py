#!/usr/bin/env python3
"""Headless smoke test: exercises the exact same functions the GUI calls
(model discovery -> Job -> adapter.run() -> validate_results) without
importing tkinter or matplotlib at all.

Run this directly on a machine with the physics dependencies installed
(numpy always; cupy/CUDA or numba only if you want xigma-i/delta's checks
to actually execute rather than being reported as unavailable) -- no
display needed:

    python3 scripts/headless_test.py

Exit code is 0 if every *available* model passed all its checks, 1
otherwise. xigma-i/delta being unavailable (no cupy/GPU and no numba) is
reported but does not fail the run.
"""

from __future__ import annotations

import os
import sys
import traceback

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np

from gammaforge.io.bunch import GaussianElectronBeam, sample_gaussian_bunch
from gammaforge.io.laser import GaussianParaxialLaser
from gammaforge.io.interaction import InteractionParameters
from gammaforge.io.units import NoConvention, PhysicalMeaning, PhysicalQuantity, TimeConvention, WidthConvention
from gammaforge.models.api import Job, OutputSpec, discover_models, validate_results

# Same rough operating point as the GUI's Electrons/Laser panel defaults.
_BEAM = GaussianElectronBeam(
    bunch_charge_C=PhysicalQuantity(1e-9, "coulomb", PhysicalMeaning.BUNCH_CHARGE),
    kinetic_energy_eV=PhysicalQuantity(2000e6, "electron_volt", PhysicalMeaning.BEAM_ENERGY),
    rel_energy_spread_rms=0.001,
    sigma_x_m=PhysicalQuantity(20e-6, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE, WidthConvention.SIGMA_INTENSITY_RMS),
    sigma_y_m=PhysicalQuantity(20e-6, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE, WidthConvention.SIGMA_INTENSITY_RMS),
    emit_geom_x_m=PhysicalQuantity(0.05e-6, "meter", PhysicalMeaning.EMITTANCE),
    emit_geom_y_m=PhysicalQuantity(0.05e-6, "meter", PhysicalMeaning.EMITTANCE),
    sigma_t_s=PhysicalQuantity(10e-12, "second", PhysicalMeaning.BUNCH_LENGTH, TimeConvention.SIGMA_INTENSITY_RMS),
    sigma_pz=0.001,
)
_LASER = GaussianParaxialLaser(
    pulse_energy_J=PhysicalQuantity(0.3, "joule", PhysicalMeaning.PULSE_ENERGY, NoConvention.PLAIN),
    wavelength_m=PhysicalQuantity(1000e-9, "meter", PhysicalMeaning.WAVELENGTH, NoConvention.PLAIN),
    waist_rms_x_m=PhysicalQuantity(5e-6, "meter", PhysicalMeaning.LASER_WIDTH, WidthConvention.SIGMA_INTENSITY_RMS),
    waist_rms_y_m=PhysicalQuantity(5e-6, "meter", PhysicalMeaning.LASER_WIDTH, WidthConvention.SIGMA_INTENSITY_RMS),
    duration_rms_s=PhysicalQuantity(3e-12, "second", PhysicalMeaning.PULSE_DURATION, TimeConvention.SIGMA_INTENSITY_RMS),
)
N_MC = 5000
SEED = 1

# All models this suite knows about, so a not-registered one (e.g. xigma-i/
# delta with no cupy/GPU and no numba) is reported as skipped rather than
# silently missing from the output.
EXPECTED_MODELS = ["kascade", "xigma-i", "delta", "analytical"]
PREVIEW_NAME = "analytical"


def check(label: str, cond: bool, detail: str = "") -> bool:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail and not cond else ""))
    return cond


def test_model(name: str, adapter) -> bool:
    print(f"\n=== {name} ===")
    ok = True
    extra = {key: default for _label, default, key in adapter.model_params()}
    electrons = sample_gaussian_bunch(_BEAM, n_particles=N_MC, rng=np.random.default_rng(SEED))
    interaction = InteractionParameters(laser=_LASER, electrons=electrons)
    job = Job(interaction=interaction, output=OutputSpec(), seed=SEED, extra=extra)

    try:
        res = adapter.run(job)
    except Exception as e:
        print(f"  [FAIL] run() raised: {e}")
        traceback.print_exc()
        return False

    problems = validate_results(res)
    ok &= check("validate_results reports no problems", not problems, "; ".join(problems))
    ok &= check("total_yield is a positive finite number",
                res.total_yield is not None and np.isfinite(res.total_yield) and res.total_yield >= 0)

    if res.temporal_envelope is not None:
        te = res.temporal_envelope
        ok &= check("temporal_envelope.t_seconds non-empty", te.t_seconds.size > 0)

    if res.spatial_distribution is not None:
        sd = res.spatial_distribution
        if hasattr(sd, "density"):
            dx = sd.x_centers[1] - sd.x_centers[0]
            dy = sd.y_centers[1] - sd.y_centers[0]
            integrated = float(sd.density.sum()) * dx * dy
            ratio = integrated / res.total_yield if res.total_yield else float("nan")
            ok &= check("spatial_distribution integrates to ~total_yield",
                        0.5 < ratio < 2.0, f"ratio={ratio:.4g}")

    if res.angular_spectrum is not None:
        ang = res.angular_spectrum
        dtx, dty = np.gradient(ang.theta_x), np.gradient(ang.theta_y)
        dE = np.gradient(ang.E_eV)
        full_integral = float(np.einsum("ijk,i,j,k->", ang.d2NdEdOmega, dtx, dty, dE))
        ratio = full_integral / res.total_yield if res.total_yield else float("nan")
        ok &= check("angular_spectrum integrates to ~total_yield",
                    0.5 < ratio < 2.0, f"ratio={ratio:.4g}")

    if hasattr(adapter, "spectrum_in_angular_range"):
        try:
            result = adapter.spectrum_in_angular_range((-1e-4, 1e-4), (-1e-4, 1e-4))
            ok &= check("spectrum_in_angular_range returns a spectrum",
                        result is not None and result.spectrum is not None)
        except Exception as e:
            print(f"  [FAIL] spectrum_in_angular_range raised: {e}")
            traceback.print_exc()
            ok = False

    print(f"  -> {name}: {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return ok


def test_preview_alongside(models: dict) -> bool:
    """Mirrors app.py's on_start(): the always-on analytical preview runs
    alongside whichever other model is selected, reusing the SAME sampled
    electron bunch. The GUI hardcodes which model is the preview
    (``self.analytical_adapter``); this test hardcodes the same name
    (``PREVIEW_NAME``) rather than reading a metadata flag."""
    preview_adapter = models.get(PREVIEW_NAME)
    print(f"\n=== always-on preview ({PREVIEW_NAME}) ===")
    if preview_adapter is None:
        print("  SKIPPED (preview model not registered)")
        return True

    ok = True
    electrons = sample_gaussian_bunch(_BEAM, n_particles=N_MC, rng=np.random.default_rng(SEED))
    interaction = InteractionParameters(laser=_LASER, electrons=electrons)
    for name, adapter in models.items():
        if name == PREVIEW_NAME:
            continue
        extra = {key: default for _label, default, key in preview_adapter.model_params()}
        job = Job(interaction=interaction, output=OutputSpec(), seed=SEED, extra=extra)
        try:
            p_res = preview_adapter.run(job)
            problems = validate_results(p_res)
            this_ok = check(f"preview runs alongside '{name}'",
                            not problems and p_res.total_yield is not None and p_res.total_yield >= 0,
                            "; ".join(problems))
        except Exception as e:
            print(f"  [FAIL] preview alongside '{name}' raised: {e}")
            traceback.print_exc()
            this_ok = False
        ok &= this_ok
    print(f"  -> preview-alongside: {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return ok


def main() -> int:
    models = discover_models()
    print(f"Discovered models: {list(models.keys())}")
    all_ok = True
    for name in EXPECTED_MODELS:
        if name not in models:
            print(f"\n=== {name} ===\n  SKIPPED (not available)")
            continue
        all_ok &= test_model(name, models[name])
    all_ok &= test_preview_alongside(models)
    print("\n" + ("=" * 40))
    print("RESULT:", "ALL PASS" if all_ok else "SOME FAILURES")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
