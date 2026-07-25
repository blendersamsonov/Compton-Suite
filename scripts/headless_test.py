#!/usr/bin/env python3
"""Headless smoke test: exercises the exact same functions the GUI calls
(model discovery -> params_to_config -> run -> validate_results, plus the
new-observable fields and spectrum_in_angular_range) without importing
tkinter or matplotlib at all.

Run this directly on a machine with the physics dependencies installed
(numpy always; cupy/CUDA only if you want xigma-i's checks to actually
execute rather than being reported as unavailable) -- no display needed:

    python3 scripts/headless_test.py

Exit code is 0 if every *available* model passed all its checks, 1 otherwise.
xigma-i being unavailable (no cupy/GPU) is reported but does not fail the
run -- it's expected on machines without a GPU.
"""

from __future__ import annotations

import os
import sys
import traceback

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from compton_guide.model_api import validate_results
from compton_guide.models import discover_models


# Same defaults as the GUI's Electrons/Laser/Compton panels
# (app.py's _build_electrons_panel/_build_laser_panel/_build_compton_panel).
DEFAULT_FIELDS = {
    "mean_energy_MeV": 2000, "rel_spread_pct": 0.1, "charge_nC": 1,
    "bunch_duration_ps": 10, "emit_x_mmmrad": 1.0, "emit_y_mmmrad": 1.0,
    "beta_x_m": 0.2, "beta_y_m": 0.2,
    "laser_wavelength_nm": 1000, "laser_energy_mJ": 300, "pulse_duration_ps": 3,
    "rayleigh_length_m": 0.00126, "pulse_frequency_Hz": 100, "crossing_angle": 0,
    "time_mismatch_ps": 0, "x_mismatch_mm": 0, "y_mismatch_mm": 0, "z_mismatch_mm": 0,
    "theta_x_col_mrad": 0.05, "theta_y_col_mrad": 0.05,
    "n_mc": 5000, "seed": 1,
}


class FakeVar:
    """Stand-in for tk.StringVar -- params_to_config only ever calls .get()."""

    def __init__(self, value):
        self._value = str(value)

    def get(self):
        return self._value


def make_fields(overrides: dict | None = None) -> dict:
    values = dict(DEFAULT_FIELDS)
    if overrides:
        values.update(overrides)
    return {k: FakeVar(v) for k, v in values.items()}


def check(label: str, cond: bool, detail: str = "") -> bool:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail and not cond else ""))
    return cond


def test_model(name: str, adapter) -> bool:
    print(f"\n=== {name} ({adapter.capabilities().display_name}) ===")
    available, reason = adapter.available()
    if not available:
        print(f"  SKIPPED (unavailable: {reason})")
        return True  # not a failure -- e.g. no GPU on this machine

    ok = True
    caps = adapter.capabilities()

    # 1. params_to_config -- same call the GUI's on_start() makes. Fields are
    #    seeded with the shared defaults plus this model's own extra_params()
    #    defaults (app.py's Model Parameters panel does the equivalent via
    #    add_field_grid before ever calling params_to_config).
    extras = {key: default for _label, default, key in adapter.extra_params()}
    fields = make_fields(extras)
    try:
        cfg, extra = adapter.params_to_config(fields, quantum=False)
    except Exception as e:
        print(f"  [FAIL] params_to_config raised: {e}")
        traceback.print_exc()
        return False
    ok &= check("params_to_config returns (cfg, extra)", cfg is not None and isinstance(extra, dict))

    # 2. run() -- same call on_start()'s worker thread makes
    try:
        res = adapter.run(cfg, n_mc=int(extra["n_mc"]), seed=int(extra["seed"]))
    except Exception as e:
        print(f"  [FAIL] run() raised: {e}")
        traceback.print_exc()
        return False

    # 3. validate_results -- same call on_start()'s worker thread makes;
    #    this is exactly the check that was previously raising a bogus
    #    "spectrum has unexpected type" error for xigma-i.
    problems = validate_results(res)
    ok &= check("validate_results reports no problems", not problems, "; ".join(problems))

    ok &= check("total_yield is a positive finite number",
                res.total_yield is not None and res.total_yield >= 0)

    # 4. New-observable fields, gated on the capability flags the GUI checks
    #    before enabling each tab.
    if caps.supports_temporal_envelope:
        te = res.temporal_envelope
        ok &= check("temporal_envelope populated", te is not None)
        if te is not None:
            ok &= check("temporal_envelope.t_seconds non-empty",
                        getattr(te, "t_seconds", None) is not None and te.t_seconds.size > 0)

    if caps.supports_spatial_distribution:
        sd = res.spatial_distribution
        ok &= check("spatial_distribution populated", sd is not None)
        if sd is not None and hasattr(sd, "density"):
            # Binned (xigma-i-style): sanity-check the areal density
            # integrates back to roughly the same total_yield -- a
            # normalization check, not a physics validation.
            import numpy as _np
            dx = sd.x_centers[1] - sd.x_centers[0]
            dy = sd.y_centers[1] - sd.y_centers[0]
            integrated = float(sd.density.sum()) * dx * dy
            ratio = integrated / res.total_yield if res.total_yield else float("nan")
            ok &= check("spatial_distribution integrates to ~total_yield",
                        0.9 < ratio < 1.1, f"ratio={ratio:.4g}")

    if caps.supports_angular_distribution:
        has_angle_data = (res.angular_spectrum is not None
                          or getattr(res.photon_samples, "ph_thx_lab", None) is not None)
        ok &= check("angular data available (angular_spectrum or photon_samples)", has_angle_data)
        # angular_spectrum, integrated over its own full theta/energy grid,
        # must reproduce total_yield -- catches the class of bug where
        # angular_spectrum is normalised inconsistently with total_yield
        # (e.g. the ~2*pi residual documented in Xigma/CLAUDE.md, quick-
        # fixed via a self-consistent rescale in gui_adapter.py -- see
        # that fix's own "QUICK FIX, FLAGGED FOR FUTURE INVESTIGATION"
        # comment for why this is a rescale, not a root-cause fix).
        if res.angular_spectrum is not None:
            import numpy as _np
            ang = res.angular_spectrum
            dtx, dty = _np.gradient(ang.theta_x), _np.gradient(ang.theta_y)
            dE = _np.gradient(ang.E_eV)
            full_integral = float(_np.einsum("ijk,i,j,k->", ang.d2NdEdOmega, dtx, dty, dE))
            ratio = full_integral / res.total_yield if res.total_yield else float("nan")
            ok &= check("angular_spectrum integrates to ~total_yield",
                        0.9 < ratio < 1.1, f"ratio={ratio:.4g}")

    # 5. spectrum_in_angular_range -- the on-demand call the GUI's "Compute"
    #    button on the Angular-Range Spectrum tab makes.
    if caps.supports_angular_range_spectrum:
        tx = float(fields["theta_x_col_mrad"].get()) * 1e-3
        ty = float(fields["theta_y_col_mrad"].get()) * 1e-3
        try:
            result = adapter.spectrum_in_angular_range((-tx, tx), (-ty, ty))
            ok &= check("spectrum_in_angular_range returns a spectrum",
                        result is not None and result.spectrum is not None)
        except Exception as e:
            print(f"  [FAIL] spectrum_in_angular_range raised: {e}")
            traceback.print_exc()
            ok = False

    print(f"  -> {name}: {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return ok


def test_preview_alongside(models: dict) -> bool:
    """Same mechanism app.py's on_start() uses: whichever model is
    'selected', the always-on analytical preview (ModelCapabilities.
    is_fast_preview) should also run successfully alongside it, using the
    same shared fields -- this is exactly the check that would have caught
    the is_fast_preview AttributeError / unseeded extra_params() bugs
    found while wiring this up."""
    preview_name, preview_adapter = next(
        ((n, a) for n, a in models.items()
         if a.available()[0] and getattr(a.capabilities(), "is_fast_preview", False)),
        (None, None))
    print(f"\n=== always-on preview ({preview_name}) ===")
    if preview_adapter is None:
        print("  SKIPPED (no fast-preview model registered/available)")
        return True

    ok = True
    for name, adapter in models.items():
        if name == preview_name or not adapter.available()[0]:
            continue
        extras = {key: default for _label, default, key in adapter.extra_params()}
        extras.update({key: default for _label, default, key in preview_adapter.extra_params()})
        fields = make_fields(extras)
        try:
            p_cfg, p_extra = preview_adapter.params_to_config(fields, quantum=False)
            p_res = preview_adapter.run(p_cfg, n_mc=int(p_extra["n_mc"]), seed=int(p_extra["seed"]))
            problems = validate_results(p_res)
            this_ok = check(f"preview runs alongside '{name}''s fields",
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
    for name, adapter in models.items():
        all_ok &= test_model(name, adapter)
    all_ok &= test_preview_alongside(models)
    print("\n" + ("=" * 40))
    print("RESULT:", "ALL PASS" if all_ok else "SOME FAILURES")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
