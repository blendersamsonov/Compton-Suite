"""Tier 2: angle-integrated spectrum shape.

Every model's spectrum field is duck-typed per this project's own
established convention (never isinstance -- see GUIde/CLAUDE.md's "The
ModelAdapter contract, and the bug it caused"): kascade returns an
unbinned SampledSpectrum (per-macro-photon energies + a scalar weight),
xigma-i/xigma-i-direct/analytical return an already-binned BinnedSpectrum
(E_eV, dNdE_per_eV). to_dNdE_on_grid() normalizes both shapes onto one
common energy grid via metrics.resample_to (binned) or a weighted
histogram (unbinned), then window_integrated_relative_error compares
them the same way Xigma's own internal figures already compare spectra.

xigma-i is used as the reference/anchor for relative-error reporting,
same choice Tier 1 makes for total_yield, for consistency -- not because
it's uniquely "correct", just a fixed basis for pairwise comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from compton_suite.io.constants import HBAR, E_CHARGE, C_LIGHT
from .metrics import resample_to, window_integrated_relative_error  # noqa: E402
from .scenarios import BASELINE, Scenario  # noqa: E402
from .runners import run_analytical, run_kascade, run_xigma, run_xigma_direct  # noqa: E402

# Same tolerance tiers as Tier 1, applied to the weighted-L1
# window-integrated relative error instead of a single total-yield ratio.
TIGHT_TOL = 0.10
LOOSE_TOL = 0.30
ANALYTICAL_TOL = 2.0


def compton_edge_eV(scenario: Scenario) -> float:
    """E_max = 4*gamma0**2*Wph (the classical/head-on Compton edge energy),
    same formula every adapter already uses internally (e.g.
    xigma_i.gui_adapter's s_scale_MeV) -- computed here directly from the
    scenario so Tier 2 doesn't depend on any one model's internals."""
    omega0 = 2.0 * np.pi * C_LIGHT / scenario.pulse.wavelength_m
    Wph_eV = HBAR * omega0 / E_CHARGE
    return 4.0 * scenario.beam.gamma0 ** 2 * Wph_eV


def _weighted_histogram(E_eV: np.ndarray, weight: float, e_centers_eV: np.ndarray) -> np.ndarray:
    edges = np.empty(e_centers_eV.size + 1)
    edges[1:-1] = 0.5 * (e_centers_eV[:-1] + e_centers_eV[1:])
    edges[0] = e_centers_eV[0] - (edges[1] - e_centers_eV[0])
    edges[-1] = e_centers_eV[-1] + (e_centers_eV[-1] - edges[-2])
    hist, _ = np.histogram(E_eV, bins=edges, weights=np.full(E_eV.size, weight))
    return hist / np.diff(edges)


def to_dNdE_on_grid(result_or_spectrum, e_centers_eV: np.ndarray) -> np.ndarray:
    """dN/dE at e_centers_eV, from any of: a CommonResults-shaped object
    with a .spectrum field (xigma-i/xigma-i-direct/analytical, via
    ModelAdapter.run()), the .spectrum field itself (BinnedSpectrum or
    SampledSpectrum), or kascade's raw Results (run via kascade.
    run_simulation directly, bypassing KascadeAdapter's CommonResults/
    SampledSpectrum wrapping -- has ph_E_eV/weight fields directly, not a
    nested .spectrum)."""
    obj = result_or_spectrum
    if hasattr(obj, "ph_E_eV"):
        return _weighted_histogram(obj.ph_E_eV, obj.weight, e_centers_eV)
    spectrum = obj.spectrum if hasattr(obj, "spectrum") else obj
    if hasattr(spectrum, "dNdE_per_eV"):
        return resample_to(e_centers_eV, spectrum.E_eV, spectrum.dNdE_per_eV)
    if hasattr(spectrum, "weight"):
        return _weighted_histogram(spectrum.E_eV, spectrum.weight, e_centers_eV)
    raise TypeError(f"not a recognized spectrum/result shape: {type(obj)!r}")


def _report(label: str, l1: float, mx: float, tol: float) -> bool:
    ok = l1 <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: weighted-L1={l1:.3g}, max={mx:.3g} (tol {tol:.2g})")
    return ok


def run(scenario: Scenario = BASELINE, results: dict | None = None) -> bool:
    print(f"=== Tier 2: spectrum shape ({scenario.name}) ===")
    results = results or {}
    res_kascade = results.get("kascade") or run_kascade(scenario)
    res_xigma = results.get("xigma_i") or run_xigma(scenario)
    res_xigma_direct = results.get("xigma_direct") or run_xigma_direct(scenario)
    res_analytical = results.get("analytical") or run_analytical(scenario)

    e_max = compton_edge_eV(scenario)
    e_centers = np.linspace(1e-3, 1.05, 256) * e_max
    mu = 0.02 * e_max  # reporting resolution, 2% of the Compton edge

    dNdE_kascade = to_dNdE_on_grid(res_kascade, e_centers)
    dNdE_xigma = to_dNdE_on_grid(res_xigma, e_centers)
    dNdE_xigma_direct = to_dNdE_on_grid(res_xigma_direct, e_centers)
    dNdE_analytical = to_dNdE_on_grid(res_analytical, e_centers)

    ok = True

    l1, mx, _ = window_integrated_relative_error(e_centers, dNdE_xigma_direct, dNdE_xigma, mu)
    ok &= _report("xigma-i-direct vs xigma-i", l1, mx, TIGHT_TOL)

    l1, mx, _ = window_integrated_relative_error(e_centers, dNdE_kascade, dNdE_xigma, mu)
    ok &= _report("kascade vs xigma-i", l1, mx, LOOSE_TOL)

    l1, mx, _ = window_integrated_relative_error(e_centers, dNdE_kascade, dNdE_xigma_direct, mu)
    ok &= _report("kascade vs xigma-i-direct", l1, mx, LOOSE_TOL)

    l1, mx, _ = window_integrated_relative_error(e_centers, dNdE_analytical, dNdE_xigma, mu)
    ok &= _report("analytical vs xigma-i (order-of-magnitude bound)", l1, mx, ANALYTICAL_TOL)

    print(f"-> Tier 2: {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
