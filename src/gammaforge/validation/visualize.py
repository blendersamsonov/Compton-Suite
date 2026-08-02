"""Visualization for the cross-model validation suite.

Generates comparison plots (total yield, angle-integrated spectrum shape,
angular collimated-fraction shape) across kascade/xigma-i/delta/
analytical for a given scenario, saved as PNGs under validation/plots/
(gitignored -- generated output, not source). Reuses tier1/tier2/tier3's
own metric/grid functions rather than reimplementing them, so a plot and
its corresponding tier [PASS]/[FAIL] number are always computed exactly
the same way.

Usage:
    conda run -n core --no-capture-output python3 validation/visualize.py
    conda run -n core --no-capture-output python3 validation/visualize.py --scenario near_a0_max
    conda run -n core --no-capture-output python3 validation/visualize.py --scenario all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gammaforge.validation.scenarios import BASELINE, LOW_A0, NEAR_A0_MAX, Scenario  # noqa: E402
from gammaforge.validation.runners import run_analytical, run_kascade, run_xigma, run_delta  # noqa: E402
from gammaforge.validation.tier1_yield import _total_yield_kascade  # noqa: E402
from gammaforge.validation.tier2_spectrum import compton_edge_eV, to_dNdE_on_grid  # noqa: E402
from gammaforge.validation.tier3_angular_shape import collimated_fraction  # noqa: E402
from gammaforge.validation.metrics import window_integrated_relative_error  # noqa: E402

PLOTS_DIR = Path(__file__).resolve().parent / "plots"

SCENARIOS = {"baseline": BASELINE, "low_a0": LOW_A0, "near_a0_max": NEAR_A0_MAX}

MODEL_ORDER = ["kascade", "xigma_i", "delta", "analytical"]
MODEL_LABELS = {"kascade": "kascade", "xigma_i": "xigma-i",
                "delta": "delta", "analytical": "analytical"}
MODEL_COLORS = {"kascade": "0.3", "xigma_i": "tab:blue",
                "delta": "tab:orange", "analytical": "tab:green"}


def run_all_models(scenario: Scenario) -> dict:
    """One run of each model for `scenario` -- shared across every plot
    below, same "run once, reuse everywhere" discipline run_cross_
    validation.py already follows for the tiers."""
    return dict(
        kascade=run_kascade(scenario),
        xigma_i=run_xigma(scenario),
        delta=run_delta(scenario),
        analytical=run_analytical(scenario),
    )


def _yield_of(key: str, res) -> float:
    return _total_yield_kascade(res) if key == "kascade" else float(res.total_yield)


def plot_yield_comparison(scenario: Scenario, results: dict, out_path: Path) -> None:
    values = [_yield_of(k, results[k]) for k in MODEL_ORDER]
    labels = [MODEL_LABELS[k] for k in MODEL_ORDER]
    colors = [MODEL_COLORS[k] for k in MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("total yield [photons/shot]")
    ax.set_title(f"Total yield -- {scenario.name}")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_spectrum_comparison(scenario: Scenario, results: dict, out_path: Path) -> None:
    e_max = compton_edge_eV(scenario)
    e_centers = np.linspace(1e-3, 1.05, 256) * e_max

    fig, ax = plt.subplots(figsize=(7, 5))
    for key in MODEL_ORDER:
        dNdE = to_dNdE_on_grid(results[key], e_centers)
        ax.plot(e_centers / 1e3, dNdE * 1e3, label=MODEL_LABELS[key], color=MODEL_COLORS[key])
    ax.set_xlabel(r"photon energy $\hbar\omega_\gamma$ [keV]")
    ax.set_ylabel("dN/dE [photons/keV]")
    ax.set_title(f"Angle-integrated spectrum -- {scenario.name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_angular_shape_comparison(scenario: Scenario, out_path: Path) -> None:
    """xigma-i vs delta's collimated fraction of total_yield, as a
    function of collimation half-angle -- the Tier 3 canary, plotted across
    a dense window sweep instead of tier3_angular_shape.py's fixed
    WINDOWS_MRAD spot-checks.

    ``results`` (cached, plain Photons) isn't usable for the angular
    recompute this needs -- computes its own fresh xigma-i/delta run via
    run_xigma_live/run_delta_live instead (see runners.py's docstring)."""
    from gammaforge.validation.runners import run_xigma_live, run_delta_live

    res_xigma, xigma_adapter = run_xigma_live(scenario)
    res_delta, delta_adapter = run_delta_live(scenario)

    windows_mrad = np.geomspace(0.02, 2.0, 14)
    frac_x = [collimated_fraction(xigma_adapter, res_xigma, w * 1e-3) for w in windows_mrad]
    frac_xd = [collimated_fraction(delta_adapter, res_delta, w * 1e-3) for w in windows_mrad]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(windows_mrad, frac_x, "o-", label="xigma-i", color=MODEL_COLORS["xigma_i"])
    ax.plot(windows_mrad, frac_xd, "s-", label="delta", color=MODEL_COLORS["delta"])
    ax.set_xscale("log")
    ax.set_xlabel("collimation half-angle [mrad]")
    ax.set_ylabel("collimated fraction of total_yield")
    ax.set_title(f"Angular shape canary (Tier 3) -- {scenario.name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_regime_boundary(out_path: Path) -> None:
    """kascade-vs-xigma-i spectrum-shape weighted-L1 at LOW_A0 vs
    NEAR_A0_MAX -- the Tier 4 finding (disagreement grows approaching
    xigma's documented a0_max validity boundary), as a bar chart."""
    labels, values = [], []
    for scenario in (LOW_A0, NEAR_A0_MAX):
        res_k = run_kascade(scenario)
        res_x = run_xigma(scenario)
        e_max = compton_edge_eV(scenario)
        e_centers = np.linspace(1e-3, 1.05, 256) * e_max
        mu = 0.02 * e_max
        dNdE_k = to_dNdE_on_grid(res_k, e_centers)
        dNdE_x = to_dNdE_on_grid(res_x, e_centers)
        l1, _, _ = window_integrated_relative_error(e_centers, dNdE_k, dNdE_x, mu)
        labels.append(scenario.name)
        values.append(l1)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, values, color=["tab:blue", "tab:red"])
    ax.set_ylabel("kascade vs xigma-i spectrum-shape weighted-L1")
    ax.set_title("Tier 4: regime boundary")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_all(scenario_name: str) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    names = list(SCENARIOS) if scenario_name == "all" else [scenario_name]
    for name in names:
        scenario = SCENARIOS[name]
        print(f"Running all models for scenario '{name}'...")
        results = run_all_models(scenario)

        plot_yield_comparison(scenario, results, PLOTS_DIR / f"{name}_yield.png")
        plot_spectrum_comparison(scenario, results, PLOTS_DIR / f"{name}_spectrum.png")
        plot_angular_shape_comparison(scenario, PLOTS_DIR / f"{name}_angular_shape.png")
        print(f"  wrote {name}_yield.png, {name}_spectrum.png, {name}_angular_shape.png")

    print("Running Tier 4 regime-boundary comparison (low_a0 vs near_a0_max)...")
    plot_regime_boundary(PLOTS_DIR / "regime_boundary.png")
    print("  wrote regime_boundary.png")
    print(f"-> all plots written to {PLOTS_DIR}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all",
                    help="which scenario(s) to plot (default: all)")
    args = p.parse_args()
    generate_all(args.scenario)


if __name__ == "__main__":
    main()
