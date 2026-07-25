"""Run each model directly (bypassing the ModelAdapter/GUI string-field
layer, same as scenarios.py's Config builders) and return its full result
object, so every tier that needs more than just total_yield (spectrum
shape, angular data, ...) can reuse the same run instead of re-running an
expensive GPU/MC computation per tier.
"""

from __future__ import annotations

from scenarios import BASELINE, DEFAULT_N_MC, DEFAULT_SEED, Scenario, build_analytical_config, build_kascade_config, build_xigma_config, build_xigma_direct_config

__all__ = ["run_kascade", "run_xigma", "run_xigma_direct", "run_analytical"]


def run_kascade(scenario: Scenario = BASELINE, n_mc: int = DEFAULT_N_MC, seed: int = DEFAULT_SEED):
    import kascade
    cfg = build_kascade_config(scenario)
    return kascade.run_simulation(cfg, n_mc=n_mc, seed=seed)


def run_xigma(scenario: Scenario = BASELINE, seed: int = DEFAULT_SEED):
    from xigma_i import gui_adapter
    cfg = build_xigma_config(scenario)
    return gui_adapter.run_simulation(cfg, seed=seed)


def run_xigma_direct(scenario: Scenario = BASELINE, seed: int = DEFAULT_SEED):
    from xigma_direct import gui_adapter
    cfg = build_xigma_direct_config(scenario)
    return gui_adapter.run_simulation(cfg, seed=seed)


def run_analytical(scenario: Scenario = BASELINE, n_mc: int = DEFAULT_N_MC, seed: int = DEFAULT_SEED):
    from compton_suite.analytical_adapter import AnalyticalAdapter
    cfg = build_analytical_config(scenario)
    return AnalyticalAdapter().run(cfg, n_mc=n_mc, seed=seed)
