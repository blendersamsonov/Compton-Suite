"""Run each model directly (bypassing the ModelAdapter/GUI string-field
layer, same as scenarios.py's Config builders) and return its full result
object, so every tier that needs more than just total_yield (spectrum
shape, angular data, ...) can reuse the same run instead of re-running an
expensive GPU/MC computation per tier.

Electron sampling is done here, once per model call, via
``compton_suite.io.bunch.sample_gaussian_bunch`` -- mirroring the same
"macrobunching" convention ``compton_suite.gui.app.py``'s ``on_start()``
uses: every model's ``run()``/``run_simulation()`` *requires* an
``electrons`` bunch (no model has its own internal sampler), so this suite
is not just "allowed" to sample here, it's the only remaining place that
can. Each scenario's ``.beam`` (a ``GaussianElectronBeam``) is the single
source of truth for what gets drawn; every function below draws
``n_mc`` particles (default ``DEFAULT_N_MC``), seeded with ``seed``
(default ``DEFAULT_SEED``) -- deterministic given (beam, n_mc, seed), so
all four models draw bit-identical bunches from each other when called
with the same (n_mc, seed) pair (the default here).

Results are cached per (model, scenario, repo commit) via
``cache.get_or_compute`` -- a clean-tree re-run with the same scenario
skips recomputation entirely. Dirty trees always recompute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compton_suite.io.bunch import sample_gaussian_bunch
from compton_suite.models.api import Job, OutputSpec

from . import cache
from .scenarios import (
    BASELINE,
    DEFAULT_N_MC,
    DEFAULT_SEED,
    Scenario,
    build_analytical_config,
    build_delta_config,
    build_kascade_config,
    build_xigma_config,
)

__all__ = ["run_kascade", "run_xigma", "run_delta", "run_analytical"]


@dataclass(frozen=True)
class _RunKey:
    """Composite cache key that includes sampling parameters (n_mc, seed)
    alongside the scenario, so different seed/n_mc combos don't collide."""
    scenario: Scenario
    n_mc: int
    seed: int


def _kascade_electrons(bunch) -> dict:
    """kascade.run_simulation's ``electrons`` parameter is a plain dict
    (``eps``/``z0``/``x_w``/``y_w``/``thx``/``thy``), not a
    ``compton_suite.io.bunch.Bunch`` -- the same boundary conversion
    ``kascade_adapter._bunch_to_kascade_electrons`` does, duplicated
    here (not imported) rather than depending on the GUI repo, since this
    validation suite deliberately runs each model directly, bypassing the
    ModelAdapter/GUI layer entirely (see module docstring)."""
    return dict(
        eps=np.asarray(bunch.gamma, dtype=float),
        z0=np.asarray(bunch.z, dtype=float),
        x_w=np.asarray(bunch.x, dtype=float),
        y_w=np.asarray(bunch.y, dtype=float),
        thx=np.asarray(bunch.thx, dtype=float),
        thy=np.asarray(bunch.thy, dtype=float),
    )


def run_kascade(scenario: Scenario = BASELINE, n_mc: int = DEFAULT_N_MC, seed: int = DEFAULT_SEED):
    from compton_suite.models.kascade import kascade

    def _compute():
        cfg = build_kascade_config(scenario)
        bunch = sample_gaussian_bunch(scenario.beam, n_particles=n_mc,
                                      rng=np.random.default_rng(seed))
        return kascade.run_simulation(cfg, n_mc=n_mc, seed=seed,
                                      electrons=_kascade_electrons(bunch))

    key = _RunKey(scenario=scenario, n_mc=n_mc, seed=seed)
    result, _ = cache.get_or_compute("kascade", key, _compute)
    return result


def run_xigma(scenario: Scenario = BASELINE, n_mc: int = DEFAULT_N_MC, seed: int = DEFAULT_SEED):
    from compton_suite.models.xigma_i.adapter import run_simulation

    def _compute():
        cfg = build_xigma_config(scenario)
        bunch = sample_gaussian_bunch(scenario.beam, n_particles=n_mc,
                                      rng=np.random.default_rng(seed))
        job = Job(interaction=cfg.interaction, electrons=bunch, output=OutputSpec(), seed=seed,
                  extra={"beta_ff": cfg.beta_ff, "phi_pol": cfg.phi_pol, "a0_max": cfg.a0_max})
        return run_simulation(job)

    key = _RunKey(scenario=scenario, n_mc=n_mc, seed=seed)
    result, _ = cache.get_or_compute("xigma", key, _compute)
    return result


def run_delta(scenario: Scenario = BASELINE, n_mc: int = DEFAULT_N_MC, seed: int = DEFAULT_SEED):
    from compton_suite.models.xigma_i.adapter import run_simulation_direct

    def _compute():
        cfg = build_delta_config(scenario)
        bunch = sample_gaussian_bunch(scenario.beam, n_particles=n_mc,
                                      rng=np.random.default_rng(seed))
        job = Job(interaction=cfg.interaction, electrons=bunch, output=OutputSpec(), seed=seed,
                  extra={"beta_ff": cfg.beta_ff, "phi_pol": cfg.phi_pol})
        return run_simulation_direct(job)

    key = _RunKey(scenario=scenario, n_mc=n_mc, seed=seed)
    result, _ = cache.get_or_compute("delta", key, _compute)
    return result


def run_analytical(scenario: Scenario = BASELINE, n_mc: int = DEFAULT_N_MC, seed: int = DEFAULT_SEED):
    from compton_suite.models.analytical import Adapter

    def _compute():
        cfg = build_analytical_config(scenario)
        bunch = sample_gaussian_bunch(scenario.beam, n_particles=n_mc,
                                      rng=np.random.default_rng(seed))
        job = Job(interaction=cfg.interaction, electrons=bunch, output=OutputSpec(), seed=seed,
                  extra={"theta_col_rad": cfg.theta_col_rad})
        return Adapter().run(job)

    key = _RunKey(scenario=scenario, n_mc=n_mc, seed=seed)
    result, _ = cache.get_or_compute("analytical", key, _compute)
    return result
