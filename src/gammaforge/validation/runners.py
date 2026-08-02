"""Run each model directly (bypassing the ModelAdapter/GUI string-field
layer, same as scenarios.py's Config builders) and return its full result
object, so every tier that needs more than just total_yield (spectrum
shape, angular data, ...) can reuse the same run instead of re-running an
expensive GPU/MC computation per tier.

Electron sampling happens once per model call, via
``scenarios.build_interaction`` (which delegates to
``gammaforge.io.bunch.sample_gaussian_bunch``) -- mirroring the same
"macrobunching" convention ``gammaforge.gui.app.py``'s ``on_start()``
uses: every model's ``run()``/``run_simulation()`` *requires* an
``electrons`` bunch (no model has its own internal sampler). Each
scenario's ``.beam`` (a ``GaussianElectronBeam``) is the single source of
truth for what gets drawn; every function below draws ``n_mc`` particles
(default ``DEFAULT_N_MC``), seeded with ``seed`` (default
``DEFAULT_SEED``) -- deterministic given (beam, n_mc, seed), so all four
models draw bit-identical bunches from each other when called with the
same (n_mc, seed) pair (the default here).

Results are cached per (model, scenario, repo commit) via
``cache.get_or_compute`` -- a clean-tree re-run with the same scenario
skips recomputation entirely. Dirty trees always recompute.

``run_xigma``/``run_delta`` return a plain, picklable ``Photons`` result --
enough for anything that only needs ``total_yield``/spectrum data. A caller
that needs an on-demand angular-range recompute (``XigmaAdapter``/
``DirectAdapter``'s own ``spectrum_in_angular_range()``, e.g. Tier 3's
angular-shape canary) needs the *live* adapter instance itself (it holds
the ``TabulatedEngine``/raw per-particle arrays that recompute reuses) --
the disk cache only stores/returns the picklable ``Photons`` result, not a
live adapter, so ``run_xigma_live``/``run_delta_live`` below always compute
fresh (never cached) and return ``(Photons, adapter)`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gammaforge.models.api import Job, OutputSpec

from . import cache
from .scenarios import (
    BASELINE,
    DEFAULT_N_MC,
    DEFAULT_SEED,
    Scenario,
    build_interaction,
    build_kascade_config,
)

__all__ = ["run_kascade", "run_xigma", "run_delta", "run_analytical",
           "run_xigma_live", "run_delta_live"]


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
    ``gammaforge.io.bunch.Bunch`` -- the same boundary conversion
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
    from gammaforge.models.kascade import kascade

    def _compute():
        interaction = build_interaction(scenario, n_mc=n_mc, seed=seed)
        cfg = build_kascade_config(scenario, interaction)
        return kascade.run_simulation(cfg, n_mc=n_mc, seed=seed,
                                      electrons=_kascade_electrons(interaction.electrons))

    key = _RunKey(scenario=scenario, n_mc=n_mc, seed=seed)
    result, _ = cache.get_or_compute("kascade", key, _compute)
    return result


def run_xigma(scenario: Scenario = BASELINE, n_mc: int = DEFAULT_N_MC, seed: int = DEFAULT_SEED):
    from gammaforge.models.xigma_i.adapter import XigmaAdapter

    def _compute():
        interaction = build_interaction(scenario, n_mc=n_mc, seed=seed)
        job = Job(interaction=interaction, output=OutputSpec(), seed=seed,
                  extra={"a0_max": scenario.a0_max})
        return XigmaAdapter().run(job)

    key = _RunKey(scenario=scenario, n_mc=n_mc, seed=seed)
    result, _ = cache.get_or_compute("xigma", key, _compute)
    return result


def run_delta(scenario: Scenario = BASELINE, n_mc: int = DEFAULT_N_MC, seed: int = DEFAULT_SEED):
    from gammaforge.models.xigma_i.adapter import DirectAdapter

    def _compute():
        interaction = build_interaction(scenario, n_mc=n_mc, seed=seed)
        job = Job(interaction=interaction, output=OutputSpec(), seed=seed, extra={})
        return DirectAdapter().run(job)

    key = _RunKey(scenario=scenario, n_mc=n_mc, seed=seed)
    result, _ = cache.get_or_compute("delta", key, _compute)
    return result


def run_xigma_live(scenario: Scenario = BASELINE, n_mc: int = DEFAULT_N_MC, seed: int = DEFAULT_SEED):
    """Like run_xigma, but always computes fresh and returns
    ``(Photons, XigmaAdapter)`` -- see module docstring."""
    from gammaforge.models.xigma_i.adapter import XigmaAdapter

    interaction = build_interaction(scenario, n_mc=n_mc, seed=seed)
    job = Job(interaction=interaction, output=OutputSpec(), seed=seed,
              extra={"a0_max": scenario.a0_max})
    adapter = XigmaAdapter()
    return adapter.run(job), adapter


def run_delta_live(scenario: Scenario = BASELINE, n_mc: int = DEFAULT_N_MC, seed: int = DEFAULT_SEED):
    """Like run_delta, but always computes fresh and returns
    ``(Photons, DirectAdapter)`` -- see module docstring."""
    from gammaforge.models.xigma_i.adapter import DirectAdapter

    interaction = build_interaction(scenario, n_mc=n_mc, seed=seed)
    job = Job(interaction=interaction, output=OutputSpec(), seed=seed, extra={})
    adapter = DirectAdapter()
    return adapter.run(job), adapter


def run_analytical(scenario: Scenario = BASELINE, n_mc: int = DEFAULT_N_MC, seed: int = DEFAULT_SEED):
    from gammaforge.models.analytical import Adapter

    def _compute():
        interaction = build_interaction(scenario, n_mc=n_mc, seed=seed)
        job = Job(interaction=interaction, output=OutputSpec(), seed=seed,
                  extra={"theta_col_rad": scenario.theta_col_rad})
        return Adapter().run(job)

    key = _RunKey(scenario=scenario, n_mc=n_mc, seed=seed)
    result, _ = cache.get_or_compute("analytical", key, _compute)
    return result
