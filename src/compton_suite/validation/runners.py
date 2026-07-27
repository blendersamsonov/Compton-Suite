"""Run each model directly (bypassing the ModelAdapter/GUI string-field
layer, same as scenarios.py's Config builders) and return its full result
object, so every tier that needs more than just total_yield (spectrum
shape, angular data, ...) can reuse the same run instead of re-running an
expensive GPU/MC computation per tier.

Electron sampling is done here, once per model call, via
``compton_suite.io.bunch.sample_gaussian_bunch`` -- mirroring the same
"macrobunching" change made to ``compton_suite.gui.app.py``'s ``on_start()``
(GUIde commit 3e779dc): every model's ``run()``/``run_simulation()`` now
*requires* an ``electrons`` bunch (their own internal self-samplers were
deleted), so this suite is not just "allowed" to sample here, it's the
only remaining place that can. Each scenario's ``.beam`` (a
``GaussianElectronBeam``) is the single source of truth for what gets
drawn; each function below sizes its own draw to its model's own
particle-count convention (``cfg.n_particles_01`` for xigma-i/
delta, ``DEFAULT_N_MC`` for kascade/analytical -- see
scenarios.py's own comment on this split) and seeds it with ``seed``
(default ``DEFAULT_SEED``). Because ``sample_gaussian_bunch`` is
deterministic given (beam, n_particles, seed), xigma-i and delta
end up drawing bit-identical bunches whenever their two Configs' own
``n_particles_01`` defaults happen to agree (they don't always -- xigma-i
defaults to 60_000, delta to 20_000 -- so this isn't guaranteed,
just a byproduct of both being sized off the same deterministic
convention); kascade and analytical always draw bit-identical bunches
from each other, since both use the same (DEFAULT_N_MC, seed) pair.
"""

from __future__ import annotations

import numpy as np

from compton_suite.io.bunch import sample_gaussian_bunch

from .scenarios import (
    BASELINE,
    DEFAULT_N_MC,
    DEFAULT_SEED,
    Scenario,
    build_analytical_config,
    build_kascade_config,
    build_xigma_config,
    build_delta_config,
)

__all__ = ["run_kascade", "run_xigma", "run_delta", "run_analytical"]


def _kascade_electrons(bunch) -> dict:
    """kascade.run_simulation's ``electrons`` parameter is a plain dict
    (``eps``/``z0``/``x_w``/``y_w``/``thx``/``thy``), not a
    ``compton_suite.io.bunch.MacroBunch`` -- the same boundary conversion
    ``kascade_adapter._macrobunch_to_kascade_electrons`` does, duplicated
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
    cfg = build_kascade_config(scenario)
    bunch = sample_gaussian_bunch(scenario.beam, n_particles=n_mc, rng=np.random.default_rng(seed))
    return kascade.run_simulation(cfg, n_mc=n_mc, seed=seed, electrons=_kascade_electrons(bunch))


def run_xigma(scenario: Scenario = BASELINE, seed: int = DEFAULT_SEED):
    from compton_suite.models.xigma_i import gui_adapter
    cfg = build_xigma_config(scenario)
    bunch = sample_gaussian_bunch(scenario.beam, n_particles=int(cfg.n_particles_01),
                                   rng=np.random.default_rng(seed))
    return gui_adapter.run_simulation(cfg, seed=seed, electrons=bunch)


def run_delta(scenario: Scenario = BASELINE, seed: int = DEFAULT_SEED):
    from compton_suite.models.delta import gui_adapter
    cfg = build_delta_config(scenario)
    bunch = sample_gaussian_bunch(scenario.beam, n_particles=int(cfg.n_particles_01),
                                   rng=np.random.default_rng(seed))
    return gui_adapter.run_simulation(cfg, seed=seed, electrons=bunch)


def run_analytical(scenario: Scenario = BASELINE, n_mc: int = DEFAULT_N_MC, seed: int = DEFAULT_SEED):
    from compton_suite.models.analytical import analytical_adapter
    cfg = build_analytical_config(scenario)
    bunch = sample_gaussian_bunch(scenario.beam, n_particles=n_mc, rng=np.random.default_rng(seed))
    return analytical_adapter.AnalyticalAdapter().run(cfg, n_mc=n_mc, seed=seed, electrons=bunch)
