"""Coverage for spectrum_in_angular_range's n_energy cap
(_MAX_LIVE_N_ENERGY_CPU/_GPU in models/xigma_i/adapter.py) -- a user
report: threading the GUI's "Energy bins" field into this GUI-automatic
(collimation-field-change-triggered, not user-"Calculate"-initiated) query
path let a large user-set value reach calculate_angular_spectrum_4d's
kernel directly. Measured cost is roughly linear in n_energy and steep:
n_energy=2048 measured at 14+ minutes on CPU/numba, ~50s on GPU/cupy --
unusable for an automatic update fired while a user is still typing.
spectrum_in_angular_range now caps n_energy internally regardless of what
the caller requests, so this can never regress silently.

Needs the dev-install (see this repo's top-level CLAUDE.md). Deliberately
uses small n_particles -- the point is to verify the RESULT SIZE reflects
the cap, not to benchmark the (still slow-ish on CPU) kernel itself.
"""

from __future__ import annotations

import pytest

from gammaforge.io.photons import AXIS_ENERGY
from gammaforge.models.api import Job, OutputSpec, SliceRequest
from gammaforge.models.xigma_i.adapter import (
    _MAX_LIVE_N_ENERGY_CPU,
    _MAX_LIVE_N_ENERGY_GPU,
    DirectAdapter,
    XigmaAdapter,
)
from gammaforge.validation.scenarios import BASELINE, build_interaction


def _job(n_particles=5_000, device="cpu"):
    interaction = build_interaction(BASELINE, n_mc=n_particles, seed=1)
    output = OutputSpec(slices=[SliceRequest(axes=(AXIS_ENERGY,), bins=(32,))])
    return Job(interaction=interaction, output=output, seed=1,
               extra={"device_preference": device})


def test_xigma_spectrum_in_angular_range_caps_n_energy_on_cpu():
    ad = XigmaAdapter()
    ad.run(_job(device="cpu"))
    result = ad.spectrum_in_angular_range((-1e-3, 1e-3), (-1e-3, 1e-3), n_energy=2048)
    n_e = result.spectrum.axes[AXIS_ENERGY].shape[0]
    assert n_e == _MAX_LIVE_N_ENERGY_CPU
    assert n_e < 2048


def test_xigma_spectrum_in_angular_range_respects_smaller_request():
    """A request below the cap must NOT be inflated up to the cap."""
    ad = XigmaAdapter()
    ad.run(_job(device="cpu"))
    result = ad.spectrum_in_angular_range((-1e-3, 1e-3), (-1e-3, 1e-3), n_energy=17)
    assert result.spectrum.axes[AXIS_ENERGY].shape[0] == 17


def test_direct_spectrum_in_angular_range_caps_n_energy_on_cpu():
    ad = DirectAdapter()
    ad.run(_job(device="cpu"))
    result = ad.spectrum_in_angular_range((-1e-3, 1e-3), (-1e-3, 1e-3), n_energy=2048)
    n_e = result.spectrum.axes[AXIS_ENERGY].shape[0]
    assert n_e == _MAX_LIVE_N_ENERGY_CPU
    assert n_e < 2048


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
