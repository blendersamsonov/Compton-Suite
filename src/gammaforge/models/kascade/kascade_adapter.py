"""ModelAdapter wrapping the existing kascade engine.

Moves today's ``Config``/``run_simulation`` call pattern behind the
``models.api.ModelAdapter`` shape without changing a single line of
kascade.py's physics.
"""

from __future__ import annotations

import numpy as np

from . import kascade as _kascade
from gammaforge.io.bunch import Bunch
from gammaforge.io.photons import PhasespaceSlice
from gammaforge.io.units import C_LIGHT
from gammaforge.models.api import (
    AXIS_ENERGY,
    AXIS_THETA_X,
    AXIS_THETA_Y,
    AXIS_TIME,
    AXIS_X,
    AXIS_Y,
    Job,
    Results,
    make_slice,
)


def _bunch_to_kascade_electrons(bunch: Bunch) -> dict:
    """Convert a Bunch into the plain dict kascade.run_simulation's
    ``electrons`` parameter expects (``eps, z0, x_w, y_w, thx, thy`` --
    kascade.py itself is untouched, this is GUI-boundary glue only).
    ``bunch.meta`` carries ``sdds_params`` (the loaded file's header, used
    by run_simulation to pick a reference energy) when the bunch came from
    ``load_elegant_ele``."""
    return dict(
        eps=np.asarray(bunch.gamma, dtype=float),
        z0=np.asarray(bunch.z, dtype=float),
        x_w=np.asarray(bunch.x, dtype=float),
        y_w=np.asarray(bunch.y, dtype=float),
        thx=np.asarray(bunch.thx, dtype=float),
        thy=np.asarray(bunch.thy, dtype=float),
        params=bunch.meta.get("sdds_params"),
    )


class KascadeAdapter:
    def __init__(self):
        self._last_results = None   # kascade.Results | None

    def model_params(self) -> list[tuple[str, float | str, str]]:
        return [
            ("Crossing angle (rad)", 0.0, "crossing_angle"),
            ("Laser focus offset x (m)", 0.0, "delta_x"),
            ("Laser focus offset y (m)", 0.0, "delta_y"),
            ("Laser focus offset z (m)", 0.0, "delta_z"),
            ("Quantum (Klein-Nishina + recoil)", "off", "quantum"),
            ("On-axis collimation gamma*theta", 0.2, "theta_onaxis_gamma"),
            ("Angular window Theta_x (rad, 0=off)", 0.0, "Theta_x"),
            ("Angular window Theta_y (rad, 0=off)", 0.0, "Theta_y"),
        ]

    def model_choices(self) -> dict[str, list[str]]:
        return {"quantum": ["off", "on"]}

    def run(self, job: Job) -> Results:
        extra = job.extra
        cfg = _kascade.Config(
            interaction=job.interaction,
            crossing_angle=float(extra.get("crossing_angle", 0.0)),
            delta_x=float(extra.get("delta_x", 0.0)),
            delta_y=float(extra.get("delta_y", 0.0)),
            delta_z=float(extra.get("delta_z", 0.0)),
            quantum=str(extra.get("quantum", "off")).lower() in ("on", "true", "1"),
            theta_onaxis_gamma=float(extra.get("theta_onaxis_gamma", 0.2)),
            Theta_x=float(extra.get("Theta_x", 0.0)),
            Theta_y=float(extra.get("Theta_y", 0.0)),
        )
        kascade_electrons = _bunch_to_kascade_electrons(job.interaction.electrons)
        res = _kascade.run_simulation(
            cfg, n_mc=job.interaction.electrons.n_particles, seed=job.seed, electrons=kascade_electrons)
        self._last_results = res
        s = res.summary

        axis_samples = {
            AXIS_ENERGY: res.ph_E_eV,
            AXIS_TIME: res.ph_t,
            AXIS_X: res.ph_x,
            AXIS_Y: res.ph_y,
            AXIS_THETA_X: res.ph_thx_lab,
            AXIS_THETA_Y: res.ph_thy_lab,
        }

        photon_slices = [PhasespaceSlice(axes={}, distr=np.asarray(s["N_gamma_total"]))]
        for sr in job.output.slices:
            if not all(axis in axis_samples for axis in sr.axes):
                continue
            samples = {axis: axis_samples[axis] for axis in sr.axes}
            bins = dict(zip(sr.axes, sr.bins))
            ranges = None
            if sr.range is not None:
                ranges = {
                    axis: r for axis, r in zip(sr.axes, sr.range) if r is not None
                }
            photon_slices.append(make_slice(samples, res.weight, bins, ranges))

        macrophoton = Bunch(
            x=res.ph_x, y=res.ph_y, z=C_LIGHT * res.ph_t,
            thx=res.ph_thx_lab, thy=res.ph_thy_lab, gamma=res.ph_E_eV,
            weight=res.weight,
        )
        electrons = Bunch(
            x=res.x_f, y=res.y_f, z=res.z_f,
            thx=res.thx_f, thy=res.thy_f, gamma=res.eps_f,
            weight=res.weight,
        )

        return Results(
            photon_slices=photon_slices,
            macrophoton=macrophoton,
            electrons=electrons,
            model_specific={
                **s,
                "photon_multiplicity": {
                    "mean_n_phot": s["measured_mean_n_phot"],
                    "frac_n0": s["frac_n0_measured"],
                    "frac_n1": s["frac_n1"],
                    "frac_n2": s["frac_n2"],
                    "frac_n3plus": s["frac_n3plus"],
                },
            },
        )
