"""The shared "model run result" contract: what every model's ``run()``
returns.

Lives here (not in the GUI package) for the same reason ``photons.py``'s
observable dataclasses do: every model already depends on ``compton_io``
(for units/constants), so constructing this shared class directly costs
nothing, whereas depending on the GUI package would be a real, avoidable
coupling in the wrong direction (a physics model has no business needing
the GUI to run). Before this move, ``xigma_i``/``xigma_direct`` each
defined their own structurally-identical-but-not-the-same local
``XigmaResults``/``DirectResults`` specifically to avoid importing the
GUI's ``compton_guide.model_api.CommonResults`` -- now that the shared
class lives here instead, that workaround is no longer needed and those
local classes are gone.

A model is free to leave any field beyond the required six as ``None`` if
it doesn't compute that observable (see the field docs below) -- this is
a model-shape difference, not an error, and ``validate_results`` never
requires more than the required six.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .photons import (
    AngularRangeSpectrumResult,
    BinnedAngularSpectrum,
    BinnedSpatialDistribution,
    BinnedSpectrum,
    BinnedTemporalEnvelope,
    ElectronFinalState,
    PhotonMultiplicity,
    SampledSpatialDistribution,
    SampledSpectrum,
    SampledTemporalEnvelope,
)

__all__ = ["CommonResults", "validate_results"]


@dataclass
class CommonResults:
    """What every model's ``run()`` must return (shape-compatibly).

    Only ``model_name``, ``cfg``, ``n_mc``, ``total_yield``, ``spectrum``
    and ``summary`` are guaranteed present and non-None. Everything else
    is optional and ``None`` when a given model doesn't compute it -- a
    caller must check before using it (see each field's doc).
    """

    model_name: str
    cfg: Any                     # exposes eps0, sigma_eps_rel, omega_L, emit_x/y,
                                  # beta_x/y, sigma_par_L, N_e, crossing_angle, quantum
    n_mc: int
    total_yield: float           # physical (weighted) total photon count
    spectrum: SampledSpectrum | BinnedSpectrum
    summary: dict                # free-form, model-specific scalar diagnostics; use
                                  # the top-level CommonResults fields (total_yield,
                                  # n_mc, ...) for anything that must be read in a
                                  # model-agnostic way
    angular_spectrum: BinnedAngularSpectrum | None = None
    photon_samples: Any | None = None          # e.g. kascade's raw per-photon Results, else None
    electron_state: ElectronFinalState | None = None
    photon_multiplicity: PhotonMultiplicity | None = None
    temporal_envelope: SampledTemporalEnvelope | BinnedTemporalEnvelope | None = None
    spatial_distribution: SampledSpatialDistribution | BinnedSpatialDistribution | None = None
    final_distribution_path: str | None = None
    warnings: list[str] = field(default_factory=list)


def validate_results(res: Any) -> list[str]:
    """Defensive duck-type check, run right after a model's ``run()``
    returns.

    Returns a list of problem descriptions; empty means OK. Never raises --
    the caller decides whether a non-empty list is fatal.

    Deliberately checks *shape* (attribute presence) for ``spectrum``, not
    ``isinstance`` against ``SampledSpectrum``/``BinnedSpectrum``: an
    unbinned (event-generator) model and a binned (semi-analytic) model
    are both valid, genuinely different shapes -- not a "which package
    defined the class" distinction.
    """
    problems: list[str] = []
    required = ("model_name", "cfg", "n_mc", "total_yield", "spectrum", "summary")
    for name in required:
        if getattr(res, name, None) is None:
            problems.append(f"missing required field: {name!r}")
    spectrum = getattr(res, "spectrum", None)
    if spectrum is not None:
        looks_sampled = hasattr(spectrum, "E_eV") and hasattr(spectrum, "weight")
        looks_binned = hasattr(spectrum, "E_eV") and hasattr(spectrum, "dNdE_per_eV")
        if not (looks_sampled or looks_binned):
            problems.append(f"spectrum has unexpected type: {type(spectrum)!r}")
    return problems
