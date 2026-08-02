"""Output-side observable representations: the spectrum/angular-spectrum/
temporal-envelope/spatial-distribution dataclasses every model reports
results through, and the GUI renders from directly.

Each observable comes in two shapes, matching the two kinds of physics
engine in this suite:

* ``Sampled*`` -- unbinned per-macroparticle arrays plus a single uniform
  ``weight`` (electrons/photons per macroparticle), from an event-generator
  Monte Carlo (kascade).
* ``Binned*`` -- smooth pre-binned density arrays, from a semi-analytic
  calculation (xigma_i, delta, analytical).

The GUI/``validate_results`` duck-type on which shape a given result
carries (``hasattr(spectrum, "weight")`` vs ``hasattr(spectrum,
"dNdE_per_eV")``) rather than assume one model's exact shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "BinnedSpectrum",
    "SampledSpectrum",
    "BinnedAngularSpectrum",
    "PhotonMultiplicity",
    "BinnedTemporalEnvelope",
    "SampledTemporalEnvelope",
    "BinnedSpatialDistribution",
    "SampledSpatialDistribution",
    "AngularRangeSpectrumResult",
    "Photons",
    "validate_results",
]


@dataclass
class BinnedSpectrum:
    E_eV: np.ndarray
    dNdE_per_eV: np.ndarray


@dataclass
class SampledSpectrum:
    E_eV: np.ndarray
    weight: float


@dataclass
class BinnedAngularSpectrum:
    theta_x: np.ndarray
    theta_y: np.ndarray
    E_eV: np.ndarray
    d2NdEdOmega: np.ndarray  # shape (theta_x.size, theta_y.size, E_eV.size)


@dataclass
class PhotonMultiplicity:
    mean_n_phot: float
    frac_n0: float
    frac_n1: float
    frac_n2: float
    frac_n3plus: float


@dataclass
class BinnedTemporalEnvelope:
    t_seconds: np.ndarray
    rate: np.ndarray


@dataclass
class SampledTemporalEnvelope:
    t_seconds: np.ndarray
    weight: float


@dataclass
class BinnedSpatialDistribution:
    x_centers: np.ndarray
    y_centers: np.ndarray
    density: np.ndarray


@dataclass
class SampledSpatialDistribution:
    x: np.ndarray
    y: np.ndarray
    weight: float


@dataclass
class AngularRangeSpectrumResult:
    spectrum: BinnedSpectrum
    theta_x_range: tuple[float, float]
    theta_y_range: tuple[float, float]
    n_photons_in_range: float | None = None


@dataclass
class Photons:
    """What every model's ``run()`` must return (shape-compatibly).

    Only ``model_name``, ``n_mc``, ``total_yield``, ``spectrum``
    and ``summary`` are guaranteed present and non-None. Everything else
    is optional and ``None`` when a given model doesn't compute it -- a
    caller must check before using it (see each field's doc). No ``cfg``
    field: no model carries a standalone ``Config`` object anymore --
    adapters own their own parameter state directly (see each adapter).
    """

    model_name: str
    n_mc: int                    # macroparticle/sample count this run used
    total_yield: float           # physical (weighted) total photon count
    spectrum: BinnedSpectrum | SampledSpectrum
    summary: dict                # free-form, model-specific scalar diagnostics; use
                                  # the top-level Photons fields (total_yield,
                                  # n_mc, ...) for anything that must be read in a
                                  # model-agnostic way

    angular_spectrum: BinnedAngularSpectrum | None = None
    final_photons: Any | None = None          # e.g. kascade's raw per-photon Results, else None
    final_electrons: Any | None = None        # e.g. a final-state gammaforge.io.bunch.Bunch, else None
    photon_multiplicity: PhotonMultiplicity | None = None
    temporal_envelope: BinnedTemporalEnvelope | SampledTemporalEnvelope | None = None
    spatial_distribution: BinnedSpatialDistribution | SampledSpatialDistribution | None = None
    warnings: list[str] = field(default_factory=list)


def validate_results(res: Any) -> list[str]:
    """Defensive duck-type check, run right after a model's ``run()``
    returns.

    Returns a list of problem descriptions; empty means OK. Never raises --
    the caller decides whether a non-empty list is fatal.

    Deliberately checks *shape* (attribute presence) for ``spectrum``, not
    ``isinstance`` against a specific class: an unbinned (event-generator)
    model and a binned (semi-analytic) model may report ``spectrum``
    differently shaped internally, as long as both expose ``E_eV`` plus
    either ``weight`` (unbinned) or ``dNdE_per_eV`` (binned).
    """
    problems: list[str] = []
    required = ("model_name", "n_mc", "total_yield", "spectrum", "summary")
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
