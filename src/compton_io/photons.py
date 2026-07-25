"""Output-side observable representations: the spectrum/angular-spectrum/
temporal-envelope/spatial-distribution dataclasses every model reports
results through, and the GUI renders from directly.

Moved (not duplicated) from ``compton_guide.model_api`` -- every model now
imports these same classes directly (this package is already a shared
dependency of every model, for units), instead of each model defining its
own structurally-identical-but-separate lookalikes. That duplication used
to exist specifically so a physics-engine repo wouldn't have to depend on
the GUI repo; now that these types live here instead, that reason is gone.
See ``GUIde/CLAUDE.md``, "The ModelAdapter contract, and the bug it
caused" for the isinstance-vs-duck-typing history this consolidation
resolves at the root.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SampledSpectrum",
    "BinnedSpectrum",
    "BinnedAngularSpectrum",
    "ElectronFinalState",
    "PhotonMultiplicity",
    "SampledTemporalEnvelope",
    "BinnedTemporalEnvelope",
    "SampledSpatialDistribution",
    "BinnedSpatialDistribution",
    "AngularRangeSpectrumResult",
]


@dataclass
class SampledSpectrum:
    """Unbinned per-macrophoton spectrum (event-generator models)."""

    E_eV: np.ndarray
    weight: float


@dataclass
class BinnedSpectrum:
    """Binned dN/dE spectrum (semi-analytic/tabulated models)."""

    E_eV: np.ndarray  # bin centers
    dNdE_per_eV: np.ndarray


@dataclass
class BinnedAngularSpectrum:
    theta_x: np.ndarray
    theta_y: np.ndarray
    E_eV: np.ndarray
    d2NdEdOmega: np.ndarray  # shape (theta_x.size, theta_y.size, E_eV.size)


@dataclass
class ElectronFinalState:
    eps_i: np.ndarray
    eps_f: np.ndarray
    weight: float


@dataclass
class PhotonMultiplicity:
    mean_n_phot: float
    frac_n0: float
    frac_n1: float
    frac_n2: float
    frac_n3plus: float


@dataclass
class SampledTemporalEnvelope:
    t_seconds: np.ndarray
    weight: float


@dataclass
class BinnedTemporalEnvelope:
    t_seconds: np.ndarray
    rate: np.ndarray


@dataclass
class SampledSpatialDistribution:
    x: np.ndarray
    y: np.ndarray
    weight: float


@dataclass
class BinnedSpatialDistribution:
    x_centers: np.ndarray
    y_centers: np.ndarray
    density: np.ndarray


@dataclass
class AngularRangeSpectrumResult:
    spectrum: "SampledSpectrum | BinnedSpectrum"
    theta_x_range: tuple[float, float]
    theta_y_range: tuple[float, float]
    n_photons_in_range: float | None = None
