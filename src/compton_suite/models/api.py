"""Model-agnostic contract between app.py and physics-engine adapters.

This module knows nothing about ``kascade`` or ``xigma_i`` -- every adapter
constructs its result (``Photons``, including fields like
``spectrum``/``angular_spectrum``) using the shared classes re-exported
below from ``compton_suite.io.photons``, which every model already depends
on (for units). This is why they can be the *literal same classes*
everywhere rather than each adapter defining its own structurally-identical
lookalikes: no physics package needs to import this module or depend on
it, only on ``compton_suite.io``, which is a shared dependency by design
already.

Three physics engines currently target this contract:

  * kascade (``kascade/kascade_adapter.py``) is an event-generator Monte
    Carlo: it samples individual macro-electrons and returns unbinned
    per-macro-photon data.
  * xigma-i (``xigma_i/adapter.py``'s ``XigmaAdapter``) is a semi-analytic,
    GPU/CPU tabulated-overlap calculation that returns smooth binned
    spectral-density arrays with no per-electron final state.
  * delta (``xigma_i/adapter.py``'s ``DirectAdapter``, registered under the
    name ``"delta"``) is xigma-i's brute-force per-macroparticle resonance-
    binning mode, reusing xigma-i's own Stage 0 physics directly -- not a
    separate model package.
  * analytical (``models/analytical.py``) is a fast closed-form estimate,
    always run alongside whichever other model is selected as a real-time
    preview (see ``ModelCapabilities.is_fast_preview``).

The GUI must branch on which ``Photons`` fields are present rather than
assume kascade's exact shape -- see ``Photons``' field docs in
``compton_suite.io.photons``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from compton_suite.io.bunch import Bunch
from compton_suite.io.interaction import InteractionParameters
from compton_suite.io.photons import (
    AngularRangeSpectrumResult,
    BinnedAngularSpectrum,
    BinnedSpatialDistribution,
    BinnedSpectrum,
    BinnedTemporalEnvelope,
    Photons,
    PhotonMultiplicity,
    SampledSpatialDistribution,
    SampledSpectrum,
    SampledTemporalEnvelope,
    validate_results,
)

__all__ = [
    "Bunch",
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
    "OutputSpec",
    "Job",
    "ModelCapabilities",
    "ModelAdapter",
    "UnavailableAdapter",
    "MODEL_REGISTRY",
    "register",
    "registered_models",
    "discover_models",
]


@dataclass(frozen=True)
class OutputSpec:
    """Model-agnostic output resolution specification.

    Models receive this in their ``run()`` method (via ``Job.output``) and
    fulfill the fields they support (e.g. kascade returns sampled data,
    xigma_i returns binned). The GUI uses these values to control display
    binning and plotting resolution.
    """

    # Spectrum
    n_energy_bins: int = 256          # for a binned spectrum (xigma_i, delta, analytical)
    n_energy_samples: int = 1024      # for an unbinned spectrum (kascade) -- GUI bins to this

    # Temporal envelope
    n_time_bins: int = 128

    # Spatial distribution
    n_spatial_bins_x: int = 64
    n_spatial_bins_y: int = 64

    # Angular distribution
    n_angular_bins_x: int = 64
    n_angular_bins_y: int = 64


@dataclass(frozen=True)
class Job:
    """The single compiled-from-UI config object the GUI passes to
    ``ModelAdapter.run()`` -- "GUI just calls model's adapter run with the
    config compiled from ui".

    ``electrons`` is required: electron sampling is the IO layer's
    (caller's) job, not any individual model's -- no adapter has its own
    internal sampler; there's exactly one place electrons get drawn from a
    beam description (``compton_suite.io.bunch.sample_gaussian_bunch``,
    typically via ``compton_suite.io.bunch.beam_from_shared_fields`` from
    whichever model's ``Config`` the caller already has -- see ``app.py``'s
    ``on_start()`` for the GUI's own draw-once-pass-to-every-model
    pattern).

    ``extra`` carries the model-specific numeric/choice fields from that
    model's own ``model_params()``/``model_choices()`` (e.g. xigma_i's
    ``device_preference``), keyed by the same ``key`` those methods use.

    ``output`` specifies desired resolution for binned outputs (spectrum,
    temporal, spatial, angular). Models should fulfill the fields they
    support and ignore the rest.
    """

    interaction: InteractionParameters
    electrons: Bunch
    output: OutputSpec = field(default_factory=OutputSpec)
    seed: int = 0
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ModelCapabilities:
    """What the GUI needs to know about a model to render it, independent
    of running it: a human-facing name, whether it's the always-on fast
    preview (shown alongside whichever other model is selected, not as a
    selectable model itself), and whether it shares the GUI's global
    MC-sample-count field or wants its own (xigma_i/delta/analytical each
    pick their own particle count independent of kascade's)."""

    display_name: str
    is_fast_preview: bool = False
    trust_level: str = "production"    # "production" | "experimental" | "unavailable"
    uses_shared_sample_count: bool = True


class ModelAdapter(Protocol):
    def capabilities(self) -> ModelCapabilities:
        """Static, cheap-to-call metadata for the GUI's model-selection
        panel -- no computation, no dependency on whether this model's own
        runtime requirements (e.g. xigma_i's cupy/CUDA) are available."""
        ...

    def model_params(self) -> list[tuple[str, float | str, str]]:
        """Model-specific parameters as (label, default, key) triples.

        Default can be float (for numeric fields) or str (for choice/enum
        fields). For choice fields, also implement ``model_choices()`` to
        provide allowed values."""
        ...

    def model_choices(self) -> dict[str, list[str]]:
        """Optional: return a dict mapping parameter keys to allowed string
        values for choice/enum fields declared in ``model_params()``. If a
        key appears here, the GUI renders a dropdown (Combobox) instead of
        an Entry. Example: ``{"device_preference": ["auto", "gpu", "cpu"]}``."""
        return {}

    def run(self, job: Job) -> Photons:
        ...


class UnavailableAdapter:
    """Stand-in ``ModelAdapter`` for a model whose runtime dependencies
    (e.g. cupy/CUDA for xigma_i) aren't available on this machine. The GUI
    can still list it (greyed out); calling ``run()`` raises with the
    reason instead of silently doing nothing."""

    def __init__(self, name: str, reason: str):
        self._name = name
        self._reason = reason

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(display_name=f"{self._name} (unavailable)", trust_level="unavailable")

    def model_params(self) -> list[tuple[str, float | str, str]]:
        return []

    def model_choices(self) -> dict[str, list[str]]:
        return {}

    def run(self, job: Job) -> Photons:
        raise RuntimeError(f"{self._name} is unavailable: {self._reason}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, ModelAdapter] = {}


def register(name: str, adapter: ModelAdapter) -> None:
    MODEL_REGISTRY[name] = adapter


def registered_models() -> dict[str, ModelAdapter]:
    return dict(MODEL_REGISTRY)


def discover_models() -> dict:
    """Populate the model registry with direct imports. kascade and
    analytical have no optional runtime dependencies; xigma_i (and
    ``"delta"``, its brute-force sibling mode) need cupy+CUDA or numba --
    fall back to :class:`UnavailableAdapter` if neither is usable."""
    from .kascade import kascade_adapter

    register("kascade", kascade_adapter.KascadeAdapter())

    try:
        from .xigma_i import adapter as xigma_adapter
        register("xigma-i", xigma_adapter.XigmaAdapter())
        register("delta", xigma_adapter.DirectAdapter())
    except Exception as exc:  # pragma: no cover - depends on local GPU/env
        register("xigma-i", UnavailableAdapter("xigma-i", str(exc)))
        register("delta", UnavailableAdapter("delta", str(exc)))

    from .analytical import Adapter as AnalyticalAdapter
    register("analytical", AnalyticalAdapter())

    return registered_models()
