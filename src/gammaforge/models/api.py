"""Model-agnostic contract between app.py and physics-engine adapters.

This module knows nothing about ``kascade`` or ``xigma_i`` -- every adapter
constructs its result (``Results``, a list of ``PhasespaceSlice`` plus
optional macrophoton/electron bunches) using the shared classes re-exported
below from ``gammaforge.io.photons``, which every model already depends
on (for units). This is why they can be the *literal same classes*
everywhere rather than each adapter defining its own structurally-identical
lookalikes: no physics package needs to import this module or depend on
it, only on ``gammaforge.io``, which is a shared dependency by design
already.

Three physics engines currently target this contract:

  * kascade (``kascade/kascade_adapter.py``) is an event-generator Monte
    Carlo: it samples individual macro-electrons and macro-photons, then
    histograms them into whatever slices ``job.output`` requests.
  * xigma-i (``xigma_i/adapter.py``'s ``XigmaAdapter``) is a semi-analytic,
    GPU/CPU tabulated-overlap calculation that returns smooth binned
    spectral-density arrays with no per-electron final state.
  * delta (``xigma_i/adapter.py``'s ``DirectAdapter``, registered under the
    name ``"delta"``) is xigma-i's brute-force per-macroparticle resonance-
    binning mode, reusing xigma-i's own Stage 0 physics directly -- not a
    separate model package.
  * analytical (``models/analytical.py``) is a fast closed-form estimate,
    always run alongside whichever other model is selected as a real-time
    preview -- the GUI hardcodes this (``self.analytical_adapter``), rather
    than reading it off a metadata flag.

A model fulfills whichever requested slices it supports and simply omits
the rest -- there is no all-or-nothing requirement, and no separate
sampled/binned shape to branch on (see ``gammaforge.io.photons``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Protocol

from gammaforge.io.bunch import Bunch
from gammaforge.io.interaction import InteractionParameters
from gammaforge.io.photons import (
    AXIS_ENERGY,
    AXIS_THETA_X,
    AXIS_THETA_Y,
    AXIS_TIME,
    AXIS_X,
    AXIS_Y,
    AngularRangeSpectrumResult,
    PhasespaceSlice,
    Results,
    find_slice,
    make_slice,
    total_yield,
    validate_results,
)

__all__ = [
    "Bunch",
    "AXIS_ENERGY",
    "AXIS_TIME",
    "AXIS_X",
    "AXIS_Y",
    "AXIS_THETA_X",
    "AXIS_THETA_Y",
    "PhasespaceSlice",
    "AngularRangeSpectrumResult",
    "Results",
    "find_slice",
    "total_yield",
    "make_slice",
    "validate_results",
    "SliceRequest",
    "find_slice_request",
    "OutputSpec",
    "Job",
    "ModelAdapter",
    "MODEL_REGISTRY",
    "register",
    "registered_models",
    "discover_models",
]


def find_slice_request(slices: tuple["SliceRequest", ...], *axis_names: str) -> "SliceRequest | None":
    """Look up a requested slice by its exact axis-name set (order-independent).
    Mirrors ``gammaforge.io.photons.find_slice`` but over ``OutputSpec.slices``."""
    wanted = frozenset(axis_names)
    for sr in slices:
        if frozenset(sr.axes) == wanted:
            return sr
    return None


@dataclass(frozen=True)
class SliceRequest:
    """One requested phase-space slice: which axes, at what resolution,
    over what range (``None`` per-axis means the model picks its own
    range). ``axes`` must be one of the groupings in
    ``gammaforge.io.photons.ALLOWED_SLICE_AXES``."""

    axes: tuple[str, ...]
    bins: tuple[int, ...]
    range: tuple[tuple[float, float] | None, ...] | None = None


@dataclass(frozen=True)
class OutputSpec:
    """Model-agnostic output resolution specification.

    Models receive this in their ``run()`` method (via ``Job.output``) and
    fulfill whichever ``slices`` entries they support (e.g. xigma_i has no
    space+angle joint slice) -- an unsupported request is simply omitted
    from the result, not an error. The GUI uses these values to control
    display binning and plotting resolution.
    """

    slices: tuple[SliceRequest, ...] = field(default_factory=lambda: (
        SliceRequest((AXIS_ENERGY,), (256,)),
        SliceRequest((AXIS_TIME,), (128,)),
        SliceRequest((AXIS_X, AXIS_Y), (64, 64)),
        SliceRequest((AXIS_THETA_X, AXIS_THETA_Y), (64, 64)),
        SliceRequest((AXIS_ENERGY, AXIS_THETA_X, AXIS_THETA_Y), (96, 33, 33)),
    ))


@dataclass(frozen=True)
class Job:
    """The single compiled-from-UI config object the GUI passes to
    ``ModelAdapter.run()`` -- "GUI just calls model's adapter run with the
    config compiled from ui".

    ``interaction.electrons`` is the already-sampled ``Bunch`` -- electron
    sampling is the IO layer's (caller's) job, not any individual model's --
    no adapter has its own internal sampler; there's exactly one place
    electrons get drawn from a beam description
    (``gammaforge.io.bunch.sample_gaussian_bunch``), before an
    ``InteractionParameters`` can even be built (see ``app.py``'s
    ``on_start()`` for the GUI's own draw-once-pass-to-every-model pattern).

    ``extra`` carries the model-specific numeric/choice fields from that
    model's own ``model_params()``/``model_choices()`` (e.g. xigma_i's
    ``device_preference``), keyed by the same ``key`` those methods use.

    ``output`` specifies desired resolution for binned outputs (spectrum,
    temporal, spatial, angular). Models should fulfill the fields they
    support and ignore the rest.
    """

    interaction: InteractionParameters
    output: OutputSpec = field(default_factory=OutputSpec)
    seed: int = 0
    extra: dict = field(default_factory=dict)


class ModelAdapter(Protocol):
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

    def run(self, job: Job) -> Results:
        ...


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
    simply not registered if neither is usable, so one missing optional
    dependency never breaks ``import gammaforge.models``."""
    from .kascade import kascade_adapter

    register("kascade", kascade_adapter.KascadeAdapter())

    try:
        from .xigma_i import adapter as xigma_adapter
        register("xigma-i", xigma_adapter.XigmaAdapter())
        register("delta", xigma_adapter.DirectAdapter())
    except Exception as exc:  # pragma: no cover - depends on local GPU/env
        warnings.warn(f"xigma-i/delta unavailable: {exc}", stacklevel=2)

    from .analytical import Adapter as AnalyticalAdapter
    register("analytical", AnalyticalAdapter())

    return registered_models()
