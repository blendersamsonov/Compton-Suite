"""Model-agnostic contract between app.py and physics-engine adapters.

This module knows nothing about ``kascade`` or ``xigma_i`` -- every adapter
constructs its result (``CommonResults``, including fields like
``spectrum``/``angular_spectrum``) using the shared classes re-exported
below from ``compton_suite.io.results``/``compton_suite.io.photons``, which every
model already depends on (for units). This is why they can be the
*literal same classes* everywhere rather than each adapter defining its
own structurally-identical lookalikes: no physics package needs to import
this GUI module or depend on it, only on ``compton_suite.io``, which is a shared
dependency by design already. (An earlier version of this module defined
these dataclasses locally, and ``xigma_i.gui_adapter``/
``delta.gui_adapter`` each kept their own separately-defined,
structurally-identical ``CommonResults`` copies specifically to avoid
depending on this module -- see git history / ``ComptonSuite/AGENTS.md``'s "The
ModelAdapter contract, and the bug it caused" for the isinstance-vs-duck-
typing bug that caused. That workaround is obsolete now that these types
live in ``compton_suite.io`` instead.)

Two physics engines currently target this contract:

  * kascade (``adapters/kascade_adapter.py``) is an event-generator Monte Carlo: it
    samples individual macro-electrons and returns unbinned per-macro-photon
    and per-macro-electron arrays (``SampledSpectrum``, ``ElectronFinalState``,
    ``PhotonMultiplicity``, ``SampledTemporalEnvelope``,
    ``SampledSpatialDistribution`` are all populated).
  * xigma-i (``xigma_i.gui_adapter``) is a semi-analytic, GPU-only calculation
    that returns smooth binned spectral-density arrays with no per-electron
    final state and no photon-multiplicity statistics (``BinnedSpectrum``/
    ``BinnedAngularSpectrum``/``BinnedTemporalEnvelope`` populated; the
    unbinned/per-electron fields and ``spatial_distribution`` are ``None``).

The GUI must branch on which of these are present rather than assume kascade's
exact shape -- see ``CommonResults`` field docs below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from compton_suite.io.bunch import MacroBunch
from compton_suite.io.photons import (
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
from compton_suite.io.results import CommonResults, validate_results

__all__ = [
    "MacroBunch",
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
    "CommonResults",
    "validate_results",
    "ModelCapabilities",
    "ModelAdapter",
    "UnavailableAdapter",
    "MODEL_REGISTRY",
    "register",
    "registered_models",
]


# ---------------------------------------------------------------------------
# Model capabilities and adapter protocol
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelCapabilities:
    name: str
    display_name: str
    requires_gpu: bool
    supports_crossing_angle: bool
    supports_quantum_toggle: bool
    supports_nonlinearity_emulation: bool   # xigma-i-only axis, orthogonal to quantum toggle
    supports_electron_final_state: bool
    supports_photon_multiplicity: bool
    supports_ele_file_io: bool
    supports_seed_reproducibility: bool
    requires_recompute_on_collimation_change: bool
    trust_level: str            # "production" | "experimental-C" | "experimental-D" | ...
    trust_note: str
    supports_temporal_envelope: bool = False
    supports_spatial_distribution: bool = False
    supports_angular_distribution: bool = False
    supports_angular_range_spectrum: bool = False
    is_fast_preview: bool = False   # True only for the always-on analytical
                                     # model (models/analytical/analytical_adapter.py)


class ModelAdapter(Protocol):
    def capabilities(self) -> ModelCapabilities: ...

    def available(self) -> tuple[bool, str]:
        """Return (True, "") if the model can actually be run right now, else
        (False, reason). Must never raise."""
        ...

    def extra_params(self) -> list[tuple[str, float | str, str]]:
        """Model-specific fields with no shared-panel analogue
        (kascade's Electrons/Laser/Compton panels), as (label, default, key)
        triples -- the same shape app.py's add_field_grid already consumes.
        The GUI renders these in a dedicated pane that's rebuilt whenever the
        active model changes, and feeds the resulting values into the same
        flat ``fields`` dict passed to ``params_to_config``. Return ``[]`` if
        the model has none (e.g. kascade today).
        
        Default can be float (for numeric fields) or str (for choice/enum fields).
        For choice fields, also implement ``extra_choices()`` to provide allowed values."""
        ...

    def extra_choices(self) -> dict[str, list[str]]:
        """Optional: return a dict mapping parameter keys to allowed string values
        for choice/enum fields declared in ``extra_params()``. If a key appears
        here, the GUI will render a dropdown (Combobox) instead of an Entry.
        Example: ``{"device_preference": ["auto", "gpu", "cpu"]}``."""
        return {}

    def params_to_config(self, fields: dict, quantum: bool) -> tuple[Any, dict]: ...

    def run(self, cfg: Any, n_mc: int, seed: int,
            *, electrons: MacroBunch) -> CommonResults:
        """``electrons`` is required: electron sampling is the IO layer's
        (caller's) job, not any individual model's -- no adapter has its
        own internal sampler; there's exactly one
        place electrons get drawn from a beam description
        (``compton_io.bunch.sample_gaussian_bunch``, typically via
        ``compton_io.bunch.beam_from_shared_fields`` from whichever
        model's ``Config`` the caller already has -- see ``app.py``'s
        ``on_start()`` for the GUI's own draw-once-pass-to-every-model
        pattern)."""
        ...

    def load_ele_file(self, path: str) -> MacroBunch:
        """Raise NotImplementedError if capabilities().supports_ele_file_io is False."""
        ...

    def ele_file_summary(self, bunch: MacroBunch) -> dict:
        """Raise NotImplementedError if capabilities().supports_ele_file_io is False."""
        ...

    def spectrum_in_angular_range(
            self, theta_x_range: tuple[float, float],
            theta_y_range: tuple[float, float],
            **kwargs) -> AngularRangeSpectrumResult:
        """Optional capability -- check
        capabilities().supports_angular_range_spectrum before calling.
        Raise NotImplementedError if unsupported."""
        ...


@dataclass
class UnavailableAdapter:
    """Stub registered when a model's real adapter couldn't be imported/used
    (e.g. cupy missing). Keeps the model visible-but-disabled in the GUI menu
    instead of silently omitting it."""

    _name: str
    _reason: str
    _display_name: str | None = None

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            name=self._name, display_name=self._display_name or self._name,
            requires_gpu=True, supports_crossing_angle=False,
            supports_quantum_toggle=False, supports_nonlinearity_emulation=False,
            supports_electron_final_state=False, supports_photon_multiplicity=False,
            supports_ele_file_io=False, supports_seed_reproducibility=False,
            requires_recompute_on_collimation_change=False,
            trust_level="unavailable", trust_note=self._reason,
        )

    def available(self) -> tuple[bool, str]:
        return False, self._reason

    def extra_params(self) -> list[tuple[str, float, str]]:
        return []

    def extra_choices(self) -> dict[str, list[str]]:
        return {}

    def params_to_config(self, fields, quantum):
        raise NotImplementedError(f"{self._name}: {self._reason}")

    def run(self, cfg, n_mc, seed, *, electrons=None):
        raise NotImplementedError(f"{self._name}: {self._reason}")

    def load_ele_file(self, path):
        raise NotImplementedError(f"{self._name}: {self._reason}")

    def ele_file_summary(self, bunch):
        raise NotImplementedError(f"{self._name}: {self._reason}")

    def spectrum_in_angular_range(self, theta_x_range, theta_y_range, **kwargs):
        raise NotImplementedError(f"{self._name}: {self._reason}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, ModelAdapter] = {}


def register(name: str, adapter: ModelAdapter) -> None:
    MODEL_REGISTRY[name] = adapter


def registered_models() -> dict[str, ModelAdapter]:
    return dict(MODEL_REGISTRY)
