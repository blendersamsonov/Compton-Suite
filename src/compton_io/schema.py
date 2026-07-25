"""What a model expects, spelled out. A ``ParameterSpec`` says: this
parameter means X, expressed with convention Y, in unit Z -- so a model's
``ModelSpec`` (e.g. ``xigma_i.params.spec.XIGMA_SPEC``,
``compton_guide.physics_params.schemas.kascade.KASCADE_SPEC``) is a
self-contained, checkable description of that model's input contract
instead of tribal knowledge encoded only in variable names."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .enums import PhysicalMeaning


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    meaning: PhysicalMeaning
    convention: Enum
    unit: str
    description: str


ModelSpec = dict[str, ParameterSpec]
