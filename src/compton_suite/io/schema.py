"""What a model expects, spelled out. A ``ParameterSpec`` says: this
parameter means X, expressed with convention Y, in unit Z -- so a model's
``ModelSpec`` (e.g. ``compton_suite.models.xigma_i.params.spec.XIGMA_SPEC``,
``compton_suite.models.kascade.params.spec.KASCADE_SPEC``) is a
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
