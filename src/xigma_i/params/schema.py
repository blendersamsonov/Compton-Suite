"""What this model expects, spelled out. A ``ParameterSpec`` says: this
parameter means X, expressed with convention Y, in unit Z -- so ``spec.py``'s
``XIGMA_SPEC`` is a self-contained, checkable description of this model's
input contract instead of tribal knowledge encoded only in variable names."""

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
