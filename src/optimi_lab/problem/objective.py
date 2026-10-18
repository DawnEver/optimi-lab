"""What is wanted: a named quantity, the way it is better, and a FIXED normalisation.

The normalisation is declared, never fitted to the data seen so far. A front normalised by its own
running min/max changes meaning every batch -- the same design scores differently on Tuesday --
so a scalarisation, a hypervolume or a distance read from it cannot be compared across runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from optimi_lab.core.errors import coerce_enum, refuse

__all__ = ['Normalization', 'Objective', 'Sense']


class Sense(StrEnum):
    """Which way an objective is better."""

    MINIMIZE = 'minimize'
    MAXIMIZE = 'maximize'


@dataclass(frozen=True, slots=True)
class _NormalizationFields:
    reference: float = 0.0
    scale: float = 1.0


@dataclass(frozen=True, slots=True, init=False)
class Normalization(_NormalizationFields):
    """``(value - reference) / scale``, both in the objective's unit; *scale* is finite and positive."""

    def __init__(self, reference: float = 0.0, scale: float = 1.0) -> None:
        reference, scale = float(reference), float(scale)
        if not math.isfinite(reference) or not (math.isfinite(scale) and scale > 0):
            refuse(f'a normalization needs a finite reference and a finite scale > 0, got {reference!r}, {scale!r}')
        _NormalizationFields.__init__(self, reference, scale)


@dataclass(frozen=True, slots=True)
class _ObjectiveFields:
    name: str
    sense: Sense = Sense.MINIMIZE
    unit: str | None = None
    normalization: Normalization = field(default_factory=Normalization)


@dataclass(frozen=True, slots=True, init=False)
class Objective(_ObjectiveFields):
    """A named quantity in *unit*, better in *sense*, compared after *normalization*."""

    def __init__(
        self,
        name: str,
        sense: Sense = Sense.MINIMIZE,
        unit: str | None = None,
        normalization: Normalization | None = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            refuse(f'an objective needs a non-empty string name, got {name!r}')
        sense = coerce_enum(Sense, sense, f'the sense of objective {name!r}')
        _ObjectiveFields.__init__(self, name, sense, unit, Normalization() if normalization is None else normalization)

    def minimized(self, value: float) -> float:
        """The value normalised and signed so that smaller is better."""
        normalised = (float(value) - self.normalization.reference) / self.normalization.scale
        return -normalised if self.sense is Sense.MAXIMIZE else normalised
