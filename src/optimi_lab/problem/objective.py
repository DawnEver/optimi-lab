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
class Normalization:
    """``(value - reference) / scale``, both in the objective's unit; *scale* is finite and positive."""

    reference: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, 'reference', float(self.reference))
        object.__setattr__(self, 'scale', float(self.scale))
        if not math.isfinite(self.reference) or not (math.isfinite(self.scale) and self.scale > 0):
            refuse(
                f'a normalization needs a finite reference and a finite scale > 0, got {self.reference!r}, {self.scale!r}'
            )


@dataclass(frozen=True, slots=True)
class Objective:
    """A named quantity in *unit*, better in *sense*, compared after *normalization*."""

    name: str
    sense: Sense = Sense.MINIMIZE
    unit: str | None = None
    normalization: Normalization = field(default_factory=Normalization)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            refuse(f'an objective needs a non-empty string name, got {self.name!r}')
        object.__setattr__(self, 'sense', coerce_enum(Sense, self.sense, f'the sense of objective {self.name!r}'))

    def minimized(self, value: float) -> float:
        """The value normalised and signed so that smaller is better."""
        normalised = (float(value) - self.normalization.reference) / self.normalization.scale
        return -normalised if self.sense is Sense.MAXIMIZE else normalised
