"""What is known ABOUT a problem besides its variables, objectives and constraints.

* :class:`Structure` -- algebraic forms of named outputs (linear, quadratic, convex, ...), which
  decide whether a structured solver is eligible at all.
* :class:`Uncertainty` -- random parameters, sampled only through an explicit generator.
* :class:`Fidelity` -- evaluation levels, cheapest first; the last level is the truth.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from optimi_lab.core.errors import refuse, require_declared_names

__all__ = ['Fidelity', 'Form', 'Level', 'Normal', 'Structure', 'Uncertainty', 'Uniform']


class Form(StrEnum):
    """An algebraic property of one output as a function of the design."""

    LINEAR = 'linear'
    QUADRATIC = 'quadratic'
    CONVEX = 'convex'
    SEPARABLE = 'separable'
    SMOOTH = 'smooth'
    SPARSE = 'sparse'


@dataclass(frozen=True, slots=True)
class Structure:
    """Output name -> the set of :class:`Form` it is declared to have."""

    forms: Mapping[str, frozenset[Form]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'forms', {name: frozenset(Form(f) for f in forms) for name, forms in sorted(self.forms.items())}
        )

    def __hash__(self) -> int:
        return hash(tuple(self.forms.items()))

    def has(self, output: str, form: Form) -> bool:
        return form in self.forms.get(output, frozenset())


@dataclass(frozen=True, slots=True)
class Normal:
    """A normally distributed parameter."""

    name: str
    mean: float
    std: float
    unit: str | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.mean) and math.isfinite(self.std) and self.std > 0):
            refuse(f'normal parameter {self.name!r} needs a finite mean and std > 0, got {self.mean!r}, {self.std!r}')

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mean, self.std, n)


@dataclass(frozen=True, slots=True)
class Uniform:
    """A uniformly distributed parameter on ``[lo, hi]``."""

    name: str
    lo: float
    hi: float
    unit: str | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.lo) and math.isfinite(self.hi) and self.lo < self.hi):
            refuse(f'uniform parameter {self.name!r} needs finite lo < hi, got {self.lo!r}, {self.hi!r}')

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lo, self.hi, n)


@dataclass(frozen=True, slots=True)
class Uncertainty:
    """Independent random parameters; drawn only through a caller-owned generator."""

    parameters: tuple[Normal | Uniform, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'parameters', tuple(self.parameters))
        require_declared_names(tuple(p.name for p in self.parameters), 'uncertainty')

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    def sample(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """*n* joint draws, one array per parameter."""
        if not isinstance(rng, np.random.Generator):
            refuse(f'sampling needs an explicit numpy Generator, got {type(rng).__name__}; global state is not a seed')
        return {p.name: p.draw(n, rng) for p in self.parameters}


@dataclass(frozen=True, slots=True)
class Level:
    """One fidelity level and its cost per evaluation, in any one unit shared by the levels."""

    name: str
    cost: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.cost) and self.cost > 0):
            refuse(f'fidelity level {self.name!r} needs a finite cost > 0, got {self.cost!r}')


@dataclass(frozen=True, slots=True)
class Fidelity:
    """Levels cheapest first, costs strictly increasing; the last level is the truth."""

    levels: tuple[Level, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'levels', tuple(self.levels))
        require_declared_names(tuple(level.name for level in self.levels), 'fidelity')
        costs = [level.cost for level in self.levels]
        if any(b <= a for a, b in itertools.pairwise(costs)):
            refuse(f'fidelity costs must be strictly increasing, cheapest first, got {costs}')

    @property
    def truth(self) -> Level:
        return self.levels[-1]

    def index(self, name: str) -> int:
        names = [level.name for level in self.levels]
        if name not in names:
            refuse(f'no fidelity level named {name!r}; the declared levels are {names}')
        return names.index(name)
