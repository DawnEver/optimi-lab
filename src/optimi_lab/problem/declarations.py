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
from scipy.special import ndtr, ndtri

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
class _StructureFields:
    forms: tuple[tuple[str, frozenset[Form]], ...]


@dataclass(frozen=True, slots=True, init=False)
class Structure(_StructureFields):
    """Output name -> the set of :class:`Form` it is declared to have (stored as sorted pairs)."""

    def __init__(self, forms: Mapping[str, frozenset[Form]]) -> None:
        pairs = tuple(sorted((str(name), frozenset(Form(f) for f in kinds)) for name, kinds in dict(forms).items()))
        _StructureFields.__init__(self, pairs)

    @property
    def outputs(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.forms)

    def has(self, output: str, form: Form) -> bool:
        return form in dict(self.forms).get(output, frozenset())


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

    def quantile(self, u: np.ndarray) -> np.ndarray:
        """The inverse CDF at unit-cube coordinates *u*."""
        return self.mean + self.std * ndtri(u)

    def from_standard(self, z: np.ndarray) -> np.ndarray:
        """The value at standard-normal coordinates *z* (the iso-probabilistic map)."""
        return self.mean + self.std * np.asarray(z, dtype=float)


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

    def quantile(self, u: np.ndarray) -> np.ndarray:
        """The inverse CDF at unit-cube coordinates *u*."""
        return self.lo + (self.hi - self.lo) * np.asarray(u, dtype=float)

    def from_standard(self, z: np.ndarray) -> np.ndarray:
        """The value at standard-normal coordinates *z* (the iso-probabilistic map)."""
        return self.quantile(ndtr(z))


@dataclass(frozen=True, slots=True)
class _UncertaintyFields:
    parameters: tuple[Normal | Uniform, ...]


@dataclass(frozen=True, slots=True, init=False)
class Uncertainty(_UncertaintyFields):
    """Independent random parameters; drawn only through a caller-owned generator."""

    def __init__(self, parameters: tuple[Normal | Uniform, ...]) -> None:
        parameters = tuple(parameters)
        require_declared_names(tuple(p.name for p in parameters), 'uncertainty')
        _UncertaintyFields.__init__(self, parameters)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    def sample(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """*n* joint draws, one array per parameter."""
        if not isinstance(rng, np.random.Generator):
            refuse(f'sampling needs an explicit numpy Generator, got {type(rng).__name__}; global state is not a seed')
        return {p.name: p.draw(n, rng) for p in self.parameters}

    def at_unit(self, u: np.ndarray) -> dict[str, np.ndarray]:
        """Rows of unit-cube points ``(n, len(parameters))`` mapped through each inverse CDF."""
        u = self._columns(u)
        if np.any((u < 0.0) | (u > 1.0)):
            refuse('unit-cube points must lie in [0, 1]')
        return {p.name: p.quantile(u[:, i]) for i, p in enumerate(self.parameters)}

    def at_standard(self, z: np.ndarray) -> dict[str, np.ndarray]:
        """Rows of standard-normal points ``(n, len(parameters))`` mapped to parameter values.

        The parameters are independent, so this is the Rosenblatt (and Nataf) transform exactly.
        """
        z = self._columns(z)
        return {p.name: p.from_standard(z[:, i]) for i, p in enumerate(self.parameters)}

    def _columns(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if points.ndim != 2 or points.shape[1] != len(self.parameters):
            refuse(f'points need {len(self.parameters)} columns, one per parameter {self.names}, got {points.shape}')
        return points


@dataclass(frozen=True, slots=True)
class Level:
    """One fidelity level and its cost per evaluation, in any one unit shared by the levels."""

    name: str
    cost: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.cost) and self.cost > 0):
            refuse(f'fidelity level {self.name!r} needs a finite cost > 0, got {self.cost!r}')


@dataclass(frozen=True, slots=True)
class _FidelityFields:
    levels: tuple[Level, ...]


@dataclass(frozen=True, slots=True, init=False)
class Fidelity(_FidelityFields):
    """Levels cheapest first, costs strictly increasing; the last level is the truth."""

    def __init__(self, levels: tuple[Level, ...]) -> None:
        levels = tuple(levels)
        require_declared_names(tuple(level.name for level in levels), 'fidelity')
        costs = [level.cost for level in levels]
        if any(b <= a for a, b in itertools.pairwise(costs)):
            refuse(f'fidelity costs must be strictly increasing, cheapest first, got {costs}')
        _FidelityFields.__init__(self, levels)

    @property
    def truth(self) -> Level:
        return self.levels[-1]

    def index(self, name: str) -> int:
        names = [level.name for level in self.levels]
        if name not in names:
            refuse(f'no fidelity level named {name!r}; the declared levels are {names}')
        return names.index(name)
