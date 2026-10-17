"""The four kinds of decision variable, and the condition that switches one on.

Every kind maps its own values onto the unit cube and back -- :meth:`encode` / :meth:`decode` --
because the unit cube is the one frame an operator can treat uniformly, and the map is the only
place that knows a variable is logarithmic, stepped, categorical or a field. A kind that has no
honest map refuses rather than inventing one: a :class:`Categorical` without an embedding has no
metric, so asking it for coordinates names the ``embedding=`` that would supply one.

``active_if`` makes a variable conditional on an earlier discrete one (:class:`When`). An inactive
variable is not part of the design at all: :class:`~optimi_lab.problem.space.Space` drops it from
the canonical form, so two designs that differ only in a switched-off value are the same design.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from optimi_lab.core.errors import coerce_enum, refuse

__all__ = ['Categorical', 'Field', 'Integer', 'Real', 'Scale', 'Variable', 'When']


class Scale(StrEnum):
    """How a :class:`Real` is spaced: evenly, or evenly in its logarithm."""

    LINEAR = 'linear'
    LOG = 'log'


@dataclass(frozen=True, slots=True)
class When:
    """``active_if``: the variable exists only while the discrete variable *name* takes one of *levels*."""

    name: str
    levels: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'levels', tuple(self.levels))
        if not self.name or not self.levels:
            refuse(f'a When needs a parent name and at least one level, got {self.name!r} and {self.levels!r}')


def _require_name(name: str, kind: str) -> None:
    if not isinstance(name, str) or not name:
        refuse(f'a {kind} variable needs a non-empty string name, got {name!r}')


def _require_interval(name: str, lo: float, hi: float) -> None:
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        refuse(f'variable {name!r} needs finite bounds with lo < hi, got lo={lo!r}, hi={hi!r}')


@dataclass(frozen=True, slots=True)
class Real:
    """A continuous variable on ``[lo, hi]`` in *unit*; ``scale=LOG`` spaces it by its logarithm."""

    name: str
    lo: float
    hi: float
    unit: str | None = None
    scale: Scale = Scale.LINEAR
    active_if: When | None = None

    def __post_init__(self) -> None:
        _require_name(self.name, 'real')
        object.__setattr__(self, 'lo', float(self.lo))
        object.__setattr__(self, 'hi', float(self.hi))
        object.__setattr__(self, 'scale', coerce_enum(Scale, self.scale, f'the scale of {self.name!r}'))
        _require_interval(self.name, self.lo, self.hi)
        if self.scale is Scale.LOG and not self.lo > 0:
            refuse(f'log variable {self.name!r} needs lo > 0, got lo={self.lo!r}; a logarithm has no zero')

    width = 1

    def contains(self, value: object) -> bool:
        return isinstance(value, (int, float)) and self.lo <= float(value) <= self.hi

    def canonical(self, value: object) -> float:
        return float(value)  # type: ignore[arg-type]

    def _ends(self) -> tuple[float, float]:
        return (math.log(self.lo), math.log(self.hi)) if self.scale is Scale.LOG else (self.lo, self.hi)

    def encode(self, value: float) -> np.ndarray:
        lo, hi = self._ends()
        x = math.log(value) if self.scale is Scale.LOG else float(value)
        return np.array([(x - lo) / (hi - lo)])

    def decode(self, unit: np.ndarray) -> float:
        lo, hi = self._ends()
        x = lo + float(np.clip(unit[0], 0.0, 1.0)) * (hi - lo)
        return float(min(max(math.exp(x), self.lo), self.hi)) if self.scale is Scale.LOG else x


@dataclass(frozen=True, slots=True)
class Integer:
    """An integer variable on the grid ``lo, lo + step, ..., hi``."""

    name: str
    lo: int
    hi: int
    step: int = 1
    active_if: When | None = None

    def __post_init__(self) -> None:
        _require_name(self.name, 'integer')
        if not all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in (self.lo, self.hi, self.step)):
            refuse(
                f'integer variable {self.name!r} needs integer lo, hi and step, got {self.lo!r}, {self.hi!r}, {self.step!r}'
            )
        _require_interval(self.name, self.lo, self.hi)
        if self.step < 1 or (self.hi - self.lo) % self.step:
            refuse(
                f'integer variable {self.name!r} needs a step >= 1 that divides hi - lo = {self.hi - self.lo}, got {self.step}'
            )

    width = 1

    @property
    def levels(self) -> tuple[int, ...]:
        return tuple(range(self.lo, self.hi + 1, self.step))

    def contains(self, value: object) -> bool:
        return (
            isinstance(value, (int, np.integer))
            and not isinstance(value, bool)
            and self.lo <= value <= self.hi
            and (value - self.lo) % self.step == 0
        )

    def canonical(self, value: object) -> int:
        return int(value)  # type: ignore[call-overload]

    def encode(self, value: int) -> np.ndarray:
        return np.array([(value - self.lo) / (self.hi - self.lo)])

    def decode(self, unit: np.ndarray) -> int:
        steps = round(float(np.clip(unit[0], 0.0, 1.0)) * (self.hi - self.lo) / self.step)
        return int(self.lo + steps * self.step)


@dataclass(frozen=True, slots=True)
class Categorical:
    """One of *levels*. No metric unless *embedding* gives each level a coordinate vector."""

    name: str
    levels: tuple[Hashable, ...]
    embedding: tuple[tuple[float, ...], ...] | None = None
    active_if: When | None = None
    _lo: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    _span: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_name(self.name, 'categorical')
        object.__setattr__(self, 'levels', tuple(self.levels))
        if len(self.levels) < 2 or len(set(self.levels)) != len(self.levels):
            refuse(f'categorical {self.name!r} needs at least two distinct levels, got {self.levels!r}')
        if self.embedding is None:
            return
        rows = tuple(tuple(float(c) for c in row) for row in self.embedding)
        object.__setattr__(self, 'embedding', rows)
        matrix = np.array(rows, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != len(self.levels) or not np.all(np.isfinite(matrix)):
            refuse(
                f'the embedding of {self.name!r} needs one finite coordinate row per level ({len(self.levels)}), got {rows!r}'
            )
        if len({tuple(row) for row in rows}) != len(rows):
            refuse(
                f'the embedding of {self.name!r} maps two levels to one point, so a decode could not tell them apart'
            )
        lo, hi = matrix.min(axis=0), matrix.max(axis=0)
        object.__setattr__(self, '_lo', lo)
        object.__setattr__(self, '_span', np.where(hi > lo, hi - lo, 1.0))

    @property
    def width(self) -> int:
        return 0 if self.embedding is None else len(self.embedding[0])

    def contains(self, value: object) -> bool:
        return value in self.levels

    def canonical(self, value: object) -> Hashable:
        return value  # type: ignore[return-value]

    def _require_metric(self) -> np.ndarray:
        if self.embedding is None:
            refuse(
                f'categorical {self.name!r} has no metric: its levels {list(self.levels)} are unordered; '
                f'supply embedding= (one coordinate row per level) to place it in the unit cube'
            )
        return (np.array(self.embedding) - self._lo) / self._span

    def encode(self, value: Hashable) -> np.ndarray:
        return self._require_metric()[self.levels.index(value)]

    def decode(self, unit: np.ndarray) -> Hashable:
        coordinates = self._require_metric()
        return self.levels[int(np.argmin(np.sum((coordinates - unit) ** 2, axis=1)))]


@dataclass(frozen=True, slots=True)
class Field:
    """*n_cells* reals on ``[lo, hi]`` with a cell adjacency (*neighbors*), e.g. a topology density."""

    name: str
    n_cells: int
    lo: float
    hi: float
    neighbors: tuple[tuple[int, int], ...] = ()
    active_if: When | None = None

    def __post_init__(self) -> None:
        _require_name(self.name, 'field')
        if not isinstance(self.n_cells, int) or self.n_cells < 1:
            refuse(f'field {self.name!r} needs n_cells >= 1, got {self.n_cells!r}')
        object.__setattr__(self, 'lo', float(self.lo))
        object.__setattr__(self, 'hi', float(self.hi))
        _require_interval(self.name, self.lo, self.hi)
        edges = tuple((int(a), int(b)) for a, b in self.neighbors)
        bad = [e for e in edges if not (0 <= e[0] < self.n_cells and 0 <= e[1] < self.n_cells) or e[0] == e[1]]
        if bad:
            refuse(f'field {self.name!r} has neighbour pairs outside cells 0..{self.n_cells - 1} or self-loops: {bad}')
        object.__setattr__(self, 'neighbors', edges)

    @property
    def width(self) -> int:
        return self.n_cells

    def contains(self, value: object) -> bool:
        array = np.asarray(value, dtype=float) if not isinstance(value, (str, bytes)) else np.array([])
        return array.shape == (self.n_cells,) and bool(np.all((array >= self.lo) & (array <= self.hi)))

    def canonical(self, value: object) -> tuple[float, ...]:
        return tuple(float(v) for v in np.asarray(value, dtype=float))

    def encode(self, value: object) -> np.ndarray:
        return (np.asarray(value, dtype=float) - self.lo) / (self.hi - self.lo)

    def decode(self, unit: np.ndarray) -> tuple[float, ...]:
        return tuple(float(v) for v in self.lo + np.clip(unit, 0.0, 1.0) * (self.hi - self.lo))


#: The declared set of variable kinds; a Space admits these and refuses a stranger.
Variable = Real | Integer | Categorical | Field
