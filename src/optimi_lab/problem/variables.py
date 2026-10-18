"""The four kinds of decision variable, and the condition that switches one on.

Every kind maps its own values onto the unit cube and back -- :meth:`encode` / :meth:`decode` --
because the unit cube is the one frame an operator can treat uniformly, and the map is the only
place that knows a variable is logarithmic, stepped, categorical or a field. A kind that has no
honest map refuses rather than inventing one: a :class:`Categorical` without an embedding has no
metric, so asking it for coordinates names the ``embedding=`` that would supply one.

``active_if`` makes a variable conditional on an earlier discrete one (:class:`When`). An inactive
variable is not part of the design at all: :class:`~optimi_lab.problem.space.Space` drops it from
the canonical form, so two designs that differ only in a switched-off value are the same design.

Each record normalises in an explicit ``__init__`` over a generated ``_*Fields`` base -- the
package's one shape for a frozen record that coerces its input (no reflection, NO-REFLECTION).
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from optimi_lab.core.errors import coerce_enum, refuse

__all__ = ['Categorical', 'Field', 'Integer', 'Real', 'Scale', 'Variable', 'When']


class Scale(StrEnum):
    """How a :class:`Real` is spaced: evenly, or evenly in its logarithm."""

    LINEAR = 'linear'
    LOG = 'log'


@dataclass(frozen=True, slots=True)
class _WhenFields:
    name: str
    levels: tuple[Hashable, ...]


@dataclass(frozen=True, slots=True, init=False)
class When(_WhenFields):
    """``active_if``: the variable exists only while the discrete variable *name* takes one of *levels*."""

    def __init__(self, name: str, levels: tuple[Hashable, ...]) -> None:
        levels = tuple(levels)
        if not name or not levels:
            refuse(f'a When needs a parent name and at least one level, got {name!r} and {levels!r}')
        _WhenFields.__init__(self, name, levels)


def _require_name(name: str, kind: str) -> None:
    if not isinstance(name, str) or not name:
        refuse(f'a {kind} variable needs a non-empty string name, got {name!r}')


def _require_interval(name: str, lo: float, hi: float) -> None:
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        refuse(f'variable {name!r} needs finite bounds with lo < hi, got lo={lo!r}, hi={hi!r}')


@dataclass(frozen=True, slots=True)
class _RealFields:
    name: str
    lo: float
    hi: float
    unit: str | None = None
    scale: Scale = Scale.LINEAR
    active_if: When | None = None


@dataclass(frozen=True, slots=True, init=False)
class Real(_RealFields):
    """A continuous variable on ``[lo, hi]`` in *unit*; ``scale=LOG`` spaces it by its logarithm."""

    def __init__(
        self,
        name: str,
        lo: float,
        hi: float,
        *,
        unit: str | None = None,
        scale: Scale = Scale.LINEAR,
        active_if: When | None = None,
    ) -> None:
        _require_name(name, 'real')
        lo, hi = float(lo), float(hi)
        scale = coerce_enum(Scale, scale, f'the scale of {name!r}')
        _require_interval(name, lo, hi)
        if scale is Scale.LOG and not lo > 0:
            refuse(f'log variable {name!r} needs lo > 0, got lo={lo!r}; a logarithm has no zero')
        _RealFields.__init__(self, name, lo, hi, unit, scale, active_if)

    @property
    def width(self) -> int:
        return 1

    def contains(self, value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and self.lo <= float(value) <= self.hi

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
class _IntegerFields:
    name: str
    lo: int
    hi: int
    step: int = 1
    active_if: When | None = None


@dataclass(frozen=True, slots=True, init=False)
class Integer(_IntegerFields):
    """An integer variable on the grid ``lo, lo + step, ..., hi``."""

    def __init__(self, name: str, lo: int, hi: int, step: int = 1, active_if: When | None = None) -> None:
        _require_name(name, 'integer')
        if not all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in (lo, hi, step)):
            refuse(f'integer variable {name!r} needs integer lo, hi and step, got {lo!r}, {hi!r}, {step!r}')
        _require_interval(name, lo, hi)
        if step < 1 or (hi - lo) % step:
            refuse(f'integer variable {name!r} needs a step >= 1 that divides hi - lo = {hi - lo}, got {step}')
        _IntegerFields.__init__(self, name, int(lo), int(hi), int(step), active_if)

    @property
    def width(self) -> int:
        return 1

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
class _CategoricalFields:
    name: str
    levels: tuple[Hashable, ...]
    embedding: tuple[tuple[float, ...], ...] | None = None
    active_if: When | None = None


@dataclass(frozen=True, slots=True, init=False)
class Categorical(_CategoricalFields):
    """One of *levels*. No metric unless *embedding* gives each level a coordinate vector."""

    def __init__(
        self,
        name: str,
        levels: tuple[Hashable, ...],
        embedding: tuple[tuple[float, ...], ...] | None = None,
        active_if: When | None = None,
    ) -> None:
        _require_name(name, 'categorical')
        levels = tuple(levels)
        if len(levels) < 2 or len(set(levels)) != len(levels):
            refuse(f'categorical {name!r} needs at least two distinct levels, got {levels!r}')
        if embedding is not None:
            embedding = tuple(tuple(float(c) for c in row) for row in embedding)
            matrix = np.array(embedding, dtype=float)
            if matrix.ndim != 2 or matrix.shape[0] != len(levels) or not np.all(np.isfinite(matrix)):
                refuse(
                    f'the embedding of {name!r} needs one finite coordinate row per level ({len(levels)}), '
                    f'got {embedding!r}'
                )
            if len(set(embedding)) != len(embedding):
                refuse(f'the embedding of {name!r} maps two levels to one point, so a decode could not tell them apart')
        _CategoricalFields.__init__(self, name, levels, embedding, active_if)

    @property
    def width(self) -> int:
        return 0 if self.embedding is None else len(self.embedding[0])

    def contains(self, value: object) -> bool:
        return value in self.levels

    def canonical(self, value: object) -> Hashable:
        return value  # type: ignore[return-value]

    def _coordinates(self) -> np.ndarray:
        """Each level's embedding row, scaled per column onto ``[0, 1]``; refused without an embedding."""
        if self.embedding is None:
            refuse(
                f'categorical {self.name!r} has no metric: its levels {list(self.levels)} are unordered; '
                f'supply embedding= (one coordinate row per level) to place it in the unit cube'
            )
        matrix = np.array(self.embedding)
        lo, hi = matrix.min(axis=0), matrix.max(axis=0)
        return (matrix - lo) / np.where(hi > lo, hi - lo, 1.0)

    def encode(self, value: Hashable) -> np.ndarray:
        return self._coordinates()[self.levels.index(value)]

    def decode(self, unit: np.ndarray) -> Hashable:
        return self.levels[int(np.argmin(np.sum((self._coordinates() - unit) ** 2, axis=1)))]


@dataclass(frozen=True, slots=True)
class _FieldFields:
    name: str
    n_cells: int
    lo: float
    hi: float
    neighbors: tuple[tuple[int, int], ...] = ()
    active_if: When | None = None


@dataclass(frozen=True, slots=True, init=False)
class Field(_FieldFields):
    """*n_cells* reals on ``[lo, hi]`` with a cell adjacency (*neighbors*), e.g. a topology density."""

    def __init__(
        self,
        name: str,
        n_cells: int,
        lo: float,
        hi: float,
        *,
        neighbors: tuple[tuple[int, int], ...] = (),
        active_if: When | None = None,
    ) -> None:
        _require_name(name, 'field')
        if not isinstance(n_cells, int) or n_cells < 1:
            refuse(f'field {name!r} needs n_cells >= 1, got {n_cells!r}')
        lo, hi = float(lo), float(hi)
        _require_interval(name, lo, hi)
        edges = tuple((int(a), int(b)) for a, b in neighbors)
        bad = [e for e in edges if not (0 <= e[0] < n_cells and 0 <= e[1] < n_cells) or e[0] == e[1]]
        if bad:
            refuse(f'field {name!r} has neighbour pairs outside cells 0..{n_cells - 1} or self-loops: {bad}')
        _FieldFields.__init__(self, name, n_cells, lo, hi, edges, active_if)

    @property
    def width(self) -> int:
        return self.n_cells

    def contains(self, value: object) -> bool:
        if isinstance(value, (str, bytes)):
            return False
        array = np.asarray(value, dtype=float)
        return array.shape == (self.n_cells,) and bool(np.all((array >= self.lo) & (array <= self.hi)))

    def canonical(self, value: object) -> tuple[float, ...]:
        return tuple(float(v) for v in np.asarray(value, dtype=float))

    def encode(self, value: object) -> np.ndarray:
        return (np.asarray(value, dtype=float) - self.lo) / (self.hi - self.lo)

    def decode(self, unit: np.ndarray) -> tuple[float, ...]:
        return tuple(float(v) for v in self.lo + np.clip(unit, 0.0, 1.0) * (self.hi - self.lo))


#: The declared set of variable kinds; a Space admits these and refuses a stranger.
Variable = Real | Integer | Categorical | Field
