"""What must hold: five constraint kinds, distinguished by WHO can evaluate them.

* :class:`Linear` -- a linear form of the numeric variables; the optimiser evaluates it exactly.
* :class:`Explicit` -- a cheap closed-form function of the design; the optimiser evaluates it.
* :class:`BlackBox` -- an oracle output; known only after an evaluation.
* :class:`Chance` -- an oracle output that must hold with a probability under the problem's
  :class:`~optimi_lab.problem.declarations.Uncertainty`.
* :class:`Path` -- an oracle output that is a trajectory and must hold at every point of it.

A violation is the distance past the bound divided by a SCALE in the constraint's unit, so
constraints in different units add into one number. The scale defaults to ``|bound|``; a zero
bound has no magnitude, so it must declare its scale -- an absolute tolerance carries a unit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from optimi_lab.core.errors import coerce_enum, refuse

__all__ = ['BlackBox', 'Chance', 'Constraint', 'Explicit', 'Linear', 'Path', 'Relation']


class Relation(StrEnum):
    """Which side of its bound a constraint is satisfied on."""

    AT_MOST = '<='
    AT_LEAST = '>='


def _resolve_scale(name: str, bound: float, scale: float | None) -> float:
    if scale is None:
        if bound == 0.0 or not math.isfinite(bound):
            refuse(f'constraint {name!r} has bound {bound!r}, which gives no magnitude; declare scale= in its unit')
        return abs(bound)
    if not (math.isfinite(scale) and scale > 0):
        refuse(f'constraint {name!r} needs a finite scale > 0, got {scale!r}')
    return float(scale)


def _bounded(name: str, relation: Relation, bound: float, scale: float | None) -> tuple[Relation, float, float]:
    """The checked ``(relation, bound, scale)`` every bounded kind shares."""
    if not isinstance(name, str) or not name:
        refuse(f'a constraint needs a non-empty string name, got {name!r}')
    relation = coerce_enum(Relation, relation, f'the relation of {name!r}')
    bound = float(bound)
    return relation, bound, _resolve_scale(name, bound, scale)


def _past(relation: Relation, bound: float, scale: float, value: float) -> float:
    if math.isnan(value):
        return math.nan
    past = value - bound if relation is Relation.AT_MOST else bound - value
    return max(past, 0.0) / scale


@dataclass(frozen=True, slots=True)
class _BlackBoxFields:
    name: str
    relation: Relation
    bound: float
    unit: str | None
    scale: float


@dataclass(frozen=True, slots=True, init=False)
class BlackBox(_BlackBoxFields):
    """The oracle output called *name* must satisfy ``relation bound``."""

    def __init__(
        self, name: str, relation: Relation, bound: float, unit: str | None = None, scale: float | None = None
    ) -> None:
        relation, bound, scale = _bounded(name, relation, bound, scale)
        _BlackBoxFields.__init__(self, name, relation, bound, unit, scale)

    @property
    def output(self) -> str:
        return self.name

    def violation(self, value: float) -> float:
        """0 when satisfied, the scaled distance past the bound otherwise, NaN when not computed."""
        return _past(self.relation, self.bound, self.scale, float(value))


@dataclass(frozen=True, slots=True)
class _ChanceFields:
    name: str
    output: str
    relation: Relation
    bound: float
    probability: float
    unit: str | None
    scale: float


@dataclass(frozen=True, slots=True, init=False)
class Chance(_ChanceFields):
    """``P(output relation bound) >= probability`` under the problem's uncertainty."""

    def __init__(
        self,
        name: str,
        output: str,
        relation: Relation,
        bound: float,
        *,
        probability: float,
        unit: str | None = None,
        scale: float | None = None,
    ) -> None:
        relation, bound, scale = _bounded(name, relation, bound, scale)
        probability = float(probability)
        if not 0.0 < probability < 1.0:
            refuse(f'chance constraint {name!r} needs 0 < probability < 1, got {probability!r}')
        _ChanceFields.__init__(self, name, output, relation, bound, probability, unit, scale)


@dataclass(frozen=True, slots=True)
class _PathFields:
    name: str
    output: str
    relation: Relation
    bound: float
    unit: str | None
    scale: float


@dataclass(frozen=True, slots=True, init=False)
class Path(_PathFields):
    """The trajectory *output* must satisfy ``relation bound`` at every point; the worst point counts."""

    def __init__(
        self,
        name: str,
        output: str,
        relation: Relation,
        bound: float,
        *,
        unit: str | None = None,
        scale: float | None = None,
    ) -> None:
        relation, bound, scale = _bounded(name, relation, bound, scale)
        _PathFields.__init__(self, name, output, relation, bound, unit, scale)

    def violation(self, value: Sequence[float]) -> float:
        return max(_past(self.relation, self.bound, self.scale, float(v)) for v in value)


@dataclass(frozen=True, slots=True)
class _ExplicitFields:
    name: str
    function: Callable[[Mapping[str, object]], float]
    relation: Relation
    bound: float
    unit: str | None
    scale: float


@dataclass(frozen=True, slots=True, init=False)
class Explicit(_ExplicitFields):
    """``function(design) relation bound``, for a cheap closed form the optimiser may evaluate itself."""

    def __init__(
        self,
        name: str,
        function: Callable[[Mapping[str, object]], float],
        relation: Relation,
        bound: float,
        *,
        unit: str | None = None,
        scale: float | None = None,
    ) -> None:
        relation, bound, scale = _bounded(name, relation, bound, scale)
        _ExplicitFields.__init__(self, name, function, relation, bound, unit, scale)

    def violation(self, design: Mapping[str, object]) -> float:
        return _past(self.relation, self.bound, self.scale, float(self.function(design)))


@dataclass(frozen=True, slots=True)
class _LinearFields:
    name: str
    coefficients: tuple[tuple[str, float], ...]
    lo: float | None = None
    hi: float | None = None
    scale: float = 1.0


@dataclass(frozen=True, slots=True, init=False)
class Linear(_LinearFields):
    """``lo <= sum(coefficients[v] * design[v]) <= hi`` over numeric variables, scaled by *scale*.

    An absent side is ``None`` -- never an infinity, which has no exact encoding in an identity. The
    coefficients are stored as sorted ``(variable, weight)`` pairs; :attr:`weights` is the mapping.
    """

    def __init__(
        self,
        name: str,
        coefficients: Mapping[str, float],
        lo: float | None = None,
        hi: float | None = None,
        scale: float = 1.0,
    ) -> None:
        pairs = tuple(sorted((str(k), float(v)) for k, v in dict(coefficients).items()))
        if not pairs:
            refuse(f'linear constraint {name!r} needs at least one coefficient')
        sides = [side for side in (lo, hi) if side is not None]
        if not sides or not all(math.isfinite(side) for side in sides) or (len(sides) == 2 and not lo <= hi):
            refuse(f'linear constraint {name!r} needs finite sides with lo <= hi, one at least, got {lo!r}, {hi!r}')
        if not (math.isfinite(scale) and scale > 0):
            refuse(f'linear constraint {name!r} needs a finite scale > 0, got {scale!r}')
        _LinearFields.__init__(self, name, pairs, lo, hi, float(scale))

    @property
    def weights(self) -> dict[str, float]:
        return dict(self.coefficients)

    def violation(self, design: Mapping[str, object]) -> float:
        total = float(np.sum([c * float(design[v]) for v, c in self.coefficients]))  # type: ignore[arg-type]
        below = 0.0 if self.lo is None else self.lo - total
        above = 0.0 if self.hi is None else total - self.hi
        return max(below, above, 0.0) / self.scale


#: The declared set of constraint kinds.
Constraint = Linear | Explicit | BlackBox | Chance | Path
