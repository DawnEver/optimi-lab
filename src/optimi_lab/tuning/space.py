"""The dimensions a hyper-parameter search ranges over, keyed by name in a plain mapping.

A tuning space is ``Mapping[str, Dimension]``: the name lives in the key, the range in the value. Each
dimension maps onto the one variable kind :mod:`optimi_lab.problem` already owns -- that is where the
unit-cube encoding lives, so this module adds no second encoding:

* :class:`RealRange` -> :class:`~optimi_lab.problem.Real`, log-scaled when ``log=True``;
* :class:`IntegerRange` -> :class:`~optimi_lab.problem.Integer`, decoded by rounding to the grid;
* :class:`OneOf` -> an unembedded :class:`~optimi_lab.problem.Categorical`, which a metric solver
  reads one-hot through :func:`~optimi_lab.solver.seeding.search_embedding`.

The names are not ``Real``/``Integer``: those are the problem layer's, and a public name has one home.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from optimi_lab.core.errors import refuse
from optimi_lab.problem import Categorical, Integer, Real, Scale, Space, Variable

__all__ = ['Dimension', 'IntegerRange', 'Kind', 'OneOf', 'RealRange', 'kind_of', 'problem_space']


class Kind(StrEnum):
    """What a search method must be able to read: a continuous, an integer or a categorical dimension."""

    REAL = 'real'
    INTEGER = 'integer'
    CHOICE = 'choice'


@dataclass(frozen=True, slots=True)
class RealRange:
    """A continuous dimension on ``[lo, hi]``; ``log=True`` searches it evenly in its logarithm."""

    lo: float
    hi: float
    log: bool = False


@dataclass(frozen=True, slots=True)
class IntegerRange:
    """An integer dimension on ``lo, lo + 1, ..., hi``."""

    lo: int
    hi: int


@dataclass(frozen=True, slots=True)
class _OneOfFields:
    values: tuple[Hashable, ...]


@dataclass(frozen=True, slots=True, init=False)
class OneOf(_OneOfFields):
    """One of *values*, unordered (any iterable is kept as a tuple)."""

    def __init__(self, values: tuple[Hashable, ...]) -> None:
        _OneOfFields.__init__(self, tuple(values))


#: The declared set of tuning dimensions.
Dimension = RealRange | IntegerRange | OneOf

_KINDS: dict[type, Kind] = {RealRange: Kind.REAL, IntegerRange: Kind.INTEGER, OneOf: Kind.CHOICE}


def kind_of(dimension: object) -> Kind:
    """The :class:`Kind` of *dimension*; a stranger is refused naming the declared set."""
    if type(dimension) not in _KINDS:
        refuse(f'a tuning dimension is one of {sorted(t.__name__ for t in _KINDS)}, got {dimension!r}')
    return _KINDS[type(dimension)]


def _variable(name: str, dimension: Dimension) -> Variable:
    if isinstance(dimension, RealRange):
        return Real(name, dimension.lo, dimension.hi, scale=Scale.LOG if dimension.log else Scale.LINEAR)
    if isinstance(dimension, IntegerRange):
        return Integer(name, dimension.lo, dimension.hi)
    return Categorical(name, dimension.values)


def problem_space(space: Mapping[str, Dimension]) -> Space:
    """The :class:`~optimi_lab.problem.Space` of *space*, one variable per name in declared order."""
    if not space:
        refuse('a tuning space needs at least one dimension')
    for dimension in space.values():
        kind_of(dimension)
    return Space(tuple(_variable(name, dimension) for name, dimension in space.items()))
