"""How a search reads a tuning space: learn-lab's dimensions, mapped onto this package's problem variables.

The space vocabulary is :mod:`learn_lab.space` -- the leaf library that declares what every model family
tunes; optimi-lab imports it and never the reverse, so one set of types serves the declaration and the
search. A tuning space is ``Mapping[str, Dimension]`` (a :class:`learn_lab.space.Space` is one). Each
dimension maps onto the one variable kind :mod:`optimi_lab.problem` already owns -- that is where the
unit-cube encoding lives, so this module adds no second encoding:

* ``learn_lab.space.Real`` -> :class:`~optimi_lab.problem.Real`, log-scaled when ``log=True``;
* ``learn_lab.space.Integer`` -> :class:`~optimi_lab.problem.Integer`, decoded by rounding to the grid;
* ``learn_lab.space.Choice`` -> an unembedded :class:`~optimi_lab.problem.Categorical`, which a metric
  solver reads one-hot through :func:`~optimi_lab.solver.seeding.search_embedding`.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

from learn_lab import space as tuned

from optimi_lab.core.errors import refuse
from optimi_lab.problem import Categorical, Integer, Real, Scale, Space, Variable

__all__ = ['Kind', 'kind_of', 'problem_space']


class Kind(StrEnum):
    """What a search method must be able to read: a continuous, an integer or a categorical dimension."""

    REAL = 'real'
    INTEGER = 'integer'
    CHOICE = 'choice'


_KINDS: dict[type, Kind] = {tuned.Real: Kind.REAL, tuned.Integer: Kind.INTEGER, tuned.Choice: Kind.CHOICE}


def kind_of(dimension: object) -> Kind:
    """The :class:`Kind` of *dimension*; a stranger is refused naming the declared set."""
    if type(dimension) not in _KINDS:
        refuse(f'a tuning dimension is one of {sorted(t.__name__ for t in _KINDS)}, got {dimension!r}')
    return _KINDS[type(dimension)]


def _variable(name: str, dimension: tuned.Dimension) -> Variable:
    if isinstance(dimension, tuned.Real):
        return Real(name, dimension.lo, dimension.hi, scale=Scale.LOG if dimension.log else Scale.LINEAR)
    if isinstance(dimension, tuned.Integer):
        return Integer(name, dimension.lo, dimension.hi)
    return Categorical(name, dimension.values)


def problem_space(space: Mapping[str, tuned.Dimension]) -> Space:
    """The :class:`~optimi_lab.problem.Space` of *space*, one variable per name in declared order."""
    if not space:
        refuse('a tuning space needs at least one dimension')
    for dimension in space.values():
        kind_of(dimension)
    return Space(tuple(_variable(name, dimension) for name, dimension in space.items()))
