"""What a run optimizes over: named variables with bounds, named objectives with a direction.

The old package encoded both in the STRING ``key@oc<N>@max|min`` and recovered them by splitting
on ``@`` into three parallel matrices (``obj_name_matrix``, ``err_value_matrix``,
``max_obj_flags``) whose indices then had to agree with a ``case_id`` allocated in encounter
order elsewhere. The two orders were never checked against each other, so a row listing ``oc1``
before ``oc0`` silently swapped the two conditions' values whenever they shared an objective
name. Here direction and operating point are FIELDS on :class:`Objective`, there is no name
grammar to parse, and no index is derived from a position in a list.

The sets are frozen: a set that can be mutated after the matrices built from it were cached is
how a shape and its declaration drift apart.
"""

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from optimi_lab.core.errors import coerce_enum, refuse, require_axis, require_finite

__all__ = ['Direction', 'Objective', 'ObjectiveSet', 'Variable', 'VariableSet']


def _require_declared_names(names: tuple[str, ...], what: str) -> None:
    """Refuse an empty declaration or a repeated name, naming the duplicates and the whole set.

    One function for both sets: "the names must be non-empty and unique" is a property of a
    declaration, not of variables or of objectives, and two copies of it would drift apart.
    """
    if not names:
        refuse(f'a {what} needs at least one entry, got none')
    duplicated = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicated:
        refuse(f'{what} names must be unique, duplicated: {duplicated}, declared: {list(names)}')


class Direction(StrEnum):
    """Which way an objective is better.

    A field of :class:`Objective`, never a suffix of its name: the old grammar decided the
    internals of ``_obj_func_normalized`` — a flag read out of a string chose whether the
    returned matrix was negated in place and negated back on the way out.
    """

    MINIMIZE = 'minimize'
    MAXIMIZE = 'maximize'


@dataclass(frozen=True, slots=True)
class Variable:
    """One search dimension: a name, an inclusive interval, and the unit it is stated in.

    The unit is carried for the caller's report and nothing else — bounds are used exactly as
    given, and no conversion happens anywhere in this package.
    """

    name: str
    lower: float
    upper: float
    unit: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            refuse('a variable name must be a non-empty string')
        require_finite(np.array([self.lower, self.upper], dtype=float), f'the bounds of variable {self.name!r}')
        if not self.lower < self.upper:
            refuse(f'variable {self.name!r} needs lower < upper, got lower={self.lower!r}, upper={self.upper!r}')

    @property
    def span(self) -> float:
        """The width of the search interval."""
        return float(self.upper - self.lower)


@dataclass(frozen=True, slots=True)
class Objective:
    """One quantity to optimize: a name, a direction, and the operating point that produces it.

    ``operating_point`` is descriptive — it groups objectives for a report and tells an
    evaluator which condition to drive; ``None`` means "the run evaluates one condition".
    """

    name: str
    direction: Direction = Direction.MINIMIZE
    operating_point: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            refuse('an objective name must be a non-empty string')
        if not isinstance(self.direction, Direction):
            coerced = coerce_enum(Direction, self.direction, f'the direction of objective {self.name!r}')
            object.__setattr__(self, 'direction', coerced)

    @property
    def maximize(self) -> bool:
        """Whether this objective is better when larger."""
        return self.direction is Direction.MAXIMIZE


@dataclass(frozen=True, slots=True)
class VariableSet:
    """A non-empty collection of uniquely named variables, in declaration order.

    Order is the contract: column ``i`` of every matrix this package passes around is
    ``variables[i]``, and it is the only thing that says so — there is no name matrix.
    """

    variables: tuple[Variable, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'variables', tuple(self.variables))
        _require_declared_names(self.names, 'variable set')

    @property
    def n_var(self) -> int:
        """Number of search dimensions."""
        return len(self.variables)

    @property
    def names(self) -> tuple[str, ...]:
        """Variable names, in column order."""
        return tuple(variable.name for variable in self.variables)

    @property
    def units(self) -> tuple[str | None, ...]:
        """Variable units, in column order."""
        return tuple(variable.unit for variable in self.variables)

    @property
    def lower_bounds(self) -> np.ndarray:
        """Lower bound per column, shape ``(n_var,)``."""
        return np.array([variable.lower for variable in self.variables], dtype=float)

    @property
    def upper_bounds(self) -> np.ndarray:
        """Upper bound per column, shape ``(n_var,)``."""
        return np.array([variable.upper for variable in self.variables], dtype=float)

    def index(self, name: str) -> int:
        """Return the column index of variable ``name``, or refuse naming every declared name."""
        names = self.names
        if name not in names:
            refuse(f'no variable named {name!r}; the declared names are {list(names)}')
        return names.index(name)

    def scale(self, unit_values: np.ndarray) -> np.ndarray:
        """Map a ``(n_points, n_var)`` sample of ``[0, 1]`` onto the declared bounds."""
        unit_values = require_axis(unit_values, self.n_var, 1, 'a unit-cube sample')
        return self.lower_bounds + unit_values * (self.upper_bounds - self.lower_bounds)

    def clip(self, values: np.ndarray) -> np.ndarray:
        """Return a copy of ``values`` with every column inside its declared bounds.

        A copy, never in place: a proposer's own population must not be rewritten by asking
        whether it is feasible.
        """
        values = require_axis(values, self.n_var, 1, 'a point matrix')
        return np.clip(values, self.lower_bounds, self.upper_bounds)

    def __len__(self) -> int:
        return len(self.variables)

    def __iter__(self) -> Iterator[Variable]:
        return iter(self.variables)


@dataclass(frozen=True, slots=True)
class ObjectiveSet:
    """A non-empty collection of uniquely named objectives, in declaration order.

    Column ``i`` of an objective matrix is ``objectives[i]``, for every evaluation, every
    acquisition and every report. Nothing reorders it.
    """

    objectives: tuple[Objective, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'objectives', tuple(self.objectives))
        _require_declared_names(self.names, 'objective set')

    @property
    def n_obj(self) -> int:
        """Number of objectives."""
        return len(self.objectives)

    @property
    def names(self) -> tuple[str, ...]:
        """Objective names, in column order."""
        return tuple(objective.name for objective in self.objectives)

    @property
    def directions(self) -> tuple[Direction, ...]:
        """Directions, in column order."""
        return tuple(objective.direction for objective in self.objectives)

    @property
    def operating_points(self) -> tuple[str | None, ...]:
        """Operating points, in column order."""
        return tuple(objective.operating_point for objective in self.objectives)

    @property
    def maximize_mask(self) -> np.ndarray:
        """Boolean mask, shape ``(n_obj,)``, True where larger is better."""
        return np.array([objective.maximize for objective in self.objectives], dtype=bool)

    def to_minimization(self, values: np.ndarray) -> np.ndarray:
        """Negate every maximizing column of a ``(n_points, n_obj)`` matrix, on a copy.

        One function, two frames: negating is its own inverse, so calling this again returns the
        native values. That identity is why there is no second function for the way back — the
        old pair negated the field in place, negated it back on the outbound path, and negated it
        once more inside the surrogate branch.
        """
        values = require_axis(values, self.n_obj, 1, 'an objective matrix').copy()
        values[:, self.maximize_mask] *= -1.0
        return values

    def __len__(self) -> int:
        return len(self.objectives)

    def __iter__(self) -> Iterator[Objective]:
        return iter(self.objectives)
