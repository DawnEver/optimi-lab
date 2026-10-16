"""What a run must satisfy: named constraints, each a bound with a sense.

A constraint is the third declaration of a problem, beside its variables and objectives
(:mod:`optimi_lab.core.space`). Feasibility is therefore a fact of the PROBLEM rather than a
penalty folded into an objective column -- the fold is what a consumer without this module wrote,
and it squashed the crowding distance of every feasible point onto the penalty's scale.

A shortfall is measured RELATIVE to the bound, ``(value - bound) / |bound|`` past it, so
constraints stated in different units add into one violation without a unit library, and the sum
is :func:`optimi_lab.core.pareto.total_violation` -- the one formula every ranking reads.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from optimi_lab.core.errors import coerce_enum, refuse, require_declared_names
from optimi_lab.core.pareto import total_violation

__all__ = ['Constraint', 'ConstraintSet', 'Sense']


class Sense(StrEnum):
    """Which side of its bound a constraint is satisfied on. A field, never a parse of a string."""

    AT_MOST = '<='
    AT_LEAST = '>='


@dataclass(frozen=True, slots=True)
class _ConstraintFields:
    name: str
    sense: Sense
    bound: float
    unit: str | None = None


@dataclass(frozen=True, slots=True, init=False)
class Constraint(_ConstraintFields):
    """One quantity that must stay on one side of a bound: ``value <= bound`` or ``value >= bound``.

    A shortfall is measured RELATIVE to the bound -- ``(value - bound) / |bound|`` past it -- so
    constraints stated in different units add into one violation without a unit library. A zero
    bound has no magnitude to normalise by and is refused: state such a limit against a nonzero
    reference instead. ``unit`` is carried for the caller's report and nothing else.
    """

    def __init__(self, name: str, sense: Sense, bound: float, unit: str | None = None) -> None:
        if not name:
            refuse('a constraint name must be a non-empty string')
        sense = coerce_enum(Sense, sense, f'the sense of constraint {name!r}')
        bound = float(bound)
        if not np.isfinite(bound) or bound == 0.0:
            refuse(
                f'constraint {name!r} needs a finite nonzero bound, got {bound!r}; its shortfall is normalised by '
                f'|bound|, which a zero bound does not have'
            )
        _ConstraintFields.__init__(self, name, sense, bound, unit)

    def shortfall(self, values: np.ndarray) -> np.ndarray:
        """How far each value is past the bound, divided by ``|bound|``; 0 when satisfied, NaN kept."""
        values = np.asarray(values, dtype=float)
        past = values - self.bound if self.sense is Sense.AT_MOST else self.bound - values
        return np.where(np.isnan(past), np.nan, np.maximum(past, 0.0)) / abs(self.bound)


@dataclass(frozen=True, slots=True)
class _ConstraintSetFields:
    constraints: tuple[Constraint, ...]


@dataclass(frozen=True, slots=True, init=False)
class ConstraintSet(_ConstraintSetFields):
    """A non-empty collection of uniquely named constraints, in declaration order.

    Column ``i`` of every constraint matrix is ``constraints[i]``. A run with no constraint passes
    ``None`` rather than an empty set, so "unconstrained" has one spelling.
    """

    def __init__(self, constraints: tuple[Constraint, ...]) -> None:
        constraints = tuple(constraints)
        require_declared_names(tuple(constraint.name for constraint in constraints), 'constraint set')
        _ConstraintSetFields.__init__(self, constraints)

    @property
    def n_con(self) -> int:
        """Number of constraints."""
        return len(self.constraints)

    @property
    def names(self) -> tuple[str, ...]:
        """Constraint names, in column order."""
        return tuple(constraint.name for constraint in self.constraints)

    def shortfall(self, values: np.ndarray) -> np.ndarray:
        """The normalised shortfall of a ``(n_points, n_con)`` matrix, column by column, on a copy."""
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_con:
            refuse(
                f'a constraint matrix needs shape (n_points, {self.n_con}), one column per constraint '
                f'{list(self.names)}, got {values.shape}'
            )
        return np.column_stack([constraint.shortfall(values[:, index]) for index, constraint in enumerate(self)])

    def violation(self, values: np.ndarray) -> np.ndarray:
        """Per point, the sum of the normalised shortfalls; NaN where any constraint did not compute."""
        return total_violation(self.shortfall(values))

    def feasible_mask(self, values: np.ndarray) -> np.ndarray:
        """Per point, True where every constraint computed and is satisfied."""
        return self.violation(values) == 0.0

    def __len__(self) -> int:
        return len(self.constraints)

    def __iter__(self) -> Iterator[Constraint]:
        return iter(self.constraints)
