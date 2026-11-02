"""The design space: an ordered set of uniquely named variables, some conditional on others.

A DESIGN is a mapping ``name -> value``. Its CANONICAL form holds exactly the active variables, each
value in its kind's canonical type, keys in declaration order; it is what an identity is computed
from, so a switched-off value cannot make two equal designs look different.

The unit-cube vector (:meth:`Space.encode`) concatenates each variable's own map in declaration
order. An inactive variable's dimensions hold NaN -- it has no value, and a number there would be
read as one -- and :meth:`Space.decode` ignores them, deciding activity from the decoded parents.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, replace

import numpy as np

from optimi_lab.core.errors import refuse, require_declared_names
from optimi_lab.core.sampling import SampleSpec, sample
from optimi_lab.problem.variables import Categorical, Field, Integer, Real, Variable

__all__ = ['Space']

_KINDS = (Real, Integer, Categorical, Field)


@dataclass(frozen=True, slots=True)
class _SpaceFields:
    variables: tuple[Variable, ...]


@dataclass(frozen=True, slots=True, init=False)
class Space(_SpaceFields):
    """Variables in declaration order; a ``When`` parent must be declared earlier and be discrete."""

    def __init__(self, variables: tuple[Variable, ...]) -> None:
        variables = tuple(variables)
        strangers = [v for v in variables if not isinstance(v, _KINDS)]
        if strangers:
            refuse(f'a Space holds Real, Integer, Categorical or Field variables, got {strangers!r}')
        require_declared_names(tuple(v.name for v in variables), 'space', empty=True)
        seen: dict[str, Variable] = {}
        for variable in variables:
            condition = variable.active_if
            if condition is not None:
                parent = seen.get(condition.name)
                if parent is None:
                    refuse(f'{variable.name!r} is active_if {condition.name!r}, which must be declared earlier')
                if not isinstance(parent, (Integer, Categorical)):
                    refuse(
                        f'{variable.name!r} is active_if {condition.name!r}, which is not discrete (Integer or Categorical)'
                    )
                missing = [level for level in condition.levels if not parent.contains(level)]
                if missing:
                    refuse(
                        f'{variable.name!r} is active_if {condition.name!r} in {missing}, which is not a level of it'
                    )
            seen[variable.name] = variable
        _SpaceFields.__init__(self, variables)

    @property
    def _slices(self) -> tuple[slice, ...]:
        offsets = np.cumsum([0, *(v.width for v in self.variables)])
        return tuple(slice(int(a), int(b)) for a, b in itertools.pairwise(offsets))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.variables)

    @property
    def width(self) -> int:
        """Dimensions of the unit-cube vector."""
        return sum(v.width for v in self.variables)

    def __getitem__(self, name: str) -> Variable:
        for variable in self.variables:
            if variable.name == name:
                return variable
        return refuse(f'no variable named {name!r}; the declared names are {list(self.names)}')

    def __iter__(self) -> Iterator[Variable]:
        return iter(self.variables)

    def __len__(self) -> int:
        return len(self.variables)

    def _is_active(self, variable: Variable, values: Mapping[str, object], active: set[str]) -> bool:
        condition = variable.active_if
        return condition is None or (condition.name in active and values[condition.name] in condition.levels)

    def active(self, design: Mapping[str, object]) -> tuple[str, ...]:
        """The names active under *design*, in declaration order."""
        active: set[str] = set()
        for variable in self.variables:
            if self._is_active(variable, design, active):
                if variable.name not in design:
                    refuse(f'the design has no value for active variable {variable.name!r}; given {sorted(design)}')
                active.add(variable.name)
        return tuple(name for name in self.names if name in active)

    def canonical(self, design: Mapping[str, object]) -> dict[str, object]:
        """The design reduced to its active variables, each value checked and in canonical type."""
        strangers = sorted(set(design) - set(self.names))
        if strangers:
            refuse(f'no variable named {strangers[0]!r}; the declared names are {list(self.names)}')
        result: dict[str, object] = {}
        for name in self.active(design):
            variable = self[name]
            if not variable.contains(design[name]):
                refuse(f'value {design[name]!r} of {name!r} is outside its declaration {variable!r}')
            result[name] = variable.canonical(design[name])
        return result

    def encode(self, design: Mapping[str, object]) -> np.ndarray:
        """The unit-cube vector of *design*; an inactive variable's dimensions are NaN."""
        for variable in self.variables:
            if isinstance(variable, Categorical) and variable.embedding is None:
                variable.encode(variable.levels[0])  # refuses, naming the missing embedding
        values = self.canonical(design)
        unit = np.full(self.width, np.nan)
        for variable, part in zip(self.variables, self._slices, strict=True):
            if variable.name in values:
                unit[part] = variable.encode(values[variable.name])
        return unit

    def decode(self, unit: np.ndarray) -> dict[str, object]:
        """The design at a unit-cube vector; activity follows the decoded parents."""
        unit = np.asarray(unit, dtype=float)
        if unit.shape != (self.width,):
            refuse(f'a unit vector of this space has shape ({self.width},), got {unit.shape}')
        design: dict[str, object] = {}
        for variable, part in zip(self.variables, self._slices, strict=True):
            if self._is_active(variable, design, set(design)):
                design[variable.name] = variable.decode(unit[part])
        return design

    def natural_sample(self, spec: SampleSpec) -> np.ndarray:
        """*spec*'s unit-cube sample decoded to natural coordinates, one row per point in name order.

        A declared ``anchor`` is a NATURAL design here: it is encoded first, so it decodes back to
        itself as row 0.
        """
        if spec.anchor is not None:
            unit_anchor = self.encode(dict(zip(self.names, spec.anchor, strict=True)))
            spec = replace(spec, anchor=tuple(float(c) for c in unit_anchor))
        designs = (self.decode(unit) for unit in sample(self.width, spec))
        rows = [[design[name] for name in self.names] for design in designs]
        return np.asarray(rows, dtype=float)

    def to_plain(self) -> list[dict[str, object]]:
        """The declaration as plain data, for an identity."""
        return [{'kind': type(v).__name__, **asdict(v)} for v in self.variables]
