"""What an oracle returned for one action: an ``outcomes.Cell`` plus what a cell cannot hold.

The cell carries the status, the scalar output values, the reason, the evidence and the cost --
the II.3 vocabulary, never a local copy of it. Around it: derivatives (``gradient``, ``jvp``,
``hvp``), per-output ``uncertainty`` (a standard deviation in the output's unit), ``quality``
indicators (residuals, mesh error), ``provenance`` (versions, hosts, seeds) and ``series`` -- the
trajectories a :class:`~optimi_lab.problem.Path` constraint is judged on.

Only a cell that carries values carries derivatives or series: a point that did not compute has no
slope either. A gradient is taken with respect to the space's unit-cube coordinates
(:meth:`~optimi_lab.problem.Space.encode`), so its width is the space width.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from lab_commons.outcomes import Cell, Status

from optimi_lab.core.errors import refuse
from optimi_lab.oracle.action import Action
from optimi_lab.problem import BlackBox, Chance, Path, Problem

__all__ = ['Result']

_CARRIERS = frozenset({Status.OK, Status.PARTIAL})
_VECTORS = ('gradient', 'hvp')
_SCALARS = ('jvp', 'uncertainty', 'quality')


def _frozen(mapping: Mapping[str, Any] | None, convert: Callable[[Any], Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(k): convert(v) for k, v in (mapping or {}).items()})


def _vector(value: object) -> np.ndarray:
    array = np.array(value, dtype=float)
    array.setflags(write=False)
    return array


def _series(value: object) -> tuple[float, ...]:
    return tuple(float(x) for x in value)  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True, eq=False)
class Result:
    """The *action*, its *cell*, and the extras around it (see the module docstring)."""

    action: Action
    cell: Cell
    gradient: Mapping[str, np.ndarray] = field(default_factory=dict)
    jvp: Mapping[str, float] = field(default_factory=dict)
    hvp: Mapping[str, np.ndarray] = field(default_factory=dict)
    uncertainty: Mapping[str, float] = field(default_factory=dict)
    quality: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping[str, str] = field(default_factory=dict)
    series: Mapping[str, tuple[float, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in _VECTORS:
            object.__setattr__(self, name, _frozen(getattr(self, name), _vector))
        for name in _SCALARS:
            object.__setattr__(self, name, _frozen(getattr(self, name), float))
        object.__setattr__(self, 'provenance', _frozen(self.provenance, str))
        object.__setattr__(self, 'series', _frozen(self.series, _series))
        carried = [name for name in ('gradient', 'jvp', 'hvp', 'uncertainty', 'series') if getattr(self, name)]
        if carried and self.cell.status not in _CARRIERS:
            refuse(f'a {self.cell.status.name} result carries no derivative, uncertainty or series, got {carried}')
        negative = sorted(k for k, v in self.uncertainty.items() if not v >= 0)
        if negative:
            refuse(f'an uncertainty is a standard deviation >= 0; {negative} are not')

    def conform(self, problem: Problem) -> Result:
        """Return self after checking every part against *problem*'s outputs and space width."""
        paths = {c.output for c in problem.constraints if isinstance(c, Path)}
        if self.cell.status is Status.OK:
            missing = sorted(set(problem.outputs) - set(self.cell.values) - paths) + sorted(paths - set(self.series))
            if missing:
                refuse(f'an OK result must report every output of the problem; missing {missing}')
        unknown = sorted((set(self.cell.values) | set(self.gradient) | set(self.series)) - set(problem.outputs))
        if unknown:
            refuse(
                f'the result reports {unknown}, which the problem does not declare; outputs are {list(problem.outputs)}'
            )
        for name, vector in (*self.gradient.items(), *self.hvp.items()):
            if vector.shape != (problem.space.width,):
                refuse(f'the derivative of {name!r} has shape {vector.shape}; the space width is {problem.space.width}')
        return self

    def violation(self, problem: Problem) -> float:
        """The total scaled violation of every constraint of *problem* at this result -- the one formula.

        A chance constraint is refused: one result is one sample, and a probability is not.
        """
        total = 0.0
        for c in problem.constraints:
            if isinstance(c, Chance):
                refuse(f'Chance constraint {c.name!r} needs an estimator over replicates, not one result')
            if isinstance(c, BlackBox):
                total += c.violation(self.cell.values[c.output])
            elif isinstance(c, Path):
                total += c.violation(self.series[c.output])
            else:
                total += c.violation(self.action.design)
        return total

    def extras(self) -> dict[str, Any]:
        """The JSON-ready extras an archive files beside the cell."""
        plain: dict[str, Any] = {}
        for name in _VECTORS:
            if getattr(self, name):
                plain[name] = {k: v.tolist() for k, v in getattr(self, name).items()}
        for name in (*_SCALARS, 'provenance'):
            if getattr(self, name):
                plain[name] = dict(getattr(self, name))
        if self.series:
            plain['series'] = {k: list(v) for k, v in self.series.items()}
        return plain

    @classmethod
    def from_extras(cls, action: Action, cell: Cell, extras: Mapping[str, Any]) -> Result:
        """The inverse of :meth:`extras`."""
        return cls(action, cell, **{k: v for k, v in extras.items() if k in cls.__dataclass_fields__})
