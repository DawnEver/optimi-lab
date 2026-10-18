"""What an oracle returned for one action: an ``outcomes.Cell`` plus what a cell cannot hold.

The cell carries the status, the scalar output values, the reason, the evidence and the cost --
the II.3 vocabulary, never a local copy of it. Around it: derivatives (``gradient``, ``jvp``,
``hvp`` -- the product with the action's ``direction``), per-output ``uncertainty`` (a standard deviation in the output's unit), ``quality``
indicators (residuals, mesh error), ``provenance`` (versions, hosts, seeds) and ``series`` -- the
trajectories a :class:`~optimi_lab.problem.Path` constraint is judged on.

Only a cell that carries values carries derivatives or series: a point that did not compute has no
slope either. A gradient is taken with respect to the space's unit-cube coordinates
(:meth:`~optimi_lab.problem.Space.encode`), so its width is the space width.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from lab_commons.outcomes import Cell, Status

from optimi_lab.core.errors import refuse
from optimi_lab.oracle.action import Action
from optimi_lab.problem import BlackBox, Chance, Path, Problem

__all__ = ['EXTRAS', 'Result']

_CARRIERS = frozenset({Status.OK, Status.PARTIAL})

#: The parts of a result beside its cell, as :meth:`Result.extras` files them.
EXTRAS = ('gradient', 'jvp', 'hvp', 'uncertainty', 'quality', 'provenance', 'series')


def _frozen(mapping: Mapping[str, Any] | None, convert: Callable[[Any], Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(k): convert(v) for k, v in (mapping or {}).items()})


def _vector(value: object) -> np.ndarray:
    array = np.array(value, dtype=float)
    array.setflags(write=False)
    return array


def _series(value: object) -> tuple[float, ...]:
    return tuple(float(x) for x in value)  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True, eq=False)
class _ResultFields:
    action: Action
    cell: Cell
    gradient: Mapping[str, np.ndarray]
    jvp: Mapping[str, float]
    hvp: Mapping[str, np.ndarray]
    uncertainty: Mapping[str, float]
    quality: Mapping[str, float]
    provenance: Mapping[str, str]
    series: Mapping[str, tuple[float, ...]]


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Result(_ResultFields):
    """The *action*, its *cell*, and the extras around it (see the module docstring)."""

    def __init__(
        self,
        action: Action,
        cell: Cell,
        *,
        gradient: Mapping[str, object] | None = None,
        jvp: Mapping[str, float] | None = None,
        hvp: Mapping[str, object] | None = None,
        uncertainty: Mapping[str, float] | None = None,
        quality: Mapping[str, float] | None = None,
        provenance: Mapping[str, str] | None = None,
        series: Mapping[str, object] | None = None,
    ) -> None:
        parts = {
            'gradient': _frozen(gradient, _vector),
            'jvp': _frozen(jvp, float),
            'hvp': _frozen(hvp, _vector),
            'uncertainty': _frozen(uncertainty, float),
            'series': _frozen(series, _series),
        }
        carried = [name for name, part in parts.items() if part]
        if carried and cell.status not in _CARRIERS:
            refuse(f'a {cell.status.name} result carries no derivative, uncertainty or series, got {carried}')
        if parts['hvp'] and action.direction is None:
            refuse('an hvp answers a direction; this action asked none (Action.of(..., direction=v))')
        negative = sorted(k for k, v in parts['uncertainty'].items() if not v >= 0)
        if negative:
            refuse(f'an uncertainty is a standard deviation >= 0; {negative} are not')
        _ResultFields.__init__(
            self,
            action,
            cell,
            parts['gradient'],
            parts['jvp'],
            parts['hvp'],
            parts['uncertainty'],
            _frozen(quality, float),
            _frozen(provenance, str),
            parts['series'],
        )

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
        parts = {
            'gradient': {k: v.tolist() for k, v in self.gradient.items()},
            'jvp': dict(self.jvp),
            'hvp': {k: v.tolist() for k, v in self.hvp.items()},
            'uncertainty': dict(self.uncertainty),
            'quality': dict(self.quality),
            'provenance': dict(self.provenance),
            'series': {k: list(v) for k, v in self.series.items()},
        }
        return {name: part for name, part in parts.items() if part}

    @classmethod
    def from_extras(cls, action: Action, cell: Cell, extras: Mapping[str, Any]) -> Result:
        """The inverse of :meth:`extras`; a key it does not know is refused."""
        unknown = sorted(set(extras) - set(EXTRAS))
        if unknown:
            refuse(f'filed extras {unknown} are not result extras; the known ones are {list(EXTRAS)}')
        return cls(action, cell, **extras)
