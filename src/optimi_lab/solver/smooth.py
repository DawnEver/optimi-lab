"""The smooth problem a gradient method sees, and what it reports when it halts.

A gradient method is a PURE FUNCTION ``method(evaluate, x0, lo, hi, **settings) -> Solution``:
``evaluate(x)`` returns the :class:`~optimi_lab.operator.Linearization` at ``x`` (or raises
:class:`Unevaluable`), and nothing else reaches it -- no oracle, no clock, no file. That is what lets
one body be checked directly against a reference on a closed form, and driven by the run loop
through :class:`~optimi_lab.solver.gradient.GradientSolver`.

:class:`SmoothView` turns a :class:`~optimi_lab.problem.Problem` and an oracle :class:`Result` into a
linearisation in the space's unit cube and the minimised, normalised objective frame:

* each objective is :meth:`~optimi_lab.problem.Objective.minimized`, its gradient scaled to match;
* each BlackBox constraint and each side of a Linear or Explicit one is a scaled ``c <= 0`` -- the
  same scale :meth:`~optimi_lab.oracle.Result.violation` divides by; a Linear with ``lo == hi`` is an
  equality ``h = 0``;
* design constraints are differentiated here, through the decode map's derivative: a Linear one
  exactly, an Explicit one by a central difference in the design's own coordinates.

Chance and Path constraints, discrete and conditional variables are refused: none has a gradient.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from lab_commons.outcomes import Status

from optimi_lab.core.errors import refuse
from optimi_lab.operator import Linearization
from optimi_lab.oracle import Result
from optimi_lab.problem import BlackBox, Explicit, Field, Linear, Problem, Real, Relation, Scale

__all__ = ['ATOL', 'RTOL', 'Evaluate', 'Halt', 'SmoothView', 'Solution', 'Unevaluable', 'kkt_bar', 'trial']

#: ``evaluate(x) -> Linearization``; raises :class:`Unevaluable`.
Evaluate = Callable[[np.ndarray], Linearization]

#: Default relative stopping bar, against the starting gradient (dimensionless).
RTOL = 1e-8
#: Default absolute floor: normalised objective units per unit-cube width.
ATOL = 1e-10

#: The central-difference step for an Explicit constraint, relative to ``max(1, |value|)``.
_EXPLICIT_STEP = 6e-6


class Unevaluable(Exception):
    """The point produced no usable value; a line search rejects it, a method at its base point halts."""


class Halt(StrEnum):
    """Why a gradient method stopped."""

    CONVERGED = 'converged'
    ITERATION_LIMIT = 'iteration_limit'
    STALLED = 'stalled'
    INFEASIBLE = 'infeasible'
    UNEVALUABLE = 'unevaluable'


@dataclass(frozen=True, slots=True, eq=False)
class Solution:
    """Why it *halt*-ed, the last accepted *x* and its *point*, *iterations*, the KKT *residual*, and a *front*."""

    halt: Halt
    x: np.ndarray
    point: Linearization | None
    iterations: int
    residual: float
    front: tuple[Linearization, ...] = field(default=())
    reason: str = ''
    #: Named choices that shaped the answer, e.g. ``'hvp:finite_difference'`` -- never silent.
    evidence: tuple[str, ...] = ()


def trial(evaluate: Evaluate, x: np.ndarray) -> Linearization | None:
    """``evaluate(x)``, or ``None`` when the point is :class:`Unevaluable` -- the form a line search takes."""
    try:
        return evaluate(x)
    except Unevaluable:
        return None


def _decode_slope(variable: Real | Field, value: object) -> np.ndarray:
    """``d design / d unit`` of one variable at its decoded *value*."""
    if isinstance(variable, Field):
        return np.full(variable.width, variable.hi - variable.lo)
    if variable.scale is Scale.LOG:
        return np.array([float(value) * math.log(variable.hi / variable.lo)])  # type: ignore[arg-type]
    return np.array([variable.hi - variable.lo])


@dataclass(frozen=True, slots=True)
class SmoothView:
    """The smooth problem of *problem*: see the module docstring."""

    problem: Problem

    def __post_init__(self) -> None:
        unfit = [v.name for v in self.problem.space if not isinstance(v, (Real, Field)) or v.active_if is not None]
        if unfit:
            refuse(f'a gradient method needs unconditional Real or Field variables; {unfit} are not')
        unfit = [c.name for c in self.problem.constraints if not isinstance(c, (BlackBox, Linear, Explicit))]
        if unfit:
            refuse(f'constraints {unfit} have no gradient (Chance, Path); estimate or penalise them first')

    @property
    def width(self) -> int:
        return self.problem.space.width

    def _design_rows(self, design: Mapping[str, object]) -> tuple[list, list, list, list]:
        """``(c, dc, h, dh)`` of the design constraints at *design*."""
        space = self.problem.space
        slopes = {v.name: _decode_slope(v, design[v.name]) for v in space}  # type: ignore[arg-type]
        starts = np.cumsum([0, *(v.width for v in space)])
        offset = {v.name: int(s) for v, s in zip(space, starts, strict=False)}
        c, dc, h, dh = [], [], [], []
        for constraint in self.problem.design_constraints:
            row = np.zeros(self.width)
            if isinstance(constraint, Linear):
                value = sum(w * float(design[v]) for v, w in constraint.coefficients)  # type: ignore[arg-type]
                for v, w in constraint.coefficients:
                    row[offset[v]] = w * slopes[v][0]
                sides = [(constraint.lo, -1.0), (constraint.hi, 1.0)]
                if constraint.lo is not None and constraint.lo == constraint.hi:
                    h.append((value - constraint.lo) / constraint.scale)
                    dh.append(row / constraint.scale)
                    continue
                for bound, sign in sides:
                    if bound is not None:
                        c.append(sign * (value - bound) / constraint.scale)
                        dc.append(sign * row / constraint.scale)
                continue
            value = float(constraint.function(design))
            for v in space:
                base = np.atleast_1d(np.asarray(design[v.name], dtype=float))
                for k in range(base.size):
                    step = _EXPLICIT_STEP * max(1.0, abs(base[k]))
                    ahead, behind = base.copy(), base.copy()
                    ahead[k] += step
                    behind[k] -= step
                    pair = [
                        dict(design, **{v.name: p if isinstance(v, Field) else float(p[0])}) for p in (ahead, behind)
                    ]
                    slope = (float(constraint.function(pair[0])) - float(constraint.function(pair[1]))) / (2 * step)
                    row[offset[v.name] + k] = slope * slopes[v.name][k]
            sign = 1.0 if constraint.relation is Relation.AT_MOST else -1.0
            c.append(sign * (value - constraint.bound) / constraint.scale)
            dc.append(sign * row / constraint.scale)
        return c, dc, h, dh

    def objective_hvp(self, result: Result) -> np.ndarray:
        """The first objective's Hessian times the action's direction, in the minimised normalised frame."""
        if result.cell.status is not Status.OK:
            msg = f'{result.cell.status.name}: an hvp evaluation did not compute'
            raise Unevaluable(msg)
        objective = self.problem.objectives[0]
        if objective.name not in result.hvp:
            refuse(f'an OK result carries no hvp of {objective.name!r}; the oracle must answer the HVP it was asked')
        sign = -1.0 if objective.sense.value == 'maximize' else 1.0
        return sign * result.hvp[objective.name] / objective.normalization.scale

    def linearize(self, x: np.ndarray, result: Result) -> Linearization:
        """The linearisation at unit point *x* from an OK *result*; a result without values is Unevaluable."""
        if result.cell.status is not Status.OK:
            reason = result.cell.reason.detail if result.cell.reason is not None else ''
            msg = f'{result.cell.status.name}: {reason}'
            raise Unevaluable(msg)
        values, gradients = result.cell.values, result.gradient
        missing = sorted(set(self.problem.outputs) - set(gradients))
        if missing:
            refuse(f'an OK result carries no gradient of {missing}; the oracle must answer the VJP it was asked')
        f = [o.minimized(values[o.name]) for o in self.problem.objectives]
        df = [
            gradients[o.name] / o.normalization.scale * (-1.0 if o.sense.value == 'maximize' else 1.0)
            for o in self.problem.objectives
        ]
        c, dc, h, dh = self._design_rows(result.action.design)
        for constraint in self.problem.constraints:
            if isinstance(constraint, BlackBox):
                sign = 1.0 if constraint.relation is Relation.AT_MOST else -1.0
                c.append(sign * (values[constraint.output] - constraint.bound) / constraint.scale)
                dc.append(sign * gradients[constraint.output] / constraint.scale)
        n = self.width
        return Linearization(
            x=x,
            f=np.array(f),
            df=np.array(df).reshape(-1, n),
            c=np.array(c),
            dc=np.array(dc).reshape(-1, n),
            h=np.array(h),
            dh=np.array(dh).reshape(-1, n),
        )


def kkt_bar(initial_gradient: np.ndarray, *, rtol: float, atol: float) -> float:
    """The stopping bar: ``max(rtol * |grad f(x0)|_max, atol)`` -- residual <= rtol x start OR <= atol.

    *atol* is in the normalised objective's units per unit-cube width (the frame every method works
    in) and must be > 0: it is the floor that stops a start whose gradient is ~0, where the relative
    bar alone would be ~0 and never met.
    """
    if not atol > 0:
        refuse(f'atol is the floor of the stopping bar and must be > 0, got {atol!r}')
    if not rtol >= 0:
        refuse(f'rtol must be >= 0, got {rtol!r}')
    return max(rtol * float(np.max(np.abs(initial_gradient), initial=0.0)), float(atol))
