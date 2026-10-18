"""Named transforms: each turns a problem or an oracle the solver cannot use into one it can.

* :func:`finite_difference` / :func:`with_finite_differences` -- gradients from values. Forward and
  central differences cost ``n`` and ``2n`` extra evaluations; the complex step is exact to rounding
  but needs a function that accepts complex input, so it exists for a function and is refused for an
  oracle, whose designs are real.
* :class:`Penalty` -- a constrained problem as an unconstrained one: each objective is worsened by
  ``weight * total violation`` normalised units. A chance constraint is refused: one sample says
  nothing about a probability.
* :func:`check_gradient` -- not a transform but its audit: an oracle's DECLARED gradient against
  finite differences of its own values, per output. A gradient that leaves out a state's
  contribution (the partial where the total derivative is meant, B0-09) is named here.
* :func:`embed` -- every unembedded categorical gets a one-hot embedding, the one metric that does
  not depend on the order the levels were written in (B0-11).

A transform that a solver needed is listed in its result (B0-20); the capability registry names it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Final

import numpy as np
from lab_commons.outcomes import Cell, Reason, Status

from optimi_lab.core.errors import coerce_enum, refuse
from optimi_lab.oracle.action import Action
from optimi_lab.oracle.capability import Capability, Information
from optimi_lab.oracle.oracle import Oracle
from optimi_lab.oracle.result import Result
from optimi_lab.problem import Categorical, Chance, Field, Problem, Real, Sense, Space

__all__ = ['Penalty', 'Scheme', 'check_gradient', 'embed', 'finite_difference', 'with_finite_differences']


class Scheme(StrEnum):
    """A finite-difference scheme."""

    FORWARD = 'forward'
    CENTRAL = 'central'
    COMPLEX = 'complex'


#: Default relative steps: the error-balancing step of each scheme in double precision.
STEPS: Final = {Scheme.FORWARD: 1.5e-8, Scheme.CENTRAL: 6e-6, Scheme.COMPLEX: 1e-20}


def finite_difference(
    function: Callable[[np.ndarray], np.ndarray], x: np.ndarray, *, scheme: str = 'central', step: float | None = None
) -> np.ndarray:
    """The Jacobian ``(m, n)`` of *function* ``R^n -> R^m`` at *x*; the step is relative to ``max(1, |x_i|)``."""
    scheme = coerce_enum(Scheme, scheme, 'the finite-difference scheme')
    x = np.asarray(x, dtype=float)
    h = (STEPS[scheme] if step is None else step) * np.maximum(1.0, np.abs(x))
    columns = []
    base = np.asarray(function(x), dtype=float) if scheme is Scheme.FORWARD else None
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h[i]
        if scheme is Scheme.COMPLEX:
            columns.append(np.imag(np.asarray(function(x + 1j * e))) / h[i])
        elif scheme is Scheme.CENTRAL:
            columns.append(
                (np.asarray(function(x + e), dtype=float) - np.asarray(function(x - e), dtype=float)) / (2 * h[i])
            )
        else:
            columns.append((np.asarray(function(x + e), dtype=float) - base) / h[i])
    return np.column_stack(columns)


@dataclass(frozen=True, slots=True)
class _Differenced:
    base: Oracle
    problem: Problem
    scheme: Scheme
    step: float
    capability: Capability

    async def _values(self, action: Action, unit: np.ndarray) -> Result:
        design = self.problem.space.decode(unit)
        return await self.base(Action.of(self.problem, self.base.capability, design, fidelity=action.fidelity))

    async def __call__(self, action: Action) -> Result:
        plain = Action.of(self.problem, self.base.capability, action.design, fidelity=action.fidelity)
        centre = await self.base(plain)
        wanted = action.want & {Information.JVP, Information.VJP}
        if not wanted or centre.cell.status is not Status.OK:
            return replace(centre, action=action)
        u = self.problem.space.encode(action.design)
        h = self.step * np.maximum(1.0, np.abs(u))
        names = sorted(centre.cell.values)
        columns = []
        for i in range(u.size):
            e = np.zeros_like(u)
            e[i] = h[i]
            # A step past the unit cube would be clipped by the decode map; the difference turns one-sided.
            room_ahead, room_behind = u[i] + h[i] <= 1.0, u[i] - h[i] >= 0.0
            if self.scheme is Scheme.CENTRAL and room_ahead and room_behind:
                ahead, behind = await asyncio.gather(self._values(action, u + e), self._values(action, u - e))
                span = 2 * h[i]
            elif room_ahead:
                ahead, behind, span = await self._values(action, u + e), centre, h[i]
            else:
                ahead, behind, span = centre, await self._values(action, u - e), h[i]
            if ahead.cell.status is not Status.OK or behind.cell.status is not Status.OK:
                failed = ahead if ahead.cell.status is not Status.OK else behind
                reason = Reason('derivative_failed', f'a perturbed evaluation ended {failed.cell.status.name}')
                return Result(
                    action, Cell(Status.PARTIAL, values=centre.cell.values, reason=reason, cost=centre.cell.cost)
                )
            columns.append([(ahead.cell.values[n] - behind.cell.values[n]) / span for n in names])
        jacobian = np.array(columns).T
        return replace(centre, action=action, gradient={n: jacobian[k] for k, n in enumerate(names)})


def with_finite_differences(
    oracle: Oracle, problem: Problem, *, scheme: str = 'central', step: float | None = None
) -> Oracle:
    """*oracle* plus JVP/VJP by differencing its values in the unit cube of *problem*'s space."""
    scheme = coerce_enum(Scheme, scheme, 'the finite-difference scheme')
    if scheme is Scheme.COMPLEX:
        refuse(
            'a complex step needs a function that accepts complex input; an oracle evaluates real designs -- use finite_difference(function, x, scheme="complex")'
        )
    discrete = [v.name for v in problem.space if not isinstance(v, (Real, Field))]
    if discrete:
        refuse(f'finite differences need continuous variables; {discrete} are discrete -- relax them first')
    provides = oracle.capability.provides | {Information.JVP, Information.VJP}
    capability = replace(oracle.capability, provides=provides, version=f'{oracle.capability.version}+fd:{scheme}')
    return _Differenced(oracle, problem, scheme, STEPS[scheme] if step is None else step, capability)


async def check_gradient(
    oracle: Oracle, problem: Problem, design: dict[str, object], *, scheme: str = 'central', step: float | None = None
) -> dict[str, float]:
    """Per output, ``|g - g_fd|_max / |g_fd|_max`` between *oracle*'s declared gradient and differences of its values.

    The error is RELATIVE, so it carries no unit; two zero gradients agree exactly.
    """
    if Information.VJP not in oracle.capability.provides:
        refuse('the oracle declares no gradient (Information.VJP); there is nothing to check')
    want = frozenset({Information.VALUES, Information.VJP})
    declared = await oracle(Action.of(problem, oracle.capability, design, want=want))
    differenced_oracle = with_finite_differences(oracle, problem, scheme=scheme, step=step)
    differenced = await differenced_oracle(Action.of(problem, differenced_oracle.capability, design, want=want))
    for result in (declared, differenced):
        if result.cell.status is not Status.OK:
            refuse(f'the gradient check needs OK evaluations at {design}; one ended {result.cell.status.name}')
    errors = {}
    for name in problem.outputs:
        g, reference = declared.gradient[name], differenced.gradient[name]
        scale = max(float(np.max(np.abs(reference))), float(np.max(np.abs(g))))
        errors[name] = float(np.max(np.abs(g - reference))) / scale if scale > 0 else 0.0
    return errors


@dataclass(frozen=True, slots=True)
class Penalty:
    """*problem* without its constraints; :meth:`apply` worsens objectives by ``weight * violation``."""

    original: Problem
    weight: float

    def __post_init__(self) -> None:
        if not self.weight > 0:
            refuse(f'a penalty weight must be > 0, got {self.weight!r}')
        chance = [c.name for c in self.original.constraints if isinstance(c, Chance)]
        if chance:
            refuse(f'Chance constraints {chance} cannot be penalised from one sample; estimate them first')

    @property
    def problem(self) -> Problem:
        return replace(self.original, constraints=())

    def apply(self, result: Result) -> Result:
        """*result* with every objective worsened by its penalty; a result without values is returned as is."""
        if result.cell.status not in {Status.OK, Status.PARTIAL}:
            return result
        excess = self.weight * result.violation(self.original)
        if excess == 0.0:
            return result
        values = dict(result.cell.values)
        for o in self.original.objectives:
            sign = -1.0 if o.sense is Sense.MAXIMIZE else 1.0
            values[o.name] += sign * excess * o.normalization.scale
        cell = result.cell
        penalised = Cell(cell.status, values=values, reason=cell.reason, evidence=cell.evidence, cost=cell.cost)
        return replace(result, cell=penalised)


def embed(space: Space) -> Space:
    """*space* with a one-hot embedding on every categorical that has none."""
    return Space(
        tuple(
            replace(
                v, embedding=tuple(tuple(float(i == j) for j in range(len(v.levels))) for i in range(len(v.levels)))
            )
            if isinstance(v, Categorical) and v.embedding is None
            else v
            for v in space
        )
    )
