"""The structured solvers (P14) as :class:`~optimi_lab.solver.Solver` objects: coefficients in, one solve, one check.

A structured solver reads algebra, not samples. The problem DECLARES an output linear (quadratic) in
its :class:`~optimi_lab.problem.Structure`; the ORACLE supplies the coefficients, as the information
class :data:`~optimi_lab.oracle.COEFFICIENTS_FOR` names (P14 ruling 2026-10-19). So the run is:

1. ask one action at *start* wanting VALUES and the coefficients; each output then reads
   ``1/2 x.Q x + a.x + k`` over the natural design vector, ``k`` recovered from the value at *start*;
2. assemble the program -- the objective (negated to maximise), one row per BlackBox constraint
   on an output, one per side of a Linear constraint (an equality when its sides meet), the
   variable box as bounds -- and solve it with :func:`~optimi_lab.solver.lp` or
   :func:`~optimi_lab.solver.qp`;
3. ask the solution's design once, so the archive holds the optimum as the oracle evaluated it.

Only linearly spaced :class:`~optimi_lab.problem.Real` variables are a natural vector that a linear
form can be linear in; anything else is refused naming the variable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Final

import numpy as np
from lab_commons.outcomes import Status

from optimi_lab.core.errors import refuse
from optimi_lab.oracle import COEFFICIENTS_FOR, Action, Capability, Information, Result
from optimi_lab.problem import BlackBox, Linear, Problem, Real, Scale, Sense
from optimi_lab.solver.linear import lp
from optimi_lab.solver.linear_program import LinearProgram, StructuredSolution
from optimi_lab.solver.quadratic import qp
from optimi_lab.solver.requires import Eligible, Requires, match
from optimi_lab.solver.smooth import Halt

__all__ = ['STRUCTURED', 'StructuredSolver']


def _lp(program: LinearProgram, hessian: np.ndarray, **settings: object) -> StructuredSolution:
    if np.any(hessian):
        refuse('the oracle reported a quadratic term for an objective declared LINEAR; the declaration lies')
    return lp(program, **settings)  # type: ignore[arg-type]


#: Every structured method the driver runs, by its registered name.
STRUCTURED: Final[dict[str, Callable[..., StructuredSolution]]] = {
    'lp': _lp,
    'qp': qp,
}


class StructuredSolver:
    """Structured method *method* on *problem* against an oracle of *capability*, reading coefficients at *start*."""

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        method: str,
        *,
        start: Mapping[str, object] | None = None,
        **settings: object,
    ) -> None:
        if method not in STRUCTURED:
            refuse(f'no structured method {method!r}; the driven structured methods are {sorted(STRUCTURED)}')
        requires = Requires.of(method)
        verdict = match(requires, problem, capability)
        if verdict != Eligible():
            refuse(f'{method} cannot run as declared: {verdict}; apply what it names to the problem or oracle first')
        for variable in problem.space:
            if not isinstance(variable, Real) or variable.scale is not Scale.LINEAR:
                refuse(
                    f'{method} is linear in the natural design vector; {variable.name!r} is not a linear-scale Real (a log scale or a discrete kind)'
                )
        self._problem, self._capability, self._method, self._settings = problem, capability, method, settings
        self._names = problem.space.names
        self._lo = np.array([v.lo for v in problem.space])
        self._hi = np.array([v.hi for v in problem.space])
        if start is None:
            start = dict(zip(self._names, (0.5 * (self._lo + self._hi)).tolist(), strict=True))
        want = frozenset({Information.VALUES}) | {COEFFICIENTS_FOR[f] for f in requires.forms if f in COEFFICIENTS_FOR}
        self._next: Action | None = Action.of(problem, capability, start, want=want)
        self._solution: StructuredSolution | None = None
        self._optimum: Result | None = None

    def ask(self, capacity: int) -> Sequence[Action]:
        """The coefficient read, then the optimum's check; nothing once concluded."""
        return [] if self._next is None or capacity < 1 else [self._next]

    def tell(self, results: Sequence[Result]) -> None:
        """A RETRYABLE result is dropped, so its action is asked again."""
        for result in results:
            if self._next is None or result.action.identity.key != self._next.identity.key:
                refuse('a result was told that this solver did not ask for; tell each result to its own solver')
            if not result.cell.status.decided:
                continue
            if self._solution is None:
                self._solve(result)
            else:
                self._optimum, self._next = result, None

    def _solve(self, result: Result) -> None:
        if result.cell.status is not Status.OK:
            x = np.array([float(result.action.design[n]) for n in self._names])  # type: ignore[arg-type]
            reason = f'the coefficient read did not compute: {result.cell.status.name}'
            self._solution, self._next = StructuredSolution(Halt.UNEVALUABLE, x, np.nan, 0, reason=reason), None
            return
        program, hessian = self._program(result)
        self._solution = STRUCTURED[self._method](program, hessian, **self._settings)
        if self._solution.halt is Halt.CONVERGED:
            design = dict(zip(self._names, np.clip(self._solution.x, self._lo, self._hi).tolist(), strict=True))
            self._next = Action.of(self._problem, self._capability, design)
        else:
            self._next = None

    def _program(self, result: Result) -> tuple[LinearProgram, np.ndarray]:
        n = len(self._names)
        x0 = np.array([float(result.action.design[name]) for name in self._names])  # type: ignore[arg-type]

        def polynomial(output: str) -> tuple[np.ndarray, np.ndarray, float]:
            if output not in result.linear_coefficients:
                refuse(f'the oracle answered no coefficients for {output!r}; it declared {sorted(result.action.want)}')
            a = result.linear_coefficients[output]
            q = result.quadratic_coefficients.get(output, np.zeros((n, n)))
            if a.shape != (n,) or q.shape != (n, n):
                refuse(f'the coefficients of {output!r} do not fit the {n} variable(s) {list(self._names)}')
            return a, q, float(result.cell.values[output]) - float(a @ x0) - 0.5 * float(x0 @ q @ x0)

        (objective,) = self._problem.objectives
        a, hessian, _ = polynomial(objective.name)
        sign = -1.0 if objective.sense is Sense.MAXIMIZE else 1.0
        a_ub: list[np.ndarray] = []
        b_ub: list[float] = []
        a_eq: list[np.ndarray] = []
        b_eq: list[float] = []
        for c in self._problem.constraints:
            if isinstance(c, BlackBox):
                row, q, k = polynomial(c.output)
                if np.any(q):
                    refuse(f'constraint {c.name!r} has a quadratic term; {self._method} takes linear rows only')
                flip = 1.0 if c.relation == '<=' else -1.0
                a_ub.append(flip * row)
                b_ub.append(flip * (c.bound - k))
            elif isinstance(c, Linear):
                row = np.array([c.weights.get(name, 0.0) for name in self._names])
                if c.lo is not None and c.lo == c.hi:
                    a_eq.append(row)
                    b_eq.append(c.lo)
                    continue
                if c.hi is not None:
                    a_ub.append(row)
                    b_ub.append(c.hi)
                if c.lo is not None:
                    a_ub.append(-row)
                    b_ub.append(-c.lo)
        program = LinearProgram(
            sign * a,
            a_ub=np.array(a_ub).reshape(-1, n),
            b_ub=b_ub,
            a_eq=np.array(a_eq).reshape(-1, n),
            b_eq=b_eq,
            lo=self._lo,
            hi=self._hi,
        )
        return program, sign * hessian

    @property
    def done(self) -> bool:
        return self._next is None and self._solution is not None

    @property
    def solution(self) -> StructuredSolution:
        """The method's solution over the natural design vector; refused before it has concluded."""
        if self._solution is None:
            refuse('the method has not concluded; drive it with optimi_lab.run first')
        return self._solution

    @property
    def optimum(self) -> Result | None:
        """The oracle's evaluation of the solution's design; None when the method did not converge."""
        return self._optimum

    def design(self) -> dict[str, object]:
        """The design of the solution's point."""
        return dict(zip(self._names, self.solution.x.tolist(), strict=True))
