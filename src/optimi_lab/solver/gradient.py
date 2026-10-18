"""The gradient methods as :class:`~optimi_lab.solver.Solver` objects: one adapter, driven by REPLAY.

A gradient method is a pure function that calls ``evaluate(x)`` whenever it needs a point
(:mod:`optimi_lab.solver.smooth`); the run loop is the other way round -- it asks the solver what to
evaluate and tells it the result later. :class:`GradientSolver` joins the two without threads and
without a second, inverted copy of each method:

* to find its next action it RE-RUNS the method from the start over the results told so far; the
  first point the method needs and has not been told interrupts it, and that point is the action;
* when the method returns instead, the solver is done and :attr:`GradientSolver.solution` holds it.

A method is deterministic in what it was told, so each re-run retraces the last one exactly and
asks for one new point -- the same property that makes a run resume by replay. Re-running costs the
method's own arithmetic once per evaluation, which is noise beside an evaluation that is a field
solve. A RETRYABLE result is not kept, so its point is asked again; any other non-OK result makes
the point :class:`~optimi_lab.solver.smooth.Unevaluable` (a line search rejects it; a method that
needs it halts UNEVALUABLE).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Final

import numpy as np

from optimi_lab.core.errors import refuse
from optimi_lab.oracle import Action, Capability, Information, Result
from optimi_lab.problem import Problem
from optimi_lab.solver.bounded import lbfgsb, newton_cg, projected_gradient
from optimi_lab.solver.constrained import augmented_lagrangian, interior_point, sqp
from optimi_lab.solver.moving_asymptotes import gcmma, mma
from optimi_lab.solver.multiobjective import mgda, pareto_continuation
from optimi_lab.solver.requires import Eligible, Requires, match
from optimi_lab.solver.smooth import Halt, SmoothView, Solution, Unevaluable

__all__ = ['METHODS', 'GradientSolver']

#: Every gradient method by its registered name (``optimi_lab.solver.SOLVERS`` holds its needs).
METHODS: Final[dict[str, Callable[..., Solution]]] = {
    'projected_gradient': projected_gradient,
    'lbfgsb': lbfgsb,
    'newton_cg': newton_cg,
    'sqp': sqp,
    'interior_point': interior_point,
    'augmented_lagrangian': augmented_lagrangian,
    'mma': mma,
    'gcmma': gcmma,
    'mgda': mgda,
    'pareto_continuation': pareto_continuation,
}

_WANT: Final = frozenset({Information.VALUES, Information.VJP})


class _Needed(BaseException):
    """Raised through the method (and through numerics-lab) when it needs a point not told yet.

    A ``BaseException`` so that no ``except Exception`` on the way can swallow it.
    """

    def __init__(self, x: np.ndarray, direction: np.ndarray | None = None) -> None:
        super().__init__()
        self.x = x
        self.direction = direction


class GradientSolver:
    """Gradient method *method* on *problem* against an oracle of *capability*, from *start* (a design or designs)."""

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        method: str,
        *,
        start: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
        **settings: object,
    ) -> None:
        if method not in METHODS:
            refuse(f'no gradient method {method!r}; the gradient methods are {sorted(METHODS)}')
        verdict = match(Requires.of(method), problem, capability)
        if verdict != Eligible():
            refuse(f'{method} cannot run as declared: {verdict}; apply what it names to the problem or oracle first')
        self._view = SmoothView(problem)
        self._problem, self._capability = problem, capability
        if method == 'newton_cg' and 'hvp' not in settings and Information.HVP in capability.provides:
            settings = {**settings, 'hvp': self._hvp}  # the oracle's declared product
        self._method = partial(METHODS[method], **settings)
        width = problem.space.width
        if start is None:
            self._x0 = np.full(width, 0.5)
        elif isinstance(start, Mapping):
            self._x0 = problem.space.encode(start)
        else:
            self._x0 = np.array([problem.space.encode(design) for design in start])
        self._lo, self._hi = np.zeros(width), np.ones(width)
        self._told: dict[bytes, Result] = {}
        self._asked: dict[str, bytes] = {}
        self._next: Action | None = None
        self._solution: Solution | None = None
        self._advance()

    def _evaluate(self, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        result = self._told.get(x.tobytes())
        if result is None:
            raise _Needed(x)
        return self._view.linearize(x, result)

    def _hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        x, v = np.asarray(x, dtype=float), np.asarray(v, dtype=float)
        result = self._told.get(x.tobytes() + v.tobytes())
        if result is None:
            raise _Needed(x, v)
        return self._view.objective_hvp(result)

    def _advance(self) -> None:
        try:
            self._solution = self._method(self._evaluate, self._x0, self._lo, self._hi)
            self._next = None
        except _Needed as needed:
            design = self._problem.space.decode(needed.x)
            if needed.direction is None:
                self._next = Action.of(self._problem, self._capability, design, want=_WANT)
                key = needed.x.tobytes()
            else:
                want, direction = _WANT | {Information.HVP}, tuple(needed.direction)
                self._next = Action.of(self._problem, self._capability, design, want=want, direction=direction)
                key = needed.x.tobytes() + needed.direction.tobytes()
            self._asked[self._next.identity.key] = key
        except Unevaluable as failure:
            x = self._x0 if self._x0.ndim == 1 else self._x0[0]
            self._solution = Solution(Halt.UNEVALUABLE, x, None, 0, float('nan'), reason=str(failure))
            self._next = None

    def ask(self, capacity: int) -> Sequence[Action]:
        """The one point the method needs next; nothing once it has concluded."""
        return [] if self._next is None or capacity < 1 else [self._next]

    def tell(self, results: Sequence[Result]) -> None:
        """Results of asked points; a RETRYABLE one is dropped, so its point is asked again."""
        for result in results:
            key = self._asked.get(result.action.identity.key)
            if key is None:
                refuse('a result was told that this solver did not ask for; tell each result to its own solver')
            if result.cell.status.decided:
                self._told[key] = result
        self._advance()

    @property
    def done(self) -> bool:
        return self._solution is not None

    @property
    def solution(self) -> Solution:
        """The method's solution, in the unit cube; refused before it has concluded."""
        if self._solution is None:
            refuse('the method has not concluded; drive it with optimi_lab.run first')
        return self._solution

    def design(self) -> dict[str, object]:
        """The design of the solution's point."""
        return self._problem.space.decode(self.solution.x)
