"""Sequential local solvers as generators, and Nelder-Mead.

A local derivative-free method is a LOOP that needs each answer before its next question. Written
as a generator -- ``scores, violation = yield points`` -- it stays the textbook loop, and
:class:`SequentialSolver` adapts it to the ask/tell protocol: it asks the yielded batch, and once
every point is told it sends ``(scores, violation)`` back in the batch's order. A point that
produced no value comes back as NaN in both arrays, which every comparison here reads as "not
better" -- the method rejects it, and no number is ever invented for it.

The solver CONCLUDES when its generator returns (its own convergence test) or when it has been
told ``max_evaluations`` points. :attr:`SequentialSolver.best` is the best FEASIBLE point told.

:class:`NelderMead` is scipy's ``minimize(method='Nelder-Mead', bounds=...)`` step for step (the
reference oracle in the tests): the same coefficients, the same clipping onto the unit cube, the
same initial simplex and the same ``xatol``/``fatol`` test, so on one problem both visit the same
points. A constraint is an extreme barrier: an infeasible point scores NaN.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence

import numpy as np
from lab_commons.outcomes import Status

from optimi_lab.core import refuse
from optimi_lab.oracle import Action, Capability, Result
from optimi_lab.problem import Field, Problem, Real
from optimi_lab.solver.population import Scalarise, build, require_one_score, score_rows

__all__ = ['NelderMead', 'Search', 'SequentialSolver', 'nelder_mead']

#: ``scores, violation = yield points``.
Search = Generator[np.ndarray, tuple[np.ndarray, np.ndarray]]


class SequentialSolver(ABC):
    """Drive the generator :meth:`_search` from ``start`` (unit cube; None: the centre)."""

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        start: np.ndarray | None = None,
        scalarise: Scalarise | None = None,
        max_evaluations: int | None = None,
        seed: int = 0,
    ) -> None:
        require_one_score(problem, scalarise, type(self).__name__)
        discrete = [v.name for v in problem.space if not isinstance(v, (Real, Field)) or v.active_if is not None]
        if discrete:
            refuse(f'a local continuous search needs unconditional Real or Field variables; {discrete} are not')
        self._problem, self._capability, self._scalarise = problem, capability, scalarise
        self._width = problem.space.width
        self._start = np.full(self._width, 0.5) if start is None else np.clip(np.asarray(start, dtype=float), 0, 1)
        self._max = max_evaluations
        self._rng = np.random.default_rng(seed)
        self._evaluated = 0
        self._finished = False
        self.best: tuple[np.ndarray, float] | None = None
        self._search_steps = self._search()
        self._load(next(self._search_steps))

    def _load(self, points: np.ndarray) -> None:
        self._points = np.atleast_2d(points)
        space = self._problem.space
        self._batch = [Action.of(self._problem, self._capability, space.decode(p)) for p in self._points]
        self._asked, self._told = 0, {}

    def ask(self, capacity: int) -> Sequence[Action]:
        if self.done:
            return []
        start, self._asked = self._asked, min(self._asked + capacity, len(self._batch))
        return self._batch[start : self._asked]

    def tell(self, results: Sequence[Result]) -> None:
        keys = {a.identity.key for a in self._batch}
        for result in results:
            if result.action.identity.key not in keys:
                refuse('a result was told that this step did not ask for; tell each result to its own solver')
            self._told[result.action.identity.key] = result
        if len(self._told) < len(keys):
            return
        ordered = [self._told[a.identity.key] for a in self._batch]
        scores, violation = np.full(len(ordered), np.nan), np.full(len(ordered), np.nan)
        ok = [i for i, r in enumerate(ordered) if r.cell.status is Status.OK]
        if ok:
            values, shortfall = score_rows(self._problem, self._scalarise, [ordered[i] for i in ok])
            scores[ok], violation[ok] = values[:, 0], shortfall
        self._evaluated += len(ordered)
        for point, score, shortfall in zip(self._points, scores, violation, strict=True):
            if shortfall == 0.0 and (self.best is None or score < self.best[1]):
                self.best = (point.copy(), float(score))
        try:
            self._load(self._search_steps.send((scores, violation)))
        except StopIteration:
            self._finished = True

    @property
    def done(self) -> bool:
        return self._finished or (self._max is not None and self._evaluated >= self._max)

    @abstractmethod
    def _search(self) -> Search:
        """The method's loop: yield unit-cube points, receive their scores and violations."""


class NelderMead(SequentialSolver):
    """Nelder-Mead on the unit cube; ``adaptive`` takes Gao and Han's dimension-dependent coefficients."""

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        initial_simplex: np.ndarray | None = None,
        xatol: float = 1e-4,
        fatol: float = 1e-4,
        adaptive: bool = False,
        **base: object,
    ) -> None:
        self._simplex0, self._xatol, self._fatol, self._adaptive = initial_simplex, xatol, fatol, adaptive
        super().__init__(problem, capability, **base)

    def _initial(self) -> np.ndarray:
        if self._simplex0 is not None:
            simplex = np.array(self._simplex0, dtype=float)
            if simplex.shape != (self._width + 1, self._width):
                refuse(f'initial_simplex must be ({self._width + 1}, {self._width}), got {simplex.shape}')
        else:
            simplex = np.tile(self._start, (self._width + 1, 1))
            for k in range(self._width):
                simplex[k + 1, k] = 1.05 * simplex[k + 1, k] if simplex[k + 1, k] != 0 else 0.00025
        simplex = np.where(simplex > 1.0, 2.0 - simplex, simplex)
        return np.clip(simplex, 0.0, 1.0)

    def _search(self) -> Search:
        n = self._width
        if self._adaptive:
            rho, chi, psi, sigma = 1.0, 1 + 2 / n, 0.75 - 1 / (2 * n), 1 - 1 / n
        else:
            rho, chi, psi, sigma = 1.0, 2.0, 0.5, 0.5
        sim = self._initial()
        fsim = yield from self._scores(sim)
        order = np.argsort(fsim)
        sim, fsim = sim[order], fsim[order]
        while not (
            np.max(np.abs(sim[1:] - sim[0])) <= self._xatol and np.max(np.abs(fsim[0] - fsim[1:])) <= self._fatol
        ):
            xbar = sim[:-1].sum(axis=0) / n
            xr = np.clip((1 + rho) * xbar - rho * sim[-1], 0.0, 1.0)
            fxr = (yield from self._scores(xr[None]))[0]
            shrink = False
            if fxr < fsim[0]:
                xe = np.clip((1 + rho * chi) * xbar - rho * chi * sim[-1], 0.0, 1.0)
                fxe = (yield from self._scores(xe[None]))[0]
                sim[-1], fsim[-1] = (xe, fxe) if fxe < fxr else (xr, fxr)
            elif fxr < fsim[-2]:
                sim[-1], fsim[-1] = xr, fxr
            elif fxr < fsim[-1]:
                xc = np.clip((1 + psi * rho) * xbar - psi * rho * sim[-1], 0.0, 1.0)
                fxc = (yield from self._scores(xc[None]))[0]
                shrink = not fxc <= fxr
                if not shrink:
                    sim[-1], fsim[-1] = xc, fxc
            else:
                xcc = np.clip((1 - psi) * xbar + psi * sim[-1], 0.0, 1.0)
                fxcc = (yield from self._scores(xcc[None]))[0]
                shrink = not fxcc < fsim[-1]
                if not shrink:
                    sim[-1], fsim[-1] = xcc, fxcc
            if shrink:
                sim[1:] = np.clip(sim[0] + sigma * (sim[1:] - sim[0]), 0.0, 1.0)
                fsim[1:] = yield from self._scores(sim[1:])
            order = np.argsort(fsim)
            sim, fsim = sim[order], fsim[order]

    @staticmethod
    def _scores(points: np.ndarray) -> Generator[np.ndarray, tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Ask *points*; an infeasible or failed one scores NaN (the extreme barrier)."""
        scores, violation = yield points
        return np.where(violation == 0.0, scores, np.nan)


def nelder_mead(problem: Problem, capability: Capability, **settings: object) -> NelderMead:
    """Nelder-Mead; ``**settings`` are :class:`NelderMead`'s and ``start``/``scalarise``/``max_evaluations``/``seed``."""
    return build(NelderMead, problem, capability, 'nelder_mead', **settings)
