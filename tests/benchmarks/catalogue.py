"""D29 -- the benchmark catalogue: problems with KNOWN answers, declared through the P8 base.

Each :class:`Benchmark` is a :class:`~optimi_lab.problem.Problem`, an oracle over it, and the true
Pareto front in the MINIMISED frame, computed in closed form rather than from any run of this
package (a metric verified against its own output proves only that it has not changed).

The B0 counterexamples (research package ``12_assumptions_counterexamples``) are tests beside this
module; the ones a later phase answers are ``xfail(strict=True)`` naming that phase's item, so the
day it lands they turn red and are promoted.

Reference oracles (pymoo and the like) are NEVER imported here at runtime of the package: this is a
test tree.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from lab_commons.outcomes import Status

from optimi_lab.core import non_dominated_sorting
from optimi_lab.oracle import Action, Capability, Oracle, Result, from_function
from optimi_lab.problem import BlackBox, Objective, Problem, Real, Space
from optimi_lab.solver import Solver

__all__ = ['CATALOGUE', 'Benchmark', 'RandomSearch', 'drive', 'front_of']


@dataclass(frozen=True)
class Benchmark:
    """A named problem, its oracle, and its true front sampled at *n* points (minimised frame)."""

    name: str
    problem: Problem
    oracle: Oracle
    front: Callable[[int], np.ndarray]


def _space(n: int, lo: float = 0.0, hi: float = 1.0) -> Space:
    return Space(tuple(Real(f'x{i}', lo, hi) for i in range(n)))


def _vector(action: Action, n: int) -> np.ndarray:
    return np.array([action.design[f'x{i}'] for i in range(n)])


def _zdt(
    name: str, shape: Callable[[float, float], float], front: Callable[[np.ndarray], np.ndarray], n: int = 6
) -> Benchmark:
    problem = Problem(_space(n), (Objective('f1'), Objective('f2')))

    def evaluate(action: Action) -> dict[str, float]:
        x = _vector(action, n)
        g = 1.0 + 9.0 * x[1:].sum() / (n - 1)
        return {'f1': float(x[0]), 'f2': float(g * shape(x[0], g))}

    def true_front(points: int) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, points)
        return front(np.column_stack([f1, np.zeros_like(f1)]))

    return Benchmark(name, problem, from_function(evaluate, Capability(outputs=('f1', 'f2'), version=name)), true_front)


def _zdt1_front(f: np.ndarray) -> np.ndarray:
    f[:, 1] = 1.0 - np.sqrt(f[:, 0])
    return f


def _zdt2_front(f: np.ndarray) -> np.ndarray:
    f[:, 1] = 1.0 - f[:, 0] ** 2
    return f


def _dtlz2(m: int = 3, k: int = 5) -> Benchmark:
    n = m - 1 + k
    names = tuple(f'f{i + 1}' for i in range(m))
    problem = Problem(_space(n), tuple(Objective(name) for name in names))

    def evaluate(action: Action) -> dict[str, float]:
        x = _vector(action, n)
        g = float(np.sum((x[m - 1 :] - 0.5) ** 2))
        theta = x[: m - 1] * np.pi / 2
        f = []
        for i in range(m):
            value = (1 + g) * np.prod(np.cos(theta[: m - 1 - i]))
            if i > 0:
                value *= np.sin(theta[m - 1 - i])
            f.append(float(value))
        return dict(zip(names, f, strict=True))

    def true_front(points: int) -> np.ndarray:
        rng = np.random.default_rng(0)
        f = np.abs(rng.normal(size=(points, m)))
        return f / np.linalg.norm(f, axis=1, keepdims=True)

    return Benchmark('dtlz2', problem, from_function(evaluate, Capability(outputs=names, version='dtlz2')), true_front)


def _bnh() -> Benchmark:
    """Binh-Korn: two objectives, two black-box constraints; the front is known piecewise."""
    problem = Problem(
        Space((Real('x0', 0.0, 5.0), Real('x1', 0.0, 3.0))),
        (Objective('f1'), Objective('f2')),
        constraints=(BlackBox('c1', '<=', 25.0), BlackBox('c2', '>=', 7.7)),
    )

    def evaluate(action: Action) -> dict[str, float]:
        x, y = action.design['x0'], action.design['x1']
        return {
            'f1': 4 * x**2 + 4 * y**2,
            'f2': (x - 5) ** 2 + (y - 5) ** 2,
            'c1': (x - 5) ** 2 + y**2,
            'c2': (x - 8) ** 2 + (y + 3) ** 2,
        }

    def true_front(points: int) -> np.ndarray:
        a = np.linspace(0.0, 3.0, points // 2)
        b = np.linspace(3.0, 5.0, points - points // 2)
        x = np.concatenate([a, b])
        y = np.concatenate([a, np.full_like(b, 3.0)])
        return np.column_stack([4 * x**2 + 4 * y**2, (x - 5) ** 2 + (y - 5) ** 2])

    capability = Capability(outputs=('f1', 'f2', 'c1', 'c2'), version='bnh')
    return Benchmark('bnh', problem, from_function(evaluate, capability), true_front)


CATALOGUE: dict[str, Benchmark] = {
    b.name: b
    for b in (
        _zdt('zdt1', lambda f1, g: 1 - np.sqrt(f1 / g), _zdt1_front),
        _zdt('zdt2', lambda f1, g: 1 - (f1 / g) ** 2, _zdt2_front),
        _dtlz2(),
        _bnh(),
    )
}


class RandomSearch:
    """A seeded uniform search over the unit cube: the simplest solver that drives the P8 loop."""

    def __init__(self, problem: Problem, capability: Capability, seed: int, batch: int = 10) -> None:
        self._problem, self._capability, self._batch = problem, capability, batch
        self._rng = np.random.default_rng(seed)
        self.told: list[Result] = []

    def ask(self, capacity: int) -> Sequence[Action]:
        space = self._problem.space
        return [
            Action.of(self._problem, self._capability, space.decode(self._rng.random(space.width)))
            for _ in range(min(self._batch, capacity))
        ]

    def tell(self, results: Sequence[Result]) -> None:
        self.told.extend(results)

    @property
    def done(self) -> bool:
        return False


def drive(solver: Solver, oracle: Oracle, evaluations: int) -> list[Result]:
    """Ask, evaluate and tell in memory until *evaluations* are spent or the solver concludes.

    The run layer's record costs two fsyncs per evaluation; an ALGORITHM gate measures the algorithm,
    so it drives the solver directly. An identity asked twice is evaluated once, as the run does.
    """

    async def loop() -> list[Result]:
        seen: dict[str, Result] = {}
        spent = 0
        while spent < evaluations and not solver.done:
            asked = list(solver.ask(evaluations - spent))
            if not asked:
                break
            answers = []
            for action in asked:
                if action.identity.key not in seen:
                    seen[action.identity.key] = await oracle(action)
                    spent += 1
                answers.append(seen[action.identity.key])
            solver.tell(answers)
        return list(seen.values())

    return asyncio.run(loop())


def front_of(problem: Problem, results: Sequence[Result]) -> np.ndarray:
    """The feasible non-dominated objective rows of *results*, minimised frame."""
    kept = [r for r in results if r.cell.status is Status.OK and r.violation(problem) == 0.0]
    if not kept:
        return np.zeros((0, len(problem.objectives)))
    values = np.array([[o.minimized(r.cell.values[o.name]) for o in problem.objectives] for r in kept])
    return values[non_dominated_sorting(values)]
