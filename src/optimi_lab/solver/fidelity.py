"""Multi-fidelity schedules: successive halving, Hyperband and ASHA over the problem's fidelity levels.

The rungs are :class:`~optimi_lab.problem.Fidelity`'s levels, cheapest first, the last the truth;
an action at rung ``k`` is asked at that level (``Action.of(..., fidelity=...)``), so its identity --
and its record -- is that level's. Scores are one per point (``scalarise`` for several objectives),
ranked by Deb's rule; a point that produced no value is not promoted.

* :class:`SuccessiveHalving` (Jamieson and Talwalkar, 2016) -- ``configs`` points at the start rung,
  the best ``1/eta`` of each rung promoted to the next, until the truth rung has been evaluated.
* :func:`hyperband` (Li et al., 2018) -- successive-halving brackets trading the count against the
  start rung, chained.
* :class:`ASHA` (Li et al., 2020) -- asynchronous: each free slot promotes a point in the top
  ``1/eta`` of its rung that has not been promoted, else starts a new point at the bottom rung.
"""

from collections.abc import Sequence
from math import ceil

import numpy as np
from lab_commons.outcomes import Status

from optimi_lab.core import refuse
from optimi_lab.oracle import Action, Capability, Result
from optimi_lab.problem import Problem
from optimi_lab.solver.meta import Chain
from optimi_lab.solver.population import Scalarise, build, require_one_score, score_rows

__all__ = ['ASHA', 'SuccessiveHalving', 'asha', 'hyperband']


def _rungs(problem: Problem, capability: Capability) -> tuple[str, ...]:
    if problem.fidelity is None:
        refuse('a multi-fidelity schedule needs problem.fidelity levels to promote through')
    names = tuple(level.name for level in problem.fidelity.levels)
    missing = [name for name in names if name not in capability.fidelities]
    if missing:
        refuse(f'the oracle does not run fidelity levels {missing}; it runs {list(capability.fidelities)}')
    return names


def _rank(problem: Problem, scalarise: Scalarise | None, results: Sequence[Result]) -> np.ndarray:
    """Indices of OK *results*, best first by violation then score."""
    ok = [i for i, r in enumerate(results) if r.cell.status is Status.OK]
    if not ok:
        return np.array([], dtype=np.intp)
    values, violation = score_rows(problem, scalarise, [results[i] for i in ok])
    return np.array(ok, dtype=np.intp)[np.lexsort((values[:, 0], violation))]


class SuccessiveHalving:
    """One successive-halving bracket of *configs* uniform points from rung *start*."""

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        configs: int,
        eta: int = 3,
        start: int = 0,
        scalarise: Scalarise | None = None,
        seed: int = 0,
    ) -> None:
        require_one_score(problem, scalarise, 'successive halving')
        self._rungs = _rungs(problem, capability)
        if eta < 2 or configs < 1 or not 0 <= start < len(self._rungs):
            refuse(f'need eta >= 2, configs >= 1 and 0 <= start < {len(self._rungs)}; got {eta}, {configs}, {start}')
        self._problem, self._capability, self._scalarise, self._eta = problem, capability, scalarise, eta
        self._rung = start
        self._points = np.random.default_rng(seed).random((configs, problem.space.width))
        self._load()

    def _load(self) -> None:
        space, level = self._problem.space, self._rungs[self._rung]
        self._batch = [
            Action.of(self._problem, self._capability, space.decode(p), fidelity=level) for p in self._points
        ]
        self._asked, self._told = 0, {}

    def ask(self, capacity: int) -> Sequence[Action]:
        if self.done:
            return []
        start, self._asked = self._asked, min(self._asked + capacity, len(self._batch))
        return self._batch[start : self._asked]

    def tell(self, results: Sequence[Result]) -> None:
        for result in results:
            self._told[result.action.identity.key] = result
        if len(self._told) < len({a.identity.key for a in self._batch}):
            return
        order = _rank(self._problem, self._scalarise, [self._told[a.identity.key] for a in self._batch])
        self._rung += 1
        if self._rung < len(self._rungs) and len(order):
            self._points = self._points[order[: max(1, len(self._points) // self._eta)]]
            self._load()
        else:
            self._rung = len(self._rungs)

    @property
    def done(self) -> bool:
        return self._rung >= len(self._rungs)


def hyperband(problem: Problem, capability: Capability, *, eta: int = 3, seed: int = 0, **settings: object) -> Chain:
    """Hyperband: brackets ``s = s_max .. 0`` of ``ceil((s_max + 1) / (s + 1) * eta ** s)`` points from rung ``s_max - s``."""
    top = len(_rungs(problem, capability)) - 1
    return Chain(
        SuccessiveHalving(
            problem,
            capability,
            configs=ceil((top + 1) / (s + 1) * eta**s),
            eta=eta,
            start=top - s,
            seed=seed + s,
            **settings,
        )
        for s in range(top, -1, -1)
    )


class ASHA:
    """Asynchronous successive halving, *workers* points at a time; it never concludes by itself."""

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        eta: int = 3,
        workers: int = 1,
        scalarise: Scalarise | None = None,
        seed: int = 0,
    ) -> None:
        require_one_score(problem, scalarise, 'asha')
        if workers < 1:
            refuse(f'workers is the number of points asked at once, at least 1, got {workers!r}')
        self._workers = workers
        self._rungs = _rungs(problem, capability)
        if eta < 2:
            refuse(f'eta must be at least 2, got {eta!r}')
        self._problem, self._capability, self._scalarise, self._eta = problem, capability, scalarise, eta
        self._rng = np.random.default_rng(seed)
        self._points: list[np.ndarray] = []
        self._scores: list[dict[int, tuple[float, float]]] = [{} for _ in self._rungs]
        self._promoted: list[set[int]] = [set() for _ in self._rungs]
        self._pending: dict[str, tuple[int, int]] = {}

    def _promotable(self) -> tuple[int, int] | None:
        for rung in range(len(self._rungs) - 2, -1, -1):
            ranked = sorted(self._scores[rung], key=lambda p: (self._scores[rung][p][1], self._scores[rung][p][0]))
            for point in ranked[: len(ranked) // self._eta]:
                if point not in self._promoted[rung]:
                    return rung, point
        return None

    def ask(self, capacity: int) -> Sequence[Action]:
        actions = []
        for _ in range(min(capacity, self._workers)):
            found = self._promotable()
            if found is None:
                self._points.append(self._rng.random(self._problem.space.width))
                rung, point = 0, len(self._points) - 1
            else:
                rung, point = found[0] + 1, found[1]
                self._promoted[found[0]].add(point)
            design = self._problem.space.decode(self._points[point])
            action = Action.of(self._problem, self._capability, design, fidelity=self._rungs[rung])
            self._pending[action.identity.key] = (rung, point)
            actions.append(action)
        return actions

    def tell(self, results: Sequence[Result]) -> None:
        for result in results:
            rung, point = self._pending.pop(result.action.identity.key)
            if result.cell.status is Status.OK:
                values, violation = score_rows(self._problem, self._scalarise, [result])
                self._scores[rung][point] = (float(values[0, 0]), float(violation[0]))

    @property
    def done(self) -> bool:
        return False


def asha(problem: Problem, capability: Capability, **settings: object) -> ASHA:
    """ASHA; ``**settings`` are :class:`ASHA`'s."""
    return build(ASHA, problem, capability, 'asha', **settings)
