"""D66 -- TuRBO-1 (Eriksson et al., 2019): Bayesian optimisation inside one trust region, Thompson-sampled.

A trust region is a box around the incumbent of the CURRENT restart, side
``length * w_i / prod(w)^(1/d)`` per coordinate (``w`` the fitted GP's length scales, so the box
stretches along the directions the model finds smooth), clipped to the unit cube. Each batch:

* the GP is refitted on the restart's told points only;
* ``candidates`` scrambled Sobol points fill the box, each coordinate perturbed with probability
  ``min(1, 20 / d)`` and the rest held at the incumbent (at least one perturbed);
* ``batch`` joint posterior draws over the candidates (one draw per pick, numerics-lab's jittered
  Cholesky via :meth:`~optimi_lab.model.FittedGaussianProcess.draw`) each pick their minimiser,
  without repeats.

When a batch is told, it is a SUCCESS if it improves the restart's best by more than ``1e-3`` of
its magnitude, else a FAILURE. ``success_tolerance`` successes in a row double ``length`` (to at
most ``length_max``), ``failure_tolerance`` failures in a row halve it (default
``ceil(max(4, d) / batch)``). Below ``length_min`` the restart is over: a fresh maximin Latin
hypercube starts the next one with ``length`` reset; told data of earlier restarts is kept only for
:attr:`TuRBO.best` -- the told OK result of lowest score across every restart (B0-08).

One score (``scalarise`` for several); constraints are refused (penalise them first).
"""

from collections.abc import Sequence
from math import ceil

import numpy as np
from lab_commons.outcomes import Status

from optimi_lab.core import refuse
from optimi_lab.core.qmc import maximin_latin_hypercube, sobol
from optimi_lab.model import GaussianProcess
from optimi_lab.oracle import Action, Capability, Result
from optimi_lab.problem import Problem
from optimi_lab.solver.bayesian import require_modelable
from optimi_lab.solver.population import Scalarise, build, require_one_score, score_rows

__all__ = ['TuRBO', 'turbo']


class TuRBO:
    """One trust region, Thompson sampling inside it, restarts when it collapses."""

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        initial: int | None = None,
        batch: int = 1,
        length: float = 0.8,
        length_min: float = 0.5**7,
        length_max: float = 1.6,
        success_tolerance: int = 3,
        failure_tolerance: int | None = None,
        candidates: int | None = None,
        surrogate: GaussianProcess | None = None,
        scalarise: Scalarise | None = None,
        max_evaluations: int | None = None,
        seed: int = 0,
    ) -> None:
        require_one_score(problem, scalarise, 'turbo')
        require_modelable(problem, constraints=False)
        width = problem.space.width
        if not 0 < length_min < length <= length_max or batch < 1 or success_tolerance < 1:
            refuse(f'need 0 < length_min < length <= length_max, batch >= 1, success_tolerance >= 1; got '
                   f'{length_min}, {length}, {length_max}, {batch}, {success_tolerance}')  # fmt: skip
        if surrogate is not None and not isinstance(surrogate, GaussianProcess):
            refuse(f'Thompson sampling draws a joint posterior; pass a GaussianProcess, got {surrogate!r}')
        self._problem, self._capability, self._scalarise, self._width = problem, capability, scalarise, width
        self._initial = 2 * width if initial is None else initial
        if self._initial < 2:
            refuse(f'initial is at least 2 points, got {self._initial}')
        self._batch, self._length0, self._length_min, self._length_max = batch, length, length_min, length_max
        self._success_tolerance = success_tolerance
        self._failure_tolerance = ceil(max(4, width) / batch) if failure_tolerance is None else failure_tolerance
        self._candidates = min(100 * width, 1000) if candidates is None else candidates
        self._surrogate = surrogate or GaussianProcess(restarts=2, seed=seed)
        self._max = max_evaluations
        self._rng = np.random.default_rng(seed)
        self._told = 0
        self._results: list[Result] = []
        self._seen: set[str] = set()
        self.restarts = 0
        self._restart()

    def _restart(self) -> None:
        self._length, self._successes, self._failures = self._length0, 0, 0
        self._x: list[np.ndarray] = []
        self._y: list[float] = []
        self._queue = list(maximin_latin_hypercube(self._initial, self._width, self._rng))
        self._pending: dict[str, np.ndarray] = {}
        self._outstanding: list[float] = []

    @property
    def length(self) -> float:
        """The current trust region's base side, unit-cube units."""
        return self._length

    def ask(self, capacity: int) -> Sequence[Action]:
        room = capacity if self._max is None else min(capacity, self._max - self._told - len(self._pending))
        if self.done or room <= 0 or (self._pending and not self._queue):
            return []
        if not self._queue:
            if not self._y:
                refuse('no point of this restart produced a value; there is nothing to model')
            self._queue = list(self._thompson())
        rows, self._queue = self._queue[:room], self._queue[room:]
        space, actions = self._problem.space, []
        for row in rows:
            action = Action.of(self._problem, self._capability, space.decode(row))
            self._pending[action.identity.key] = space.encode(action.design)
            actions.append(action)
        return actions

    def tell(self, results: Sequence[Result]) -> None:
        for result in results:
            key = result.action.identity.key
            if key not in self._pending:
                if key in self._seen:
                    continue
                refuse('a result was told that this solver did not ask for; tell each result to its own solver')
            x = self._pending.pop(key)
            self._seen.add(key)
            self._told += 1
            if result.cell.status is Status.OK:
                score = float(score_rows(self._problem, self._scalarise, [result])[0][0, 0])
                self._results.append(result)
                self._outstanding.append(score)
                self._x.append(x)
                self._y.append(score)
        if not self._pending and not self._queue and self._outstanding:
            self._adapt()

    def _adapt(self) -> None:
        """Fold one told batch into the success/failure counters and the length."""
        new = min(self._outstanding)
        before = self._y[: len(self._y) - len(self._outstanding)]
        self._outstanding = []
        if not before:
            return
        incumbent = min(before)
        if new < incumbent - 1e-3 * abs(incumbent):
            self._successes, self._failures = self._successes + 1, 0
        else:
            self._successes, self._failures = 0, self._failures + 1
        if self._successes >= self._success_tolerance:
            self._length, self._successes = min(2.0 * self._length, self._length_max), 0
        elif self._failures >= self._failure_tolerance:
            self._length, self._failures = self._length / 2.0, 0
        if self._length < self._length_min:
            self.restarts += 1
            self._restart()

    def _thompson(self) -> np.ndarray:
        x, y = np.array(self._x), np.array(self._y)
        fitted = self._surrogate.fit(x, y[:, None])
        lengths = np.exp(fitted.hyperparameters[0][1:-1])
        weights = lengths / np.prod(lengths) ** (1.0 / self._width)
        centre = x[int(np.argmin(y))]
        lo = np.clip(centre - weights * self._length / 2.0, 0.0, 1.0)
        hi = np.clip(centre + weights * self._length / 2.0, 0.0, 1.0)
        points = lo + (hi - lo) * sobol(self._candidates, self._width, rng=self._rng)
        mask = self._rng.random(points.shape) <= min(1.0, 20.0 / self._width)
        mask[np.arange(len(mask)), self._rng.integers(self._width, size=len(mask))] = True
        points = np.where(mask, points, centre)
        draws = fitted.draw(points, self._rng.standard_normal((self._batch, len(points), 1)))[:, :, 0]
        chosen: list[int] = []
        for draw in draws:
            draw[chosen] = np.inf
            chosen.append(int(np.argmin(draw)))
        return points[chosen]

    @property
    def done(self) -> bool:
        return self._max is not None and self._told >= self._max

    @property
    def best(self) -> Result | None:
        """The told OK result of lowest score over every restart (B0-08)."""
        if not self._results:
            return None
        scores = score_rows(self._problem, self._scalarise, self._results)[0][:, 0]
        return self._results[int(np.argmin(scores))]


def turbo(problem: Problem, capability: Capability, **settings: object) -> TuRBO:
    """TuRBO-1; ``**settings`` are :class:`TuRBO`'s."""
    return build(TuRBO, problem, capability, 'turbo', **settings)
