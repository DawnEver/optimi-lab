"""D65 -- multi-fidelity Bayesian optimisation: AR1 co-Kriging and the augmented expected improvement.

The levels are :class:`~optimi_lab.problem.Fidelity`'s, cheapest first, the last the truth; an action
at level ``l`` is asked at that level (``Action.of(..., fidelity=...)``). The model is
:class:`~optimi_lab.model.CoKriging` over every told point at its level.

THE INITIAL DESIGN IS NESTED: one maximin Latin hypercube, the truth level taking its first
``initial[-1]`` rows and each cheaper level the first ``initial[l]`` (non-increasing toward the truth,
at least 3 per level above the cheapest -- co-Kriging's scale, offset and discrepancy). The default
is ``2 (d + 1)`` at the truth and three times the level above at each cheaper one: with fewer, the
cheapest level's likelihood is maximised by calling all of it noise (measured on Forrester's pair
at 6 + 3 points), and the AR1 scale that follows is meaningless.

THE PICK (Huang, Allen, Notz and Miller, 2006) is the point AND the level maximising

    AEI(x, l) = EI_L(x) * corr(f_l(x), f_L(x)) * (1 - sqrt(noise_l) / sqrt(var_l(x) + noise_l)) * cost_L / cost_l

with ``EI_L`` the truth level's expected improvement over the EFFECTIVE best ``min (mean_L + std_L)``
at the told points, ``corr = prod_{t > l} scale_t sqrt(var_l / var_L)`` under the AR1 chain (clipped to
``[0, 1]``), and the level's cost from its declaration. Each level's acquisition is maximised by
:func:`~optimi_lab.solver.model_based.maximise`; one point is asked at a time.

:attr:`MultiFidelityBO.best` is the told OK TRUTH-level result of lowest score (B0-08): a cheap level
steers, the truth reports. Constraints are refused (penalise them first).
"""

import itertools
from collections.abc import Sequence

import numpy as np
from lab_commons.outcomes import Status

from optimi_lab.core import refuse
from optimi_lab.core.qmc import maximin_latin_hypercube
from optimi_lab.model import CoKriging, FittedCoKriging
from optimi_lab.operator import expected_improvement
from optimi_lab.oracle import Action, Capability, Result
from optimi_lab.problem import Problem
from optimi_lab.solver.model_based import maximise, require_modelable
from optimi_lab.solver.population import Scalarise, build, require_one_score, score_rows

__all__ = ['MultiFidelityBO', 'mfbo']


class MultiFidelityBO:
    """Augmented-EI multi-fidelity BO over the problem's fidelity levels."""

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        initial: Sequence[int] | None = None,
        surrogate: CoKriging | None = None,
        scalarise: Scalarise | None = None,
        candidates: int | None = None,
        polish: int = 3,
        max_evaluations: int | None = None,
        seed: int = 0,
    ) -> None:
        require_one_score(problem, scalarise, 'mfbo')
        require_modelable(problem, constraints=False)
        if problem.fidelity is None or len(problem.fidelity.levels) < 2:
            refuse('multi-fidelity BO needs problem.fidelity with at least two levels')
        levels = problem.fidelity.levels
        missing = [level.name for level in levels if level.name not in capability.fidelities]
        if missing:
            refuse(f'the oracle does not run fidelity levels {missing}; it runs {list(capability.fidelities)}')
        width = problem.space.width
        top = 2 * (width + 1)
        counts = [top * 3 ** (len(levels) - 1 - i) for i in range(len(levels))] if initial is None else list(initial)
        if len(counts) != len(levels) or any(b > a for a, b in itertools.pairwise(counts)):
            refuse(f'initial is one non-increasing count per level ({len(levels)}), got {counts}')
        if counts[0] < 2 or min(counts[1:]) < 3:
            refuse(f'co-Kriging needs at least 2 cheapest points and 3 per level above, got {counts}')
        if surrogate is not None and not isinstance(surrogate, CoKriging):
            refuse(f'the augmented EI reads the AR1 chain; pass a CoKriging, got {surrogate!r}')
        self._problem, self._capability, self._scalarise, self._width = problem, capability, scalarise, width
        self._names = tuple(level.name for level in levels)
        self._costs = np.array([level.cost for level in levels], dtype=float)
        self._rng = np.random.default_rng(seed)
        self._surrogate = surrogate or CoKriging(seed=seed)
        self._candidates = min(256 * width, 4096) if candidates is None else candidates
        self._polish, self._max = polish, max_evaluations
        design = maximin_latin_hypercube(counts[0], width, self._rng)
        self._design = [(row, level) for level, n in enumerate(counts) for row in design[:n]]
        self._asked_design = 0
        self._pending: dict[str, tuple[np.ndarray, int]] = {}
        self._seen: set[str] = set()
        self._told = 0
        self._x: list[np.ndarray] = []
        self._level: list[int] = []
        self._results: list[Result] = []

    def ask(self, capacity: int) -> Sequence[Action]:
        room = capacity if self._max is None else min(capacity, self._max - self._told - len(self._pending))
        if self.done or room <= 0:
            return []
        if self._asked_design < len(self._design):
            picks = self._design[self._asked_design : self._asked_design + room]
            self._asked_design += len(picks)
        elif self._pending:
            return []
        else:
            picks = [self._pick()]
        space, actions = self._problem.space, []
        for row, level in picks:
            action = Action.of(self._problem, self._capability, space.decode(row), fidelity=self._names[level])
            self._pending[action.identity.key] = (space.encode(action.design), level)
            actions.append(action)
        return actions

    def tell(self, results: Sequence[Result]) -> None:
        for result in results:
            key = result.action.identity.key
            if key not in self._pending:
                if key in self._seen:
                    continue
                refuse('a result was told that this solver did not ask for; tell each result to its own solver')
            x, level = self._pending.pop(key)
            self._seen.add(key)
            self._told += 1
            if result.cell.status is Status.OK:
                self._x.append(x)
                self._level.append(level)
                self._results.append(result)

    @property
    def done(self) -> bool:
        return self._max is not None and self._told >= self._max

    @property
    def best(self) -> Result | None:
        """The told OK truth-level result of lowest score -- an observation at the truth (B0-08)."""
        truth = [r for r, level in zip(self._results, self._level, strict=True) if level == len(self._names) - 1]
        if not truth:
            return None
        return truth[int(np.argmin(self._scores(truth)))]

    def _scores(self, results: list[Result]) -> np.ndarray:
        return score_rows(self._problem, self._scalarise, results)[0][:, 0]

    def _pick(self) -> tuple[np.ndarray, int]:
        x, levels = np.array(self._x), np.array(self._level)
        present = sorted(set(levels.tolist()))
        if present != list(range(len(self._names))):
            refuse(f'every fidelity level needs a told value to model; levels {present} have one')
        fitted = self._surrogate.fit(x, self._scores(self._results)[:, None], levels)
        top = len(self._names) - 1
        p = fitted.predict(x, variance=True)
        best = float(np.min(p.mean[:, 0] + np.sqrt(p.variance[:, 0])))
        chains = [FittedCoKriging(fitted.base, fitted.steps[:level]) for level in range(top + 1)]
        noise = [fitted.base.noise[0], *(level[0].discrepancy.noise[0] for level in fitted.steps)]
        rho = np.abs(fitted.scales[:, 0]) if top else np.zeros(0)

        def augmented(level: int):
            def acquisition(points: np.ndarray) -> np.ndarray:
                truth = fitted.predict(points, variance=True)
                var_top = np.maximum(truth.variance[:, 0], 1e-300)
                var_l = chains[level].predict(points, variance=True).variance[:, 0]
                corr = np.clip(np.prod(rho[level:]) * np.sqrt(var_l / var_top), 0.0, 1.0)
                damping = 1.0 - np.sqrt(noise[level]) / np.sqrt(var_l + noise[level])
                ei = expected_improvement(truth.mean[:, 0], np.sqrt(var_top), best)
                return ei * corr * damping * self._costs[top] / self._costs[level]

            return acquisition

        options = []
        for level in range(top + 1):
            acquisition = augmented(level)
            point = maximise(acquisition, self._width, rng=self._rng, candidates=self._candidates, polish=self._polish)
            options.append((float(acquisition(point[None])[0]), -level, point))
        _score, negative_level, point = max(options, key=lambda option: (option[0], option[1]))
        return point, -negative_level


def mfbo(problem: Problem, capability: Capability, **settings: object) -> MultiFidelityBO:
    """Multi-fidelity BO; ``**settings`` are :class:`MultiFidelityBO`'s."""
    return build(MultiFidelityBO, problem, capability, 'mfbo', **settings)
