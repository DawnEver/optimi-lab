"""D61-D64 -- Bayesian optimisation: Gaussian processes, an acquisition maximised on them, fantasies for the pending.

THE LOOP. ``initial`` points of a maximin Latin hypercube (unit cube) are asked first; after that,
each pick refits the models on every OK told point -- objectives in the MINIMISED frame, each
:class:`~optimi_lab.problem.BlackBox` constraint as its signed scaled margin (``<= 0`` satisfied) --
and maximises an acquisition (:mod:`optimi_lab.operator.acquisition`) with :func:`maximise`:
``candidates`` scrambled Sobol points scored, the best ``polish`` of them polished by numerics-lab's
L-BFGS (``minimize_lbfgs``) in logit coordinates of the box, on central differences of the
acquisition -- the MODEL is differenced, never the oracle.

FANTASIES (D64). A point asked and not yet told is PENDING. Each pick reads the models CONDITIONED
on the pending points at their own posterior mean -- the Kriging believer (Ginsbourger, Le Riche and
Carraro, 2010) -- at the fitted hyperparameters, never a refit
(:meth:`~optimi_lab.model.FittedGaussianProcess.condition`). A batch of ``batch`` is built one pick
at a time the same way, so a synchronous batch and an asynchronous free slot are one mechanism. NEHVI
instead carries the pending points in its BASELINE, drawn jointly (Daulton et al., 2021).

OBSERVED, NEVER PREDICTED (B0-08). :attr:`BayesianOptimisation.best` and :attr:`ModelBased.front`
read only told OK results: a prediction steers an ask and is never reported.

* :func:`bo` (D61) -- one score (``scalarise`` for several): ``log_ei`` (default), ``ei``, ``pi``,
  ``log_pi``, ``ucb`` (``beta`` default Srinivas et al.'s ``2 log(d t^2 pi^2 / 6 delta)``, delta 0.1),
  ``kg`` (knowledge gradient over the training points and the candidate, Scott et al.'s KGCP). With
  constraints the EI/PI family is multiplied by the probability of feasibility (constrained EI),
  and while no told point is feasible the acquisition IS that probability; UCB and KG take none.
* :func:`parego` (D62) -- Knowles (2006): every pick draws a Das-Dennis weight, scores the told
  rows by :func:`~optimi_lab.operator.acquisition.parego_scores`, fits one GP to the scores and
  maximises its log EI.
* :func:`ehvi` / :func:`nehvi` (D63) -- one GP per objective; exact EHVI over the told feasible
  front plus the believed pending, or NEHVI over ``samples`` joint draws. ``reference`` (minimised
  frame) defaults to the told maximum plus a tenth of the told range, re-read at each pick.

A told point that is not OK leaves the pending set and the data alike. A variable must be
unconditional and every categorical embedded (the model needs a metric); a constraint other than
BlackBox is refused, naming the ``penalise`` transform.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Final

import numpy as np
from lab_commons.outcomes import Status
from numerics_lab.minimize import minimize_lbfgs

from optimi_lab.core import non_dominated_sorting, refuse
from optimi_lab.core.qmc import maximin_latin_hypercube, sobol
from optimi_lab.model import FittedGaussianProcess, GaussianProcess
from optimi_lab.model.likelihood import Box, factor
from optimi_lab.operator import (
    das_dennis,
    expected_hypervolume_improvement,
    expected_improvement,
    hypervolume_improvement,
    knowledge_gradient,
    log_expected_improvement,
    log_probability_of_feasibility,
    log_probability_of_improvement,
    parego_scores,
    probability_of_improvement,
    upper_confidence_bound,
)
from optimi_lab.oracle import Action, Capability, Result
from optimi_lab.problem import BlackBox, Categorical, Problem, Relation
from optimi_lab.solver.population import Scalarise, build, require_one_score

__all__ = [
    'ACQUISITIONS',
    'Acquisition',
    'BayesianOptimisation',
    'HypervolumeImprovement',
    'ModelBased',
    'ParEGO',
    'bo',
    'ehvi',
    'margins',
    'maximise',
    'nehvi',
    'parego',
    'require_modelable',
]

#: Unit-cube rows ``(n, d)`` -> one score per row ``(n,)``, larger is better.
Acquisition = Callable[[np.ndarray], np.ndarray]

#: The single-score acquisitions :func:`bo` takes.
ACQUISITIONS: Final = ('log_ei', 'ei', 'pi', 'log_pi', 'ucb', 'kg')
_CONSTRAINED: Final = frozenset({'log_ei', 'ei', 'pi', 'log_pi'})

_STEP: Final = 1e-6
"""Central-difference step of the acquisition, in logit coordinates of the box."""
_FLAT: Final = 1e-150
"""Acquisition units per logit unit: a slope below it squares to zero, and the search would see no descent."""


def margins(problem: Problem, result: Result) -> np.ndarray:
    """Each BlackBox constraint's signed scaled margin at an OK *result*: ``<= 0`` is satisfied."""
    rows = []
    for c in problem.constraints:
        value = float(result.cell.values[c.output])
        rows.append((value - c.bound if c.relation is Relation.AT_MOST else c.bound - value) / c.scale)
    return np.array(rows)


def maximise(
    acquisition: Acquisition,
    width: int,
    *,
    rng: np.random.Generator,
    candidates: int,
    polish: int,
    lo: np.ndarray | None = None,
    hi: np.ndarray | None = None,
) -> np.ndarray:
    """The best point found for *acquisition* inside ``[lo, hi]`` (default the unit cube): Sobol, then L-BFGS."""
    lo = np.zeros(width) if lo is None else np.asarray(lo, dtype=float)
    hi = np.ones(width) if hi is None else np.asarray(hi, dtype=float)
    points = lo + (hi - lo) * sobol(candidates, width, rng=rng)
    scores = np.nan_to_num(np.asarray(acquisition(points), dtype=float), nan=-np.inf)
    order = np.argsort(-scores, kind='stable')
    best, best_score = points[order[0]], scores[order[0]]
    box, shift = Box(lo, hi), np.eye(width) * _STEP

    def negative(u: np.ndarray) -> tuple[float, np.ndarray]:
        values = acquisition(box.theta(np.vstack([u, u + shift, u - shift])))
        if not np.all(np.isfinite(values)):
            return np.inf, np.zeros(width)
        return -float(values[0]), -(values[1 : width + 1] - values[width + 1 :]) / (2 * _STEP)

    for start in order[:polish]:
        if not np.isfinite(scores[start]):
            break
        x = box.theta(minimize_lbfgs(negative, box.u(points[start]), rtol=1e-6, atol=_FLAT, max_iter=50).x)
        score = float(acquisition(x[None])[0])
        if score > best_score:
            best, best_score = x, score
    return best


def require_modelable(problem: Problem, *, constraints: bool) -> None:
    """Refuse what a model over the unit cube cannot read: a conditional or unembedded variable, a foreign constraint.

    With ``constraints=False`` every constraint is refused, naming the ``penalise`` transform.
    """
    conditional = [v.name for v in problem.space if v.active_if is not None]
    unembedded = [v.name for v in problem.space if isinstance(v, Categorical) and v.embedding is None]
    if conditional or unembedded:
        refuse(f'a model needs every variable active and metric; {conditional + unembedded} are not (embed first)')
    foreign = [c.name for c in problem.constraints if not (constraints and isinstance(c, BlackBox))]
    if foreign:
        kind = 'BlackBox constraints only' if constraints else 'no constraint'
        refuse(f'this solver models {kind}; apply optimi_lab.oracle.penalise to {foreign}')


class ModelBased(ABC):
    """The ask/tell shell every Gaussian-process solver shares: initial design, pending set, data, fantasies."""

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        initial: int | None = None,
        surrogate: GaussianProcess | None = None,
        batch: int = 1,
        candidates: int | None = None,
        polish: int = 3,
        max_evaluations: int | None = None,
        seed: int = 0,
    ) -> None:
        require_modelable(problem, constraints=True)
        if surrogate is not None and not isinstance(surrogate, GaussianProcess):
            refuse(f'the acquisitions read a joint posterior and fantasies; pass a GaussianProcess, got {surrogate!r}')
        width = problem.space.width
        initial = 2 * (width + 1) if initial is None else initial
        if initial < 2 or batch < 1 or polish < 0:
            refuse(f'need initial >= 2, batch >= 1 and polish >= 0; got {initial}, {batch} and {polish}')
        self._problem, self._capability, self._width = problem, capability, width
        self._rng = np.random.default_rng(seed)
        self._surrogate = surrogate or GaussianProcess(restarts=2, seed=seed)
        self._batch, self._polish, self._max = batch, polish, max_evaluations
        self._candidates = min(256 * width, 4096) if candidates is None else candidates
        self._design = maximin_latin_hypercube(initial, width, self._rng)
        self._asked_design = 0
        self._pending: dict[str, np.ndarray] = {}
        self._seen: set[str] = set()
        self._told = 0
        self._x: list[np.ndarray] = []
        self._y: list[np.ndarray] = []
        self._g: list[np.ndarray] = []
        self._results: list[Result] = []
        self._cache: dict[str, tuple[int, FittedGaussianProcess]] = {}

    # -- the protocol ------------------------------------------------------------------------------

    def ask(self, capacity: int) -> Sequence[Action]:
        room = capacity if self._max is None else min(capacity, self._max - self._told - len(self._pending))
        if self.done or room <= 0:
            return []
        if self._asked_design < len(self._design):
            rows = self._design[self._asked_design : self._asked_design + room]
            self._asked_design += len(rows)
        elif not self._x:
            if self._pending:
                return []
            refuse(f'none of the {self._told} initial point(s) produced a value; there is nothing to model')
        else:
            rows = self._propose(min(room, self._batch))
        space = self._problem.space
        actions = [Action.of(self._problem, self._capability, space.decode(row)) for row in rows]
        for action in actions:
            self._pending[action.identity.key] = space.encode(action.design)
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
                self._x.append(x)
                self._y.append(np.array([o.minimized(result.cell.values[o.name]) for o in self._problem.objectives]))
                self._g.append(margins(self._problem, result))
                self._results.append(result)

    @property
    def done(self) -> bool:
        return self._max is not None and self._told >= self._max

    @property
    def front(self) -> list[Result]:
        """The told OK feasible results no other such result dominates -- observations only (B0-08)."""
        feasible = [i for i in range(len(self._results)) if self._feasible[i]]
        if not feasible:
            return []
        keep = non_dominated_sorting(self._objectives[feasible])
        return [self._results[feasible[i]] for i in np.atleast_1d(keep)]

    # -- what a pick reads ---------------------------------------------------------------------------

    @property
    def _inputs(self) -> np.ndarray:
        return np.array(self._x).reshape(-1, self._width)

    @property
    def _objectives(self) -> np.ndarray:
        return np.array(self._y).reshape(-1, len(self._problem.objectives))

    @property
    def _feasible(self) -> np.ndarray:
        return np.all(np.array(self._g).reshape(len(self._x), len(self._problem.constraints)) <= 0.0, axis=1)

    def _fit(self, name: str, targets: np.ndarray) -> FittedGaussianProcess:
        """The surrogate on the told inputs -> *targets*, refitted only when the data has grown."""
        cached = self._cache.get(name)
        if cached is None or cached[0] != len(self._x):
            cached = (len(self._x), self._surrogate.fit(self._inputs, targets))
            self._cache[name] = cached
        return cached[1]

    @staticmethod
    def _believed(fitted: FittedGaussianProcess, rows: np.ndarray) -> FittedGaussianProcess:
        """*fitted* conditioned on *rows* at its own posterior mean: the Kriging believer."""
        return fitted.condition(rows, fitted.predict(rows).mean) if len(rows) else fitted

    def _log_feasibility(self, rows: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """``log P(every margin <= 0)`` at points, from the believed constraint models (0 without constraints)."""
        if not self._problem.constraints:
            return lambda points: np.zeros(len(points))
        model = self._believed(self._fit('margins', np.array(self._g)), rows)

        def log_pf(points: np.ndarray) -> np.ndarray:
            p = model.predict(points, variance=True)
            return log_probability_of_feasibility(p.mean, np.sqrt(p.variance))

        return log_pf

    def _maximise(self, acquisition: Acquisition) -> np.ndarray:
        return maximise(acquisition, self._width, rng=self._rng, candidates=self._candidates, polish=self._polish)

    def _propose(self, count: int) -> np.ndarray:
        """*count* picks, each conditioned on the pending points and on the picks before it."""
        picks: list[np.ndarray] = []
        for _ in range(count):
            rows = np.array([*self._pending.values(), *picks]).reshape(-1, self._width)
            picks.append(self._pick(rows))
        return np.array(picks)

    @abstractmethod
    def _pick(self, pending: np.ndarray) -> np.ndarray:
        """The next unit-cube point, given the *pending* rows ``(p, d)`` it must not repeat."""


class BayesianOptimisation(ModelBased):
    """D61: one score, one acquisition from :data:`ACQUISITIONS`."""

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        acquisition: str = 'log_ei',
        beta: float | None = None,
        scalarise: Scalarise | None = None,
        **base: object,
    ) -> None:
        require_one_score(problem, scalarise, 'bo')
        if acquisition not in ACQUISITIONS:
            refuse(f'acquisition is one of {list(ACQUISITIONS)}, got {acquisition!r}')
        if problem.constraints and acquisition not in _CONSTRAINED:
            refuse(f'{acquisition} takes no constraint; the constrained acquisitions are {sorted(_CONSTRAINED)}')
        super().__init__(problem, capability, **base)
        self._acquisition, self._beta, self._scalarise = acquisition, beta, scalarise

    def _scores(self) -> np.ndarray:
        values = self._objectives
        return (values if self._scalarise is None else np.asarray(self._scalarise(values), dtype=float)).reshape(-1)

    @property
    def best(self) -> Result | None:
        """The told OK feasible result of lowest score -- an observation, never a prediction (B0-08)."""
        if not self._results:
            return None
        scores = np.where(self._feasible, self._scores(), np.inf)
        return self._results[int(np.argmin(scores))] if np.isfinite(scores.min()) else None

    def _pick(self, pending: np.ndarray) -> np.ndarray:
        model = self._believed(self._fit('score', self._scores()[:, None]), pending)
        log_pf = self._log_feasibility(pending)
        feasible = self._scores()[self._feasible]
        if not len(feasible):
            return self._maximise(log_pf)
        best = float(feasible.min())
        if self._acquisition == 'kg':
            return self._maximise(lambda points: self._knowledge_gradient(model, points))

        def acquisition(points: np.ndarray) -> np.ndarray:
            p = model.predict(points, variance=True)
            mean, std = p.mean[:, 0], np.sqrt(p.variance[:, 0])
            if self._acquisition == 'ucb':
                return upper_confidence_bound(mean, std, self._beta_t())
            log_pf_here = log_pf(points)
            if self._acquisition == 'log_ei':
                return log_expected_improvement(mean, std, best) + log_pf_here
            if self._acquisition == 'log_pi':
                return log_probability_of_improvement(mean, std, best) + log_pf_here
            closed = expected_improvement if self._acquisition == 'ei' else probability_of_improvement
            return closed(mean, std, best) * np.exp(log_pf_here)

        return self._maximise(acquisition)

    def _beta_t(self) -> float:
        if self._beta is not None:
            return self._beta
        t = len(self._x) + 1
        return float(2.0 * np.log(self._width * t**2 * np.pi**2 / (6 * 0.1)))

    @staticmethod
    def _knowledge_gradient(model: FittedGaussianProcess, points: np.ndarray) -> np.ndarray:
        """KGCP: the value of observing each point, over the training points and the point itself."""
        known = model.predict(model.x).mean[:, 0]
        cross = model.posterior_covariance(model.x, points)[0]
        p = model.predict(points, variance=True)
        spread = np.sqrt(p.variance[:, 0] + model.noise[0])
        return np.array([
            knowledge_gradient(-np.append(known, p.mean[j, 0]), np.append(cross[:, j], p.variance[j, 0]) / spread[j])
            for j in range(len(points))
        ])


class ParEGO(ModelBased):
    """D62: a fresh Das-Dennis weight per pick, a GP on the ParEGO scores, log EI."""

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        rho: float = 0.05,
        divisions: int | None = None,
        **base: object,
    ) -> None:
        m = len(problem.objectives)
        if m < 2:
            refuse('ParEGO scalarises several objectives; one objective is bo()')
        super().__init__(problem, capability, **base)
        self._rho = rho
        self._weights = das_dennis(m, {2: 10, 3: 4}.get(m, 3) if divisions is None else divisions)

    def _pick(self, pending: np.ndarray) -> np.ndarray:
        weight = self._weights[self._rng.integers(len(self._weights))]
        scores = parego_scores(self._objectives, weight, rho=self._rho)
        model = self._believed(self._surrogate.fit(self._inputs, scores[:, None]), pending)
        log_pf = self._log_feasibility(pending)
        if not self._feasible.any():
            return self._maximise(log_pf)
        best = float(scores[self._feasible].min())

        def acquisition(points: np.ndarray) -> np.ndarray:
            p = model.predict(points, variance=True)
            return log_expected_improvement(p.mean[:, 0], np.sqrt(p.variance[:, 0]), best) + log_pf(points)

        return self._maximise(acquisition)


class HypervolumeImprovement(ModelBased):
    """D63: one GP per objective; exact EHVI (``noisy=False``) or NEHVI over ``samples`` joint draws."""

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        noisy: bool = False,
        samples: int = 32,
        reference: Sequence[float] | None = None,
        **base: object,
    ) -> None:
        m = len(problem.objectives)
        if m < 2:
            refuse('a hypervolume acquisition needs at least two objectives; one objective is bo()')
        if reference is not None and len(reference) != m:
            refuse(f'reference is one minimised value per objective ({m}), got {len(reference)}')
        if samples < 2:
            refuse(f'NEHVI averages at least 2 joint draws, got {samples}')
        super().__init__(problem, capability, **base)
        self._noisy, self._samples = noisy, samples
        self._reference = None if reference is None else np.asarray(reference, dtype=float)

    def reference(self) -> np.ndarray:
        """The declared reference, else the told maximum plus a tenth of the told range (minimised frame)."""
        if self._reference is not None:
            return self._reference
        values = self._objectives
        span = values.max(axis=0) - values.min(axis=0)
        return values.max(axis=0) + 0.1 * np.where(span > 0, span, 1.0)

    def _pick(self, pending: np.ndarray) -> np.ndarray:
        model = self._fit('objectives', self._objectives)
        log_pf = self._log_feasibility(pending)
        if self._problem.constraints and not self._feasible.any():
            return self._maximise(log_pf)
        reference = self.reference()
        observed = self._objectives[self._feasible]
        if not self._noisy:
            believed = self._believed(model, pending)
            front = np.vstack([observed, believed.predict(pending).mean]) if len(pending) else observed

            def acquisition(points: np.ndarray) -> np.ndarray:
                p = believed.predict(points, variance=True)
                gain = expected_hypervolume_improvement(p.mean, np.sqrt(p.variance), front, reference)
                return gain * np.exp(log_pf(points))

            return self._maximise(acquisition)
        return self._maximise(self._nehvi(model, np.vstack([self._inputs[self._feasible], pending]), reference, log_pf))

    def _nehvi(
        self,
        model: FittedGaussianProcess,
        baseline: np.ndarray,
        reference: np.ndarray,
        log_pf: Callable[[np.ndarray], np.ndarray],
    ) -> Acquisition:
        """NEHVI with the baseline drawn once and each candidate drawn CONDITIONALLY on it (exactly joint)."""
        m, s = len(self._problem.objectives), self._samples
        drawn = model.draw(baseline, self._rng.standard_normal((s, len(baseline), m))) if len(baseline) else None
        own = self._rng.standard_normal((s, m))
        mean_b = model.predict(baseline).mean if len(baseline) else None
        chols = [factor(block) for block in model.posterior_covariance(baseline, baseline)] if len(baseline) else []
        if any(c is None for c in chols):
            refuse('the baseline posterior covariance cannot be factored even with jitter')

        def acquisition(points: np.ndarray) -> np.ndarray:
            p = model.predict(points, variance=True)
            draws = np.repeat(p.mean[None], s, axis=0)
            variance = p.variance.copy()
            if drawn is not None:
                cross = model.posterior_covariance(points, baseline)
                for k, chol in enumerate(chols):
                    gain = chol.solve(cross[k].T).T
                    draws[:, :, k] += (drawn[:, :, k] - mean_b[:, k]) @ gain.T
                    variance[:, k] -= np.sum(gain * cross[k], axis=1)
            draws += np.sqrt(np.maximum(variance, 0.0))[None] * own[:, None, :]
            fronts = drawn if drawn is not None else np.zeros((s, 0, m))
            gains = np.mean(
                [hypervolume_improvement(d, f, reference) for d, f in zip(draws, fronts, strict=True)], axis=0
            )
            return gains * np.exp(log_pf(points))

        return acquisition


def bo(problem: Problem, capability: Capability, **settings: object) -> BayesianOptimisation:
    """Bayesian optimisation; ``**settings`` are :class:`BayesianOptimisation`'s and :class:`ModelBased`'s."""
    return build(BayesianOptimisation, problem, capability, 'bo', **settings)


def parego(problem: Problem, capability: Capability, **settings: object) -> ParEGO:
    """ParEGO; ``**settings`` are :class:`ParEGO`'s and :class:`ModelBased`'s."""
    return build(ParEGO, problem, capability, 'parego', **settings)


def ehvi(problem: Problem, capability: Capability, **settings: object) -> HypervolumeImprovement:
    """Exact-EHVI Bayesian optimisation; ``**settings`` are :class:`HypervolumeImprovement`'s but ``noisy``."""
    return build(HypervolumeImprovement, problem, capability, 'ehvi', **settings)


def nehvi(problem: Problem, capability: Capability, **settings: object) -> HypervolumeImprovement:
    """NEHVI Bayesian optimisation; ``**settings`` are :class:`HypervolumeImprovement`'s but ``noisy``."""
    return build(HypervolumeImprovement, problem, capability, 'nehvi', noisy=True, **settings)
