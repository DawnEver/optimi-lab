"""The shell every Gaussian-process solver of P11 shares, and the one acquisition maximiser.

THE LOOP. ``initial`` points of a maximin Latin hypercube (unit cube) are asked first; after that,
each pick refits the models on every OK told point -- objectives in the MINIMISED frame, each
:class:`~optimi_lab.problem.BlackBox` constraint as its signed scaled margin (``<= 0`` satisfied,
:func:`margins`) -- and maximises an acquisition (:mod:`optimi_lab.operator.acquisition`) with
:func:`maximise`: ``candidates`` scrambled Sobol points scored, the best ``polish`` of them polished by
numerics-lab's L-BFGS (``minimize_lbfgs``) in logit coordinates of the box, on central differences
of the acquisition -- the MODEL is differenced, never the oracle.

FANTASIES (D64). A point asked and not yet told is PENDING. Each pick reads the models CONDITIONED
on the pending points at their own posterior mean -- the Kriging believer (Ginsbourger, Le Riche and
Carraro, 2010) -- at the fitted hyperparameters, never a refit
(:meth:`~optimi_lab.model.FittedGaussianProcess.condition`). A batch of ``batch`` is built one pick
at a time the same way, so a synchronous batch and an asynchronous free slot are one mechanism.

OBSERVED, NEVER PREDICTED (B0-08). :attr:`ModelBased.front` (and each solver's ``best``) reads only
told OK results: a prediction steers an ask and is never reported.

A told point that is not OK leaves the pending set and the data alike. :func:`require_modelable` is
the one readiness check: a space with conditional variables or unembedded categoricals is modelled
through the declared ``embedding`` (:class:`optimi_lab.problem.Embedding`) and refused without one;
a Chance constraint is refused (a model estimates no probability); any other constraint than
BlackBox is refused, naming the ``penalise`` transform.

ROBUST (a problem declaring :class:`~optimi_lab.problem.Uncertainty`). ``scenarios`` draws are taken
once from the seed and shared by every design (:class:`optimi_lab.solver.scenarios.Scenarios`); a
pick is asked once per scenario, and only when every scenario of it is told OK does it enter the
data -- as ONE row whose objectives and constraint margins are each the declared ``risk`` over its
scenarios. So the model is fitted on the risk estimate, the acquisition reads the risk at its
predicted mean, and :attr:`ModelBased.front` reports observed risk rows. A design with any scenario
not OK is dropped whole.
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
from optimi_lab.model.likelihood import Box
from optimi_lab.operator import Risk, log_probability_of_feasibility
from optimi_lab.oracle import Action, Capability, Result
from optimi_lab.problem import BlackBox, Chance, Embedding, Problem, Relation
from optimi_lab.solver.scenarios import Scenarios

__all__ = ['Acquisition', 'ModelBased', 'margins', 'maximise', 'refuse_uncertainty', 'require_modelable']

#: Unit-cube rows ``(n, d)`` -> one score per row ``(n,)``, larger is better.
Acquisition = Callable[[np.ndarray], np.ndarray]

_STEP: Final = 1e-6
"""Central-difference step of the acquisition, in logit coordinates of the box."""


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
        x = box.theta(minimize_lbfgs(negative, box.u(points[start]), rtol=1e-6, max_iter=50).x)
        score = float(acquisition(x[None])[0])
        if score > best_score:
            best, best_score = x, score
    return best


def require_modelable(problem: Problem, *, constraints: bool, embedding: Embedding | None = None) -> Embedding:
    """The embedding a model reads *problem* through, refusing what it cannot read.

    A conditional or unembedded variable needs a declared *embedding*; a Chance constraint is
    refused (a model estimates no probability); with ``constraints=False`` every constraint is
    refused, naming the ``penalise`` transform.
    """
    embedding = Embedding.declared(problem.space, embedding)
    chance = [c.name for c in problem.constraints if isinstance(c, Chance)]
    if chance:
        refuse(
            f'a model estimates no probability, so Chance constraints {chance} cannot be modelled; '
            f'a population solver estimates them over its scenarios'
        )
    foreign = [c.name for c in problem.constraints if not (constraints and isinstance(c, BlackBox))]
    if foreign:
        kind = 'BlackBox constraints only' if constraints else 'no constraint'
        refuse(f'this solver models {kind}; apply optimi_lab.oracle.penalise to {foreign}')
    return embedding


def refuse_uncertainty(problem: Problem, what: str, why: str) -> None:
    """Refuse a problem that declares uncertainty for a solver with no risk estimator, saying *why*."""
    if problem.uncertainty is not None:
        refuse(f'{what} has no risk estimator over the problem uncertainty: {why}')


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
        risk: Risk | None = None,
        scenarios: int | None = None,
        embedding: Embedding | None = None,
    ) -> None:
        self._embedding = require_modelable(problem, constraints=True, embedding=embedding)
        if surrogate is not None and not isinstance(surrogate, GaussianProcess):
            refuse(f'the acquisitions read a joint posterior and fantasies; pass a GaussianProcess, got {surrogate!r}')
        width = self._embedding.width
        initial = 2 * (width + 1) if initial is None else initial
        if initial < 2 or batch < 1 or polish < 0:
            refuse(f'need initial >= 2, batch >= 1 and polish >= 0; got {initial}, {batch} and {polish}')
        self._problem, self._capability, self._width = problem, capability, width
        self._rng = np.random.default_rng(seed)
        self._robust = Scenarios.of(problem, risk, scenarios, self._rng)
        self._per = 1 if self._robust is None else len(self._robust.scenarios)
        self._surrogate = surrogate or GaussianProcess(restarts=2, seed=seed)
        self._batch, self._polish, self._max = batch, polish, max_evaluations
        self._candidates = min(256 * width, 4096) if candidates is None else candidates
        self._design = maximin_latin_hypercube(initial, width, self._rng)
        self._asked_design = 0
        self._pending: dict[str, tuple[int, np.ndarray]] = {}
        self._groups: dict[int, list[Result]] = {}
        self._picks = 0
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
        room //= self._per  # in designs: each is asked once per scenario
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
        embedding = self._embedding
        draws = (None,) if self._robust is None else self._robust.scenarios
        actions: list[Action] = []
        for row in rows:
            design = embedding.decode(row)
            group, self._picks = self._picks, self._picks + 1
            self._groups[group] = []
            for scenario in draws:
                action = Action.of(self._problem, self._capability, design, scenario=scenario)
                self._pending[action.identity.key] = (group, embedding.encode(action.design))
                actions.append(action)
        return actions

    def tell(self, results: Sequence[Result]) -> None:
        for result in results:
            key = result.action.identity.key
            if key not in self._pending:
                if key in self._seen:
                    continue
                refuse('a result was told that this solver did not ask for; tell each result to its own solver')
            group, x = self._pending.pop(key)
            self._seen.add(key)
            self._told += 1
            told = self._groups[group]
            told.append(result)
            if len(told) == self._per:
                del self._groups[group]
                self._admit(x, told)

    def _admit(self, x: np.ndarray, told: list[Result]) -> None:
        """One design's results (one per scenario) as one data row -- the risk of each column -- if all are OK."""
        if any(r.cell.status is not Status.OK for r in told):
            return
        objectives = self._problem.objectives
        y = np.array([[o.minimized(r.cell.values[o.name]) for o in objectives] for r in told])
        g = np.array([margins(self._problem, r) for r in told]).reshape(len(told), -1)
        if self._robust is not None:
            risk = self._robust.risk
            y = np.array([risk.of(column) for column in y.T]).reshape(1, -1)
            g = np.array([risk.of(column) for column in g.T]).reshape(1, -1)
        self._x.append(x)
        self._y.append(y[0])
        self._g.append(g[0])
        self._results.append(told[0])

    @property
    def done(self) -> bool:
        return self._max is not None and self._max - self._told < self._per and not self._pending

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
            pending = dict(self._pending.values())  # one row per design, not one per scenario
            rows = np.array([*pending.values(), *picks]).reshape(-1, self._width)
            picks.append(self._pick(rows))
        return np.array(picks)

    @abstractmethod
    def _pick(self, pending: np.ndarray) -> np.ndarray:
        """The next unit-cube point, given the *pending* rows ``(p, d)`` it must not repeat."""
