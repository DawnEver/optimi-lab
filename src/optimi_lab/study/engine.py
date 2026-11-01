"""The study engine: a bounded, routed population search, and the evaluation of named points.

Both run over a :class:`~optimi_lab.policy.ScreeningOracle` and one record. The archive owns every
measured number; :class:`Searched` is a view of it plus the run's metadata, never a second store.
An answer is filed the moment it completes, so a paid evaluation survives a later failure.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import Lock

import numpy as np
from lab_commons.outcomes import SUBMITTED, Evidence, Record

from optimi_lab.archive import Archive
from optimi_lab.core import SampleSpec
from optimi_lab.core.errors import Refusal, refuse
from optimi_lab.operator.reference import reference_select
from optimi_lab.operator.selection import crowding_select, tournament_select
from optimi_lab.oracle import Action, Oracle, Result
from optimi_lab.policy import Batches, DecidedReplay, Router, RoutingPolicy, ScreeningOracle, Steer, VariationPolicy
from optimi_lab.problem import Problem
from optimi_lab.run import Budget, Dispatch, Run, run
from optimi_lab.solver.evolutionary import nsga2, nsga3
from optimi_lab.solver.population import EvolutionarySolver
from optimi_lab.study.strategy import algorithm_factory

__all__ = ['Searched', 'evaluate', 'opened', 'search']

#: Arguments the engine owns; a solver setting may not repeat them.
_ENGINE_OWNED = frozenset({'problem', 'capability', 'spec', 'initial', 'seed'})


@dataclass(frozen=True)
class Searched:
    """The *archive* a search or evaluation filed, why a search stopped early, and its batch ends.

    *solved* / *served* count, for :func:`evaluate`, the fresh evaluations and the decided reuses.
    """

    archive: Archive
    search_refusal: str | None = None
    evidence: Evidence = Evidence.OBSERVED
    batch_ends: tuple[int, ...] = ()
    execution: Run | None = None
    solved: int = 0
    served: int = 0

    def ordered_results(self) -> tuple[Result, ...]:
        """First-submitted order, independent of completion order or screening; then the predictions."""
        observed = {result.action.identity.key: result for result in self.archive.results()}
        keys = dict.fromkeys(event.key for event in self.archive.record.replay() if event.kind == SUBMITTED)
        return (*(observed[key] for key in keys if key in observed), *self.archive.predictions())


def opened(problem: Problem, oracle: Oracle, record: Record) -> Archive:
    """The archive of *record*, refused when an action in it belongs to another problem or oracle build."""
    archive = Archive(problem, record)
    for action in (*archive.pending(), *(result.action for result in archive.results())):
        expected = Action.of(problem, oracle.capability, action.design, fidelity=action.fidelity)
        if expected.identity.key != action.identity.key:
            refuse('the record belongs to another problem or evaluator identity')
    return archive


def _publish(oracle: ScreeningOracle, actions: Sequence[Action], file: Callable[[Result], None]) -> dict[str, Result]:
    """Every action answered once through ``publish_batch``, each answer filed as it completes."""
    remaining = {action.identity.key: action for action in actions}
    answered: dict[str, Result] = {}
    lock = Lock()

    def on_result(result: Result) -> None:
        with lock:
            if remaining.pop(result.action.identity.key, None) is None:
                refuse('a publisher answered an unrequested or repeated action')
            file(result)
            answered[result.action.identity.key] = result

    oracle.publish_batch(tuple(actions), on_result)
    if remaining:
        refuse('a publisher returned before answering every requested action')
    return answered


def evaluate(
    problem: Problem,
    oracle: ScreeningOracle,
    points: np.ndarray,
    record: Record,
    *,
    evidence: Evidence = Evidence.OBSERVED,
) -> Searched:
    """Evaluate natural *points* once each; a decided result (or held prediction) costs no evaluation."""
    names = problem.space.names
    matrix = np.asarray(points, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        refuse('the points have shape (points, design axes) in natural coordinates')
    archive = opened(problem, oracle, record)
    actions = {
        action.identity.key: action
        for action in (
            Action.of(problem, oracle.capability, dict(zip(names, map(float, row), strict=True))) for row in matrix
        )
    }
    predictions = {r.action.identity.key: r for r in archive.predictions()} if evidence is Evidence.PREDICTED else {}
    fresh = [
        action
        for key, action in actions.items()
        if (known := archive.result(key) or predictions.get(key)) is None or not known.cell.status.decided
    ]
    for action in fresh:
        archive.submit(action)

    def file(result: Result) -> None:
        if result.cell.evidence is Evidence.PREDICTED and evidence is Evidence.PREDICTED:
            archive.predict(result)
        else:
            archive.tell(result)

    if fresh:
        _publish(oracle, fresh, file)
    return Searched(archive, evidence=evidence, solved=len(fresh), served=len(actions) - len(fresh))


def search(
    problem: Problem,
    oracle: ScreeningOracle,
    record: Record,
    *,
    algorithm: str | Callable = 'nsga2',
    population: int = 8,
    generations: int = 1,
    seed: int = 0,
    initial: np.ndarray | None = None,
    settings: Mapping[str, object] | None = None,
    router: Router | None = None,
    steer: Steer | None = None,
    dispatch: Dispatch | None = None,
) -> Searched:
    """A value-only population search of ``generations + 1`` told batches; *initial* is natural coordinates.

    *router* screens or routes offspring (:class:`~optimi_lab.policy.RoutingPolicy`); *steer* sets the
    SBX/PM rates per generation and needs nsga2 or nsga3; *dispatch* replaces the default, which
    publishes each batch through ``oracle.publish_batch`` and files every answer as it completes.
    """
    factory = algorithm_factory(algorithm)
    if population < 1 or generations < 1:
        refuse('a search needs population >= 1 and generations >= 1; named points use evaluate')
    opened(problem, oracle, record)  # refuse a foreign record BEFORE evaluating or mutating it
    spec: SampleSpec | None = SampleSpec(n_samples=population, seed=seed)
    unit_initial: np.ndarray | None = None
    if initial is not None:
        natural = np.asarray(initial, dtype=float)
        if natural.ndim != 2 or natural.shape[1] != len(problem.space.names) or len(natural) < 1:
            refuse('initial is a non-empty (points, design axes) natural-coordinate matrix')
        unit_initial = np.array([
            problem.space.encode(dict(zip(problem.space.names, point, strict=True))) for point in natural
        ])
        spec, population = None, len(natural)
    extra = dict(settings or {})
    if overlap := set(extra) & _ENGINE_OWNED:
        refuse(f'settings repeat engine-owned declarations {sorted(overlap)}')
    control = None
    if steer is None:
        solver = factory(problem, oracle.capability, seed=seed, spec=spec, initial=unit_initial, **extra)
    else:
        if factory not in (nsga2, nsga3):
            refuse('SBX/PM variation control requires nsga2 or nsga3; the chosen solver has other operator settings')
        control = VariationPolicy(tuple, steer, generations, extra)
        solver = EvolutionarySolver(
            problem,
            oracle.capability,
            seed=seed,
            spec=spec,
            initial=unit_initial,
            variation=control,
            selection=crowding_select if factory is nsga2 else reference_select,
            parent_selection=tournament_select,
        )
    bounded = Batches(solver, generations + 1)
    policy = RoutingPolicy(bounded, oracle, problem, record, router)
    if control is not None:
        control.read = policy.results

    async def published(_current: Oracle, actions: Sequence[Action]) -> AsyncIterator[Result]:
        archive = Archive(problem, record)
        answered = _publish(oracle, actions, archive.tell)
        for action in actions:
            yield answered[action.identity.key]

    budget = Budget(evaluations=population * (generations + 1))
    try:
        execution = run(problem, oracle, policy, DecidedReplay(record), budget, dispatch=dispatch or published)
    except Refusal as error:
        if bounded.last_batch_complete is not False:
            raise
        return Searched(opened(problem, oracle, record), str(error), batch_ends=tuple(policy.batch_ends))
    return Searched(opened(problem, oracle, record), batch_ends=tuple(policy.batch_ends), execution=execution)
