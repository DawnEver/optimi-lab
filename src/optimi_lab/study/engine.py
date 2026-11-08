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
from optimi_lab.operator import Risk
from optimi_lab.operator.reference import reference_select
from optimi_lab.operator.selection import crowding_select, tournament_select
from optimi_lab.oracle import Action, Capability, Oracle, Result
from optimi_lab.policy import (
    Batches,
    DecidedReplay,
    RouteRule,
    RoutingPolicy,
    ScreeningOracle,
    VariationPolicy,
    VariationRule,
)
from optimi_lab.problem import Problem
from optimi_lab.run import Budget, Dispatch, Run, run
from optimi_lab.solver import (
    SEEDING,
    SEQUENTIAL,
    Ineligible,
    NeedsTransform,
    Requires,
    Seeding,
    Solver,
    match,
    search_embedding,
)
from optimi_lab.solver.evolutionary import nsga2, nsga3
from optimi_lab.solver.population import EvolutionarySolver
from optimi_lab.solver.scenarios import DEFAULT_SCENARIOS
from optimi_lab.study.strategy import algorithm_factory

__all__ = ['Searched', 'build_solver', 'evaluate', 'opened', 'search']

#: Arguments the engine owns; a solver setting may not repeat them.
_ENGINE_OWNED = frozenset({
    'problem',
    'capability',
    'spec',
    'initial',
    'seed',
    'embedding',
    'population',
    'start',
    'starts',
    'max_evaluations',
    'generations',
})
#: Solvers whose schedule adapts over the whole run, so they take the search's ``generations``.
_TAKES_GENERATIONS = frozenset({'rvea'})


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
    """The archive of *record*, refused when an action in it belongs to another problem or oracle build.

    Each action is rebuilt with everything it asked -- information, replicate, direction -- so a
    slope-reading run reopens; only the problem and the oracle build may refuse it.
    """
    archive = Archive(problem, record)
    for action in (*archive.pending(), *(result.action for result in archive.results())):
        expected = Action.of(
            problem,
            oracle.capability,
            action.design,
            fidelity=action.fidelity,
            want=action.want,
            replicate=action.replicate,
            direction=action.direction,
            scenario=action.scenario,
        )
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


def _robust(problem: Problem, risk: Risk | None, scenarios: int | None) -> dict[str, object]:
    """The solver settings a robust declaration adds; none for a deterministic problem that declares none."""
    if problem.uncertainty is None:
        if risk is not None or scenarios is not None:
            refuse('risk and scenarios estimate over the problem uncertainty, and the problem declares none')
        return {}
    return {'risk': risk, 'scenarios': scenarios}


def _draws(problem: Problem, scenarios: int | None) -> int:
    """Evaluations ONE design costs: one per shared scenario under declared uncertainty, else one."""
    if problem.uncertainty is None:
        return 1
    return DEFAULT_SCENARIOS if scenarios is None else scenarios


def _refuse_solves(solves: int | None) -> None:
    if solves is not None and solves < 1:
        refuse(f'solves={solves!r} is a true-solve budget of at least 1')


def _refuse_choosers(algorithm: str | Callable, oracle: ScreeningOracle, router: RouteRule | None) -> None:
    """Exactly ONE chooser picks each point's evaluator leg: a *router*, or a solver that picks its own level.

    A ``Seeding.LEVELS`` solver asks every action at a level of ``problem.fidelity``; a router would overwrite
    that pick, so the two together are refused. An oracle of several legs with neither would answer every
    action on leg 0, so the cheaper legs it declares would be a declaration nothing reads.
    """
    levels = sorted(name for name, row in SEEDING.items() if row is Seeding.LEVELS)
    picks_level = isinstance(algorithm, str) and algorithm in levels
    if picks_level and router is not None:
        refuse(f'{algorithm} picks its own level for every point; a router choosing the leg too is a second chooser')
    if not picks_level and router is None and getattr(oracle, 'evaluators', 1) > 1:
        refuse(
            f'{oracle.evaluators} evaluator legs need an explicit selection policy: a router, or a solver that picks '
            f'its own level ({levels})'
        )


def build_solver(
    name: str | Callable,
    problem: Problem,
    capability: Capability,
    *,
    seed: int,
    population: int,
    generations: int,
    initial: np.ndarray | None = None,
    settings: Mapping[str, object] | None = None,
    risk: Risk | None = None,
    scenarios: int | None = None,
) -> Solver:
    """Solver *name* over *problem*, matched by the library and seeded by its ``solver.SEEDING`` row.

    *initial* is unit-cube rows: a POPULATION solver's first generation (None draws *population* by
    LHS); a START or SLOPES solver's ONE start (None: the cube's centre); a solver that draws its own
    first batch is refused rows and draws *population* itself. A name is refused by its own
    ``Requires`` (``solver.requires.match``); the one transform applied is a conditional space's
    embedding. A bare callable is a population solver, called unmatched.

    *risk* and *scenarios* are the robust declaration over ``problem.uncertainty``: the solver asks each
    design once per shared scenario and ranks it on *risk* over them (``solver.scenarios``); declaring
    either on a problem with no uncertainty is refused.
    """
    factory = algorithm_factory(name)
    extra = dict(settings or {})
    if overlap := set(extra) & _ENGINE_OWNED:
        refuse(f'settings repeat engine-owned declarations {sorted(overlap)}')
    extra.update(_robust(problem, risk, scenarios))
    spec = SampleSpec(n_samples=population, seed=seed) if initial is None else None
    if callable(name):
        return factory(problem, capability, seed=seed, spec=spec, initial=initial, **extra)
    verdict = match(Requires.of(name), problem, capability)
    if isinstance(verdict, Ineligible):
        refuse(f'{name} cannot run this problem: {verdict.reason}')
    if isinstance(verdict, NeedsTransform) and set(verdict.transforms) - {'embed'}:
        refuse(f'{name} runs this problem only after {list(verdict.transforms)}, which a study does not apply')
    if (embedding := search_embedding(problem)) is not None:
        extra['embedding'] = embedding
    if name in _TAKES_GENERATIONS:
        extra['generations'] = generations
    if SEEDING[name] is Seeding.POPULATION:
        seeded: dict[str, object] = {'spec': spec, 'initial': initial}
    else:
        seeded = _seeded(name, extra, problem=problem, population=population, generations=generations, initial=initial)
    return factory(problem, capability, seed=seed, **seeded, **extra)


def _seeded(
    name: str,
    extra: dict[str, object],
    *,
    problem: Problem,
    population: int,
    generations: int,
    initial: np.ndarray | None,
) -> dict[str, object]:
    """The first-batch and length settings *name*'s seeding hands its factory (*extra* loses what they consume)."""
    seeding = SEEDING[name]
    evaluations = population * (generations + 1)
    if seeding in (Seeding.START, Seeding.SLOPES) and initial is not None and len(initial) != 1:
        refuse(f'{name} walks from ONE start point; it was handed {len(initial)} rows')
    if seeding is Seeding.SLOPES:
        return {} if initial is None else {'start': problem.space.decode(initial[0])}
    if seeding is Seeding.START:
        return {'max_evaluations': evaluations, **({} if initial is None else {'start': initial[0]})}
    if initial is not None:
        refuse(f'{name} draws its own first batch ({seeding.value}); it was handed {len(initial)} rows it would ignore')
    if seeding in (Seeding.STARTS, Seeding.SCAN):
        local = extra.pop('local', None)
        starts = sorted(key for key, row in SEEDING.items() if row is Seeding.START)
        if local not in starts:
            refuse(f'{name} chains a local search: settings.local must name one of {starts}, got {local!r}')
        chained = algorithm_factory(local)
        if seeding is Seeding.STARTS:
            return {'local': chained, 'starts': population, 'max_evaluations': generations + 1}
        if 'weights' not in extra:
            refuse(f'{name} needs settings.weights, one weight row per scalarisation')
        weights = np.atleast_2d(np.asarray(extra.pop('weights'), dtype=float))
        return {'local': chained, 'weights': weights, 'max_evaluations': max(1, evaluations // len(weights))}
    if name == 'mfbo' and problem.fidelity is not None and len(problem.fidelity.levels) >= 2:
        # The nested design: *population* points at the truth, each cheaper level three times the one above.
        top = len(problem.fidelity.levels) - 1
        return {'initial': tuple(population * 3 ** (top - level) for level in range(top + 1))}
    return {Seeding.MODEL: {'initial': population}, Seeding.SELF: {'population': population}}.get(seeding, {})


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
    router: RouteRule | None = None,
    variation: VariationRule | None = None,
    dispatch: Dispatch | None = None,
    solves: int | None = None,
    risk: Risk | None = None,
    scenarios: int | None = None,
) -> Searched:
    """A search by *algorithm* of ``generations + 1`` told batches; *initial* is natural coordinates.

    *algorithm* names any seeded library solver, matched and seeded by :func:`build_solver`, or is a
    population-solver callable; a SEQUENTIAL solver's run is bounded by ``population * (generations + 1)``
    evaluations rather than batches.

    *router* screens or routes offspring (:class:`~optimi_lab.policy.RoutingPolicy`); *variation* sets the
    SBX/PM rates per generation and needs nsga2 or nsga3; *dispatch* replaces the default, which
    publishes each batch through ``oracle.publish_batch`` and files every answer as it completes.
    *solves* is a TRUE-SOLVE budget: the search stops once that many points were evaluated, however many
    generations it takes, so a screening router pays the same solves as a run that screens nothing;
    None keeps the ``generations + 1`` batch stop.
    *risk* and *scenarios* are the robust declaration (:func:`build_solver`); each scenario of a design is
    one evaluation, so the generations stop counts ``population * scenarios`` per batch.
    """
    factory = algorithm_factory(algorithm)
    if population < 1 or generations < 1:
        refuse('a search needs population >= 1 and generations >= 1; named points use evaluate')
    _refuse_solves(solves)
    _refuse_choosers(algorithm, oracle, router)
    opened(problem, oracle, record)  # refuse a foreign record BEFORE evaluating or mutating it
    unit_initial: np.ndarray | None = None
    if initial is not None:
        natural = np.asarray(initial, dtype=float)
        if natural.ndim != 2 or natural.shape[1] != len(problem.space.names) or len(natural) < 1:
            refuse('initial is a non-empty (points, design axes) natural-coordinate matrix')
        unit_initial = np.array([
            problem.space.encode(dict(zip(problem.space.names, point, strict=True))) for point in natural
        ])
        population = len(natural)
    control = None
    if variation is None:
        solver = build_solver(
            algorithm,
            problem,
            oracle.capability,
            seed=seed,
            population=population,
            generations=generations,
            initial=unit_initial,
            settings=settings,
            risk=risk,
            scenarios=scenarios,
        )
    else:
        if factory not in (nsga2, nsga3):
            refuse('SBX/PM variation control requires nsga2 or nsga3; the chosen solver has other operator settings')
        extra = dict(settings or {})
        if overlap := set(extra) & _ENGINE_OWNED:
            refuse(f'settings repeat engine-owned declarations {sorted(overlap)}')
        control = VariationPolicy(tuple, variation, generations, extra)
        solver = EvolutionarySolver(
            problem,
            oracle.capability,
            seed=seed,
            spec=SampleSpec(n_samples=population, seed=seed) if unit_initial is None else None,
            initial=unit_initial,
            variation=control,
            selection=crowding_select if factory is nsga2 else reference_select,
            parent_selection=tournament_select,
            **_robust(problem, risk, scenarios),
        )
    evaluations = population * (generations + 1) * _draws(problem, scenarios)
    # A SEQUENTIAL solver asks a few points per round, so its rounds are bounded by evaluations.
    rounds = evaluations if isinstance(algorithm, str) and algorithm in SEQUENTIAL else generations + 1
    # Under a solve budget a batch can repeat decided points and solve nothing new, so the batches
    # stay bounded -- by the budget, since every productive batch pays at least one solve.
    bounded = Batches(solver, rounds if solves is None else max(rounds, solves))
    policy = RoutingPolicy(bounded, oracle, problem, record, router)
    if control is not None:
        control.read = policy.results

    async def published(_current: Oracle, actions: Sequence[Action]) -> AsyncIterator[Result]:
        archive = Archive(problem, record)
        answered = _publish(oracle, actions, archive.tell)
        for action in actions:
            yield answered[action.identity.key]

    budget = Budget(evaluations=evaluations if solves is None else solves)
    try:
        execution = run(problem, oracle, policy, DecidedReplay(record), budget, dispatch=dispatch or published)
    except Refusal as error:
        if bounded.last_batch_complete is not False:
            raise
        return Searched(opened(problem, oracle, record), str(error), batch_ends=tuple(policy.batch_ends))
    return Searched(opened(problem, oracle, record), batch_ends=tuple(policy.batch_ends), execution=execution)
