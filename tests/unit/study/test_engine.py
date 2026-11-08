"""The study engine: the algorithm menu, search over a screening oracle, evaluate, and surrogate rounds."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
from lab_commons.outcomes import Cell, Evidence, Identity, Record, Status
from study_toys import AXES, PROBLEM, READINGS, SPACE, Codec, Publisher, Toy

from optimi_lab.core import SampleSpec
from optimi_lab.core.errors import Refusal
from optimi_lab.oracle import Action, Capability, Information, Result
from optimi_lab.policy import RouteChoice, surrogate_candidates
from optimi_lab.problem import Fidelity, Level, Objective, Problem, Real, Space
from optimi_lab.solver import SEEDING, cmaes
from optimi_lab.solver.evolutionary import nsga2
from optimi_lab.study import (
    ALGORITHMS,
    ROUND_SEED_STRIDE,
    EvaluationPolicy,
    FidelityEvaluation,
    FixedDesign,
    Request,
    SampleDesign,
    SearchBinding,
    SearchStrategy,
    SurrogateStrategy,
    algorithm_factory,
    build_solver,
    evaluate,
    opened,
    search,
    study,
)
from optimi_lab.study.rounds import searched_study

START = np.array([[0.1, 2.5], [0.4, 3.0], [0.7, 3.5], [0.9, 2.2]])


def test_the_algorithm_menu_is_every_seeded_library_solver_and_passes_a_callable_through() -> None:
    assert tuple(sorted(SEEDING)) == ALGORITHMS
    assert ROUND_SEED_STRIDE == 1000
    assert algorithm_factory('nsga2') is nsga2
    assert algorithm_factory('cmaes') is cmaes
    assert callable(algorithm_factory('lbfgsb'))

    def custom(problem, capability, **settings):
        return nsga2(problem, capability, **settings)

    assert algorithm_factory(custom) is custom
    with pytest.raises(Refusal, match='registered solvers'):
        algorithm_factory('no_such_solver')
    with pytest.raises(Refusal, match='first batch'):
        algorithm_factory('lp')  # registered, but it reads coefficients, not a first batch


SMOOTH = Problem(Space((Real('a', 0.0, 1.0), Real('b', 2.0, 4.0))), (Objective('f'),))
OPTIMUM = (0.3, 3.5)


class Sloped(Toy):
    """A screening oracle declaring its gradient: f = (a - 0.3)^2 + (b - 3.5)^2, slopes in the unit cube."""

    capability = Capability(outputs=('f',), provides=frozenset({Information.VALUES, Information.VJP}), version='s')

    def answer(self, action: Action) -> Result:
        a, b = float(action.design['a']), float(action.design['b'])
        cell = Cell(Status.OK, values={'f': (a - OPTIMUM[0]) ** 2 + (b - OPTIMUM[1]) ** 2})
        if Information.VJP not in action.want:
            return Result(action, cell)
        slope = np.array([2 * (a - OPTIMUM[0]), 2 * (b - OPTIMUM[1]) * 2.0])  # d b / d unit = 2
        return Result(action, cell, gradient={'f': slope})


def _best(searched) -> tuple[float, float]:
    best = min(searched.archive.results(), key=lambda r: r.cell.values['f'])
    return float(best.action.design['a']), float(best.action.design['b'])


def test_a_named_gradient_solver_walks_from_its_one_start_on_the_declared_slope(tmp_path: Path) -> None:
    oracle = Sloped()
    searched = search(SMOOTH, oracle, Record(tmp_path / 'g.jsonl'), algorithm='lbfgsb', initial=[[0.9, 2.2]], solves=40)
    assert searched.search_refusal is None
    assert oracle.published[0] == (0.9, 2.2)
    np.testing.assert_allclose(_best(searched), OPTIMUM, atol=1e-4)
    assert len(set(oracle.published)) <= 40


def test_a_study_strategy_names_a_gradient_solver(tmp_path: Path) -> None:
    strategy = SearchStrategy(FixedDesign(((0.9, 2.2),)), algorithm='lbfgsb', solves=40)
    bound = SearchBinding(SMOOTH, Sloped(), Record(tmp_path / 's.jsonl'))
    searched, *_ = searched_study(bound, strategy, EvaluationPolicy(), np.array([[0.9, 2.2]]), Evidence.OBSERVED)
    np.testing.assert_allclose(_best(searched), OPTIMUM, atol=1e-4)


def test_a_gradient_run_records_and_reopens_its_slope_reading_actions(tmp_path: Path) -> None:
    record = Record(tmp_path / 'g.jsonl')
    search(SMOOTH, Sloped(), record, algorithm='lbfgsb', initial=[[0.9, 2.2]], solves=40)
    archive = opened(SMOOTH, Sloped(), Record(tmp_path / 'g.jsonl'))  # closed, then reopened from disk
    sloped = [r.action for r in archive.results() if Information.VJP in r.action.want]
    assert sloped
    # Planted control: the values-only rebuild the reopen used to make names another action.
    values_only = Action.of(SMOOTH, Sloped.capability, sloped[0].design)
    assert values_only.identity.key != sloped[0].identity.key
    rebuilt = Sloped()
    rebuilt.capability = Capability(outputs=('f',), provides=Sloped.capability.provides, version='s2')
    with pytest.raises(Refusal, match='another problem or evaluator'):
        opened(SMOOTH, rebuilt, record)  # another oracle build is still refused


def test_a_named_solver_is_refused_by_its_own_requires(tmp_path: Path) -> None:
    with pytest.raises(Refusal, match='lbfgsb cannot run'):
        search(SMOOTH, Toy(), Record(tmp_path / 'v.jsonl'), algorithm='lbfgsb', initial=[[0.9, 2.2]])
    with pytest.raises(Refusal, match='single-objective'):
        search(PROBLEM, Toy(), Record(tmp_path / 'm.jsonl'), algorithm='cmaes')
    with pytest.raises(Refusal, match='ONE start'):
        search(SMOOTH, Sloped(), Record(tmp_path / 'two.jsonl'), algorithm='lbfgsb', initial=START)
    with pytest.raises(Refusal, match='draws its own'):
        search(SMOOTH, Sloped(), Record(tmp_path / 'bo.jsonl'), algorithm='bo', initial=START)


def test_build_solver_validates_a_name_without_evaluating() -> None:
    solver = build_solver('lbfgsb', SMOOTH, Sloped.capability, seed=0, population=1, generations=1)
    assert dict(solver.ask(1)[0].design) == {'a': 0.5, 'b': 3.0}  # no start: the cube's centre
    with pytest.raises(Refusal, match='cannot run'):
        build_solver('lbfgsb', SMOOTH, Toy.capability, seed=0, population=1, generations=1)


def test_a_self_seeded_solver_runs_by_count_under_the_solve_budget(tmp_path: Path) -> None:
    oracle = Sloped()
    searched = search(SMOOTH, oracle, Record(tmp_path / 'c.jsonl'), algorithm='cmaes', population=6, solves=30)
    assert searched.search_refusal is None
    assert 6 <= len(set(oracle.published)) <= 30


def test_a_search_files_every_batch_and_a_rerun_on_its_record_publishes_nothing(tmp_path: Path) -> None:
    record = Record(tmp_path / 'r.jsonl')
    toy = Toy()
    searched = search(PROBLEM, toy, record, population=4, generations=2, seed=3, initial=START)
    assert searched.search_refusal is None and searched.execution is not None
    assert len(searched.batch_ends) == 3 and searched.batch_ends[0] == 4
    assert len(searched.ordered_results()) == len(searched.archive.results()) == len(set(toy.published))
    np.testing.assert_allclose(
        [[r.action.design['a'], r.action.design['b']] for r in searched.ordered_results()[:4]], START
    )
    again = Toy()
    replayed = search(PROBLEM, again, Record(tmp_path / 'r.jsonl'), population=4, generations=2, seed=3, initial=START)
    assert again.published == [] and replayed.batch_ends == searched.batch_ends


def test_a_screening_router_spends_no_solve_and_steering_needs_an_sbx_solver(tmp_path: Path) -> None:
    toy = Toy()
    screened = search(
        PROBLEM,
        toy,
        Record(tmp_path / 'r.jsonl'),
        population=4,
        generations=1,
        initial=START,
        router=lambda points, _results: np.full(len(points), RouteChoice.SCREENED),
    )
    assert len(toy.published) == 4
    statuses = {r.cell.status for r in screened.archive.results()}
    assert statuses == {Status.OK, Status.SCREENED}
    with pytest.raises(Refusal, match='nsga2 or nsga3'):
        search(PROBLEM, Toy(), Record(tmp_path / 's.jsonl'), algorithm='mode', initial=START, variation=lambda _r: None)


def test_a_true_solve_budget_makes_a_screening_search_pay_the_same_solves_as_one_that_screens_nothing(
    tmp_path: Path,
) -> None:
    def every_other(points, _results):
        return np.where(np.arange(len(points)) % 2, RouteChoice.SCREENED, 0)

    paid = {}
    for name, router in (('plain', None), ('screened', every_other)):
        toy = Toy()
        search(
            PROBLEM,
            toy,
            Record(tmp_path / f'{name}.jsonl'),
            population=4,
            generations=1,
            initial=START,
            router=router,
            solves=11,
        )
        paid[name] = len(set(toy.published))
    assert paid == {'plain': 11, 'screened': 11}
    toy = Toy()
    search(PROBLEM, toy, Record(tmp_path / 'g.jsonl'), population=4, generations=1, initial=START, router=every_other)
    assert len(set(toy.published)) < 8  # without a solve budget the generations stop, screened points unpaid


def test_a_solve_budget_below_one_is_refused(tmp_path: Path) -> None:
    with pytest.raises(Refusal, match='true-solve budget'):
        search(PROBLEM, Toy(), Record(tmp_path / 'r.jsonl'), initial=START, solves=0)
    with pytest.raises(Refusal, match='true-solve budget'):
        SearchStrategy(FixedDesign(tuple(map(tuple, START))), solves=0)


def test_evaluate_reuses_decided_results_and_counts_what_it_solved(tmp_path: Path) -> None:
    record = Record(tmp_path / 'r.jsonl')
    toy = Toy()
    first = evaluate(PROBLEM, toy, START[:2], record)
    second = evaluate(PROBLEM, toy, START[1:3], record)
    assert (first.solved, first.served, second.solved, second.served) == (2, 0, 1, 1)
    assert len(toy.published) == 3 and len(second.ordered_results()) == 3
    with pytest.raises(ValueError, match='design axes'):
        evaluate(PROBLEM, toy, np.zeros((2, 3)), record)


def test_a_surrogate_study_steers_at_the_round_seed_and_records_its_evaluators(tmp_path: Path) -> None:
    rounds = []
    codec = Codec()
    strategy = SearchStrategy(
        SampleDesign(SampleSpec(n_samples=10, seed=2)),
        generations=0,
        seed=7,
        surrogate=SurrogateStrategy(rounds=1, candidates=6, on_round=rounds.append),
    )
    outcome = study(
        Request(
            SPACE,
            READINGS,
            Publisher(),
            strategy,
            {'case': 'c'},
            axes=AXES,
            codec=codec,
            record_path=tmp_path / 'r.jsonl',
        )
    )
    assert [r.index for r in rounds] == [0, 1] and outcome.steering is rounds[1].steering
    assert outcome.steering.refusal is None and outcome.steering.solved >= 1
    drawn = {tuple(p) for p in surrogate_candidates(SPACE, candidates=6, seed=7 + 1 * ROUND_SEED_STRIDE)}
    assert set(codec.oracle.published[10:]) <= drawn
    assert len(outcome.result.ordered_results()) == len(codec.oracle.published)


def test_a_fidelity_policy_is_recorded_as_one_study_evaluator_event(tmp_path: Path) -> None:
    fidelity = FidelityEvaluation(Publisher(), {'leg': 'fast', 'mesh': 2}, 1.0)
    policy = EvaluationPolicy(
        router=lambda points, _results: np.zeros(len(points), dtype=int), fidelities=(fidelity,), primary_cost=4.0
    )
    strategy = SearchStrategy(FixedDesign(tuple(map(tuple, START))), generations=1)
    path = tmp_path / 'r.jsonl'
    study(
        Request(
            SPACE,
            READINGS,
            Publisher(),
            strategy,
            {'case': 'c'},
            axes=AXES,
            codec=Codec(),
            record_path=path,
            policy=policy,
        )
    )
    lines = [
        line for line in path.read_text().splitlines() if '"study_evaluator"' in line or '"study_strategy"' in line
    ]
    assert lines == [
        '{"data":{"leg":"fast","mesh":2},"key":"280cc995e290dcf53d9a2381ca59dad1dc7b5831fd8b3dc98ed0ffd1fda24d49","kind":"study_evaluator","schema":3}',
        '{"data":{"kind":"SearchStrategy","slot":0},"key":"strategy:0","kind":"study_strategy","schema":3}',
    ]
    assert Identity.of({'leg': 'fast', 'mesh': 2}).key.startswith('280cc995')
    # No router is a declaration too: the SOLVER picks each point's level (``Seeding.LEVELS``); the search refuses a
    # solver that cannot (test_a_router_and_a_level_seeded_solver_are_two_choosers_and_many_legs_need_one).
    assert EvaluationPolicy(fidelities=(fidelity,), primary_cost=4.0).router is None


#: Two levels of one objective: the cheap level is a biased, scaled copy of the truth.
LEVELS = {'coarse': 1.0, 'fine': 10.0}
LEVELLED = Problem(
    SMOOTH.space, SMOOTH.objectives, fidelity=Fidelity(tuple(Level(name, cost) for name, cost in LEVELS.items()))
)


class Levelled(Toy):
    """A two-leg screening oracle: leg 0 is the truth ``fine``, leg 1 the cheap ``coarse``; it records each level."""

    capability = Capability(outputs=('f',), fidelities=tuple(LEVELS), version='levelled')
    evaluators = 2

    def __init__(self) -> None:
        super().__init__()
        self.levels: list[str | None] = []

    def answer(self, action: Action) -> Result:
        a, b = float(action.design['a']), float(action.design['b'])
        truth = (a - OPTIMUM[0]) ** 2 + (b - OPTIMUM[1]) ** 2
        value = truth if action.fidelity in ('fine', None) else 0.5 * truth + 0.1 * a
        return Result(action, Cell(Status.OK, values={'f': value}))

    def routed(self, action: Action, index: int) -> Action:
        return Action.of(LEVELLED, self.capability, action.design, fidelity=('fine', 'coarse')[index])

    def publish_batch(self, actions, on_result) -> None:
        self.levels.extend(action.fidelity for action in actions)
        super().publish_batch(actions, on_result)


def test_a_level_seeded_solver_picks_each_points_level_and_its_population_is_the_truth_design(tmp_path: Path) -> None:
    oracle = Levelled()
    searched = search(LEVELLED, oracle, Record(tmp_path / 'mf.jsonl'), algorithm='mfbo', population=3, solves=16)
    assert searched.search_refusal is None
    # The nested design: the declared population at the truth, three times it at the cheaper level.
    assert sorted(oracle.levels[:12]) == ['coarse'] * 9 + ['fine'] * 3
    assert len(oracle.levels) == 16 and set(oracle.levels[12:]) <= set(LEVELS)


def test_a_router_and_a_level_seeded_solver_are_two_choosers_and_many_legs_need_one(tmp_path: Path) -> None:
    with pytest.raises(Refusal, match='picks its own level'):
        search(
            LEVELLED,
            Levelled(),
            Record(tmp_path / 'two.jsonl'),
            algorithm='mfbo',
            population=3,
            router=lambda points, _results: np.zeros(len(points), dtype=int),
        )
    with pytest.raises(Refusal, match='selection policy'):
        search(LEVELLED, Levelled(), Record(tmp_path / 'none.jsonl'), algorithm='cmaes', population=4)


class PredictingToy(Toy):
    """The toy whose every answer is a PREDICTED cell -- a model standing in for the solver."""

    def answer(self, action):
        observed = super().answer(action)
        return Result(action, Cell(Status.OK, values=dict(observed.cell.values), evidence=Evidence.PREDICTED))


def test_steering_under_predicted_evidence_counts_and_scores_the_predictions_it_got_back(tmp_path: Path) -> None:
    codec = Codec(PredictingToy())
    strategy = SearchStrategy(
        SampleDesign(SampleSpec(n_samples=10, seed=2)),
        generations=0,
        seed=7,
        surrogate=SurrogateStrategy(rounds=1, candidates=6),
    )
    outcome = study(
        Request(
            SPACE,
            READINGS,
            Publisher(),
            strategy,
            {'case': 'c'},
            axes=AXES,
            codec=codec,
            record_path=tmp_path / 'r.jsonl',
            evidence=Evidence.PREDICTED,
        )
    )
    steering = outcome.steering
    assert steering.refusal is None and steering.solved >= 1
    assert steering.added == steering.solved
    assert steering.accuracy is not None and steering.accuracy['n'] == steering.solved


def _search_request(path: Path, strategy: SearchStrategy) -> Request:
    return Request(SPACE, READINGS, Publisher(), strategy, {'case': 'c'}, axes=AXES, codec=Codec(), record_path=path)


def _searches(path: Path) -> dict[str, dict]:
    return {e.key: e.data for e in Record(path).replay() if e.kind == 'study_search'}


def test_an_identical_search_rerun_resumes_its_slot(tmp_path: Path) -> None:
    path = tmp_path / 'r.jsonl'
    strategy = SearchStrategy(FixedDesign(tuple(map(tuple, START))), generations=1, seed=3)
    study(_search_request(path, strategy))
    again = Toy()
    study(
        Request(SPACE, READINGS, Publisher(), strategy, {'case': 'c'}, axes=AXES, codec=Codec(again), record_path=path)
    )
    assert [e.key for e in Record(path).replay() if e.kind == 'study_strategy'] == ['strategy:0']
    assert list(_searches(path)) == ['strategy:0:search']
    assert again.published == []


def test_a_search_with_different_parameters_opens_a_new_slot_and_reuses_paid_evidence(tmp_path: Path) -> None:
    """Ruling A-63: different canonical parameters are a NEW strategy reading the record's shared archive."""
    path = tmp_path / 'r.jsonl'
    strategy = SearchStrategy(FixedDesign(tuple(map(tuple, START))), generations=1, seed=3)
    study(_search_request(path, strategy))
    changes = (
        SearchStrategy(strategy.initial, generations=1, seed=4),
        SearchStrategy(strategy.initial, generations=2, seed=3),
        SearchStrategy(strategy.initial, algorithm='nsga3', generations=1, seed=3),
        SearchStrategy(strategy.initial, generations=1, seed=3, surrogate=SurrogateStrategy(candidates=4)),
    )
    for slot, changed in enumerate(changes, start=1):
        toy = Toy()
        request = Request(
            SPACE, READINGS, Publisher(), changed, {'case': 'c'}, axes=AXES, codec=Codec(toy), record_path=path
        )
        study(request)
        assert not {tuple(point) for point in START.tolist()} & set(toy.published)
        searches = _searches(path)
        assert len(searches) == slot + 1 and f'strategy:{slot}:search' in searches
    searches = _searches(path)
    assert searches['strategy:0:search']['seed'] == 3 and searches['strategy:1:search']['seed'] == 4
    assert searches['strategy:2:search']['generations'] == 2
    resumed = Toy()
    study(
        Request(
            SPACE, READINGS, Publisher(), changes[0], {'case': 'c'}, axes=AXES, codec=Codec(resumed), record_path=path
        )
    )
    assert len(_searches(path)) == 5 and resumed.published == []


def test_the_study_docstring_states_the_slot_rule() -> None:
    module = importlib.import_module('optimi_lab.study.study')
    doc = ' '.join(module.__doc__.split())
    assert 'differ from every recorded search open a new slot' in doc
    assert 'equal a recorded slot' in doc and 'resume' in doc


def test_a_rejected_request_files_no_strategy_event(tmp_path: Path) -> None:
    path = tmp_path / 'r.jsonl'
    with pytest.raises(TypeError, match='fixed/sample design'):
        study(_search_request(path, object()))  # type: ignore[arg-type]
    with pytest.raises(Refusal, match='different design space'):
        study(_search_request(path, SearchStrategy(FixedDesign(((0.1, 2.5, 9.0),)), generations=1)))
    assert not [e for e in Record(path).replay() if e.kind == 'study_strategy']
