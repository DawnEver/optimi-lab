"""The study engine: the algorithm menu, search over a screening oracle, evaluate, and surrogate rounds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from lab_commons.outcomes import Identity, Record, Status
from study_toys import AXES, PROBLEM, READINGS, SPACE, Codec, Publisher, Toy

from optimi_lab.core import SampleSpec
from optimi_lab.core.errors import Refusal
from optimi_lab.policy import RouteChoice, surrogate_candidates
from optimi_lab.solver.evolutionary import nsga2
from optimi_lab.study import (
    ALGORITHMS,
    ROUND_SEED_STRIDE,
    EvaluationPolicy,
    FidelityEvaluation,
    FixedDesign,
    Request,
    SampleDesign,
    SearchStrategy,
    SurrogateStrategy,
    algorithm_factory,
    evaluate,
    search,
    study,
)

START = np.array([[0.1, 2.5], [0.4, 3.0], [0.7, 3.5], [0.9, 2.2]])


def test_the_algorithm_menu_resolves_names_and_passes_a_callable_through() -> None:
    assert ALGORITHMS == ('mode', 'moead', 'mopso', 'nsga2', 'nsga3')
    assert ROUND_SEED_STRIDE == 1000
    assert algorithm_factory('nsga2') is nsga2

    def custom(problem, capability, **settings):
        return nsga2(problem, capability, **settings)

    assert algorithm_factory(custom) is custom
    with pytest.raises(ValueError, match='unknown algorithm'):
        algorithm_factory('cmaes')


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
        search(PROBLEM, Toy(), Record(tmp_path / 's.jsonl'), algorithm='mode', initial=START, steer=lambda _r: None)


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
    outcome = study(Request(SPACE, AXES, READINGS, Publisher(), codec, strategy, {'case': 'c'}, tmp_path / 'r.jsonl'))
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
    study(Request(SPACE, AXES, READINGS, Publisher(), Codec(), strategy, {'case': 'c'}, path, policy=policy))
    lines = [
        line for line in path.read_text().splitlines() if '"study_evaluator"' in line or '"study_strategy"' in line
    ]
    assert lines == [
        '{"data":{"leg":"fast","mesh":2},"key":"280cc995e290dcf53d9a2381ca59dad1dc7b5831fd8b3dc98ed0ffd1fda24d49","kind":"study_evaluator","schema":3}',
        '{"data":{"kind":"SearchStrategy","slot":0},"key":"strategy:0","kind":"study_strategy","schema":3}',
    ]
    assert Identity.of({'leg': 'fast', 'mesh': 2}).key.startswith('280cc995')
    with pytest.raises(ValueError, match='selection policy'):
        EvaluationPolicy(fidelities=(fidelity,), primary_cost=4.0)
