"""L8 surrogate steering: the gate, the closed-form pieces, the seed offset and one steered round."""

from __future__ import annotations

import math

import numpy as np
import pytest
from lab_commons.outcomes import Cell, Reason, Status

from optimi_lab.core import SampleSpec
from optimi_lab.oracle import Action, Capability, Result
from optimi_lab.policy import (
    FALLBACK_PROPOSALS,
    MIN_SURROGATE_POINTS,
    SURROGATE_POOL,
    SURROGATE_STEERING_SEED_OFFSET,
    Evaluation,
    ScoredModel,
    Steering,
    best_model,
    feasible_front,
    fit_surrogate,
    probability_of_feasibility,
    proposal_accuracy,
    steer,
    steering_refusal,
    surrogate_candidates,
)
from optimi_lab.problem import BlackBox, Objective, Problem, Real, Relation, Sense, Space

SPACE = Space((Real('x', 0.0, 1.0), Real('y', 0.0, 1.0)))
PROBLEM = Problem(SPACE, (Objective('f'), Objective('g', Sense.MAXIMIZE)))
CAPABILITY = Capability(outputs=('f', 'g'), version='lin')


def _answer(point) -> Result:
    x, y = float(point[0]), float(point[1])
    action = Action.of(PROBLEM, CAPABILITY, {'x': x, 'y': y})
    return Result(action, Cell(Status.OK, values={'f': x + y, 'g': x - 2 * y}))


def test_the_steering_constants_are_the_ones_the_records_were_drawn_with() -> None:
    assert SURROGATE_POOL == ('polynomial', 'radial_basis', 'nearest_neighbours')
    assert (MIN_SURROGATE_POINTS, SURROGATE_STEERING_SEED_OFFSET, FALLBACK_PROPOSALS) == (8, 1, 8)


def test_the_candidate_batch_is_the_space_sample_at_the_offset_seed() -> None:
    drawn = surrogate_candidates(SPACE, candidates=6, seed=41)
    np.testing.assert_array_equal(drawn, SPACE.natural_sample(SampleSpec(n_samples=6, seed=42)))


def test_the_gate_names_the_best_member_and_refuses_at_or_below_the_mean() -> None:
    members = (ScoredModel('a', {'r2': -0.5}, None), ScoredModel('b', {}, None), ScoredModel('c', {'r2': 0.0}, None))
    assert best_model(members)[:2] == ('c', 0.0)
    assert 'best: c r2 0' in steering_refusal(members)
    assert 'no pool member reported' in steering_refusal((ScoredModel('b', {}, None),))
    assert steering_refusal((ScoredModel('d', {'r2': 0.1}, None),)) is None
    with pytest.raises(ValueError, match='never both'):
        Steering(best=('d', 0.1), refusal='no')


def test_feasibility_is_a_product_of_normal_tails_and_the_front_falls_back_to_the_most_probable() -> None:
    bounds = (BlackBox('c', Relation.AT_MOST, 1.0), BlackBox('d', Relation.AT_LEAST, 0.0, scale=1.0))
    mean = np.array([[1.0, 0.0], [0.0, 1.0]])
    pof = probability_of_feasibility(mean, np.ones((2, 2)), bounds)
    phi1 = 0.5 * (1 + math.erf(1 / math.sqrt(2)))
    np.testing.assert_allclose(pof, [0.25, phi1 * phi1])
    low = np.array([0.1, 0.3, 0.2])
    np.testing.assert_array_equal(feasible_front(np.zeros((3, 2)), low), [1, 2, 0])
    np.testing.assert_array_equal(np.sort(feasible_front(np.array([[0, 1], [1, 0], [2, 2]]), np.full(3, 0.9))), [0, 1])


def test_proposal_accuracy_in_closed_form() -> None:
    got = proposal_accuracy(
        np.array([[1.0], [3.0]]),
        np.array([[2.0], [3.0]]),
        span=np.array([2.0]),
        sd=np.array([[1.0], [1.0]]),
        pof=np.array([1.0, 0.5]),
        feasible=np.array([True, False]),
    )
    assert got == {
        'n': 2,
        'nrmse': pytest.approx(math.sqrt(0.5) / 2),
        'coverage95': 1.0,
        'feasible_share': 0.5,
        'pof_mean': 0.75,
        'brier': 0.125,
    }


def test_a_fit_needs_eight_ok_results_and_skips_the_ones_that_did_not_compute() -> None:
    rng = np.random.default_rng(0)
    results = [_answer(p) for p in rng.random((7, 2))]
    failed = Result(Action.of(PROBLEM, CAPABILITY, {'x': 0.5, 'y': 0.5}), Cell(Status.ERROR, reason=Reason('error')))
    assert fit_surrogate(PROBLEM, [*results, failed]) is None
    fit = fit_surrogate(PROBLEM, [*results, failed, _answer((0.9, 0.1))])
    assert fit is not None and len(fit.results) == 8 and [m.key for m in fit.models] == list(SURROGATE_POOL)


def test_a_steered_round_evaluates_the_predicted_front_and_counts_what_was_new() -> None:
    known = [_answer(p) for p in np.random.default_rng(1).random((12, 2))]
    fit = fit_surrogate(PROBLEM, known)
    assert steer(None, known, lambda _p: Evaluation((), 0, 0), candidates=4, seed=0).refusal is not None
    asked: list[np.ndarray] = []

    def evaluate(points: np.ndarray) -> Evaluation:
        asked.append(points)
        return Evaluation(tuple(_answer(p) for p in points), len(points), 0)

    steering = steer(fit, known, evaluate, candidates=16, seed=5)
    assert steering.refusal is None and steering.best[0] == 'polynomial' and steering.best[1] > 0.99
    assert steering.proposed == steering.solved == steering.added == len(asked[0]) >= 1
    assert steering.accuracy['nrmse'] < 1e-6
    drawn = surrogate_candidates(SPACE, candidates=16, seed=5)
    assert {tuple(p) for p in asked[0]} <= {tuple(p) for p in drawn}
