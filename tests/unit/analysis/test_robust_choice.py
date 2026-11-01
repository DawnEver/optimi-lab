"""Candidate thinning, the scenario box, the three robust formulations and their validation.

Every expected choice is read off a table small enough to judge by hand; the validation bootstrap
is pinned against a copy of the vectorised resampling it replaces.
"""

import numpy as np
import pytest

from optimi_lab.analysis import box_corners, box_draw, choose, robust_validation, thin_candidates
from optimi_lab.core import Refusal


def test_thinning_keeps_the_front_starts_at_the_best_feasible_and_spreads():
    f = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.9, 0.9], [0.25, 0.75]])
    picked = thin_candidates(f, 3, feasible=np.array([False, True, True, True, True]))
    assert picked[0] == 4  # least first objective among feasible front members
    assert set(picked) == {4, 2, 0}  # then the farthest, then the farthest from both
    assert 3 not in thin_candidates(f, 4, feasible=np.ones(5, dtype=bool))
    with pytest.raises(Refusal):
        thin_candidates(f, 5, feasible=np.ones(5, dtype=bool))


def test_thinning_without_a_feasible_member_starts_at_the_least_first_objective():
    f = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    assert thin_candidates(f, 1, feasible=np.zeros(3, dtype=bool)) == [0]


def test_box_draw_is_uniform_on_the_box_from_its_seed():
    lower, upper = np.array([70.0, -0.05, 0.98]), np.array([130.0, 0.05, 1.02])
    draws = box_draw(500, lower, upper, seed=6001)
    unit = np.random.default_rng(6001).random((500, 3))
    np.testing.assert_allclose(draws, lower + unit * (upper - lower))
    assert np.all(draws >= lower) and np.all(draws <= upper)


def test_box_corners_pick_a_bound_per_axis():
    corners = box_corners(np.array([0.0, -1.0]), np.array([1.0, 1.0]), [[True, False], [False, True]])
    assert corners.tolist() == [[1.0, -1.0], [0.0, 1.0]]


def _table():
    """Columns: nominal, one corner, four train draws; rows: designs a, b, c."""
    score = np.array([
        [10.0, 11.0, 10.0, 10.0, 10.0, 10.0],  # a: cheapest nominal, violates at the corner and one draw
        [12.0, 12.0, 12.0, 12.0, 12.0, 12.0],  # b: robust everywhere
        [11.0, 11.5, 11.0, 11.0, 11.0, 11.0],  # c: violates one train draw only
    ])
    violation = np.array([
        [0.0, 0.2, 0.0, 0.0, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.3, 0.0, 0.0],
    ])
    return ['a', 'b', 'c'], score, violation


def test_the_three_formulations_choose_by_hand():
    names, score, violation = _table()
    out = choose(names, score, violation, nominal=0, corners=[1], train=[2, 3, 4, 5], chance_level=0.75)
    assert (out['DET'].design, out['DET'].feasible, out['DET'].score) == ('a', True, 10.0)
    assert (out['WC'].design, out['WC'].feasible, out['WC'].score) == ('b', True, 12.0)
    # one miss in four is allowed at 0.75: a and c qualify, a has the lower mean train score
    assert (out['CC'].design, out['CC'].feasible, out['CC'].score) == ('a', True, 10.0)


def test_a_formulation_nobody_satisfies_returns_the_least_violating_marked_infeasible():
    names, score, violation = _table()
    violation = violation + 0.05
    out = choose(names, score, violation, nominal=0, corners=[1], train=[2, 3, 4, 5], chance_level=0.95)
    assert (out['DET'].design, out['DET'].feasible) == ('a', False)  # ties on violation, least score
    assert (out['WC'].design, out['WC'].feasible) == ('b', False)


def _reference_bootstrap(values, stat, draws, seed):
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(values), size=(draws, len(values)))
    sample = np.array([stat(values[p]) for p in picks])
    return float(stat(values)), float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))


def test_validation_summaries_match_the_replaced_vectorised_bootstrap():
    rng = np.random.default_rng(4)
    score, floor = rng.normal(1000.0, 30.0, 64), rng.normal(62.0, 1.0, 64)
    violated = floor < 61.0
    v = robust_validation(score, floor, violated, draws=500, seed=0)
    assert v['n'] == 64
    assert v['score_mean'] == pytest.approx(_reference_bootstrap(score, np.mean, 500, 0), abs=0.0)
    assert v['score_p95'] == pytest.approx(_reference_bootstrap(score, lambda s: np.quantile(s, 0.95), 500, 0), abs=0.0)
    assert v['floor_p5'] == pytest.approx(_reference_bootstrap(floor, lambda s: np.quantile(s, 0.05), 500, 0), abs=0.0)
    assert v['floor_min'] == float(floor.min())
    assert v['violation'][0] == violated.mean()
