"""A constraint is a declared bound with a sense; violation is the sum of normalised shortfalls; Deb ranks."""

import numpy as np
import pytest

from optimi_lab import (
    Constraint,
    ConstraintSet,
    Refusal,
    Sense,
    constrained_dominates,
    constrained_non_dominated_sorting,
    non_dominated_sorting,
)
from optimi_lab.core.pareto import total_violation

LIMITS = ConstraintSet((Constraint('ripple', '<=', 0.05), Constraint('torque', '>=', 200.0)))


def test_a_constraint_coerces_its_sense_and_refuses_a_stranger():
    assert Constraint('ripple', '<=', 0.05).sense is Sense.AT_MOST
    assert Constraint('torque', '>=', 2.0).sense is Sense.AT_LEAST
    with pytest.raises(Refusal, match="'<='"):
        Constraint('ripple', '<', 0.05)


@pytest.mark.parametrize('bound', [0.0, float('nan'), float('inf')])
def test_a_constraint_refuses_a_bound_it_cannot_normalise_by(bound):
    with pytest.raises(Refusal, match='bound'):
        Constraint('ripple', '<=', bound)


def test_a_constraint_refuses_an_empty_name_and_a_set_refuses_a_repeated_one():
    with pytest.raises(Refusal, match='non-empty'):
        Constraint('', '<=', 1.0)
    with pytest.raises(Refusal, match='duplicated'):
        ConstraintSet((Constraint('a', '<=', 1.0), Constraint('a', '>=', 2.0)))


def test_the_shortfall_is_normalised_by_the_magnitude_of_the_bound_and_zero_when_satisfied():
    raw = np.array([[0.04, 250.0], [0.10, 250.0], [0.04, 150.0], [0.10, 100.0]])
    expected = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.25], [1.0, 0.5]]
    assert np.allclose(LIMITS.shortfall(raw), expected)
    assert np.allclose(LIMITS.violation(raw), [0.0, 1.0, 0.25, 1.5])
    assert LIMITS.feasible_mask(raw).tolist() == [True, False, False, False]


def test_a_negative_bound_normalises_by_its_magnitude_not_its_sign():
    floor = ConstraintSet((Constraint('margin', '>=', -2.0),))
    assert np.allclose(floor.shortfall(np.array([[-3.0], [-1.0]])), [[0.5], [0.0]])


def test_a_constraint_that_did_not_compute_is_not_feasible_known():
    raw = np.array([[np.nan, 250.0]])
    assert np.isnan(LIMITS.violation(raw)[0])
    assert not LIMITS.feasible_mask(raw)[0]


def test_a_constraint_matrix_of_the_wrong_width_is_refused_naming_the_set():
    with pytest.raises(Refusal, match='ripple'):
        LIMITS.shortfall(np.zeros((2, 3)))


def test_total_violation_sums_shortfalls_and_keeps_an_unknown_unknown():
    assert total_violation(np.array([[0.0, 0.5], [0.0, 0.0], [np.nan, 0.0]])).tolist()[:2] == [0.5, 0.0]
    assert np.isnan(total_violation(np.array([[np.nan, 0.0]]))[0])
    assert total_violation(np.zeros((3, 0))).tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ('obj_1', 'obj_2', 'v_1', 'v_2', 'expected'),
    [
        ([1.0, 1.0], [2.0, 2.0], 0.0, 0.0, True),  # both feasible: Pareto
        ([2.0, 2.0], [1.0, 1.0], 0.0, 0.0, False),
        ([1.0, 2.0], [2.0, 1.0], 0.0, 0.0, False),  # mutually non-dominated
        ([9.0, 9.0], [1.0, 1.0], 0.0, 0.1, True),  # feasible beats infeasible, whatever the objectives
        ([1.0, 1.0], [9.0, 9.0], 0.1, 0.0, False),
        ([9.0, 9.0], [1.0, 1.0], 0.1, 0.2, True),  # two infeasible: smaller violation
        ([1.0, 1.0], [9.0, 9.0], 0.2, 0.2, False),  # equal violation: neither
    ],
)
def test_constrained_dominates_is_debs_rule(obj_1, obj_2, v_1, v_2, expected):
    assert bool(constrained_dominates(np.array(obj_1), np.array(obj_2), v_1, v_2)) is expected


def test_the_constrained_sort_ranks_feasible_fronts_first_then_infeasible_by_violation():
    values = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [0.0, 0.0], [0.5, 0.5], [0.1, 0.1]])
    violation = np.array([0.0, 0.0, 0.0, 0.3, 0.1, 0.1])
    fronts = constrained_non_dominated_sorting(values, violation, only_first_front=False)
    assert [sorted(front.tolist()) for front in fronts] == [[0, 1], [2], [4, 5], [3]]
    assert sorted(constrained_non_dominated_sorting(values, violation).tolist()) == [0, 1]


def test_with_no_feasible_row_the_first_front_is_the_least_violating():
    values = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert constrained_non_dominated_sorting(values, np.array([0.5, 0.2])).tolist() == [1]


def test_with_every_row_feasible_the_constrained_sort_is_the_plain_sort_bit_for_bit():
    """The planted control: zero violation must not move a single index or its order."""
    values = np.random.default_rng(7).random((40, 3))
    plain = non_dominated_sorting(values, only_first_front=False)
    constrained = constrained_non_dominated_sorting(values, np.zeros(40), only_first_front=False)
    assert len(plain) == len(constrained)
    for left, right in zip(plain, constrained, strict=True):
        assert left.dtype == right.dtype
        assert np.array_equal(left, right)


def test_the_constrained_sort_refuses_an_unknown_or_negative_violation_or_a_misaligned_one():
    values = np.zeros((2, 2))
    with pytest.raises(Refusal, match='finite'):
        constrained_non_dominated_sorting(values, np.array([0.0, np.nan]))
    with pytest.raises(Refusal, match='negative'):
        constrained_non_dominated_sorting(values, np.array([0.0, -1.0]))
    with pytest.raises(Refusal, match='one violation per row'):
        constrained_non_dominated_sorting(values, np.zeros(3))
