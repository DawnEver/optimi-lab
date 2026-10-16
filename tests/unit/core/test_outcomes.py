"""An Evaluation validates its shapes, blanks a failed cell, and carries one constraint row per point."""

import numpy as np
import pytest

from optimi_lab import Constraint, ConstraintSet, Evaluation, Objective, ObjectiveSet, Outcome, Refusal

OBJECTIVES = ObjectiveSet((Objective('torque', 'maximize'), Objective('mass')))
LIMITS = ConstraintSet((Constraint('ripple', '<=', 0.05),))


def test_an_ok_cell_must_hold_a_finite_number():
    with pytest.raises(Refusal, match="marked 'ok'"):
        Evaluation([[0.0]], [[np.inf]])


def test_a_failed_cell_is_blanked_whatever_the_caller_passed():
    evaluation = Evaluation([[0.0]], [[3.0, 4.0]], [['ok', 'timeout']])
    assert np.isnan(evaluation.values[0, 1])
    assert evaluation.point_outcomes() == (Outcome.TIMEOUT,)
    with pytest.raises(Refusal, match="'timeout'"):
        evaluation.value(0, 1)


@pytest.mark.parametrize(
    ('inputs', 'values', 'outcomes', 'match'),
    [
        ([0.0, 1.0], [[1.0]], None, 'inputs must be a 2-D'),
        ([[0.0]], [1.0], None, 'values must be a 2-D'),
        ([[0.0], [1.0]], [[1.0]], None, 'one row per point'),
        ([[0.0]], [[1.0]], [['ok', 'ok']], 'one entry per value'),
        ([[0.0]], [[1.0]], [['lost']], 'outcome at cell'),
    ],
)
def test_a_misshapen_evaluation_is_refused(inputs, values, outcomes, match):
    with pytest.raises(Refusal, match=match):
        Evaluation(inputs, values, outcomes)


def test_an_evaluation_without_constraints_carries_an_empty_constraint_matrix():
    evaluation = Evaluation([[0.0], [1.0]], [[1.0, 2.0], [3.0, 4.0]])
    assert evaluation.constraints.shape == (2, 0)
    assert evaluation.n_con == 0


def test_the_constraint_matrix_needs_one_row_per_point_and_refuses_an_infinity():
    with pytest.raises(Refusal, match='one row per point'):
        Evaluation([[0.0]], [[1.0, 2.0]], constraints=[[0.1], [0.2]])
    with pytest.raises(Refusal, match='did not compute'):
        Evaluation([[0.0]], [[1.0, 2.0]], constraints=[[np.inf]])


def test_a_point_whose_constraint_did_not_compute_is_not_complete():
    evaluation = Evaluation([[0.0], [1.0]], [[1.0, 2.0], [3.0, 4.0]], constraints=[[0.01], [np.nan]])
    assert evaluation.complete_mask().tolist() == [True, False]
    assert len(evaluation.complete()) == 1
    assert evaluation.complete().constraints.tolist() == [[0.01]]
    with pytest.raises(Refusal, match=r'\[1\]'):
        evaluation.require_complete()


def test_stack_carries_constraints_and_refuses_batches_that_disagree_on_them():
    first = Evaluation([[0.0]], [[1.0, 2.0]], constraints=[[0.01]])
    second = Evaluation([[1.0]], [[3.0, 4.0]], constraints=[[0.09]])
    assert Evaluation.stack([first, second]).constraints.tolist() == [[0.01], [0.09]]
    with pytest.raises(Refusal, match='n_con'):
        Evaluation.stack([first, Evaluation([[1.0]], [[3.0, 4.0]])])


def test_the_minimization_frame_negates_maxima_and_turns_constraints_into_shortfalls():
    evaluation = Evaluation([[0.0], [1.0]], [[10.0, 2.0], [20.0, 4.0]], constraints=[[0.04], [0.10]])
    minimized = evaluation.to_minimization(OBJECTIVES, LIMITS)
    assert minimized.values.tolist() == [[-10.0, 2.0], [-20.0, 4.0]]
    assert np.allclose(minimized.constraints, [[0.0], [1.0]])
    assert evaluation.constraints.tolist() == [[0.04], [0.10]]


def test_the_minimization_frame_refuses_constraints_without_their_declaration_and_the_reverse():
    constrained = Evaluation([[0.0]], [[10.0, 2.0]], constraints=[[0.04]])
    with pytest.raises(Refusal, match='ConstraintSet'):
        constrained.to_minimization(OBJECTIVES)
    with pytest.raises(Refusal, match='ripple'):
        Evaluation([[0.0]], [[10.0, 2.0]]).to_minimization(OBJECTIVES, LIMITS)


def test_a_default_outcome_is_the_member_not_its_string():
    """`np.full` stored the member's plain `str`, so every default cell failed `is Outcome.OK`."""
    evaluation = Evaluation([[0.0]], [[1.0]])
    assert evaluation.outcomes[0, 0] is Outcome.OK
    assert evaluation.point_outcomes() == (Outcome.OK,)
