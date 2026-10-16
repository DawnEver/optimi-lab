"""Every proposer ranks by Deb's rule when constraints exist, and by Pareto alone when none do.

The problem is ZDT1 with ``x >= 0.8``: its unconstrained front spans ``x`` in ``[0, 1]``, so a
proposer blind to the constraint keeps spreading most of its population over the forbidden 80%,
while one that ranks feasible points first moves its batches into the allowed strip.
"""

import numpy as np
import pytest

from optimi_lab import (
    Constraint,
    ConstraintSet,
    Evaluation,
    Objective,
    ObjectiveSet,
    SampleSpec,
    Variable,
    VariableSet,
    optimize,
    zdt1,
)
from optimi_lab.intelligent_algorithm import mode, moead, mopso, nsga2, nsga3
from optimi_lab.intelligent_algorithm.selection import crowding_select, reference_select, tournament_select

ALGORITHMS = (nsga2, nsga3, mode, moead, mopso)
PROBLEM = zdt1(n_var=3)
SPACE = VariableSet(tuple(Variable(f'x{index}', 0.0, 1.0) for index in range(3)))
OBJECTIVES = ObjectiveSet((Objective('f0'), Objective('f1')))
STRIP = ConstraintSet((Constraint('x0_floor', '>=', 0.8),))
SLACK = ConstraintSet((Constraint('x0_cap', '<=', 10.0),))


def _run(factory, constraints: ConstraintSet | None, n_batches: int = 25, seed: int = 3):
    def evaluate(points: np.ndarray) -> Evaluation:
        extra = {} if constraints is None else {'constraints': points[:, :1]}
        return Evaluation(points, PROBLEM.evaluate(points), **extra)

    return optimize(
        space=SPACE,
        objectives=OBJECTIVES,
        constraints=constraints,
        evaluate=evaluate,
        build_proposer=factory(SPACE, seed=seed),
        spec=SampleSpec(n_samples=20, seed=seed),
        n_batches=n_batches,
    )


@pytest.mark.parametrize('factory', ALGORITHMS)
def test_a_constraint_that_never_binds_moves_no_point(factory):
    """The planted control: zero violation everywhere is the unconstrained run, bit for bit."""
    free = _run(factory, None, n_batches=8)
    slack = _run(factory, SLACK, n_batches=8)
    assert np.array_equal(free.evaluation.inputs, slack.evaluation.inputs)


@pytest.mark.parametrize('factory', ALGORITHMS)
def test_a_binding_constraint_moves_the_last_batch_into_the_feasible_strip(factory):
    record = _run(factory, STRIP)
    last = record.evaluation.inputs[-20:]
    assert np.mean(last[:, 0] >= 0.8) >= 0.5
    assert np.all(record.pareto().inputs[:, 0] >= 0.8)


def test_a_survivor_selection_keeps_the_feasible_rows_before_any_infeasible_one():
    values = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0], [0.1, 0.1]])
    violation = np.array([0.5, 0.0, 0.0, 0.2])
    assert sorted(crowding_select(values, 2, violation=violation).tolist()) == [1, 2]
    assert sorted(reference_select(values, 3, violation=violation).tolist()) == [1, 2, 3]


def test_a_tournament_between_a_feasible_and_an_infeasible_row_goes_to_the_feasible_one():
    values = np.array([[0.0, 0.0], [5.0, 5.0]])
    winners = tournament_select(values, 2, np.random.default_rng(0), violation=np.array([0.1, 0.0]))
    assert winners.tolist() == [1, 1]
