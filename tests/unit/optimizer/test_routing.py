"""A router decides which evaluator -- if any -- spends its budget on each point.

The routed evaluator is an :class:`Evaluator` like any other, so :func:`optimize` and every
proposer run unchanged; a point the router screens out carries :attr:`Outcome.SCREENED` and no
number, so no proposer can rank it as good or bad.
"""

import numpy as np
import pytest

from optimi_lab import (
    SCREENED,
    Evaluation,
    Objective,
    ObjectiveSet,
    Outcome,
    Refusal,
    SampleSpec,
    Variable,
    VariableSet,
    optimize,
    routed,
)
from optimi_lab.intelligent_algorithm import nsga2

OBJECTIVES = ObjectiveSet([Objective('f1', 'minimize'), Objective('f2', 'minimize')])


class Counting:
    """An evaluator that scores ``(x0, offset + x1)`` and remembers every point it was handed."""

    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset
        self.seen = []

    def __call__(self, inputs: np.ndarray) -> Evaluation:
        self.seen.append(inputs.copy())
        return Evaluation(inputs=inputs, values=np.column_stack([inputs[:, 0], self.offset + inputs[:, 1]]))


def test_points_go_to_the_routed_evaluator_and_come_back_in_order():
    cheap, dear = Counting(0.0), Counting(100.0)
    evaluate = routed(OBJECTIVES, [cheap, dear], lambda _: np.array([1, SCREENED, 0, 1]))
    x = np.arange(8.0).reshape(4, 2)
    got = evaluate(x)
    assert np.array_equal(got.inputs, x)
    assert [o.value for o in got.point_outcomes()] == ['ok', 'screened', 'ok', 'ok']
    assert got.values[0].tolist() == [0.0, 101.0]
    assert got.values[2].tolist() == [4.0, 5.0]
    assert np.isnan(got.values[1]).all()
    assert [len(s) for s in cheap.seen] == [1] and [len(s) for s in dear.seen] == [2]


def test_an_evaluator_with_no_point_is_not_called():
    cheap, dear = Counting(), Counting()
    routed(OBJECTIVES, [cheap, dear], lambda x: np.zeros(len(x), dtype=int))(np.ones((3, 2)))
    assert dear.seen == []


def test_a_batch_screened_entirely_still_has_the_declared_width():
    got = routed(OBJECTIVES, [Counting()], lambda x: np.full(len(x), SCREENED))(np.ones((2, 2)))
    assert got.values.shape == (2, 2)
    assert all(o is Outcome.SCREENED for o in got.point_outcomes())


def test_screened_produces_no_value():
    assert not Outcome.SCREENED.produced_a_value


@pytest.mark.parametrize(
    ('route', 'match'),
    [
        (np.array([0, 2]), 'between -1 and 0'),
        (np.array([0]), 'one route per point'),
        (np.array([0.5, 0.0]), 'integer'),
    ],
)
def test_a_route_outside_the_declared_evaluators_is_refused(route, match):
    with pytest.raises(Refusal, match=match):
        routed(OBJECTIVES, [Counting()], lambda _: route)(np.ones((2, 2)))


def test_no_evaluator_is_refused():
    with pytest.raises(Refusal, match='at least one evaluator'):
        routed(OBJECTIVES, [], lambda x: x)


def test_an_evaluator_that_reorders_its_points_is_refused():
    def shuffled(inputs):
        return Evaluation(inputs=inputs[::-1], values=np.zeros((len(inputs), 2)))

    with pytest.raises(Refusal, match='points other than'):
        routed(OBJECTIVES, [shuffled], lambda x: np.zeros(len(x), dtype=int))(np.arange(4.0).reshape(2, 2))


def test_an_optimizer_runs_through_a_screening_router_and_never_ranks_a_screened_point():
    space = VariableSet([Variable('x0', 0.0, 1.0), Variable('x1', 0.0, 1.0)])
    expensive = Counting()

    def router(x):
        return np.where(x[:, 0] > 0.5, SCREENED, 0)

    record = optimize(
        space=space,
        objectives=OBJECTIVES,
        evaluate=routed(OBJECTIVES, [expensive], router),
        build_proposer=nsga2(space, seed=1),
        spec=SampleSpec(n_samples=12, seed=1),
        n_batches=4,
    )
    screened = [o is Outcome.SCREENED for o in record.evaluation.point_outcomes()]
    assert any(screened)
    assert sum(len(s) for s in expensive.seen) == len(record) - sum(screened)
    assert (record.pareto().inputs[:, 0] <= 0.5).all()
