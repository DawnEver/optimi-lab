"""The ask/tell loop: one count, one evaluation per batch, and a front of feasible rows only."""

import numpy as np
import pytest

from optimi_lab import (
    Constraint,
    ConstraintSet,
    Evaluation,
    Objective,
    ObjectiveSet,
    Refusal,
    SampleSpec,
    Variable,
    VariableSet,
    optimize,
)

SPACE = VariableSet((Variable('x', 0.0, 1.0), Variable('y', 0.0, 1.0)))
OBJECTIVES = ObjectiveSet((Objective('f0'), Objective('f1')))
LIMIT = ConstraintSet((Constraint('x_cap', '<=', 0.5),))


class Replay:
    """A proposer that asks for a fixed batch and records what it was told."""

    def __init__(self, pop_size: int, batch: np.ndarray | None = None) -> None:
        self.pop_size = pop_size
        self.batch = np.full((pop_size, 2), 0.25) if batch is None else batch
        self.told: list[Evaluation] = []

    def ask(self) -> np.ndarray:
        return self.batch

    def tell(self, evaluation: Evaluation) -> None:
        self.told.append(evaluation)


def _objectives(points: np.ndarray) -> np.ndarray:
    return np.column_stack([points[:, 0], 1.0 - points[:, 0] + points[:, 1]])


def _evaluate(points: np.ndarray) -> Evaluation:
    return Evaluation(points, _objectives(points))


def _evaluate_constrained(points: np.ndarray) -> Evaluation:
    return Evaluation(points, _objectives(points), constraints=points[:, :1])


def test_the_population_size_is_the_sample_row_count_and_every_batch_is_told():
    built: list[Replay] = []

    def build(pop_size: int) -> Replay:
        built.append(Replay(pop_size))
        return built[-1]

    seen: list[int] = []
    record = optimize(
        space=SPACE,
        objectives=OBJECTIVES,
        evaluate=_evaluate,
        build_proposer=build,
        spec=SampleSpec(n_samples=6, seed=2),
        n_batches=3,
        on_batch=lambda run: seen.append(len(run)),
    )
    assert built[0].pop_size == 6
    assert [len(told) for told in built[0].told] == [6, 6, 6]
    assert seen == [6, 12, 18]
    assert len(record) == 18
    assert record.constraints is None


@pytest.mark.parametrize(
    ('evaluate', 'match'),
    [
        (_objectives, 'not an Evaluation'),
        (lambda points: Evaluation(points[::-1], _objectives(points)), 'other than the'),
        (lambda points: Evaluation(points, _objectives(points)[:, :1]), 'objective column'),
        (lambda points: Evaluation(points, _objectives(points), constraints=points[:, :1]), 'constraint column'),
    ],
)
def test_an_evaluator_that_does_not_answer_the_batch_it_was_asked_is_refused(evaluate, match):
    with pytest.raises(Refusal, match=match):
        optimize(space=SPACE, objectives=OBJECTIVES, evaluate=evaluate, build_proposer=Replay, n_batches=1)


def test_a_declared_constraint_the_evaluator_does_not_return_is_refused():
    with pytest.raises(Refusal, match='x_cap'):
        optimize(
            space=SPACE,
            objectives=OBJECTIVES,
            constraints=LIMIT,
            evaluate=_evaluate,
            build_proposer=Replay,
            n_batches=1,
        )


@pytest.mark.parametrize(('batch', 'match'), [(None, 'returned None'), (np.zeros((0, 2)), 'no point')])
def test_an_empty_ask_is_refused(batch, match):
    class Empty(Replay):
        def ask(self):
            return batch

    with pytest.raises(Refusal, match=match):
        optimize(space=SPACE, objectives=OBJECTIVES, evaluate=_evaluate, build_proposer=Empty, n_batches=2)


def test_a_builder_that_returns_no_proposer_and_a_zero_batch_count_are_refused():
    with pytest.raises(Refusal, match='ask'):
        optimize(space=SPACE, objectives=OBJECTIVES, evaluate=_evaluate, build_proposer=lambda _: object())
    with pytest.raises(Refusal, match='n_batches'):
        optimize(space=SPACE, objectives=OBJECTIVES, evaluate=_evaluate, build_proposer=Replay, n_batches=0)


def test_a_proposer_is_told_constraint_shortfalls_and_the_record_keeps_raw_values():
    built: list[Replay] = []

    def build(pop_size: int) -> Replay:
        built.append(Replay(pop_size))
        return built[-1]

    matrix = np.array([[0.25, 0.5], [1.0, 0.5]])
    record = optimize(
        space=SPACE,
        objectives=OBJECTIVES,
        constraints=LIMIT,
        evaluate=_evaluate_constrained,
        build_proposer=build,
        spec=SampleSpec(kind='import', matrix=matrix),
        n_batches=1,
    )
    assert record.constraints is LIMIT
    assert record.evaluation.constraints.tolist() == [[0.25], [1.0]]
    assert np.allclose(built[0].told[0].constraints, [[0.0], [1.0]])


def test_the_front_holds_only_feasible_rows_even_when_an_infeasible_row_dominates_them():
    # The infeasible row (x = 0.9) dominates both feasible rows on both objectives.
    values = {0.4: [0.4, 0.6], 0.9: [0.0, 0.0], 0.1: [0.1, 1.8]}

    def evaluate(points: np.ndarray) -> Evaluation:
        return Evaluation(points, [values[x] for x in points[:, 0]], constraints=points[:, :1])

    record = optimize(
        space=SPACE,
        objectives=OBJECTIVES,
        constraints=LIMIT,
        evaluate=evaluate,
        build_proposer=Replay,
        spec=SampleSpec(kind='import', matrix=np.array([[0.4, 0.0], [0.9, 0.0], [0.1, 0.0]])),
        n_batches=1,
    )
    front = record.pareto()
    assert np.all(front.constraints[:, 0] <= 0.5)
    assert sorted(front.inputs[:, 0].tolist()) == [0.1, 0.4]
    assert record.feasible().inputs[:, 0].tolist() == [0.4, 0.1]


def test_a_run_with_no_feasible_row_has_no_front():
    matrix = np.array([[0.8, 0.1], [0.9, 0.2]])
    record = optimize(
        space=SPACE,
        objectives=OBJECTIVES,
        constraints=LIMIT,
        evaluate=_evaluate_constrained,
        build_proposer=Replay,
        spec=SampleSpec(kind='import', matrix=matrix),
        n_batches=1,
    )
    with pytest.raises(Refusal, match='feasible'):
        record.pareto()


def test_a_row_whose_constraint_did_not_compute_is_never_on_the_front():
    def evaluate(points: np.ndarray) -> Evaluation:
        return Evaluation(points, _objectives(points), constraints=[[np.nan], [0.4]])

    record = optimize(
        space=SPACE,
        objectives=OBJECTIVES,
        constraints=LIMIT,
        evaluate=evaluate,
        build_proposer=Replay,
        spec=SampleSpec(kind='import', matrix=np.array([[0.0, 0.0], [0.4, 0.9]])),
        n_batches=1,
    )
    assert record.pareto().inputs.tolist() == [[0.4, 0.9]]
