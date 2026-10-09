"""A steer retunes the proposer between batches -- the search's own settings, decided as it runs.

``optimize(steer=...)`` is handed the run so far after every batch but the last and returns the
settings the NEXT generation uses, or nothing. The proposer is the one that knows which settings
exist, so a setting it does not take is refused naming the ones it does.
"""

import numpy as np
import pytest

from optimi_lab import (
    Evaluation,
    Objective,
    ObjectiveSet,
    Refusal,
    SampleSpec,
    Tunable,
    Variable,
    VariableSet,
    optimize,
)
from optimi_lab.intelligent_algorithm import mode, moead, nsga2

SPACE = VariableSet([Variable('x0', 0.0, 1.0), Variable('x1', 0.0, 1.0)])
OBJECTIVES = ObjectiveSet([Objective('f1', 'minimize'), Objective('f2', 'minimize')])


def _evaluate(inputs: np.ndarray) -> Evaluation:
    return Evaluation(inputs=inputs, values=np.column_stack([inputs[:, 0], 1.0 + inputs[:, 1] - inputs[:, 0]]))


def _run(build, steer, n_batches: int = 4):
    return optimize(
        space=SPACE,
        objectives=OBJECTIVES,
        evaluate=_evaluate,
        build_proposer=build,
        spec=SampleSpec(n_samples=8, seed=0),
        n_batches=n_batches,
        steer=steer,
    )


def test_the_steer_sees_every_batch_but_the_last() -> None:
    seen = []
    _run(nsga2(SPACE, seed=0), lambda record: seen.append(len(record)))
    assert seen == [8, 16, 24]


def test_a_steer_that_returns_nothing_is_the_unsteered_run() -> None:
    plain = _run(nsga2(SPACE, seed=0), None)
    steered = _run(nsga2(SPACE, seed=0), lambda _record: None)
    assert np.array_equal(plain.evaluation.inputs, steered.evaluation.inputs)


def test_a_retuned_setting_changes_the_next_generation() -> None:
    plain = _run(nsga2(SPACE, seed=0), None)
    frozen = _run(nsga2(SPACE, seed=0), lambda _record: {'mutation_rate': 0.0, 'crossover_rate': 0.0})
    assert np.array_equal(plain.evaluation.inputs[:8], frozen.evaluation.inputs[:8])
    assert not np.array_equal(plain.evaluation.inputs[8:], frozen.evaluation.inputs[8:])


def test_no_variation_leaves_every_child_a_parent() -> None:
    record = _run(nsga2(SPACE, seed=0), lambda _record: {'mutation_rate': 0.0, 'crossover_rate': 0.0}, n_batches=2)
    parents = {tuple(row) for row in record.evaluation.inputs[:8]}
    assert {tuple(row) for row in record.evaluation.inputs[8:]} <= parents


def test_an_unknown_setting_is_refused_naming_the_valid_ones() -> None:
    with pytest.raises(Refusal, match=r'eta_crossover'):
        _run(nsga2(SPACE, seed=0), lambda _record: {'temperature': 2.0})


def test_mode_retunes_its_own_settings() -> None:
    record = _run(mode(SPACE, seed=0), lambda _record: {'sizing_factor': 0.9})
    assert len(record) == 32
    with pytest.raises(Refusal, match=r'sizing_factor'):
        _run(mode(SPACE, seed=0), lambda _record: {'eta_mutation': 5.0})


def test_a_proposer_that_cannot_be_retuned_is_refused_by_name() -> None:
    with pytest.raises(Refusal, match=r'DecompositionProposer'):
        _run(moead(SPACE, seed=0), lambda _record: {'mutation_rate': 0.5})


def test_an_evolutionary_proposer_is_tunable() -> None:
    assert isinstance(nsga2(SPACE, seed=0)(8), Tunable)
    assert not isinstance(moead(SPACE, seed=0)(8), Tunable)
