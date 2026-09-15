"""The five proposers against a trivial two-objective problem, and the declarations they owe.

Every algorithm is run through :func:`optimi_lab.core.optimize.optimize` -- the path the user
takes -- and the per-batch sizes are traced, because the defect being ratcheted here (a first
generation that is the whole sample grid rather than the population) is only visible in the
sizes a run actually asks for.
"""

import inspect
import pathlib

import numpy as np
import pytest

import optimi_lab.intelligent_algorithm as package
from optimi_lab import (
    Evaluation,
    Objective,
    ObjectiveSet,
    Outcome,
    Refusal,
    SampleSpec,
    Variable,
    VariableSet,
    optimize,
)
from optimi_lab.intelligent_algorithm import mode, moead, mopso, nsga2, nsga3

ALGORITHMS = (nsga2, nsga3, mode, moead, mopso)
SPACE = VariableSet([Variable('x', 0.0, 1.0), Variable('y', 0.0, 1.0)])
OBJECTIVES = ObjectiveSet([Objective('to_origin'), Objective('to_one')])
MODULES = sorted(pathlib.Path(package.__file__).parent.glob('*.py'))


def evaluate(points: np.ndarray) -> Evaluation:
    """Two quadratic bowls, so the Pareto front is the quarter circle from ``(1, 0)`` to ``(0, 1)``."""
    values = np.column_stack([np.sum(points**2, axis=1), np.sum((points - 1.0) ** 2, axis=1)])
    return Evaluation(inputs=points, values=values)


def run(factory, objectives: ObjectiveSet = OBJECTIVES, n_samples: int = 8, n_batches: int = 3, **settings):
    """Drive one algorithm through the driver, with the evaluator matching the declared objectives."""

    def scoring(points: np.ndarray) -> Evaluation:
        values = np.zeros((points.shape[0], len(objectives)))
        for column in range(len(objectives)):
            values[:, column] = np.sum((points - column) ** 2, axis=1)
        return Evaluation(inputs=points, values=values)

    return optimize(
        space=SPACE,
        objectives=objectives,
        evaluate=scoring,
        build_proposer=factory(SPACE, **settings),
        spec=SampleSpec(n_samples=n_samples, seed=1),
        n_batches=n_batches,
    )


@pytest.mark.parametrize('factory', ALGORITHMS)
def test_every_algorithm_finds_a_front_of_points_that_produced_values(factory):
    """Construct, minimise, and assert a non-empty front whose every outcome is ``OK``."""
    record = run(factory)

    assert [outcome.value for outcome in record.evaluation.outcomes.ravel()] == ['ok'] * len(record) * 2
    front = record.pareto()
    assert len(front) > 0
    assert all(outcome is Outcome.OK for outcome in front.outcomes.ravel())


@pytest.mark.parametrize('factory', ALGORITHMS)
@pytest.mark.parametrize('n_samples', [8, 9])
def test_every_batch_holds_the_population_the_sample_drew(factory, n_samples):
    """One count, not two: the sample's row count IS the population, whatever the sampler returned.

    The old run took its first batch from ``variable_space.var_space_matrix`` and stepped a
    population of its own size, measured ``[121, 8, 8, 8]`` for ``pop_size=8``; a grid that is not
    the population (9 rows here) also broadcast an 8-particle swarm against it.
    """
    sizes = []

    def tracing(points: np.ndarray) -> Evaluation:
        sizes.append(points.shape[0])
        return evaluate(points)

    record = optimize(
        space=SPACE,
        objectives=OBJECTIVES,
        evaluate=tracing,
        build_proposer=factory(SPACE),
        spec=SampleSpec(n_samples=n_samples, seed=1),
        n_batches=4,
    )

    assert sizes == [n_samples] * 4, 'every batch must be the count the sample set'
    assert len(record) == n_samples * 4


@pytest.mark.parametrize('factory', ALGORITHMS)
def test_a_point_that_produced_no_value_is_dropped_and_never_scored(factory):
    """A point that did not compute leaves the population; it never enters a ranking as a number."""
    calls = []

    def flaky(points: np.ndarray) -> Evaluation:
        values = evaluate(points).values
        outcomes = np.full(values.shape, Outcome.OK, dtype=object)
        outcomes[0] = Outcome.ERROR
        calls.append(points.shape[0])
        return Evaluation(inputs=points, values=values, outcomes=outcomes)

    record = optimize(
        space=SPACE,
        objectives=OBJECTIVES,
        evaluate=flaky,
        build_proposer=factory(SPACE),
        spec=SampleSpec(n_samples=8, seed=1),
        n_batches=3,
    )

    assert calls[0] == 8, 'the first batch is the sample the driver drew'
    assert len(calls) == 3 and min(calls) >= 1, 'the run continues after a point failed'
    assert any(outcome is Outcome.ERROR for outcome in record.evaluation.outcomes.ravel())
    front = record.pareto()
    assert len(front) > 0
    assert all(np.isfinite(value) for value in front.values.ravel())


@pytest.mark.parametrize('factory', ALGORITHMS)
def test_a_proposer_refuses_to_ask_before_it_is_told(factory):
    """The first batch is the driver's, so a proposer asked to invent one refuses and says why."""
    with pytest.raises(Refusal, match=r'ask\(\) was called before tell'):
        factory(SPACE)(4).ask()


@pytest.mark.parametrize('factory', ALGORITHMS)
def test_selection_operator_is_spelled_the_same_by_every_algorithm(factory):
    """Every algorithm that takes an environmental selector calls it ``selection_operator``."""
    parameters = inspect.signature(factory).parameters

    assert 'select_operator' not in parameters, 'MODE spelled it `select_operator` once; only one spelling survives'
    if factory is not moead:
        assert 'selection_operator' in parameters


def test_one_objective_is_refused_by_the_methods_that_need_two():
    """MOEA/D has nothing to decompose and NSGA-III no grid to refine, so both refuse and name it."""
    single = ObjectiveSet([Objective('to_origin')])

    for factory in (moead, nsga3):
        with pytest.raises(Refusal, match='at least 2 objectives'):
            run(factory, objectives=single, n_batches=2)


def test_a_setting_the_algorithm_does_not_take_is_refused_with_the_valid_set():
    """A misspelled setting is a refusal naming the ones accepted, not a ``TypeError`` later."""
    with pytest.raises(Refusal) as refusal:
        nsga2(SPACE, crossover_rat=0.5)

    assert 'crossover_rat' in str(refusal.value)
    assert 'crossover_rate' in str(refusal.value)


def test_the_package_holds_its_own_declarations():
    """Seven modules, each naming its exports first, staying under 400 lines, suppressing nothing."""
    expected = [
        '__init__.py',
        'algorithms.py',
        'decomposition.py',
        'proposers.py',
        'selection.py',
        'swarm.py',
        'variation.py',
    ]
    assert [module.name for module in MODULES] == expected

    for module in MODULES:
        lines = module.read_text(encoding='utf-8').splitlines()
        assert len(lines) <= 400, f'{module.name} is {len(lines)} lines; the ceiling is 400'
        first_definition = next(
            (index for index, line in enumerate(lines) if line.startswith(('def ', 'class '))), len(lines)
        )
        assert any(line.startswith('__all__') for line in lines[:first_definition]), (
            f'{module.name} declares no __all__ first'
        )
        for suppression in ('noqa', 'type: ignore', 'pragma: no cover'):
            assert not any(suppression in line for line in lines), (
                f'{module.name} suppresses a finding with {suppression}'
            )


def test_every_public_name_of_the_subpackage_is_reachable():
    assert sorted(package.__all__) == package.__all__
    assert len(package.__all__) == len(set(package.__all__))
    for name in package.__all__:
        assert getattr(package, name, None) is not None, f'{name} is declared and missing'
