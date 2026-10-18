"""The population solvers against a trivial two-objective problem, and the declarations they owe.

Every algorithm is driven by :func:`optimi_lab.run.run` -- the path the user takes -- and the
per-batch sizes are traced, because the defect being ratcheted here (a first generation that is the
whole sample grid rather than the population) is only visible in the sizes a run actually asks for.
"""

import asyncio
import inspect

import numpy as np
import pytest
from lab_commons.outcomes import Cell, Reason, Record, Status

from optimi_lab.core import Refusal, SampleSpec
from optimi_lab.oracle import Action, Capability, Result, from_function
from optimi_lab.problem import Objective, Problem, Real, Space
from optimi_lab.run import Budget, run
from optimi_lab.solver import mode, moead, mopso, nsga2, nsga3, spea2, unsga3

ALGORITHMS = (nsga2, nsga3, unsga3, spea2, mode, moead, mopso)
SPACE = Space((Real('x', 0.0, 1.0), Real('y', 0.0, 1.0)))
TWO = (Objective('to_origin'), Objective('to_one'))


def _bowls(action: Action) -> dict[str, float]:
    """Two quadratic bowls, so the Pareto front is the quarter circle from ``(1, 0)`` to ``(0, 1)``."""
    p = np.array([action.design['x'], action.design['y']])
    return {'to_origin': float(np.sum(p**2)), 'to_one': float(np.sum((p - 1.0) ** 2))}


def _drive(factory, path, *, objectives=TWO, n_samples=8, n_batches=3, evaluate=_bowls, **settings):
    problem = Problem(SPACE, objectives)
    names = tuple(o.name for o in objectives)
    oracle = from_function(
        lambda a: {k: v for k, v in dict(evaluate(a)).items() if k in names}, Capability(outputs=names)
    )
    solver = factory(problem, oracle.capability, spec=SampleSpec(n_samples=n_samples, seed=1), **settings)
    return run(problem, oracle, solver, Record(path), Budget(evaluations=n_samples * n_batches))


@pytest.mark.parametrize('factory', ALGORITHMS)
def test_every_algorithm_finds_a_front_of_points_that_produced_values(factory, tmp_path):
    outcome = _drive(factory, tmp_path / 'r.jsonl')
    assert all(r.cell.status is Status.OK for r in outcome.archive.results())
    assert len(outcome.archive.front()) > 0


@pytest.mark.parametrize('factory', ALGORITHMS)
@pytest.mark.parametrize('n_samples', [8, 9])
def test_every_batch_holds_the_population_the_sample_drew(factory, n_samples):
    """One count, not two: the sample's row count IS the population, whatever the sampler returned."""
    problem = Problem(SPACE, TWO)
    oracle = from_function(_bowls, Capability(outputs=('to_origin', 'to_one')))
    solver = factory(problem, oracle.capability, spec=SampleSpec(n_samples=n_samples, seed=1))
    sizes = []
    for _ in range(4):
        asked = solver.ask(10_000)
        sizes.append(len(asked))
        solver.tell([asyncio.run(oracle(a)) for a in asked])
    assert sizes == [n_samples] * 4, 'every generation must be the count the sample set'


@pytest.mark.parametrize('factory', ALGORITHMS)
def test_a_point_that_produced_no_value_is_dropped_and_never_scored(factory, tmp_path):
    """A point that did not compute leaves the population; it never enters a ranking as a number."""
    seen = []

    def flaky(action: Action) -> dict[str, float] | Result:
        seen.append(action)
        if len(seen) % 8 == 1:
            return Result(action, Cell(Status.ERROR, reason=Reason('error', 'planted')))
        return _bowls(action)

    outcome = _drive(factory, tmp_path / 'r.jsonl', evaluate=flaky)
    assert any(r.cell.status is Status.ERROR for r in outcome.archive.results())
    assert outcome.ledger.evaluations == 24, 'the run continues after a point failed'
    front = outcome.archive.front()
    assert front and all(r.cell.status is Status.OK for r in front)


@pytest.mark.parametrize('factory', ALGORITHMS)
def test_selection_operator_is_spelled_the_same_by_every_algorithm(factory):
    """Every algorithm that takes an environmental selector calls it ``selection_operator``."""
    parameters = inspect.signature(factory).parameters

    assert 'select_operator' not in parameters, 'MODE spelled it `select_operator` once; only one spelling survives'
    if factory in (nsga2, nsga3, mode, mopso):
        assert 'selection_operator' in parameters


def test_one_objective_is_refused_by_the_methods_that_need_two(tmp_path):
    """MOEA/D has nothing to decompose and NSGA-III no grid to refine, so both refuse and name it."""
    for factory in (moead, nsga3, unsga3):
        with pytest.raises(Refusal, match='at least 2 objectives'):
            _drive(factory, tmp_path / f'{factory.__name__}.jsonl', objectives=(Objective('to_origin'),), n_batches=2)


def test_a_setting_the_algorithm_does_not_take_is_refused_with_the_valid_set():
    """A misspelled setting is a refusal naming the ones accepted, not a ``TypeError`` later."""
    with pytest.raises(Refusal) as refusal:
        nsga2(Problem(SPACE, TWO), Capability(outputs=('to_origin', 'to_one')), crossover_rat=0.5)

    assert 'crossover_rat' in str(refusal.value)
    assert 'crossover_rate' in str(refusal.value)


def test_a_moead_child_is_scored_under_each_neighbours_own_weight():
    """A child displaces neighbour j when it beats j's incumbent on j's OWN Tchebycheff weight.

    The child (0, 1) is proposed for subproblem 2 (weight 0.1/0.9), where it loses; under subproblem
    0's weight (0.9/0.1) it scores 0.1 against the incumbent's 0.45 and must take that slot. Scored
    once, under the proposer's weight, it would win nowhere -- the MOEA/D-on-DTLZ2 stall P12 left open.
    """
    problem = Problem(Space((Real('x', 0.0, 1.0),)), TWO)
    solver = moead(problem, Capability(outputs=('f', 'g')), spec=SampleSpec(n_samples=3, seed=1), seed=1)
    solver._weights = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])
    solver._neighbors = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]])
    solver._ideal = np.zeros(2)
    solver._population = np.array([[0.5], [0.5], [0.5]])
    solver._values = np.full((3, 2), 0.5)
    solver._violation = np.zeros(3)
    solver._replace(np.array([[0.0]]), np.array([[0.0, 1.0]]), np.zeros(1), np.array([2]))
    assert solver._values.tolist() == [[0.0, 1.0], [0.5, 0.5], [0.5, 0.5]]
