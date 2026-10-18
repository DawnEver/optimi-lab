"""Every population solver ranks by Deb's rule when constraints exist, and by Pareto alone when none do.

The problem is ZDT1 with ``x >= 0.8``: its unconstrained front spans ``x`` in ``[0, 1]``, so a
solver blind to the constraint keeps spreading most of its population over the forbidden 80%,
while one that ranks feasible points first moves its batches into the allowed strip.
"""

from pathlib import Path

import numpy as np
import pytest
from lab_commons.outcomes import Record

from optimi_lab.core import SampleSpec
from optimi_lab.operator import crowding_select, reference_select, tournament_select
from optimi_lab.oracle import Action, Capability, from_function
from optimi_lab.problem import BlackBox, Objective, Problem, Real, Space
from optimi_lab.run import Budget, run
from optimi_lab.solver import mode, moead, mopso, nsga2, nsga3, spea2, unsga3

ALGORITHMS = (nsga2, nsga3, unsga3, spea2, mode, moead, mopso)
SPACE = Space(tuple(Real(f'x{index}', 0.0, 1.0) for index in range(3)))
STRIP = (BlackBox('x0', '>=', 0.8),)
SLACK = (BlackBox('x0', '<=', 10.0),)


def _zdt1(action: Action) -> dict[str, float]:
    x = np.array([action.design[f'x{i}'] for i in range(3)])
    g = 1.0 + 9.0 * x[1:].sum() / 2
    return {'f0': x[0], 'f1': g * (1 - np.sqrt(x[0] / g)), 'x0': x[0]}


def _run(factory, constraints, path: Path, n_batches: int = 25, seed: int = 3):
    problem = Problem(SPACE, (Objective('f0'), Objective('f1')), constraints=constraints)
    outputs = ('f0', 'f1', 'x0') if constraints else ('f0', 'f1')
    oracle = from_function(lambda a: {k: v for k, v in _zdt1(a).items() if k in outputs}, Capability(outputs=outputs))
    solver = factory(problem, oracle.capability, spec=SampleSpec(n_samples=20, seed=seed), seed=seed)
    batches: list[list] = []
    tell = solver.tell
    solver.tell = lambda results: (batches.append(list(results)), tell(results))
    return run(problem, oracle, solver, Record(path), Budget(evaluations=20 * n_batches)), batches


@pytest.mark.parametrize('factory', ALGORITHMS)
def test_a_constraint_that_never_binds_moves_no_point(factory, tmp_path):
    """The planted control: zero violation everywhere is the unconstrained run, design for design."""
    free, _ = _run(factory, (), tmp_path / 'free.jsonl', n_batches=8)
    slack, _ = _run(factory, SLACK, tmp_path / 'slack.jsonl', n_batches=8)
    assert [dict(r.action.design) for r in free.archive.results()] == [
        dict(r.action.design) for r in slack.archive.results()
    ]


@pytest.mark.parametrize('factory', ALGORITHMS)
def test_a_binding_constraint_moves_the_last_batch_into_the_feasible_strip(factory, tmp_path):
    outcome, batches = _run(factory, STRIP, tmp_path / 'strip.jsonl')
    last = [r.action.design['x0'] for r in [batch for batch in batches if len(batch) == 20][-1]]
    assert np.mean(np.array(last) >= 0.8) >= 0.5
    assert all(r.action.design['x0'] >= 0.8 for r in outcome.archive.front())


def test_a_survivor_selection_keeps_the_feasible_rows_before_any_infeasible_one():
    values = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0], [0.1, 0.1]])
    violation = np.array([0.5, 0.0, 0.0, 0.2])
    assert sorted(crowding_select(values, 2, violation=violation).tolist()) == [1, 2]
    assert sorted(reference_select(values, 3, violation=violation).tolist()) == [1, 2, 3]


def test_a_tournament_between_a_feasible_and_an_infeasible_row_goes_to_the_feasible_one():
    values = np.array([[0.0, 0.0], [5.0, 5.0]])
    winners = tournament_select(values, 2, np.random.default_rng(0), violation=np.array([0.1, 0.0]))
    assert winners.tolist() == [1, 1]
