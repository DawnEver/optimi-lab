"""A global search hands its best designs to a gradient method: ``memetic`` (population -> local refinement).

The function has two basins; the shallow one holds the centre of the box, where a gradient method started
at the default ``0.5`` settles. The global stage finds the deep basin, the refinement lands on its minimum
to the method's tolerance -- which neither stage reaches alone within the same budget.
"""

from __future__ import annotations

import numpy as np
import pytest
from lab_commons.outcomes import Cell, Record, Status

from optimi_lab.core import SampleSpec
from optimi_lab.core.errors import Refusal
from optimi_lab.oracle import Action, Capability, Information, Result, from_function
from optimi_lab.problem import Objective, Problem, Real, Space
from optimi_lab.run import Budget, run
from optimi_lab.solver import GradientSolver, ga, memetic

SPACE = Space((Real('x', -2.0, 2.0),))
PROBLEM = Problem(SPACE, (Objective('f'),))
DEEP = -1.5  # the global minimiser: f = (x + 1.5)^2 (x - 1)^2 - 0.5 exp(-(x + 1.5)^2)


def _f(x: float) -> float:
    return (x - DEEP) ** 2 * (x - 1.0) ** 2 - 0.5 * np.exp(-((x - DEEP) ** 2))


def _df(x: float) -> float:
    return 2 * (x - DEEP) * (x - 1.0) ** 2 + 2 * (x - 1.0) * (x - DEEP) ** 2 + (x - DEEP) * np.exp(-((x - DEEP) ** 2))


def _oracle():
    def evaluate(action: Action) -> Result:
        x = float(action.design['x'])
        cell = Cell(Status.OK, values={'f': _f(x)})
        if Information.VJP not in action.want:
            return Result(action, cell)
        return Result(action, cell, gradient={'f': np.array([4.0 * _df(x)])})  # d x / d unit = 4

    return from_function(
        evaluate, Capability(outputs=('f',), provides=frozenset({Information.VALUES, Information.VJP}))
    )


def test_the_refinement_starts_from_the_global_stage_best_and_lands_on_the_deep_minimum(tmp_path):
    oracle = _oracle()
    explore = ga(PROBLEM, oracle.capability, spec=SampleSpec(n_samples=8, seed=3))
    solver = memetic(
        PROBLEM,
        oracle.capability,
        explore=explore,
        evaluations=24,
        refine=lambda p, c, start: GradientSolver(p, c, 'lbfgsb', start=start),
    )
    outcome = run(PROBLEM, oracle, solver, Record(tmp_path / 'r.jsonl'), Budget(evaluations=200))
    assert solver.done
    (local,) = solver.finished[1:]
    assert abs(local.design()['x'] - DEEP) < 1e-4
    assert outcome.ledger.evaluations < 200


def test_polish_refines_that_many_of_the_best_designs_one_after_another(tmp_path):
    oracle = _oracle()
    solver = memetic(
        PROBLEM,
        oracle.capability,
        explore=ga(PROBLEM, oracle.capability, spec=SampleSpec(n_samples=8, seed=3)),
        evaluations=16,
        refine=lambda p, c, start: GradientSolver(p, c, 'lbfgsb', start=start),
        polish=3,
    )
    run(PROBLEM, oracle, solver, Record(tmp_path / 'r.jsonl'), Budget(evaluations=400))
    assert len(solver.finished) == 1 + 3


def test_an_exploration_with_no_complete_design_is_refused_rather_than_refining_from_nothing(tmp_path):
    def broken(action: Action) -> Result:
        return Result(action, Cell(Status.ERROR))

    oracle = from_function(
        broken, Capability(outputs=('f',), provides=frozenset({Information.VALUES, Information.VJP}))
    )
    solver = memetic(
        PROBLEM,
        oracle.capability,
        explore=ga(PROBLEM, oracle.capability, spec=SampleSpec(n_samples=4, seed=0)),
        evaluations=4,
        refine=lambda p, c, start: GradientSolver(p, c, 'lbfgsb', start=start),
    )
    with pytest.raises(Refusal, match='no complete design'):
        run(PROBLEM, oracle, solver, Record(tmp_path / 'r.jsonl'), Budget(evaluations=50))
