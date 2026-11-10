"""An Explicit design constraint may declare its exact gradient; a gradient method then differences nothing.

``min (a - 2)^2 + (b - 2)^2`` over ``[0, 3]^2`` s.t. the Explicit ``a^2 + b^2 <= 2``: the optimum is
``(1, 1)``. The constraint function records every design it is called at, so a finite difference shows
up as a call at a design the run never evaluated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from lab_commons.outcomes import Cell, Record, Status

from optimi_lab.archive import Archive
from optimi_lab.core.errors import Refusal
from optimi_lab.oracle import Action, Capability, Information, Result, from_function
from optimi_lab.problem import Explicit, Objective, Problem, Real, Relation, Space
from optimi_lab.run import Budget, run
from optimi_lab.solver import GradientSolver, Halt, check_explicit_gradient

SPACE = Space((Real('a', 0.0, 3.0), Real('b', 0.0, 3.0)))
START = {'a': 0.5, 'b': 0.2}


class Disk:
    """``a^2 + b^2`` as :attr:`function`, remembering each design it was called at."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, float]] = []

        def disk(design) -> float:
            self.calls.append((float(design['a']), float(design['b'])))
            return float(design['a']) ** 2 + float(design['b']) ** 2

        self.function = disk


def _exact(design) -> dict[str, float]:
    return {'a': 2.0 * float(design['a']), 'b': 2.0 * float(design['b'])}


def _problem(disk: Disk, gradient=None) -> Problem:
    constraint = Explicit('disk', disk.function, Relation.AT_MOST, 2.0, gradient=gradient)
    return Problem(SPACE, (Objective('f'),), constraints=(constraint,))


def _oracle():
    def evaluate(action: Action) -> Result:
        a, b = float(action.design['a']), float(action.design['b'])
        cell = Cell(Status.OK, values={'f': (a - 2.0) ** 2 + (b - 2.0) ** 2})
        if Information.VJP not in action.want:
            return Result(action, cell)
        return Result(action, cell, gradient={'f': 3.0 * np.array([2 * (a - 2.0), 2 * (b - 2.0)])})  # d x / d u = 3

    provides = frozenset({Information.VALUES, Information.VJP})
    return from_function(evaluate, Capability(outputs=('f',), provides=provides))


def _drive(tmp_path: Path, problem: Problem) -> tuple[GradientSolver, set[tuple[float, float]]]:
    oracle = _oracle()
    solver = GradientSolver(problem, oracle.capability, 'sqp', start=START)
    run(problem, oracle, solver, Record(tmp_path / 'r.jsonl'), Budget(evaluations=500))
    archive = Archive(problem, Record(tmp_path / 'r.jsonl'))
    evaluated = {(float(r.action.design['a']), float(r.action.design['b'])) for r in archive.results()}
    return solver, evaluated


def test_an_exact_constraint_gradient_is_used_and_no_design_is_differenced(tmp_path: Path) -> None:
    disk = Disk()
    solver, evaluated = _drive(tmp_path, _problem(disk, _exact))
    assert solver.solution.halt is Halt.CONVERGED, solver.solution.reason
    found = solver.solution.x * 3.0
    np.testing.assert_allclose(found, (1.0, 1.0), atol=1e-5)
    assert disk.calls
    assert set(disk.calls) <= evaluated  # every call at an evaluated design: no differencing step


def test_planted_control_without_a_gradient_the_constraint_is_differenced_off_the_evaluated_designs(
    tmp_path: Path,
) -> None:
    disk = Disk()
    solver, evaluated = _drive(tmp_path, _problem(disk))
    np.testing.assert_allclose(solver.solution.x * 3.0, (1.0, 1.0), atol=1e-5)
    assert set(disk.calls) - evaluated


def test_a_gradient_check_holds_the_exact_gradient_against_differences_and_catches_a_wrong_one() -> None:
    design = {'a': 0.7, 'b': 1.3}
    assert check_explicit_gradient(_problem(Disk(), _exact), design)['disk'] < 1e-8
    wrong = _problem(Disk(), lambda d: {'a': 2.0 * float(d['a']), 'b': float(d['b'])})
    assert check_explicit_gradient(wrong, design)['disk'] > 0.1
    with pytest.raises(Refusal, match='nothing to check'):
        check_explicit_gradient(_problem(Disk()), design)


def test_a_gradient_over_an_unknown_variable_or_of_the_wrong_width_is_refused() -> None:
    for gradient, message in ((lambda _: {'c': 1.0}, 'not design variables'), (lambda _: {'a': [1.0, 2.0]}, 'width')):
        with pytest.raises(Refusal, match=message):
            check_explicit_gradient(_problem(Disk(), gradient), {'a': 0.7, 'b': 1.3})
    with pytest.raises(Refusal, match='callable'):
        Explicit('disk', Disk().function, Relation.AT_MOST, 2.0, gradient=1.0)  # type: ignore[arg-type]
