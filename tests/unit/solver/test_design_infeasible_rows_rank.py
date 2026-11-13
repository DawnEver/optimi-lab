"""A refused design that breaks a DESIGN constraint is ranked by its violation, never dropped unscored.

The planted box is mostly infeasible: an ``Explicit`` ball of radius 0.03 around ``(0.9, 0.9, 0.9)`` in the unit
cube (about 0.01 % of it), and the oracle refuses every design outside it before any solve -- the shape of a
manufacturing screen. A solver that drops every non-``OK`` row has nothing to breed from and only redraws the
box; one that ranks the refused rows by their design violation (Deb's rule) walks into the ball.
"""

from pathlib import Path

import numpy as np
import pytest
from lab_commons.outcomes import Cell, Reason, Record, Status

from optimi_lab.core import SampleSpec, constrained_non_dominated_sorting
from optimi_lab.operator import crowding_select
from optimi_lab.oracle import Action, Capability, Infeasible, Result, from_function
from optimi_lab.problem import BlackBox, Explicit, Objective, Problem, Real, Space
from optimi_lab.run import Budget, run
from optimi_lab.solver import ga, nsga2

SPACE = Space(tuple(Real(f'x{index}', 0.0, 1.0) for index in range(3)))
CENTRE, RADIUS = 0.9, 0.03
OUTPUTS = ('f0', 'f1', 'cap')


def _distance(design) -> float:
    return float(np.sqrt(sum((float(design[f'x{i}']) - CENTRE) ** 2 for i in range(3))))


BALL = Explicit('ball', _distance, '<=', RADIUS)
#: An output constraint too: a refused row has no reading of it, so its rank is its design violation alone.
CAP = BlackBox('cap', '<=', 10.0)


def _problem(objectives: int, *, declared: bool = True) -> Problem:
    constraints = (BALL, CAP) if declared else (CAP,)
    return Problem(SPACE, tuple(Objective(f'f{k}') for k in range(objectives)), constraints=constraints)


def _run(factory, objectives: int, path: Path, *, declared: bool = True, seed: int = 1, batches: int = 20):
    problem = _problem(objectives, declared=declared)

    def solve(action: Action) -> dict[str, float]:
        if _distance(action.design) > RADIUS:
            msg = 'outside the ball: refused before any solve'
            raise Infeasible(msg)
        x = [float(action.design[f'x{i}']) for i in range(3)]
        read = {'f0': x[0] + x[1] + x[2], 'f1': -x[0], 'cap': x[0]}
        return {name: read[name] for name in problem.outputs}

    oracle = from_function(solve, Capability(outputs=problem.outputs))
    solver = factory(problem, oracle.capability, spec=SampleSpec(n_samples=20, seed=seed), seed=seed)
    return run(problem, oracle, solver, Record(path), Budget(evaluations=20 * batches))


@pytest.mark.parametrize(('factory', 'objectives'), [(ga, 1), (nsga2, 2)])
def test_a_mostly_infeasible_box_is_walked_into_by_ranking_the_refused_rows(factory, objectives, tmp_path):
    outcome = _run(factory, objectives, tmp_path / 'ranked.jsonl')
    assert outcome.archive.feasible(), 'no design inside the ball was ever solved'


@pytest.mark.parametrize(('factory', 'objectives'), [(ga, 1), (nsga2, 2)])
def test_the_control_with_the_ball_undeclared_drops_every_refusal_and_solves_nothing(factory, objectives, tmp_path):
    """The planted control: the same refusals with no design constraint to rank them by are dropped unscored, the
    population is never seeded, and the same budget only redraws the box -- the old behaviour.
    """
    outcome = _run(factory, objectives, tmp_path / 'blind.jsonl', declared=False)
    assert not outcome.archive.complete()


def test_a_refused_row_carries_its_design_violation_and_reads_no_output():
    problem = _problem(1)
    action = Action.of(problem, Capability(outputs=OUTPUTS), {'x0': 0.0, 'x1': 0.9, 'x2': 0.9})
    refused = Result(action, Cell(Status.REJECTED, {}, reason=Reason('infeasible', 'screened')))
    assert refused.design_violation(problem) == pytest.approx((0.9 - RADIUS) / BALL.scale)


def test_a_refused_row_inside_every_design_constraint_is_still_dropped_unranked():
    """Nothing about its design is wrong, so nothing ranks it: a refusal for another reason scores nothing."""
    problem = _problem(1)
    action = Action.of(problem, Capability(outputs=OUTPUTS), {'x0': 0.9, 'x1': 0.9, 'x2': 0.9})
    refused = Result(action, Cell(Status.REJECTED, {}, reason=Reason('invalid', 'unbuildable')))
    assert refused.design_violation(problem) == 0.0


def test_an_evaluated_feasible_row_outranks_every_refused_row_even_with_the_worst_objective():
    """The refused rows carry the generation's WORST scored value; Deb's rule never reads it -- the solved feasible
    design wins on feasibility, and the refused ones follow ordered by their violation.
    """
    problem = _problem(1)
    designs = (
        {'x0': 0.0, 'x1': 0.0, 'x2': 0.0},  # far outside the ball
        {'x0': CENTRE, 'x1': CENTRE, 'x2': CENTRE},  # inside: solved, feasible
        {'x0': 0.8, 'x1': CENTRE, 'x2': CENTRE},  # just outside
    )
    solver = ga(problem, Capability(outputs=OUTPUTS), initial=np.array([problem.space.encode(d) for d in designs]))
    admitted = []
    admit = solver._admit
    solver._admit = lambda *rows: (admitted.append(rows), admit(*rows))
    results = []
    for action in solver.ask(len(designs)):
        if _distance(action.design) > RADIUS:
            cell = Cell(Status.REJECTED, {}, reason=Reason('infeasible', 'screened'))
        else:
            cell = Cell(Status.OK, {'f0': 3 * CENTRE, 'cap': CENTRE})
        results.append(Result(action, cell))
    solver.tell(results)
    _, values, violation, mask = admitted[0]
    assert mask.all()
    shortfall = [(_distance(d) - RADIUS) / BALL.scale if _distance(d) > RADIUS else 0.0 for d in designs]
    assert violation.tolist() == pytest.approx(shortfall)
    assert values[:, 0].tolist() == [values[1, 0]] * 3  # the placeholder is the one scored row's (worst) value
    assert crowding_select(values, 1, violation=violation).tolist() == [1]
    order = constrained_non_dominated_sorting(values, violation, only_first_front=False)
    assert [front.tolist() for front in order] == [[1], [2], [0]]
