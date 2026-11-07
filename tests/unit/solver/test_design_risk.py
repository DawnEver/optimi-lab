"""A robust run read per design: grouped scenario results through the solver's own estimator."""

from __future__ import annotations

import pytest
from lab_commons.outcomes import Cell, Reason, Status

from optimi_lab.core import Refusal
from optimi_lab.operator import Risk
from optimi_lab.oracle import Action, Capability, Result
from optimi_lab.problem import BlackBox, Normalization, Objective, Problem, Real, Sense, Space, Uncertainty, Uniform
from optimi_lab.solver import DesignRisk, design_front, design_risk

PROBLEM = Problem(
    Space((Real('x', 0.0, 1.0),)),
    (Objective('f'), Objective('g', Sense.MAXIMIZE, normalization=Normalization(reference=1.0, scale=2.0))),
    constraints=(BlackBox('c', '<=', 0.0),),
    uncertainty=Uncertainty((Uniform('xi', -1.0, 1.0),)),
)
CAPABILITY = Capability(outputs=('f', 'g', 'c'))
XI = (-0.5, 0.5)


def _results(x: float, f: tuple[float, float], g: tuple[float, float], c: tuple[float, float] = (0.0, 0.0)):
    out = []
    for xi, fv, gv, cv in zip(XI, f, g, c, strict=True):
        action = Action.of(PROBLEM, CAPABILITY, {'x': x}, scenario={'xi': xi})
        out.append(Result(action, Cell(Status.OK, values={'f': fv, 'g': gv, 'c': cv})))
    return out


def test_each_design_reads_its_risk_in_natural_units() -> None:
    (design,) = design_risk(PROBLEM, _results(0.5, (1.0, 3.0), (5.0, 9.0)), risk=Risk('mean'), scenarios=2)
    assert design.design == {'x': 0.5}
    assert design.objectives == pytest.approx({'f': 2.0, 'g': 7.0})
    assert design.loss == pytest.approx((2.0, -3.0))  # g minimised: -(7 - 1) / 2
    assert design.feasible and design.statistic == 'mean over 2 scenarios'
    assert design.scores() == pytest.approx((2.0, 7.0))


def test_the_worst_case_reads_the_worse_scenario_in_each_sense() -> None:
    (design,) = design_risk(PROBLEM, _results(0.5, (1.0, 3.0), (5.0, 9.0)), risk=Risk('worst_case'), scenarios=2)
    assert design.objectives == pytest.approx({'f': 3.0, 'g': 5.0})


def test_a_violated_scenario_makes_the_design_infeasible() -> None:
    (design,) = design_risk(
        PROBLEM, _results(0.5, (1.0, 1.0), (1.0, 1.0), c=(0.0, 2.0)), risk=Risk('mean'), scenarios=2
    )
    assert design.complete and not design.feasible and design.violation > 0.0


def test_a_design_short_of_scenarios_or_with_a_failed_one_is_incomplete() -> None:
    short = design_risk(PROBLEM, _results(0.5, (1.0, 1.0), (1.0, 1.0))[:1], risk=Risk('mean'), scenarios=2)
    rows = _results(0.5, (1.0, 1.0), (1.0, 1.0))
    failed = [rows[0], Result(rows[1].action, Cell(Status.REJECTED, reason=Reason('infeasible')))]
    for (design,) in (short, design_risk(PROBLEM, failed, risk=Risk('mean'), scenarios=2)):
        assert not design.complete and not design.rankable and design.loss is None
        with pytest.raises(Refusal, match='incomplete'):
            design.scores()


def test_the_design_front_keeps_feasible_non_dominated_designs() -> None:
    results = [
        *_results(0.1, (1.0, 1.0), (9.0, 9.0)),  # best f, best g: dominates the next
        *_results(0.2, (2.0, 2.0), (5.0, 5.0)),
        *_results(0.3, (0.0, 0.0), (9.0, 9.0), c=(1.0, 1.0)),  # better, infeasible
    ]
    designs = design_risk(PROBLEM, results, risk=Risk('mean'), scenarios=2)
    assert [d.design['x'] for d in designs] == [0.1, 0.2, 0.3]
    assert [d.design['x'] for d in design_front(designs)] == [0.1]
    assert design_front(()) == ()
    assert all(isinstance(d, DesignRisk) for d in designs)
