"""D67-D68 -- the surrogate-prescreened EA against plain NSGA-II, trust-region model management on closed forms.

IGD is in ZDT1's (dimensionless) objective units; objective gaps in each function's unit; positions
in the unit cube. Measured values (2026-10-18) are quoted beside each bar.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import rosen

from optimi_lab.core import Refusal, SampleSpec
from optimi_lab.metric import igd
from optimi_lab.model import Polynomial, Prediction
from optimi_lab.oracle import Capability, from_function
from optimi_lab.problem import Objective, Problem, Real, Space
from optimi_lab.solver import model_trust_region, nsga2, prescreened

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
from catalogue import CATALOGUE, drive, front_of

#: ZDT1 at 160 evaluations, population 20, seed 1; measured prescreened 0.240, NSGA-II 0.746.
ZDT1_IGD = 0.35
#: Objective units; a 3-D quadratic bowl, measured 2.3e-9 at 70 evaluations.
BOWL_GAP = 1e-7
#: Objective units; 2-D Rosenbrock on [-2, 2]^2 from the centre at 300 evaluations, measured 0.030.
ROSENBROCK_GAP = 0.1


def _zdt1_igd(factory) -> float:
    b = CATALOGUE['zdt1']
    solver = factory(b.problem, b.oracle.capability, spec=SampleSpec(n_samples=20, seed=1), seed=1)
    return igd(front_of(b.problem, drive(solver, b.oracle, 160)), b.front(200))


def test_prescreening_reaches_the_zdt1_front_at_a_budget_nsga2_cannot():
    assert _zdt1_igd(prescreened) <= ZDT1_IGD


def test_planted_the_same_generation_without_prescreening_misses_the_bar():
    assert _zdt1_igd(nsga2) > ZDT1_IGD


def test_prescreening_refusals():
    b = CATALOGUE['zdt1']
    with pytest.raises(Refusal, match='gives none'):
        prescreened(b.problem, b.oracle.capability, surrogate=Polynomial())
    with pytest.raises(Refusal, match='oversample'):
        prescreened(b.problem, b.oracle.capability, oversample=0)
    with pytest.raises(Refusal, match='valid settings'):
        prescreened(b.problem, b.oracle.capability, oversampling=3)
    prescreened(b.problem, b.oracle.capability, surrogate=Polynomial(), kappa=0.0)


def _bowl() -> tuple[Problem, object]:
    problem = Problem(Space(tuple(Real(n, 0.0, 1.0) for n in 'xyz')), (Objective('f'),))

    def f(action):
        x, y, z = (action.design[n] for n in 'xyz')
        return {'f': (x - 0.3) ** 2 + 10 * (y - 0.6) ** 2 + 3 * (z - 0.45) ** 2 + (x - 0.3) * (y - 0.6)}

    return problem, from_function(f, Capability(outputs=('f',)))


def test_model_management_converges_on_a_bowl_and_concludes():
    problem, oracle = _bowl()
    solver = model_trust_region(problem, oracle.capability, seed=3)
    drive(solver, oracle, 300)
    assert solver.done and solver.best[1] <= BOWL_GAP


def test_model_management_descends_rosenbrocks_valley():
    problem = Problem(Space(tuple(Real(f'x{i}', -2.0, 2.0) for i in range(2))), (Objective('f'),))
    oracle = from_function(
        lambda a: {'f': float(rosen(np.array([a.design['x0'], a.design['x1']])))}, Capability(outputs=('f',))
    )
    solver = model_trust_region(problem, oracle.capability, seed=3, start=np.full(2, 0.5))
    drive(solver, oracle, 300)
    assert solver.best[1] <= ROSENBROCK_GAP


@dataclass(frozen=True)
class _Contrary:
    """A planted surrogate that predicts the NEGATED fit of a quadratic: every step it proposes goes uphill."""

    @property
    def gives(self):
        return frozenset()

    def fit(self, x, y):
        inner = Polynomial(degree=2).fit(x, y) if len(x) >= 10 else Polynomial(degree=1).fit(x, y)
        return _Negated(inner)


@dataclass(frozen=True)
class _Negated:
    inner: object

    @property
    def gives(self):
        return frozenset()

    def predict(self, x, *, variance=False, gradient=False):
        return Prediction(-self.inner.predict(x).mean)


def test_planted_a_contrary_model_never_moves_the_centre_uphill_and_the_run_concludes():
    """The control: the ratio test is what protects the search -- a model that is always wrong is never trusted."""
    problem, oracle = _bowl()
    solver = model_trust_region(problem, oracle.capability, seed=3, surrogate=_Contrary())
    results = drive(solver, oracle, 300)
    first = results[0].cell.values['f']
    assert solver.done and solver.best[1] <= first and len(results) < 300


def test_model_management_refuses_a_disordered_radius():
    problem, oracle = _bowl()
    with pytest.raises(Refusal, match='radius_end'):
        model_trust_region(problem, oracle.capability, radius=1e-5)
