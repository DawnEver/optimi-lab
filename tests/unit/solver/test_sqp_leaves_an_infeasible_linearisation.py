"""SQP's l1 merit slope is the merit's own derivative along ANY step, and an infeasible linearisation still moves.

The incident (a downstream motor-design search, 2026-11-13): an SQP refine of a spline barrier, its objective gradient matching central
differences to 2e-5, started below its own radius floors and ended ``stalled -- no step decreased the l1 merit``.
When the linearised constraints admit no step inside the box, the QP returns its least-violating step, which removes
only PART of the violation. The slope handed to Armijo was ``grad f . d - nu * violation`` -- the slope of a step
that removes ALL of it -- so it promised a decrease the merit never had, at every step length.

Planted: minimise ``0.25 x0 + (x0 - 0.1)^2 + (x1 - 0.5)^2`` on the unit box under ``x0^2 >= 0.25`` from
``(0.1, 0.5)``. There the linearised floor needs ``d0 >= 1.2`` and the box allows 0.9: the least-violating step
removes 0.18 of the 0.24 violation while the objective rises at 0.225, so with ``nu = 1`` the merit RISES along it at
every length (+0.045 at the start, and the curvature keeps it above) although the old slope said -0.015 -- the
stall, on exact gradients. The constrained minimum is ``(0.5, 0.5)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from optimi_lab.operator import Linearization, active_set_qp, l1_merit, l1_slope
from optimi_lab.solver import Halt, sqp

_LO, _HI = np.zeros(2), np.ones(2)
_START = np.array([0.1, 0.5])


def _evaluate(x: np.ndarray) -> Linearization:
    f = 0.25 * x[0] + (x[0] - 0.1) ** 2 + (x[1] - 0.5) ** 2
    g = np.array([0.25 + 2.0 * (x[0] - 0.1), 2.0 * (x[1] - 0.5)])
    return Linearization(x=x, f=[f], df=[g], c=[0.25 - x[0] ** 2], dc=[[-2.0 * x[0], 0.0]])


def _least_violating_step(point: Linearization) -> np.ndarray:
    n = point.x.size
    qp = active_set_qp(
        np.eye(n),
        point.gradient,
        a_in=np.vstack([point.dc, np.eye(n), -np.eye(n)]),
        b_in=np.concatenate([-point.c, _HI - point.x, point.x - _LO]),
    )
    assert not qp.feasible  # the planted start's linearisation admits no step inside the box
    return qp.x


def test_the_slope_is_the_merits_derivative_along_a_least_violating_step() -> None:
    point = _evaluate(_START)
    d = _least_violating_step(point)
    t = 1e-7
    derivative = (l1_merit(_evaluate(point.x + t * d), 1.0) - l1_merit(point, 1.0)) / t
    assert derivative > 0.0  # the planted control: the merit rises along this step at nu = 1
    assert l1_slope(point, d, 1.0) == pytest.approx(derivative, abs=1e-5)


def test_sqp_from_an_infeasible_linearisation_reaches_the_constrained_minimum() -> None:
    found = sqp(_evaluate, _START, _LO, _HI)
    assert found.halt is Halt.CONVERGED, found.reason
    assert found.x == pytest.approx([0.5, 0.5], abs=1e-6)
