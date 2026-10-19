"""D33-D41: every gradient method agrees with scipy.optimize (a REFERENCE ORACLE, tests only) on closed forms.

Each method is the pure function, called on exact derivatives. Every comparison is :func:`agrees`;
the PLANTED CONTROL runs the same method on the same problem with one gradient component's sign
flipped and asserts the same check refuses what comes back -- a comparison that a wrong gradient
also passes is no evidence that the right one is used.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
import pytest
from scipy.optimize import NonlinearConstraint, minimize, nnls, rosen, rosen_der, rosen_hess_prod

from optimi_lab.operator import Linearization
from optimi_lab.solver import (
    Halt,
    augmented_lagrangian,
    interior_point,
    lbfgsb,
    mgda,
    mma,
    newton_cg,
    pareto_continuation,
    projected_gradient,
    sqp,
)

#: Agreement in the variables' own scale (these closed forms are O(1)).
TOLERANCE = 1e-5
#: MMA converges linearly; its KKT tolerance is 1e-6, so its point agrees to this.
MMA_TOLERANCE = 1e-4


def agrees(found: np.ndarray, reference: np.ndarray, tolerance: float = TOLERANCE) -> bool:
    return bool(np.max(np.abs(np.asarray(found) - np.asarray(reference))) <= tolerance)


def _flipped(evaluate: Callable[[np.ndarray], Linearization]) -> Callable[[np.ndarray], Linearization]:
    """*evaluate* with the sign of every objective gradient's first component flipped -- the planted defect."""

    def wrong(x: np.ndarray) -> Linearization:
        p = evaluate(x)
        df = np.array(p.df)
        df[:, 0] *= -1.0
        return Linearization(x=p.x, f=p.f, df=df, c=p.c, dc=p.dc, h=p.h, dh=p.dh)

    return wrong


# --- box problems: Rosenbrock in 4 variables, interior optimum and an active bound -----------------

ROSEN_X0 = np.array([-1.2, 1.0, -0.5, 0.3])
BOXES = {'interior': (-2 * np.ones(4), 2 * np.ones(4)), 'active': (-2 * np.ones(4), np.array([2.0, 2.0, 0.8, 2.0]))}


def rosenbrock(x: np.ndarray) -> Linearization:
    return Linearization(x=x, f=[rosen(x)], df=[rosen_der(x)])


def _box_reference(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    found = minimize(
        rosen,
        ROSEN_X0,
        jac=rosen_der,
        method='L-BFGS-B',
        bounds=list(zip(lo, hi, strict=True)),
        options={'ftol': 1e-15, 'gtol': 1e-12, 'maxiter': 10_000},
    )
    assert found.success, found.message
    return found.x


#: Newton-CG takes scipy's exact Rosenbrock Hessian-vector product, declared -- not the named fallback.
BOX_METHODS = {
    'lbfgsb': lbfgsb,
    'projected_gradient': partial(projected_gradient, rtol=1e-11),  # linear rate: a 1e-5 point needs this bar
    'newton_cg': partial(newton_cg, hvp=rosen_hess_prod),
}


@pytest.mark.parametrize('box', sorted(BOXES))
@pytest.mark.parametrize('name', sorted(BOX_METHODS))
def test_d33_d34_d35_box_methods_agree_with_scipy(name: str, box: str) -> None:
    lo, hi = BOXES[box]
    found = BOX_METHODS[name](rosenbrock, ROSEN_X0, lo, hi)
    assert found.halt is Halt.CONVERGED, found.reason
    assert agrees(found.x, _box_reference(lo, hi))


def test_d34_lbfgsb_halts_naming_a_failed_wolfe_search() -> None:
    """A jump at x = 0.5 holds no Wolfe point; the search's failure is the halt, not an Armijo detour."""

    def jump(x: np.ndarray) -> Linearization:
        return Linearization(x=x, f=[-x[0] if x[0] < 0.5 else 1.0], df=[[-1.0]])

    found = lbfgsb(jump, np.zeros(1), np.zeros(1), np.ones(1))
    assert found.halt is Halt.STALLED and found.reason.startswith('the Wolfe search failed')


def test_d35_newton_cg_agrees_with_scipy_trust_ncg_on_the_interior_optimum() -> None:
    reference = minimize(
        rosen, ROSEN_X0, jac=rosen_der, hessp=rosen_hess_prod, method='trust-ncg', options={'gtol': 1e-10}
    )
    assert reference.success
    assert agrees(BOX_METHODS['newton_cg'](rosenbrock, ROSEN_X0, *BOXES['interior']).x, reference.x)


@pytest.mark.parametrize('name', sorted(BOX_METHODS))
def test_d33_d34_d35_planted_control_a_wrong_gradient_does_not_agree(name: str) -> None:
    lo, hi = BOXES['interior']
    found = BOX_METHODS[name](_flipped(rosenbrock), ROSEN_X0, lo, hi, max_iterations=200)
    assert not (found.halt is Halt.CONVERGED and agrees(found.x, _box_reference(lo, hi)))


# --- constrained: Hock-Schittkowski 71 -------------------------------------------------------------

HS_X0, HS_LO, HS_HI = np.array([1.0, 5.0, 5.0, 1.0]), np.ones(4), 5 * np.ones(4)


def hs071(x: np.ndarray) -> Linearization:
    """``min x0 x3 (x0 + x1 + x2) + x2`` s.t. ``prod(x) >= 25``, ``|x|^2 = 40``, ``1 <= x <= 5``; both scaled."""
    f = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
    g = np.array([x[3] * (2 * x[0] + x[1] + x[2]), x[0] * x[3], x[0] * x[3] + 1, x[0] * (x[0] + x[1] + x[2])])
    products = np.array([x[1] * x[2] * x[3], x[0] * x[2] * x[3], x[0] * x[1] * x[3], x[0] * x[1] * x[2]])
    return Linearization(
        x=x, f=[f], df=[g], c=[(25 - np.prod(x)) / 25], dc=[-products / 25], h=[(x @ x - 40) / 40], dh=[2 * x / 40]
    )


def _hs071_reference() -> np.ndarray:
    def objective(x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    found = minimize(
        objective,
        HS_X0,
        jac=lambda x: hs071(x).df[0],
        method='SLSQP',
        bounds=list(zip(HS_LO, HS_HI, strict=True)),
        constraints=[
            {'type': 'ineq', 'fun': lambda x: np.prod(x) - 25, 'jac': lambda x: -25 * hs071(x).dc[0]},
            {'type': 'eq', 'fun': lambda x: x @ x - 40, 'jac': lambda x: 2 * x},
        ],
        # ftol 1e-12 on an O(17) objective asks SLSQP's line search for a decrease it cannot resolve once
        # at the optimum: with one BLAS thread (the forge's Linux node) it stops at the optimum with status 8
        # 'Positive directional derivative'. At 1e-10 it converges to the same x on every platform measured.
        options={'ftol': 1e-10, 'maxiter': 500},
    )
    assert found.success, found.message
    trust = minimize(
        objective,
        HS_X0,
        method='trust-constr',
        bounds=list(zip(HS_LO, HS_HI, strict=True)),
        constraints=[NonlinearConstraint(np.prod, 25, np.inf), NonlinearConstraint(lambda x: x @ x, 40, 40)],
        options={'gtol': 1e-12, 'xtol': 1e-14, 'maxiter': 5000},
    )
    assert agrees(found.x, trust.x, 1e-4)  # the two references agree with each other first
    return found.x


CONSTRAINED = {
    'sqp': sqp,
    'interior_point': interior_point,
    'interior_point_sparse': lambda *a, **k: interior_point(*a, sparse=True, **k),
    'augmented_lagrangian': augmented_lagrangian,
}


@pytest.mark.parametrize('name', sorted(CONSTRAINED))
def test_d36_d37_d38_constrained_methods_agree_with_scipy_on_hs071(name: str) -> None:
    found = CONSTRAINED[name](hs071, HS_X0, HS_LO, HS_HI)
    assert found.halt is Halt.CONVERGED, found.reason
    assert found.residual <= 1e-8 * (1.0 + np.max(np.abs(hs071(HS_X0).df[0])))  # the relative KKT bar
    assert agrees(found.x, _hs071_reference())


def test_d37_the_dense_and_sparse_interior_point_take_the_same_path() -> None:
    dense = interior_point(hs071, HS_X0, HS_LO, HS_HI)
    sparse = interior_point(hs071, HS_X0, HS_LO, HS_HI, sparse=True)
    assert dense.iterations == sparse.iterations and agrees(dense.x, sparse.x, 1e-10)


@pytest.mark.parametrize('name', sorted(CONSTRAINED))
def test_d36_d37_d38_planted_control_a_wrong_gradient_does_not_agree(name: str) -> None:
    found = CONSTRAINED[name](_flipped(hs071), HS_X0, HS_LO, HS_HI)
    assert not (found.halt is Halt.CONVERGED and agrees(found.x, _hs071_reference()))


# --- MMA / GCMMA: Svanberg's two-sphere problem ----------------------------------------------------

MMA_X0, MMA_LO, MMA_HI = np.array([4.0, 3.0, 2.0]), np.zeros(3), 5 * np.ones(3)
CENTRES = np.array([[5.0, 2.0, 1.0], [3.0, 4.0, 3.0]])


def spheres(x: np.ndarray) -> Linearization:
    """``min |x|^2`` s.t. ``|x - centre_i|^2 <= 9`` for two centres."""
    return Linearization(x=x, f=[x @ x], df=[2 * x], c=np.sum((x - CENTRES) ** 2, axis=1) - 9, dc=2 * (x - CENTRES))


def _spheres_reference() -> np.ndarray:
    found = minimize(
        lambda x: x @ x,
        MMA_X0,
        method='SLSQP',
        bounds=list(zip(MMA_LO, MMA_HI, strict=True)),
        constraints=[{'type': 'ineq', 'fun': lambda x: 9 - np.sum((x - CENTRES) ** 2, axis=1)}],
        options={'ftol': 1e-12},
    )
    assert found.success
    return found.x


@pytest.mark.parametrize('conservative', [False, True], ids=['mma', 'gcmma'])
def test_d39_mma_and_gcmma_agree_with_scipy(conservative: bool) -> None:
    found = mma(spheres, MMA_X0, MMA_LO, MMA_HI, conservative=conservative)
    assert found.halt is Halt.CONVERGED, found.reason
    assert agrees(found.x, _spheres_reference(), MMA_TOLERANCE)


@pytest.mark.parametrize('conservative', [False, True], ids=['mma', 'gcmma'])
def test_d39_planted_control_a_wrong_gradient_does_not_agree(conservative: bool) -> None:
    found = mma(_flipped(spheres), MMA_X0, MMA_LO, MMA_HI, conservative=conservative, max_iterations=100)
    assert not (found.halt is Halt.CONVERGED and agrees(found.x, _spheres_reference(), MMA_TOLERANCE))


def test_d39_gcmma_reaches_the_optimum_from_behind_a_concave_constraint() -> None:
    # A concave constraint is where an MMA approximation can under-estimate; GCMMA's inner loop
    # must still reach the same optimum.
    def concave(x: np.ndarray) -> Linearization:
        return Linearization(
            x=x,
            f=[(x[0] - 2) ** 2 + (x[1] - 2) ** 2],
            df=[2 * (x - 2)],
            c=[0.5 - np.sqrt(x[0] * x[1] + 1e-12) / 2],
            dc=[[-x[1] / (4 * np.sqrt(x[0] * x[1] + 1e-12)), -x[0] / (4 * np.sqrt(x[0] * x[1] + 1e-12))]],
        )

    found = mma(concave, np.array([0.5, 0.5]), np.full(2, 0.1), np.full(2, 3.0), conservative=True)
    assert found.halt is Halt.CONVERGED and agrees(found.x, np.array([2.0, 2.0]), MMA_TOLERANCE)


# --- several objectives ------------------------------------------------------------------------------

TWO_LO, TWO_HI = -3 * np.ones(2), 3 * np.ones(2)


def two(x: np.ndarray) -> Linearization:
    """``|x|^2`` and ``|x - (1, 0.5)|^2 + x0``: the Pareto set is a segment, the front convex."""
    target = np.array([1.0, 0.5])
    return Linearization(x=x, f=[x @ x, (x - target) @ (x - target) + x[0]], df=[2 * x, 2 * (x - target) + [1.0, 0.0]])


def _critical_weights(point: Linearization) -> tuple[np.ndarray, float]:
    """Weights on the simplex minimising ``|G^T a|`` (scipy nnls), and that minimum: zero iff Pareto-critical."""
    system = np.vstack([point.df.T, 1e3 * np.ones(point.f.size)])
    weights, residual = nnls(system, np.concatenate([np.zeros(point.x.size), [1e3]]))
    return weights, residual


def test_d40_every_mgda_point_is_pareto_critical_and_is_scipys_weighted_minimiser() -> None:
    starts = np.array([[2.0, -1.0], [-1.0, 2.5], [0.3, 0.3]])
    found = mgda(two, starts, TWO_LO, TWO_HI)
    assert found.halt is Halt.CONVERGED and len(found.front) == 3
    for point in found.front:
        weights, residual = _critical_weights(point)
        assert residual <= 1e-6
        reference = minimize(
            lambda x, w=weights: w @ two(x).f,
            point.x + 0.3,
            jac=lambda x, w=weights: w @ two(x).df,
            method='BFGS',
            options={'gtol': 1e-12},
        )
        assert agrees(point.x, reference.x)


def test_d40_planted_control_a_wrong_gradient_is_not_critical() -> None:
    found = mgda(_flipped(two), np.array([[2.0, -1.0]]), TWO_LO, TWO_HI, max_iterations=200)
    assert all(_critical_weights(two(p.x))[1] > 1e-6 for p in found.front)


def test_d41_pareto_continuation_agrees_with_scipy_epsilon_constraints() -> None:
    found = pareto_continuation(two, np.zeros(2), TWO_LO, TWO_HI, points=7)
    assert found.halt is Halt.CONVERGED and len(found.front) == 7
    for point in found.front[1:-1]:
        level = point.f[1]
        reference = minimize(
            lambda x: two(x).f[0],
            point.x + 0.2,
            jac=lambda x: two(x).df[0],
            method='SLSQP',
            constraints=[{'type': 'ineq', 'fun': lambda x, e=level: e - two(x).f[1], 'jac': lambda x: -two(x).df[1]}],
            options={'ftol': 1e-12},
        )
        assert agrees(point.x, reference.x)
    seconds = [p.f[1] for p in found.front]
    assert np.all(np.diff(seconds) < 0)  # the levels walk from one anchor to the other


def test_d41_pareto_continuation_refuses_three_objectives() -> None:
    def three(x: np.ndarray) -> Linearization:
        return Linearization(x=x, f=[x @ x, x[0], x[1]], df=[2 * x, [1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match='two objectives'):
        pareto_continuation(three, np.zeros(2), TWO_LO, TWO_HI)


def test_d41_planted_control_a_wrong_gradient_does_not_trace_the_front() -> None:
    right = pareto_continuation(two, np.zeros(2), TWO_LO, TWO_HI, points=5)
    wrong = pareto_continuation(_flipped(two), np.zeros(2), TWO_LO, TWO_HI, points=5, max_iterations=60)
    traced = wrong.halt is Halt.CONVERGED and all(
        agrees(a.x, b.x) for a, b in zip(right.front, wrong.front, strict=True)
    )
    assert not traced
