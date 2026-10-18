"""D31 KKT assembly and D32 merit functions: each derivative is checked against finite differences."""

from __future__ import annotations

import numpy as np
import pytest

from optimi_lab.operator import (
    Linearization,
    augmented_lagrangian_merit,
    damped_secant,
    kkt_residual,
    l1_merit,
    l1_penalty,
    l1_slope,
    lagrangian_gradient,
)
from optimi_lab.oracle import finite_difference


def _point(x: np.ndarray) -> Linearization:
    """``f = x0^2 + 2 x1^2``, ``h = x0 + x1 - 1``, ``c = x0^2 - x1`` -- with exact derivatives."""
    return Linearization(
        x=x,
        f=np.array([x[0] ** 2 + 2 * x[1] ** 2]),
        df=np.array([[2 * x[0], 4 * x[1]]]),
        c=np.array([x[0] ** 2 - x[1]]),
        dc=np.array([[2 * x[0], -1.0]]),
        h=np.array([x[0] + x[1] - 1.0]),
        dh=np.array([[1.0, 1.0]]),
    )


def test_d31_the_lagrangian_gradient_is_the_gradient_of_the_lagrangian() -> None:
    x, y, z = np.array([0.3, 0.4]), np.array([0.7]), np.array([1.5])

    def lagrangian(v: np.ndarray) -> np.ndarray:
        p = _point(v)
        return np.array([p.f[0] + y @ p.h + z @ p.c])

    assert np.allclose(lagrangian_gradient(_point(x), y, z), finite_difference(lagrangian, x)[0], atol=1e-8)


def test_d31_the_kkt_residual_is_zero_at_a_kkt_point_and_names_each_part() -> None:
    # min x^2 on [1, 2]: x* = 1, the lower bound is active with multiplier 2 (absorbed by projection).
    p = Linearization(x=np.array([1.0]), f=np.array([1.0]), df=np.array([[2.0]]))
    assert kkt_residual(p, lo=np.array([1.0]), hi=np.array([2.0])).worst == 0.0
    off = kkt_residual(
        Linearization(x=np.array([1.5]), f=np.array([2.25]), df=np.array([[3.0]])), np.array([1.0]), np.array([2.0])
    )
    assert off.stationarity == pytest.approx(0.5) and off.worst == pytest.approx(0.5)
    infeasible = Linearization(
        x=np.zeros(1), f=np.zeros(1), df=np.zeros((1, 1)), c=np.array([0.25]), dc=np.ones((1, 1))
    )
    residual = kkt_residual(infeasible, np.array([-1.0]), np.array([1.0]), z=np.array([0.0]))
    assert residual.feasibility == pytest.approx(0.25)
    slack = Linearization(x=np.zeros(1), f=np.zeros(1), df=np.array([[-2.0]]), c=np.array([-0.5]), dc=np.array([[1.0]]))
    assert kkt_residual(slack, np.array([-1.0]), np.array([1.0]), z=np.array([2.0])).complementarity == pytest.approx(
        1.0
    )


def test_d32_the_l1_slope_bounds_the_merit_derivative_along_a_linearised_step() -> None:
    p, nu = _point(np.array([0.9, 0.5])), 3.0
    # a step solving the linearised constraints exactly: h + dh d = 0 and c + dc d = 0
    d = np.linalg.solve(np.vstack([p.dh, p.dc]), -np.concatenate([p.h, p.c]))
    t = 1e-7
    numeric = (l1_merit(_point(p.x + t * d), nu) - l1_merit(p, nu)) / t
    assert numeric <= l1_slope(p, d, nu) + 1e-5


def test_d32_the_l1_penalty_dominates_the_multipliers_and_never_falls() -> None:
    assert l1_penalty(np.array([1.0, -4.0]), previous=1.0) > 4.0
    assert l1_penalty(np.array([0.1]), previous=7.0) == 7.0


def test_d32_the_augmented_lagrangian_gradient_matches_finite_differences() -> None:
    y, z, rho = np.array([0.5]), np.array([0.2]), 10.0
    for x in (np.array([0.9, 0.5]), np.array([0.1, 0.9])):  # one active inequality, one inactive
        value, gradient = augmented_lagrangian_merit(_point(x), y, z, rho)

        def merit(v: np.ndarray) -> np.ndarray:
            return np.array([augmented_lagrangian_merit(_point(v), y, z, rho)[0]])

        assert np.allclose(gradient, finite_difference(merit, x)[0], atol=1e-6)
        assert np.isfinite(value)


def test_d32_powell_damping_keeps_the_curvature_positive_and_leaves_a_good_pair_alone() -> None:
    b = np.eye(2)
    s = np.array([1.0, 0.0])
    assert np.array_equal(damped_secant(b, s, np.array([2.0, 0.0])), np.array([2.0, 0.0]))
    damped = damped_secant(b, s, np.array([-1.0, 0.0]))
    assert damped @ s >= 0.2 * s @ b @ s - 1e-12


def test_d32_planted_control_a_wrong_gradient_is_caught_by_the_same_check() -> None:
    y, z, rho, x = np.array([0.5]), np.array([0.2]), 10.0, np.array([0.9, 0.5])
    _, gradient = augmented_lagrangian_merit(_point(x), y, z, rho)

    def merit(v: np.ndarray) -> np.ndarray:
        return np.array([augmented_lagrangian_merit(_point(v), y, z, rho)[0]])

    assert not np.allclose(gradient + np.array([1e-3, 0.0]), finite_difference(merit, x)[0], atol=1e-6)
