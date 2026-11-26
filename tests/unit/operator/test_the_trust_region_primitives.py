"""Trust-region agreement, radius update and dogleg."""

from __future__ import annotations

import math

import numpy as np
import pytest

from optimi_lab.operator.trust_region import agreement, dogleg, update_radius

_RNG = np.random.default_rng(11)
_N = 12
_M = _RNG.standard_normal((_N, _N))
_SPD = _M @ _M.T + _N * np.eye(_N)
_B = _RNG.standard_normal(_N)


def test_the_radius_shrinks_grows_or_holds_by_agreement() -> None:
    assert agreement(1.0, 2.0) == 0.5
    assert agreement(1.0, 0.0) == -math.inf
    assert update_radius(1.0, 0.1, 0.8, max_radius=10.0) == 0.25 * 0.8
    assert update_radius(1.0, 0.9, 1.0, max_radius=10.0) == 2.0
    assert update_radius(8.0, 0.9, 8.0, max_radius=10.0) == 10.0
    # Planted control: a good agreement INSIDE the ball is no reason to grow.
    assert update_radius(1.0, 0.9, 0.5, max_radius=10.0) == 1.0
    assert update_radius(1.0, 0.5, 1.0, max_radius=10.0) == 1.0


def test_dogleg_takes_newton_inside_and_cuts_on_the_boundary() -> None:
    newton = -np.linalg.solve(_SPD, _B)
    apply = lambda v: _SPD @ v  # noqa: E731
    np.testing.assert_allclose(dogleg(_B, apply, newton, 1e6), newton)
    for radius in (1e-4, 0.5 * float(np.linalg.norm(newton))):
        step = dogleg(_B, apply, newton, radius)
        assert math.isclose(float(np.linalg.norm(step)), radius, rel_tol=1e-10)
        model = lambda p: float(_B @ p + 0.5 * p @ _SPD @ p)  # noqa: E731
        assert model(step) < 0.0


def test_dogleg_refuses_a_nonconvex_model() -> None:
    with pytest.raises(ValueError, match='positive definite'):
        dogleg(np.ones(2), lambda v: -v, np.full(2, 10.0), 1.0)
