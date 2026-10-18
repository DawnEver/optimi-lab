"""P9 ruling (a): the generalised Cauchy point of L-BFGS-B, checked along the projected path itself.

The GCP is the FIRST local minimiser of the quadratic model ``m(z) = g.z + z.B z / 2`` along the
projected steepest-descent path ``P(x - t g) - x``. The check walks that path on a fine grid: the
model must not rise before the GCP's ``t`` and must not fall just after it. The planted control
displaces the point along the path and the same check refuses it.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from optimi_lab.operator import cauchy_point


def _model(g: np.ndarray, b: np.ndarray, z: np.ndarray) -> float:
    return float(g @ z + 0.5 * z @ b @ z)


def _path(x, g, lo, hi, t):
    return np.clip(x - t * g, lo, hi) - x


def first_local_minimum(x, g, b, box, z) -> bool:
    """*z* is on the path, the model never rises before it, and does not fall right after it."""
    lo, hi = box
    ts = np.linspace(0.0, 10.0, 1_000_001)
    distance = np.max(np.abs(np.clip(x - ts[:, None] * g, lo, hi) - x - z), axis=1)
    k = int(np.argmin(distance))
    if distance[k] > 1e-4:
        return False
    t_star = ts[k]
    values = [_model(g, b, _path(x, g, lo, hi, t)) for t in np.linspace(0.0, t_star, 400)]
    before = all(v2 <= v1 + 1e-9 for v1, v2 in itertools.pairwise(values))
    after = _model(g, b, _path(x, g, lo, hi, t_star + 1e-3)) >= _model(g, b, z) - 1e-9
    return before and after


@pytest.mark.parametrize('seed', range(10))
def test_p9a_the_cauchy_point_is_the_first_minimiser_along_the_projected_path(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 5
    m = rng.normal(size=(n, n))
    b = m @ m.T + 0.5 * np.eye(n)
    lo, hi = -np.ones(n), np.ones(n)
    x = rng.uniform(-1, 1, size=n)
    g = 3 * rng.normal(size=n)
    z = cauchy_point(x, g, b, lo, hi)
    assert np.all(x + z >= lo - 1e-12) and np.all(x + z <= hi + 1e-12)
    assert first_local_minimum(x, g, b, (lo, hi), z)


def test_p9a_planted_control_a_displaced_point_is_refused() -> None:
    rng = np.random.default_rng(3)
    n = 4
    m = rng.normal(size=(n, n))
    b, g = m @ m.T + np.eye(n), 3 * rng.normal(size=n)
    x, lo, hi = np.zeros(n), -np.ones(n), np.ones(n)
    z = cauchy_point(x, g, b, lo, hi)
    assert first_local_minimum(x, g, b, (lo, hi), z)
    assert not first_local_minimum(x, g, b, (lo, hi), z + 0.05 * _path(x, g, lo, hi, 1.0))


def test_p9a_a_point_already_stationary_on_the_box_does_not_move() -> None:
    x, g = np.array([1.0, -1.0]), np.array([-2.0, 3.0])  # both gradients push out of the box
    assert np.array_equal(cauchy_point(x, g, np.eye(2), -np.ones(2), np.ones(2)), np.zeros(2))
