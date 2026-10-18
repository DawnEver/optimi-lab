"""D124 sample average approximation, with the Mak-Morton-Wood optimality-gap bounds.

``min_x E[exp(x - xi) - 2 x]`` with ``xi ~ N(0.5, 0.4)``: ``x* = ln 2 + 0.5 - 0.08`` and
``v* = 2 - 2 x*`` (closed form, ``E[exp(-xi)] = exp(-0.5 + 0.4^2 / 2)``).
"""

from __future__ import annotations

import numpy as np

from optimi_lab.solver import Halt, saa, sample_average

X_STAR = np.log(2.0) + 0.5 - 0.08
V_STAR = 2.0 - 2.0 * X_STAR
#: Agreement of the SAA optimum with x*, in the design's own unit, at 4000 scenarios.
X_TOLERANCE = 0.05


def response(x: np.ndarray, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = np.exp(x[0] - xi)
    return e - 2 * x[0], (e - 2)[:, None]


def draw(mean: float):
    return lambda n, rng: rng.normal(mean, 0.4, n)


def test_d124_the_sample_average_is_a_deterministic_problem_with_the_mean_gradient() -> None:
    xi = np.array([0.0, 1.0])
    point = sample_average(response, xi)(np.array([0.5]))
    assert abs(point.f[0] - (np.mean(np.exp(0.5 - xi)) - 1.0)) <= 1e-12  # dimensionless objective
    assert abs(point.df[0, 0] - np.mean(np.exp(0.5 - xi) - 2)) <= 1e-12


def test_d124_saa_finds_the_optimum_and_brackets_its_value() -> None:
    found = saa(
        response, draw(0.5), [0.0], [-5.0], [5.0], n=4000, batches=10, n_check=20_000, rng=np.random.default_rng(0)
    )
    assert found.solution.halt is Halt.CONVERGED
    assert abs(found.solution.x[0] - X_STAR) <= X_TOLERANCE
    assert found.lower.value - 4 * found.lower.std_error <= V_STAR <= found.upper.value + 4 * found.upper.std_error
    assert found.gap.value >= -4 * found.gap.std_error


def test_d124_planted_control_scenarios_from_the_wrong_law_miss_the_optimum() -> None:
    found = saa(
        response, draw(0.0), [0.0], [-5.0], [5.0], n=4000, batches=10, n_check=20_000, rng=np.random.default_rng(0)
    )
    assert abs(found.solution.x[0] - X_STAR) > 0.3
