"""The analytic benchmark problems: every front point is optimal by its closed form.

ZDT: on the optimal set (x_2.. at their g-minimising value) f2 is the front's formula of f1.
DTLZ2: on the optimal set (x_m.. = 0.5) the objectives lie on the unit sphere.
"""

import numpy as np
import pytest

from optimi_lab.benchmark import PROBLEMS, Benchmark, dtlz2, zdt1, zdt2, zdt3, zdt4
from optimi_lab.core import Refusal, non_dominated_sorting


def _optimal(b: Benchmark, n: int, rest: float) -> np.ndarray:
    x = np.full((n, b.n_var), rest)
    x[:, 0] = np.linspace(0.0, 1.0, n)
    return x


@pytest.mark.parametrize(
    ('make', 'shape'),
    [(zdt1, lambda f1: 1 - np.sqrt(f1)), (zdt2, lambda f1: 1 - f1**2), (zdt4, lambda f1: 1 - np.sqrt(f1))],
)
def test_zdt_optimal_set_lies_on_its_front(make, shape):
    b = make(6)
    f = b.evaluate(_optimal(b, 21, 0.0))
    np.testing.assert_allclose(f[:, 1], shape(f[:, 0]), atol=1e-12)
    front = b.pareto_front(11)
    np.testing.assert_allclose(front[:, 1], shape(front[:, 0]), atol=1e-12)


def test_zdt3_front_is_its_formula_filtered_to_non_dominated():
    front = zdt3(6).pareto_front(50)
    np.testing.assert_allclose(front[:, 1], 1 - np.sqrt(front[:, 0]) - front[:, 0] * np.sin(10 * np.pi * front[:, 0]))
    assert len(non_dominated_sorting(front)) == len(front)


def test_dtlz2_optimal_set_is_on_the_unit_sphere():
    b = dtlz2(7, 3)
    x = np.random.default_rng(0).random((30, 7))
    x[:, 2:] = 0.5
    np.testing.assert_allclose(np.linalg.norm(b.evaluate(x), axis=1), 1.0, atol=1e-12)
    front = b.pareto_front(20)
    assert len(front) >= 20
    np.testing.assert_allclose(np.linalg.norm(front, axis=1), 1.0, atol=1e-12)


def test_zdt4_bounds_and_the_problem_table():
    b = zdt4(4)
    assert b.xl.tolist() == [0.0, -5.0, -5.0, -5.0] and b.xu.tolist() == [1.0, 5.0, 5.0, 5.0]
    assert set(PROBLEMS) == {'ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'DTLZ2'}
    assert all(PROBLEMS[name](6).name == name for name in PROBLEMS)


def test_a_benchmark_refuses_too_few_variables():
    with pytest.raises(Refusal):
        zdt1(1)
