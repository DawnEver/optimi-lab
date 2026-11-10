"""Pearson's r against arithmetic: an exact line is +-1, and a hand-computed sample matches its formula."""

import numpy as np
import pytest

from optimi_lab.metric import pearson


def test_an_exact_line_is_plus_or_minus_one():
    x = np.linspace(0.0, 1.0, 11)
    assert pearson(np.column_stack([x, 3.0 * x + 2.0])) == pytest.approx(1.0, abs=1e-12)
    assert pearson(np.column_stack([x, -0.5 * x])) == pytest.approx(-1.0, abs=1e-12)


def test_a_hand_computed_pair():
    # x = 1..4, y = (1, 3, 2, 4): Sxy = 4, Sxx = Syy = 5, so r = 0.8 exactly.
    assert pearson(np.array([[1, 1], [2, 3], [3, 2], [4, 4]])) == pytest.approx(0.8, abs=1e-12)


def test_three_columns_give_the_matrix_and_a_convex_curve_is_below_one():
    x = np.linspace(0.0, 1.0, 21)
    matrix = pearson(np.column_stack([x, x, np.exp(5 * x)]))
    assert matrix.shape == (3, 3)
    assert matrix[0, 1] == pytest.approx(1.0, abs=1e-12)
    assert 0.5 < matrix[0, 2] < 0.95
