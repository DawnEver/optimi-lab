"""Spacing, generalized spread, coverage, the fixed normalization, front_metrics and within_budget.

Every expected value is arithmetic on a front small enough to do by hand; none is a captured run.
"""

import numpy as np
import pytest

from optimi_lab.core import Refusal
from optimi_lab.metric import (
    coverage,
    fixed_normalization,
    front_metrics,
    generalized_spread,
    spacing,
    within_budget,
)


def test_spacing_by_hand():
    """L1 nearest distances 1, 1, 2: mean 4/3, squared deviations 6/9, over n - 1 = 2."""
    assert spacing(np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])) == pytest.approx(np.sqrt(1 / 3))
    assert spacing(np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])) == 0.0
    assert np.isnan(spacing(np.array([[0.0, 1.0]])))


def test_generalized_spread_is_zero_on_an_even_front_through_the_extremes():
    front = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    assert generalized_spread(front, front) == 0.0


def test_generalized_spread_by_hand():
    """Extremes (0, 1) and (1, 0); front (0, 1), (0.25, 0.75): d_e = 0 + |(0.25, 0.75) - (1, 0)|, nearest both d."""
    front = np.array([[0.0, 1.0], [0.25, 0.75]])
    reference = np.array([[0.0, 1.0], [1.0, 0.0]])
    d_e, d = np.hypot(0.75, 0.75), np.hypot(0.25, 0.25)
    assert generalized_spread(front, reference) == pytest.approx(d_e / (d_e + 2 * d))


def test_coverage_counts_the_weakly_dominated_reference_share():
    reference = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.6, 0.6]])
    assert coverage(np.array([[0.5, 0.5]]), reference) == 0.5
    assert coverage(np.empty((0, 2)), reference) == 0.0


def test_fixed_normalization_takes_the_union_minimum_and_the_front_maximum():
    ideal, nadir = fixed_normalization(np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0], [3.0, 3.0]]))
    assert ideal.tolist() == [0.0, 0.0] and nadir.tolist() == [2.0, 2.0]
    ideal, nadir = fixed_normalization(np.array([[1.0, 5.0], [1.0, 5.0]]))
    assert nadir.tolist() == [2.0, 6.0]
    with pytest.raises(Refusal):
        fixed_normalization(np.empty((0, 2)))


def test_front_metrics_on_the_two_corner_front():
    """HV of (0, 1) and (1, 0) at r: 2 r (r - 1) - (r - 1)^2; IGD+ 0, coverage 1, spacing 0, spread 0."""
    front = np.array([[0.0, 1.0], [1.0, 0.0]])
    out = front_metrics(front, front)
    for r in (1.01, 1.1, 2.0):
        assert out[f'HV({r})'] == pytest.approx(2 * r * (r - 1) - (r - 1) ** 2)
    assert (out['IGD+'], out['coverage'], out['spacing'], out['spread']) == (0.0, 1.0, 0.0, 0.0)


def test_front_metrics_of_an_empty_run():
    out = front_metrics(np.empty((0, 2)), np.array([[0.0, 1.0], [1.0, 0.0]]), references=(1.1,))
    assert out['HV(1.1)'] == 0.0 and out['IGD+'] == float('inf') and out['coverage'] == 0.0
    assert np.isnan(out['spacing']) and np.isnan(out['spread'])


def test_within_budget_counts_the_leading_rows_a_budget_pays_for():
    assert within_budget([1.0, 1.0, 1.0], 2.0) == 2
    assert within_budget([1.0, 1.0, 1.0], 2.5) == 2
    assert within_budget([0.5] * 4, 2.0) == 4
    assert within_budget([0.1] * 30, 3.0) == 30
    assert within_budget([], 1.0) == 0
