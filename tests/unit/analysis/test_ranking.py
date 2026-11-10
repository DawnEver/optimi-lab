"""Rankings of sensitivity results against functions whose ordering is known in closed form.

``f = 4 x1 + 2 x2 + 0 x3`` on ``[0, 1]^3`` is additive and linear: its total Sobol' indices are
``(16, 4, 0) / 20``, its Morris ``mu*`` the coefficients ``(4, 2, 0)``, and its input-output
correlations order the same way. Ties and the sign of a correlation are checked on hand-built rows.
"""

import numpy as np
import pytest

from optimi_lab.analysis import (
    CORRELATIONS,
    SensitivityMeasure,
    SensitivityRanking,
    morris,
    rank_by_correlation,
    rank_morris,
    rank_sobol,
    sobol_indices,
)
from optimi_lab.core.errors import Refusal
from optimi_lab.problem import Real, Space

SPACE = Space(tuple(Real(f'x{i}', 0.0, 1.0) for i in (1, 2, 3)))
COEFFICIENTS = np.array([4.0, 2.0, 0.0])


def linear(x: np.ndarray) -> np.ndarray:
    return x @ COEFFICIENTS


def test_sobol_ranks_by_the_total_index_with_its_interval():
    ranking = rank_sobol(sobol_indices(SPACE, linear, 2**10, seed=1, n_bootstrap=200), confidence=0.95)
    rows = ranking.rows()
    assert [r.name for r in rows] == ['x1', 'x2', 'x3']
    assert [r.rank for r in rows] == [1, 2, 3]
    assert rows[0].value == pytest.approx(0.8, abs=0.02)
    assert rows[1].value == pytest.approx(0.2, abs=0.02)
    assert rows[0].interval[0] <= rows[0].value <= rows[0].interval[1]
    assert set(rows[0].measures) == {'total', 'first_order'}
    assert not ranking.correlational


def test_morris_ranks_by_mu_star_and_carries_the_standard_error():
    ranking = rank_morris(morris(SPACE, linear, trajectories=10, seed=2), trajectories=10)
    rows = ranking.rows()
    assert [r.name for r in rows] == ['x1', 'x2', 'x3']
    assert [r.value for r in rows] == pytest.approx([4.0, 2.0, 0.0], abs=1e-9)
    assert rows[0].measures['mu_standard_error'] == pytest.approx(0.0, abs=1e-9)
    assert rows[0].interval is None


def test_correlation_ranks_by_magnitude_and_flags_association():
    rng = np.random.default_rng(3)
    x = rng.uniform(size=(400, 3))
    y = -4.0 * x[:, 0] + 2.0 * x[:, 1] + 0.01 * rng.normal(size=400)
    ranking = rank_by_correlation(x, y, ('a', 'b', 'c'), by='pearson', n_bootstrap=200, seed=4)
    rows = ranking.rows()
    assert ranking.correlational
    assert [r.name for r in rows] == ['a', 'b', 'c']
    assert rows[0].value < -0.85  # a negative coefficient still ranks first
    low, high = rows[0].interval
    assert low <= rows[0].value <= high
    assert set(rows[0].measures) == set(CORRELATIONS)
    assert abs(rows[2].value) < 0.15


def test_a_tie_keeps_declaration_order():
    ranking = SensitivityRanking(
        'hand',
        ('p', 'q', 'r'),
        (SensitivityMeasure('m', np.array([1.0, 2.0, 2.0])),),
        'm',
        0,
        None,
        correlational=False,
    )
    assert ranking.order() == (1, 2, 0)


def test_strangers_are_refused():
    with pytest.raises(Refusal, match='ranks by one of'):
        SensitivityRanking('hand', ('p',), (SensitivityMeasure('m', np.ones(1)),), 'nope', 0, None, correlational=False)
    with pytest.raises(Refusal, match='correlation is one of'):
        rank_by_correlation(np.ones((4, 1)), np.arange(4.0), ('p',), by='nope', seed=0)
    with pytest.raises(Refusal, match='pairs'):
        rank_by_correlation(np.ones((4, 2)), np.arange(4.0), ('p',), seed=0)
