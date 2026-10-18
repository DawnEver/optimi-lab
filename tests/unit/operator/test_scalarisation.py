"""D83, D84 -- each scalarisation's formula, and its DECLARED reach checked against what it reaches.

The reach claim is the one that matters (B0-02): on the concave front ``f2 = 1 - f1**2`` a weighted
sum's minimiser is an end point for every weight, while a whole-front scalarisation lands in the
middle for some weight. A scalarisation whose behaviour contradicts its ``reach`` reds here.
"""

import numpy as np
import pytest

from optimi_lab.core import Refusal
from optimi_lab.operator import (
    SCALARISATIONS,
    Reach,
    asf,
    epsilon_constraint,
    nbi,
    pbi,
    tchebycheff,
    weighted_sum,
)

F1 = np.linspace(0.0, 1.0, 401)
CONCAVE = np.column_stack([F1, 1.0 - F1**2])
IDEAL = np.zeros(2)
POINT = np.array([[0.6, 0.3]])
W = np.array([[0.25, 0.75]])


def _minimisers(scalarisation) -> set[int]:
    """The front rows a scan of the scalarisation's own parameter selects.

    Every scalarisation is scanned over its weights, except the epsilon-constraint, whose parameter
    is the bound on the second objective (the weight one-hot on the first).
    """
    out = set()
    for w in np.linspace(0.01, 0.99, 99):
        if scalarisation is epsilon_constraint:
            score = scalarisation(CONCAVE, np.array([[1.0, 0.0]]), IDEAL, bounds=np.array([np.inf, w]))
        else:
            score = scalarisation(CONCAVE, np.array([[w, 1.0 - w]]), IDEAL)
        out.add(int(np.argmin(score)))
    return out


@pytest.mark.parametrize('scalarisation', SCALARISATIONS, ids=lambda s: s.name)
def test_the_declared_reach_is_what_the_scalarisation_reaches_on_a_concave_front(scalarisation):
    middle = {i for i in _minimisers(scalarisation) if 0.05 < F1[i] < 0.95}
    if scalarisation.reach is Reach.CONVEX_HULL:
        assert not middle, 'a convex-hull scalarisation reached the middle of a concave front'
    elif scalarisation.reach is Reach.WHOLE_FRONT:
        assert len(middle) > 20, 'a whole-front scalarisation missed the middle of a concave front'


def test_the_formulas_are_the_published_ones():
    assert weighted_sum(POINT, W, IDEAL).tolist() == pytest.approx([0.25 * 0.6 + 0.75 * 0.3])
    assert tchebycheff(POINT, W, IDEAL).tolist() == pytest.approx([max(0.25 * 0.6, 0.75 * 0.3)])
    assert asf(POINT, W, IDEAL, rho=0.0).tolist() == pytest.approx([max(0.6 / 0.25, 0.3 / 0.75)])
    d1 = (0.6 * 0.25 + 0.3 * 0.75) / np.hypot(0.25, 0.75)
    d2 = np.linalg.norm(POINT[0] - d1 * W[0] / np.hypot(0.25, 0.75))
    assert pbi(POINT, W, IDEAL, theta=5.0).tolist() == pytest.approx([d1 + 5.0 * d2])
    assert epsilon_constraint(POINT, np.array([[1.0, 0.0]]), IDEAL, bounds=np.array([np.inf, 0.4])).tolist() == [0.6]
    over = epsilon_constraint(POINT, np.array([[1.0, 0.0]]), IDEAL, bounds=np.array([np.inf, 0.2]), penalty=10.0)
    assert over.tolist() == pytest.approx([0.6 + 10.0 * 0.1])


def test_nbi_moves_along_the_quasi_normal_of_the_hull_of_the_individual_minima():
    """On the linear front f1 + f2 = 1 the hull IS the front, so every weight's minimiser is ON its foot."""
    linear = np.column_stack([F1, 1.0 - F1])
    anchors = np.array([[0.0, 1.0], [1.0, 0.0]])
    for w in (0.2, 0.5, 0.8):
        best = linear[np.argmin(nbi(linear, np.array([[w, 1 - w]]), IDEAL, anchors=anchors))]
        assert best == pytest.approx(w * anchors[0] + (1 - w) * anchors[1], abs=3e-3)


def test_tchebycheff_refuses_a_negative_weight():
    with pytest.raises(Refusal, match='non-negative'):
        tchebycheff(POINT, np.array([[-0.1, 1.1]]), IDEAL)


def test_every_scalarisation_declares_a_reach():
    assert {s.name for s in SCALARISATIONS} == {
        'weighted_sum',
        'tchebycheff',
        'asf',
        'pbi',
        'epsilon_constraint',
        'nbi',
    }
    assert all(isinstance(s.reach, Reach) for s in SCALARISATIONS)
    assert weighted_sum.reach == 'convex_hull'
