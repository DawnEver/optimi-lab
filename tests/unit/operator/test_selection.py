"""D78-D82 -- environmental selection by reference direction, angle penalty, strength and hypervolume.

Each operator is checked against a hand-computed instance of its published definition (objective
values dimensionless, minimised frame). The NSGA-III niche test is the planted control for the
P8 fill, which counted niches over the overflowing front alone: it picks the other candidate.
"""

import numpy as np
import pytest

from optimi_lab.core import Refusal
from optimi_lab.operator import (
    angle_penalized_select,
    aspiration_directions,
    das_dennis,
    hypervolume_select,
    niching_tournament,
    reference_select,
    spea2_fitness,
    spea2_select,
)

SPEA = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0], [3.0, 3.0], [4.0, 4.0]])


def test_das_dennis_is_every_composition_on_the_simplex():
    assert das_dennis(2, 2).tolist() == [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
    grid = das_dennis(3, 4)
    assert len(grid) == 15 and np.allclose(grid.sum(axis=1), 1.0)


def test_spea2_raw_fitness_sums_the_strengths_of_every_dominator():
    """D81: strengths A1 B2 C1 D1 E0; raw fitness D = S(B) = 2, E = 1 + 2 + 1 + 1 = 5."""
    fitness = spea2_fitness(SPEA)
    assert np.floor(fitness).tolist() == [0.0, 0.0, 0.0, 2.0, 5.0]
    assert np.all((fitness - np.floor(fitness) > 0.0) & (fitness - np.floor(fitness) < 0.5)), 'density in (0, 1/2)'


def test_spea2_truncation_removes_the_most_crowded_non_dominated_point():
    assert sorted(spea2_select(SPEA, 3).tolist()) == [0, 1, 2]
    assert sorted(spea2_select(SPEA, 2).tolist()) == [0, 2], 'B is nearest to both A and C'
    assert sorted(spea2_select(SPEA, 4).tolist()) == [0, 1, 2, 3], 'then the best dominated by fitness'


def test_hypervolume_selection_drops_the_least_exclusive_contributor():
    """D82: contributions to (4, 4) are 1.0, 3.6 and 1.2, so the first point goes."""
    front = np.array([[0.0, 3.0], [1.0, 1.2], [3.0, 0.0], [3.5, 3.5]])
    assert sorted(hypervolume_select(front, 3, reference=np.array([4.0, 4.0])).tolist()) == [0, 1, 2]
    assert sorted(hypervolume_select(front, 2, reference=np.array([4.0, 4.0])).tolist()) == [1, 2]


def test_reference_selection_spreads_a_front_over_its_directions_whatever_the_scale():
    for scale in (1.0, 100.0):
        front = np.array([[0.0, 1.0], [0.05, 0.95], [0.1, 0.9], [0.5, 0.5], [1.0, 0.0]]) * [1.0, scale]
        assert sorted(reference_select(front, 3, directions=das_dennis(2, 2)).tolist()) == [0, 3, 4]


def test_reference_selection_counts_niches_over_every_survivor_not_only_the_last_front():
    """Front 1 holds niche (0, 1) twice and (1, 0) once, so NSGA-III takes the (1, 0) candidate, 4.

    The P8 fill counted the overflowing front alone -- both niches empty -- and took 3.
    """
    values = np.array([[0.0, 1.0], [0.05, 0.9], [1.0, 0.0], [0.02, 1.1], [1.1, 0.05]])
    assert sorted(reference_select(values, 4, directions=das_dennis(2, 2)).tolist()) == [0, 1, 2, 4]


def test_a_niching_tournament_is_won_by_rank_inside_one_niche():
    values = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
    winners = niching_tournament(values, 300, np.random.default_rng(0), directions=das_dennis(2, 2))
    assert 2 not in winners.tolist(), 'the worst row of a shared niche never wins a tournament'


def test_aspiration_directions_cluster_on_the_simplex_around_each_point():
    points = np.array([[0.2, 0.8, 0.3], [0.9, 0.1, 0.1]])
    directions, owner = aspiration_directions(points, partitions=4, spread=0.1)
    assert np.allclose(directions.sum(axis=1), 1.0) and np.all(directions >= 0.0)
    projected = points / points.sum(axis=1, keepdims=True)
    assert np.all(np.abs(directions - projected[owner]).max(axis=1) <= 0.1 + 1e-12)
    assert set(owner.tolist()) == {0, 1}


def _apd(values, direction, progress, alpha, gamma):
    shifted = values - values.min(axis=0)
    norm = np.linalg.norm(shifted, axis=1)
    cosine = shifted @ direction / np.maximum(norm, 1e-300) / np.linalg.norm(direction)
    return (1.0 + values.shape[1] * progress**alpha * np.arccos(np.clip(cosine, -1, 1)) / gamma) * norm


@pytest.mark.parametrize('progress', [0.0, 0.5, 1.0])
def test_rvea_keeps_the_least_angle_penalized_member_of_every_niche(progress):
    """D80: APD = (1 + M (t/T)^alpha theta / gamma) |f - z| (Cheng et al., 2016), z the population ideal."""
    values = np.random.default_rng(4).random((40, 3))
    directions = das_dennis(3, 3)
    chosen = angle_penalized_select(values, directions=directions, progress=progress, alpha=2.0)
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    shifted = values - values.min(axis=0)
    niche = np.argmax(shifted @ unit.T / np.maximum(np.linalg.norm(shifted, axis=1, keepdims=True), 1e-300), axis=1)
    cosines = np.clip(unit @ unit.T, -1, 1)
    np.fill_diagonal(cosines, -1)
    gamma = np.arccos(cosines.max(axis=1))
    expected = sorted(
        int(np.flatnonzero(niche == j)[np.argmin(_apd(values, unit[j], progress, 2.0, gamma[j])[niche == j])])
        for j in sorted(set(niche.tolist()))
    )
    assert sorted(chosen.tolist()) == expected


def test_rvea_progress_outside_the_unit_interval_is_refused():
    with pytest.raises(Refusal, match='progress'):
        angle_penalized_select(np.ones((3, 2)), directions=das_dennis(2, 2), progress=1.5)
