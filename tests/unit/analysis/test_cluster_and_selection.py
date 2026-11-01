"""k-means labels and the stratified round-robin pick.

k-means is checked on clusters separated far beyond their spread, where the labelling is unique up
to names; the pick is checked by its contract (no repeats, round-robin over sorted strata) and
against a verbatim copy of the loop it replaces on fixed seeds.
"""

from collections import defaultdict

import numpy as np
import pytest

from optimi_lab.analysis import kmeans, stratified_pick
from optimi_lab.core import Refusal


def test_kmeans_separates_well_separated_clusters():
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    x = np.vstack([c + 0.1 * rng.standard_normal((20, 2)) for c in centres])
    labels = kmeans(x, 3, np.random.default_rng(1))
    groups = [set(labels[20 * i : 20 * (i + 1)].tolist()) for i in range(3)]
    assert all(len(g) == 1 for g in groups)
    assert len(set.union(*groups)) == 3


def test_kmeans_refuses_more_clusters_than_points():
    with pytest.raises(Refusal):
        kmeans(np.zeros((2, 2)), 3, np.random.default_rng(0))


def _reference(keys, k, rng, chosen):
    chosen = list(chosen)
    strata = defaultdict(list)
    for index in rng.permutation(len(keys)):
        if index not in chosen:
            strata[keys[index]].append(int(index))
    order = sorted(strata)
    while len(chosen) < k:
        for key in order:
            if strata[key] and len(chosen) < k:
                chosen.append(strata[key].pop())
    return chosen


@pytest.mark.parametrize('seed', [0, 3, 9])
def test_stratified_pick_reproduces_the_replaced_loop(seed):
    keys = [(i % 3, bool(i % 2)) for i in range(17)]
    expected = _reference(keys, 11, np.random.default_rng(seed), [4])
    assert stratified_pick(keys, 11, np.random.default_rng(seed), chosen=[4]) == expected


def test_stratified_pick_balances_strata_and_never_repeats():
    keys = ['a'] * 10 + ['b'] * 2
    picked = stratified_pick(keys, 6, np.random.default_rng(0))
    assert len(set(picked)) == 6
    assert sum(keys[i] == 'b' for i in picked) == 2
    with pytest.raises(Refusal):
        stratified_pick(keys, 13, np.random.default_rng(0))
