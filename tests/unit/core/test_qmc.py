"""D70-D72 -- the self-implemented point sets, each against scipy.stats.qmc as a TEST-ONLY oracle (R1).

Unscrambled Sobol and Halton are deterministic sequences, so the oracle comparison is EXACT (a
tolerance of 0 in unit-cube coordinates, which are dimensionless). A scrambled set is a random draw,
so it is compared on the property scrambling exists to preserve -- its centred L2 discrepancy, which
must stay within a factor of the oracle's own scrambled draw -- and on balance (one point per
elementary interval).
"""

import numpy as np
import pytest
from scipy.stats import qmc as oracle

from optimi_lab.core import Refusal
from optimi_lab.core.qmc import MAX_SOBOL_DIMENSION, halton, latin_hypercube, maximin_latin_hypercube, sobol


@pytest.mark.parametrize('width', [1, 2, 7, 40, MAX_SOBOL_DIMENSION])
def test_unscrambled_sobol_is_the_published_sequence_exactly(width):
    ours = sobol(256, width, rng=None)
    theirs = oracle.Sobol(d=width, scramble=False, bits=32).random(256)
    assert np.array_equal(ours, theirs)


@pytest.mark.parametrize('width', [1, 3, 10])
def test_unscrambled_halton_is_the_radical_inverse_sequence_exactly(width):
    assert np.allclose(halton(100, width, rng=None), oracle.Halton(d=width, scramble=False).random(100), atol=0.0)


def test_a_planted_wrong_direction_number_is_caught_by_the_oracle_comparison(monkeypatch):
    """The control: the exact comparison above is not an echo -- one changed table entry reds it."""
    import optimi_lab.core.qmc as module  # noqa: PLC0415 -- the module whose table is planted

    table = list(module.DIRECTIONS)
    table[0] = (table[0][0], (3,))
    monkeypatch.setattr(module, 'DIRECTIONS', tuple(table))
    assert not np.array_equal(module.sobol(16, 2, rng=None), oracle.Sobol(d=2, scramble=False).random(16))


@pytest.mark.parametrize(('ours', 'theirs'), [(sobol, oracle.Sobol), (halton, oracle.Halton)])
def test_a_scrambled_set_is_as_uniform_as_the_oracles(ours, theirs):
    points = ours(128, 4, rng=np.random.default_rng(3))
    reference = theirs(d=4, seed=3).random(128)
    assert points.min() >= 0.0 and points.max() < 1.0
    assert oracle.discrepancy(points) <= 1.5 * oracle.discrepancy(reference)
    assert not np.array_equal(points, ours(128, 4, rng=None)), 'a scrambled set is not the raw sequence'
    assert np.array_equal(points, ours(128, 4, rng=np.random.default_rng(3))), 'one seed, one set'


def test_scrambled_sobol_keeps_one_point_per_elementary_interval():
    points = sobol(64, 2, rng=np.random.default_rng(0))
    assert sorted(np.floor(points[:, 0] * 64).astype(int)) == list(range(64))
    assert sorted(np.floor(points[:, 1] * 64).astype(int)) == list(range(64))


def test_sobol_beyond_the_table_is_refused_naming_the_ceiling():
    with pytest.raises(Refusal, match=str(MAX_SOBOL_DIMENSION)):
        sobol(8, MAX_SOBOL_DIMENSION + 1, rng=None)


def test_a_latin_hypercube_holds_one_point_per_stratum_in_every_column():
    points = latin_hypercube(50, 3, np.random.default_rng(1))
    for column in points.T:
        assert sorted(np.floor(column * 50).astype(int)) == list(range(50))


def test_maximin_spreads_its_closest_pair_further_than_a_plain_hypercube():
    """D70: the criterion it optimises is the smallest pairwise distance (unit-cube units)."""

    def closest(points):
        gaps = np.linalg.norm(points[:, None] - points[None], axis=-1)
        return gaps[np.triu_indices(len(points), 1)].min()

    plain = [closest(latin_hypercube(30, 2, np.random.default_rng(s))) for s in range(5)]
    spread = [closest(maximin_latin_hypercube(30, 2, np.random.default_rng(s))) for s in range(5)]
    assert min(spread) > np.median(plain)
    for column in maximin_latin_hypercube(30, 2, np.random.default_rng(0)).T:
        assert sorted(np.floor(column * 30).astype(int)) == list(range(30)), 'it is still a Latin hypercube'
