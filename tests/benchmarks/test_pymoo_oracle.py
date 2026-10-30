"""P12's owed oracle -- every population method that pymoo also implements reaches pymoo's IGD.

Same benchmark (pymoo evaluates the catalogue's own oracle), same population, same evaluation
budget, same seeds, IGD read off the whole evaluated set against the catalogue's closed-form front.
The statistic is the MEDIAN over ``SEEDS``; the test bounds the GAP ``median(optimi-lab) -
median(pymoo)`` by ``TOLERANCE``, in the benchmark's own objective units (ZDT1 and DTLZ2 are
dimensionless; BNH's objectives span 0-140, its tolerance is 0.2 of those units). Being BETTER than pymoo is never a failure.

Pairs (optimi-lab -> pymoo): nsga2 -> NSGA2, nsga3 -> NSGA3, unsga3 -> UNSGA3, spea2 -> SPEA2,
sms_emoa -> SMSEMOA, mode -> NSDE (DE variation under NSGA-II survival), moead -> MOEAD, rvea -> RVEA,
mopso -> MOPSO_CD. Not paired: rnsga3 (an aspiration-point method; IGD over the whole front does not
measure it), and pymoo's AGE-MOEA, C-TAEA, GDE3 and the rest, which optimi-lab does not implement.
"""

from functools import cache

import numpy as np
import pytest
from catalogue import CATALOGUE, RandomSearch, drive, front_of
from pymoo_reference import pymoo_igd
from test_population_gate import _igd

from optimi_lab.metric import igd

SEEDS = (1, 2, 3)

#: benchmark -> (population, evaluations, gap tolerance in objective units, methods compared).
CASES = {
    'zdt1': (40, 2400, 0.01, ('nsga2', 'nsga3', 'unsga3', 'spea2', 'sms_emoa', 'mode', 'moead', 'rvea', 'mopso')),
    'dtlz2': (60, 3600, 0.01, ('nsga2', 'nsga3', 'unsga3', 'spea2', 'mode', 'moead', 'rvea', 'mopso')),
    'bnh': (40, 2400, 0.2, ('nsga2', 'spea2')),
}

PAIRS = [(bench, name) for bench, (*_, names) in CASES.items() for name in names]


def _median(values: list[float]) -> float:
    return float(np.median(values))


@cache
def _random_median(bench: str, evaluations: int) -> float:
    """The unguided control's median IGD over ``SEEDS``, once per benchmark per process."""
    b = CATALOGUE[bench]
    return _median([
        igd(
            front_of(b.problem, drive(RandomSearch(b.problem, b.oracle.capability, seed), b.oracle, evaluations)),
            b.front(200),
        )
        for seed in SEEDS
    ])


@pytest.mark.parametrize(('bench', 'name'), PAIRS, ids=[f'{b}-{n}' for b, n in PAIRS])
def test_the_method_reaches_pymoos_igd_within_the_tolerance(bench, name):
    """Plus the planted control against the SAME reference: a random search does not reach it.

    The control rides in this test (load ruling 2026-10-30): as its own test it re-ran every pymoo
    reference, doubling the file's pymoo runs for a claim this one already computes the input to.
    """
    population, evaluations, tolerance, _ = CASES[bench]
    reference = _median([pymoo_igd(bench, name, population, evaluations, seed) for seed in SEEDS])
    ours = _median([_igd(bench, name, population, evaluations, seed) for seed in SEEDS])
    assert ours - reference <= tolerance, f'{bench}/{name}: optimi-lab {ours:.4f}, pymoo {reference:.4f}'
    unguided = _random_median(bench, evaluations)
    assert unguided - reference > tolerance, f'{bench}/{name}: random {unguided:.4f}, pymoo {reference:.4f}'
