"""P12 gate -- every Pareto population solver reaches its benchmark's TRUE front within a stated IGD.

THE ORACLE. The reference is the closed-form Pareto front of each benchmark (``catalogue.py``);
the equal-seed comparison against pymoo's implementation of each method is
``test_pymoo_oracle.py``. The tolerance is an IGD in the benchmark's own objective units (ZDT1 and DTLZ2 are
dimensionless), on the MEDIAN of three seeds at a fixed budget, read off the run's whole evaluated
set; each value below is the measured median rounded up with headroom, re-measured 2026-10-18 (MOEA/D after its replacement fix).

The solvers are driven in memory (``catalogue.drive``): the run layer's record costs two fsyncs per
evaluation, and this gate measures the algorithms. The run path has its own tests.
"""

import numpy as np
import pytest
from catalogue import CATALOGUE, drive, front_of

from optimi_lab.core import SampleSpec
from optimi_lab.metric import igd
from optimi_lab.solver import mode, moead, mopso, nsga2, nsga3, rvea, sms_emoa, spea2, unsga3

SEEDS = (1, 2, 3)

#: (benchmark, population, evaluations) -> {solver: IGD ceiling, objective units}.
GATE = {
    ('zdt1', 40, 2400): {
        'nsga2': 0.02, 'nsga3': 0.02, 'unsga3': 0.02, 'spea2': 0.02, 'mode': 0.005, 'moead': 0.01,
        'mopso': 0.01, 'rvea': 0.03, 'sms_emoa': 0.01,
    },
    ('dtlz2', 60, 3600): {
        'nsga2': 0.035, 'nsga3': 0.035, 'unsga3': 0.035, 'spea2': 0.035, 'mode': 0.035, 'moead': 0.035,
        'mopso': 0.08, 'rvea': 0.035,
    },
}  # fmt: skip

FACTORIES = {
    'nsga2': nsga2, 'nsga3': nsga3, 'unsga3': unsga3, 'spea2': spea2, 'mode': mode, 'moead': moead,
    'mopso': mopso, 'sms_emoa': sms_emoa,
    'rvea': lambda problem, capability, **settings: rvea(problem, capability, generations=60, **settings),
}  # fmt: skip

CASES = [
    (bench, pop, evals, name, ceiling) for (bench, pop, evals), row in GATE.items() for name, ceiling in row.items()
]


def _igd(bench: str, name: str, pop: int, evals: int, seed: int) -> float:
    b = CATALOGUE[bench]
    solver = FACTORIES[name](b.problem, b.oracle.capability, spec=SampleSpec(n_samples=pop, seed=seed), seed=seed)
    return igd(front_of(b.problem, drive(solver, b.oracle, evals)), b.front(200))


@pytest.mark.parametrize(('bench', 'pop', 'evals', 'name', 'ceiling'), CASES, ids=[f'{c[0]}-{c[3]}' for c in CASES])
def test_the_solver_reaches_the_true_front(bench, pop, evals, name, ceiling):
    assert float(np.median([_igd(bench, name, pop, evals, seed) for seed in SEEDS])) <= ceiling


def test_a_random_search_at_the_same_budget_does_not_pass_the_gate():
    """The control: the ceilings separate an algorithm from no algorithm at all."""
    b = CATALOGUE['zdt1']
    points = np.random.default_rng(1).random((2400, b.problem.space.width))
    results = drive(nsga2(b.problem, b.oracle.capability, initial=points, seed=1), b.oracle, len(points))
    assert igd(front_of(b.problem, results), b.front(200)) > GATE['zdt1', 40, 2400]['nsga2']
