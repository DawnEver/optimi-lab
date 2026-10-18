"""pymoo as a REFERENCE ORACLE: the same algorithm, benchmark, population, budget and seeds (tests only).

Ruling R1: the runtime never imports pymoo. This module lives in the test tree, and the dependency
is declared in the ``dev`` extra beside pytest.

THE SAME BENCHMARK. Each pymoo problem below evaluates the catalogue's own oracle, so the two runs
cannot disagree about the function; the front each IGD is read against is the catalogue's closed
form, for both sides.

THE SAME STATISTIC. The catalogue gate reads IGD off the run's WHOLE evaluated set; so does this
side -- the wrapped problem records every row pymoo evaluates, and pymoo stops at the same evaluation
budget. Each algorithm runs at pymoo's own defaults: the comparison is algorithm against algorithm,
not a tuning exercise.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import numpy as np
from catalogue import CATALOGUE, Benchmark
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
from pymoo.algorithms.moo.nsde import NSDE
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.problem import Problem as PymooProblem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from optimi_lab.core import non_dominated_sorting
from optimi_lab.core.pareto import partition_count
from optimi_lab.metric import igd
from optimi_lab.oracle import Action

__all__ = ['REFERENCE', 'pymoo_igd']


class _Recorded(PymooProblem):
    """The catalogue benchmark as a pymoo problem, keeping every feasible objective row it evaluates."""

    def __init__(self, benchmark: Benchmark) -> None:
        problem = benchmark.problem
        variables = problem.space.variables
        super().__init__(
            n_var=len(variables),
            n_obj=len(problem.objectives),
            n_ieq_constr=len(problem.constraints),
            xl=np.array([v.lo for v in variables]),
            xu=np.array([v.hi for v in variables]),
        )
        self._benchmark = benchmark
        self.rows: list[np.ndarray] = []

    def _evaluate(self, x: np.ndarray, out: dict, *args: object, **kwargs: object) -> None:
        problem, oracle = self._benchmark.problem, self._benchmark.oracle
        names = [v.name for v in problem.space.variables]

        async def each() -> list:
            return [
                await oracle(Action.of(problem, oracle.capability, dict(zip(names, map(float, row), strict=True))))
                for row in x
            ]

        results = asyncio.run(each())
        f = np.array([[o.minimized(r.cell.values[o.name]) for o in problem.objectives] for r in results])
        out['F'] = f
        if problem.constraints:
            out['G'] = np.array([[c.violation(r.cell.values[c.output]) for c in problem.constraints] for r in results])
            feasible = (out['G'] <= 0.0).all(axis=1)
        else:
            feasible = np.ones(len(f), dtype=bool)
        self.rows.extend(f[feasible])


def _directions(objectives: int, population: int) -> np.ndarray:
    return get_reference_directions('das-dennis', objectives, n_partitions=partition_count(objectives, population))


#: optimi-lab solver name -> pymoo's implementation of the same algorithm at (objectives, population).
REFERENCE: dict[str, Callable[[int, int], object]] = {
    'nsga2': lambda _m, n: NSGA2(pop_size=n),
    'nsga3': lambda m, n: NSGA3(_directions(m, n), pop_size=n),
    'unsga3': lambda m, n: UNSGA3(_directions(m, n), pop_size=n),
    'spea2': lambda _m, n: SPEA2(pop_size=n),
    'sms_emoa': lambda _m, n: SMSEMOA(pop_size=n),
    'mode': lambda _m, n: NSDE(pop_size=n),
    'moead': lambda m, n: MOEAD(_directions(m, n)),
    'rvea': lambda m, n: RVEA(_directions(m, n), pop_size=n),
    'mopso': lambda _m, n: MOPSO_CD(pop_size=n),
}


def pymoo_igd(bench: str, name: str, population: int, evaluations: int, seed: int) -> float:
    """The IGD pymoo reaches on *bench* for the algorithm optimi-lab calls *name*, whole evaluated set."""
    benchmark = CATALOGUE[bench]
    problem = _Recorded(benchmark)
    algorithm = REFERENCE[name](problem.n_obj, population)
    minimize(problem, algorithm, ('n_eval', evaluations), seed=seed, verbose=False)
    rows = np.array(problem.rows[:evaluations]) if problem.rows else np.zeros((0, problem.n_obj))
    return igd(rows[non_dominated_sorting(rows)], benchmark.front(200))
