"""L4 benchmark: analytic multi-objective test problems with closed-form Pareto fronts.

A problem with a known front is what an indicator or a solver is verified against; the front is
computed in closed form, never from a run of this package. Minimisation frame, vectorised over rows.
"""

from optimi_lab.benchmark.problems import PROBLEMS, Benchmark, dtlz2, zdt1, zdt2, zdt3, zdt4

__all__ = ['PROBLEMS', 'Benchmark', 'dtlz2', 'zdt1', 'zdt2', 'zdt3', 'zdt4']
