"""D124 sample average approximation: ``min_x E[f(x, xi)]`` replaced by the mean over drawn scenarios.

A *response* ``(x, scenarios) -> (values (N,), gradients (N, n))`` gives ``f`` and ``grad_x f`` for
every scenario at once. :func:`sample_average` turns a fixed scenario set into an ordinary
:data:`~optimi_lab.solver.smooth.Evaluate`, so ANY gradient method solves it (L-BFGS-B by default).

:func:`saa` solves one SAA problem of *n* scenarios and brackets the true optimal value by Mak, Morton
and Wood (1999):

* ``lower`` -- the mean of *batches* independent SAA optimal values (``E[v_N] <= v*`` for a
  minimisation), with its replicate standard error;
* ``upper`` -- ``f`` at the SAA solution averaged over *n_check* fresh scenarios (``>= v*``);
* ``gap = upper - lower``, the two standard errors combined in quadrature (independent samples).

Scenarios come from ``draw(n, rng)``; the rng is explicit, so a run is reproducible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from optimi_lab.operator import Estimate, Linearization, mean_estimate, replicated
from optimi_lab.solver.bounded import lbfgsb
from optimi_lab.solver.smooth import Evaluate, Solution

__all__ = ['SampleAverage', 'saa', 'sample_average']

Response = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True, slots=True, eq=False)
class SampleAverage:
    """The SAA *solution* and the Mak-Morton-Wood *lower* / *upper* bounds and *gap* on the optimal value."""

    solution: Solution
    lower: Estimate
    upper: Estimate
    gap: Estimate


def sample_average(response: Response, scenarios: np.ndarray) -> Evaluate:
    """The deterministic SAA objective over a fixed *scenarios* set, as a gradient method's evaluate."""

    def evaluate(x: np.ndarray) -> Linearization:
        values, gradients = response(np.asarray(x, dtype=float), scenarios)
        return Linearization(x, [float(np.mean(values))], [np.mean(np.atleast_2d(gradients), axis=0)])

    return evaluate


def saa(
    response: Response,
    draw: Callable[[int, np.random.Generator], np.ndarray],
    x0,
    lo,
    hi,
    *,
    n: int,
    batches: int,
    n_check: int,
    rng: np.random.Generator,
    method: Callable[..., Solution] = lbfgsb,
) -> SampleAverage:
    """Solve one SAA problem and bracket its value; see the module docstring."""
    solution = method(sample_average(response, draw(n, rng)), x0, lo, hi)
    optima = [method(sample_average(response, draw(n, rng)), x0, lo, hi).point.objective for _ in range(batches)]
    lower = replicated(Estimate(value, 0.0, n) for value in optima)
    upper = mean_estimate(response(solution.x, draw(n_check, rng))[0])
    gap = Estimate(upper.value - lower.value, float(np.hypot(upper.std_error, lower.std_error)), lower.n + upper.n)
    return SampleAverage(solution, lower, upper, gap)
