"""Analytic multi-objective benchmark problems and convergence metrics.

WHAT A LIBRARY OWES ITS CALLERS. Shipping optimizers without a problem to run them on and a metric
to score them by leaves every consumer to write their own, and a hand-rolled ZDT is a hand-rolled
bug: `f2 = 1 - sqrt(f1)` is easy to get subtly wrong and impossible to notice, because both versions
converge to *a* front. So the known-Pareto-front problems and the two standard convergence metrics
live here, beside the algorithms they judge.

Pure numerics on `numpy` only -- no motor, material or solver vocabulary, and nothing in this module
needs to change for a lab doing something unrelated to use it unchanged. `Problem` is the seam: a
box-constrained minimization over `(n_var)` inputs onto `(n_obj)` outputs, with an OPTIONAL analytic
`pareto_front` that exists for exactly one purpose -- letting :func:`igd` score a run against the
truth rather than against a previous run.

`zdt1`/`zdt2` are the two ZDT shapes (convex and concave front) that between them catch the common
failure where a decomposition-based algorithm handles one curvature and falls over on the other.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from optimi_lab.core.optimize import optimize
from optimi_lab.core.outcomes import Evaluation
from optimi_lab.core.sampling import SampleSpec
from optimi_lab.core.space import Objective, ObjectiveSet, Variable, VariableSet
from optimi_lab.intelligent_algorithm import nsga2

__all__ = [
    'Problem',
    'hypervolume_2d',
    'igd',
    'solve',
    'zdt1',
    'zdt2',
]


@dataclass(frozen=True)
class Problem:
    """A box-constrained multi-objective minimization problem.

    ``evaluate`` maps a population matrix ``X`` of shape ``(n_sample, n_var)`` to an objective
    matrix ``F`` of shape ``(n_sample, n_obj)``, minimized on every objective.

    Attributes:
        name: Human-readable identifier.
        n_var: Number of decision variables.
        n_obj: Number of objectives.
        xl: Lower bounds, shape ``(n_var,)``.
        xu: Upper bounds, shape ``(n_var,)``.
        evaluate: Vectorized objective ``X -> F``.
        pareto_front: Optional callable ``n -> (n, n_obj)`` returning ``n`` evenly sampled points on
            the analytic Pareto front, for :func:`igd`.

    """

    name: str
    n_var: int
    n_obj: int
    xl: np.ndarray
    xu: np.ndarray
    evaluate: Callable[[np.ndarray], np.ndarray]
    pareto_front: Callable[[int], np.ndarray] | None = None


def _zdt(kind: int, n_var: int) -> Problem:
    """Build a ZDT problem (kind 1 = convex front, kind 2 = concave front)."""

    def evaluate(x: np.ndarray) -> np.ndarray:
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (n_var - 1)
        ratio = f1 / g
        h = 1.0 - np.sqrt(ratio) if kind == 1 else 1.0 - ratio**2
        return np.column_stack([f1, g * h])

    def pareto_front(n: int) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n)
        f2 = 1.0 - np.sqrt(f1) if kind == 1 else 1.0 - f1**2
        return np.column_stack([f1, f2])

    return Problem(
        name=f'ZDT{kind}',
        n_var=n_var,
        n_obj=2,
        xl=np.zeros(n_var),
        xu=np.ones(n_var),
        evaluate=evaluate,
        pareto_front=pareto_front,
    )


def zdt1(n_var: int = 3) -> Problem:
    """ZDT1: two objectives, convex Pareto front ``f2 = 1 - sqrt(f1)``."""
    return _zdt(1, n_var)


def zdt2(n_var: int = 3) -> Problem:
    """ZDT2: two objectives, concave Pareto front ``f2 = 1 - f1**2``."""
    return _zdt(2, n_var)


def igd(front: np.ndarray, reference: np.ndarray) -> float:
    """Inverted Generational Distance from a reference front to ``front``.

    Mean Euclidean distance from each reference point to its nearest solution in ``front``, so
    lower is better and 0 means the reference is perfectly covered. An EMPTY ``front`` is
    ``inf`` rather than 0, because a run that found nothing is the worst result and not the best.
    """
    if front.size == 0:
        return float('inf')
    d = np.sqrt(((reference[:, None, :] - front[None, :, :]) ** 2).sum(axis=-1))
    return float(d.min(axis=1).mean())


def hypervolume_2d(front: np.ndarray, ref_point: np.ndarray) -> float:
    """Dominated hypervolume of a 2-objective ``front`` w.r.t. ``ref_point``.

    Only points that dominate ``ref_point`` contribute, and larger is better. The front is swept in
    ascending ``f1``; each step adds the rectangle it dominates. Refuses more than two objectives
    rather than silently reading the first two, because a 3-objective hypervolume is a different
    algorithm and a plausible wrong number is worse than a refusal.
    """
    if front.shape[1] != 2:
        msg = 'hypervolume_2d only supports 2 objectives.'
        raise ValueError(msg)
    pts = front[np.all(front < ref_point, axis=1)]
    if pts.size == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev_f2 = ref_point[1]
    for f1, f2 in pts:
        if f2 < prev_f2:
            hv += (ref_point[0] - f1) * (prev_f2 - f2)
            prev_f2 = f2
    return float(hv)


def solve(
    problem: Problem,
    *,
    pop_size: int = 50,
    max_iter: int = 60,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``problem`` with NSGA-II and return ``(pareto_x, pareto_f)``.

    Thin wiring over :func:`~optimi_lab.core.optimize.optimize`, and every column is a FIELD rather
    than a position in a parallel list: one ``Variable`` per decision dimension, one ``Objective``
    per column, all minimized -- which is what a :class:`Problem`'s own ``evaluate`` promises. The
    seed reaches the sampler and the proposer's own generator, so two calls with one seed are one
    run.

    The archive returned is the LIBRARY's own first front of the complete points (``Record.pareto``)
    rather than a front this function recomputes: it refuses a run in which no point computed every
    objective, because an empty front would otherwise read exactly like a run that found nothing.

    ``max_iter`` counts generations AFTER the initial sample, so a run evaluates ``max_iter + 1``
    batches -- the initial sample IS the first one.
    """
    space = VariableSet(
        tuple(
            Variable(f'x{index}', float(problem.xl[index]), float(problem.xu[index])) for index in range(problem.n_var)
        )
    )
    objectives = ObjectiveSet(tuple(Objective(f'f{index}') for index in range(problem.n_obj)))

    def evaluate(points: np.ndarray) -> Evaluation:
        return Evaluation(inputs=points, values=np.asarray(problem.evaluate(points), dtype=float))

    record = optimize(
        space=space,
        objectives=objectives,
        evaluate=evaluate,
        build_proposer=nsga2(space, seed=seed),
        spec=SampleSpec(n_samples=pop_size, seed=seed),
        n_batches=max_iter + 1,
    )
    front = record.pareto()
    return front.inputs, front.values
