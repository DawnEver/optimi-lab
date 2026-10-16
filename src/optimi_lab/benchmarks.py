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
`bnh` (Binh and Korn, 1997) is the constrained one: two quadratic objectives under two circular
constraints, the planted case for a front that must be feasible as well as non-dominated.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from optimi_lab.core.constraints import Constraint, ConstraintSet
from optimi_lab.core.errors import refuse, require_finite
from optimi_lab.core.optimize import Record, optimize
from optimi_lab.core.outcomes import Evaluation
from optimi_lab.core.pareto import non_dominated_sorting
from optimi_lab.core.protocols import Proposer
from optimi_lab.core.sampling import SampleSpec
from optimi_lab.core.space import Objective, ObjectiveSet, Variable, VariableSet
from optimi_lab.intelligent_algorithm import nsga2

__all__ = [
    'Algorithm',
    'Problem',
    'bnh',
    'hypervolume',
    'hypervolume_2d',
    'igd',
    'solve',
    'zdt1',
    'zdt2',
]


Algorithm = Callable[..., Callable[[int], Proposer]]
"""What :func:`solve` takes for ``algorithm``: a FACTORY of proposer factories.

The two levels are the package's own, not an indirection added here. Every name in
:mod:`optimi_lab.intelligent_algorithm` -- ``nsga2``, ``nsga3``, ``mode``, ``moead``, ``mopso`` --
is called with the space and its settings and hands back the ``Callable[[int], Proposer]`` that
:func:`~optimi_lab.core.optimize.optimize` builds once the sampler has fixed the population size.
So ``algorithm`` is the factory by NAME and the population count stays the one number the loop
measured, which is what keeps a second population size from existing to disagree with the first.
"""


@dataclass(frozen=True)
class Problem:
    """A box-bounded multi-objective minimization problem, optionally under constraints.

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
        constraints: The declared constraints, or ``None`` for an unconstrained problem.
        constraint_values: Vectorized ``X -> G`` of shape ``(n_sample, n_con)``, the raw values the
            constraints bound; given exactly when ``constraints`` is.

    """

    name: str
    n_var: int
    n_obj: int
    xl: np.ndarray
    xu: np.ndarray
    evaluate: Callable[[np.ndarray], np.ndarray]
    pareto_front: Callable[[int], np.ndarray] | None = None
    constraints: ConstraintSet | None = None
    constraint_values: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self) -> None:
        if (self.constraints is None) != (self.constraint_values is None):
            refuse(
                f'problem {self.name!r} must declare constraints and constraint_values together: the set names the '
                f'columns and the callable produces them, and either alone describes no constraint'
            )


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


def bnh() -> Problem:
    """BNH (Binh and Korn): two objectives over ``[0, 5] x [0, 3]``, two circular constraints.

    ``f1 = 4 x1^2 + 4 x2^2`` and ``f2 = (x1 - 5)^2 + (x2 - 5)^2``, subject to
    ``(x1 - 5)^2 + x2^2 <= 25`` and ``(x1 - 8)^2 + (x2 + 3)^2 >= 7.7``.
    """

    def evaluate(x: np.ndarray) -> np.ndarray:
        return np.column_stack([4.0 * x[:, 0] ** 2 + 4.0 * x[:, 1] ** 2, (x[:, 0] - 5.0) ** 2 + (x[:, 1] - 5.0) ** 2])

    def constraint_values(x: np.ndarray) -> np.ndarray:
        return np.column_stack([(x[:, 0] - 5.0) ** 2 + x[:, 1] ** 2, (x[:, 0] - 8.0) ** 2 + (x[:, 1] + 3.0) ** 2])

    return Problem(
        name='BNH',
        n_var=2,
        n_obj=2,
        xl=np.zeros(2),
        xu=np.array([5.0, 3.0]),
        evaluate=evaluate,
        constraints=ConstraintSet((Constraint('circle_inside', '<=', 25.0), Constraint('circle_outside', '>=', 7.7))),
        constraint_values=constraint_values,
    )


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


def hypervolume(front: np.ndarray, ref_point: np.ndarray) -> float:
    """Exact dominated hypervolume of an ``m``-objective ``front`` w.r.t. ``ref_point``, ``m >= 2``.

    Hypervolume by slicing objectives (HSO): the points are swept in ascending last objective and
    each slab up to the next value adds the ``(m - 1)``-objective volume of the points already
    swept times its depth, recursing down to :func:`hypervolume_2d`. Exact; its cost grows like
    ``n ** (m - 1)``, which is cheap for ``m <= 4`` and a few hundred points. Minimization frame;
    only points strictly dominating ``ref_point`` contribute.
    """
    front = np.asarray(front, dtype=float)
    ref_point = np.asarray(ref_point, dtype=float)
    if front.ndim != 2 or front.shape[1] < 2 or ref_point.shape != (front.shape[1],):
        refuse(
            f'a hypervolume needs a (n_points, m >= 2) front and a reference of width m; '
            f'got front {front.shape} and reference {ref_point.shape}'
        )
    require_finite(front, 'a front to measure')
    require_finite(ref_point, 'a hypervolume reference point')
    return _sliced_volume(front[np.all(front < ref_point, axis=1)], ref_point)


def _sliced_volume(points: np.ndarray, ref_point: np.ndarray) -> float:
    if points.shape[0] == 0:
        return 0.0
    if points.shape[1] == 2:
        return hypervolume_2d(points, ref_point)
    points = points[non_dominated_sorting(points)]
    points = points[np.argsort(points[:, -1])]
    depths = np.diff(np.append(points[:, -1], ref_point[-1]))
    return float(sum(
        _sliced_volume(points[: index + 1, :-1], ref_point[:-1]) * depth
        for index, depth in enumerate(depths)
        if depth > 0.0
    ))


def solve(
    problem: Problem,
    *,
    algorithm: Algorithm = nsga2,
    pop_size: int = 50,
    max_iter: int = 60,
    seed: int = 0,
) -> Record:
    """Solve ``problem`` with ``algorithm`` and return the :class:`~optimi_lab.core.optimize.Record`.

    Thin wiring over :func:`~optimi_lab.core.optimize.optimize`, and every column is a FIELD rather
    than a position in a parallel list: one ``Variable`` per decision dimension, one ``Objective``
    per column, all minimized -- which is what a :class:`Problem`'s own ``evaluate`` promises. The
    seed reaches the sampler and the proposer's own generator, so two calls with one seed are one
    run.

    ``algorithm`` IS A PARAMETER BECAUSE THE CHOICE WAS NEVER THIS LIBRARY'S TO MAKE SILENTLY.
    Until 2026-09-19 this helper named :func:`~optimi_lab.intelligent_algorithm.nsga2` and offered
    no say in it, so four of the five algorithms this package ships had no route through the one
    entry point it offers for scoring an algorithm -- a benchmark module that could benchmark a
    fifth of the thing it exists to judge. The hardcoding was not a decision taken here: it was the
    CONSUMER's, whose wrapper described itself as "thin wiring over the optimi_lab NSGA-II", and it
    outlived the rewrite of everything underneath it. Measured at ``pop_size=50, max_iter=60,
    seed=0``, the default is not even the best of the set: ``mode`` reaches IGD 0.0006 on ZDT1 and
    0.0005 on ZDT2 against NSGA-II's 0.0665 and 0.1204. ``nsga2`` remains the DEFAULT, because a
    one-call helper needs one, and it is now a default rather than a ceiling.

    THE RETURN IS THE RECORD AND NOT TWO BARE MATRICES. It was ``(pareto_x, pareto_f)`` until
    2026-09-19, which is the one shape :class:`~optimi_lab.core.optimize.Record`'s own docstring
    names as the defect it exists to remove -- "four bare matrices whose columns were named only by
    the optimizer object the caller still had to hold on to". That tuple was this library's
    CONSUMER's signature, carried unchanged across the rewrite that replaced everything underneath
    it, and it discarded three facts the loop had already computed: which variable each input
    column is, which objective each value column is, and the
    :class:`~optimi_lab.core.outcomes.Outcome` of every cell. A caller that wants only the front
    asks for it -- ``solve(problem).pareto()`` -- and THAT call is where the refusal belongs: a run
    in which no point computed every objective has no front, and an empty one would read exactly
    like a run that converged on nothing.

    ``max_iter`` counts generations AFTER the initial sample, so a run evaluates ``max_iter + 1``
    batches -- the initial sample IS the first one.

        >>> record = solve(zdt1(n_var=3), pop_size=8, max_iter=2, seed=0)
        >>> len(record), list(record.space.names), list(record.objectives.names)
        (24, ['x0', 'x1', 'x2'], ['f0', 'f1'])
        >>> record.pareto().values.shape[1]
        2

    """
    space = VariableSet(
        tuple(
            Variable(f'x{index}', float(problem.xl[index]), float(problem.xu[index])) for index in range(problem.n_var)
        )
    )
    objectives = ObjectiveSet(tuple(Objective(f'f{index}') for index in range(problem.n_obj)))

    def evaluate(points: np.ndarray) -> Evaluation:
        values = np.asarray(problem.evaluate(points), dtype=float)
        if problem.constraint_values is None:
            return Evaluation(inputs=points, values=values)
        return Evaluation(inputs=points, values=values, constraints=problem.constraint_values(points))

    return optimize(
        space=space,
        objectives=objectives,
        constraints=problem.constraints,
        evaluate=evaluate,
        build_proposer=algorithm(space, seed=seed),
        spec=SampleSpec(n_samples=pop_size, seed=seed),
        n_batches=max_iter + 1,
    )
