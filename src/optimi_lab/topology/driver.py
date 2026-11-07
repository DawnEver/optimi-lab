"""The topology loop: continuation stages of a gradient method over a :class:`~.provider.TopologyProblem`.

Each :class:`Continuation` stage sets the projection sharpness ``beta`` and runs the optimizer from the
previous stage's design; the optimizer restarts with the stage (its asymptotes and multipliers describe
the old projection, not the new one). A stage ends when the design moves less than ``tol`` (max norm),
when its ``iterations`` run out, or when the optimizer converges; ``max_iter`` caps the whole run.

Optimizers (:class:`Optimizer`): MMA and GCMMA are the library's own
(:func:`optimi_lab.solver.moving_asymptotes.mma`, any number of constraints, either sign of sensitivity);
OC is the optimality-criteria fixed point (Bendsoe 1995; "top88"), for ONE constraint whose sensitivity
is positive and an objective that only falls as density rises, as in compliance minimisation.

Every evaluation is an :class:`Iteration` of the run's history: stage, ``beta``, objective (raw, the KS
aggregate when robust), the normalised constraints (``<= 0`` is feasible), the max-norm design change and
the nominal realization's grayness. GCMMA's rejected inner candidates are evaluations too, and are recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from optimi_lab.core.errors import coerce_enum, refuse
from optimi_lab.solver.moving_asymptotes import mma
from optimi_lab.topology.density import ROBUST_ETAS, DensityFilter, DesignChain, HelmholtzFilter
from optimi_lab.topology.provider import Evaluation, PhysicsProvider, TopologyProblem

__all__ = ['DEFAULT_CONTINUATION', 'Continuation', 'Iteration', 'Optimizer', 'TopologyResult', 'topology_optimize']

_OC_DAMPING = 0.5


class Optimizer(StrEnum):
    """The update rules :func:`topology_optimize` accepts."""

    MMA = 'mma'
    GCMMA = 'gcmma'
    OC = 'oc'


@dataclass(frozen=True, slots=True)
class Continuation:
    """Projection sharpness per stage (non-decreasing), the evaluation budget per stage, the settle tolerance."""

    betas: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)
    iterations: int = 40
    tol: float = 1e-3

    def __post_init__(self) -> None:
        if not self.betas or any(b < 0.0 for b in self.betas) or list(self.betas) != sorted(self.betas):
            refuse(f'continuation betas must be a non-empty, non-decreasing run of non-negatives, got {self.betas}')
        if self.iterations < 1 or not self.tol > 0.0:
            refuse(f'a stage needs iterations >= 1 and tol > 0, got {self.iterations} and {self.tol}')


#: Five doublings of beta from 1, 40 evaluations each.
DEFAULT_CONTINUATION = Continuation()


@dataclass(frozen=True, slots=True)
class Iteration:
    """One evaluation of the run; see the module docstring."""

    stage: int
    beta: float
    objective: float
    constraints: tuple[float, ...]
    change: float
    grayness: float


@dataclass(frozen=True, slots=True, eq=False)
class TopologyResult:
    """The final design variables, its physical realizations, and the run's history."""

    design: np.ndarray
    physical: tuple[np.ndarray, ...]
    history: tuple[Iteration, ...]


class _Settled(Exception):
    def __init__(self, design: np.ndarray, *, finished: bool) -> None:
        super().__init__('settled')
        self.design, self.finished = design, finished


def _oc_step(evaluation: Evaluation, move: float) -> np.ndarray:
    """The optimality-criteria update, bisecting the multiplier on the linearised constraint."""
    point = evaluation.linearization
    if point.c.size != 1:
        refuse(f'OC takes exactly one constraint, this problem has {point.c.size} -- use mma')
    z, df, dc = point.x, np.minimum(point.df[0], 0.0), point.dc[0]
    if np.any(dc <= 0.0):
        refuse('OC needs a constraint whose sensitivity is positive everywhere -- use mma')
    low, high = 0.0, 1e9
    candidate = z
    while (high - low) > 1e-9 * (low + high):
        lam = 0.5 * (low + high)
        candidate = np.clip(z * (-df / (lam * dc)) ** _OC_DAMPING, np.maximum(0.0, z - move), np.minimum(1.0, z + move))
        if point.c[0] + dc @ (candidate - z) > 0.0:
            low = lam
        else:
            high = lam
    return candidate


def topology_optimize(
    provider: PhysicsProvider,
    density_filter: DensityFilter | HelmholtzFilter,
    *,
    optimizer: Optimizer | str = Optimizer.MMA,
    continuation: Continuation = DEFAULT_CONTINUATION,
    robust: bool = False,
    etas: tuple[float, ...] = ROBUST_ETAS,
    x0: np.ndarray | None = None,
    max_iter: int | None = None,
    ks_rho: float = 40.0,
    move: float = 0.2,
) -> TopologyResult:
    """Minimise *provider*'s objective under its constraints over density, through *density_filter*.

    *robust* projects at every threshold of *etas* (else at the middle one only); *x0* defaults to the
    uniform design at 0.5; *move* caps each step's change of a design variable.
    """
    optimizer = coerce_enum(Optimizer, optimizer, 'the topology optimizer')
    chain = DesignChain(density_filter, etas=etas if robust else (etas[len(etas) // 2],))
    problem = TopologyProblem(provider, chain, ks_rho=ks_rho)
    n = provider.domain.n_design
    design = np.full(n, 0.5) if x0 is None else np.clip(np.asarray(x0, dtype=float), 0.0, 1.0)
    budget = max_iter if max_iter is not None else len(continuation.betas) * continuation.iterations
    history: list[Iteration] = []
    for stage, beta in enumerate(continuation.betas):
        chain.beta = float(beta)
        previous = [design]
        last: list[Evaluation] = []

        def evaluate(z: np.ndarray, stage=stage, beta=beta, previous=previous, last=last):
            evaluation = problem.evaluate(z)
            change = float(np.max(np.abs(z - previous[0])))
            history.append(
                Iteration(stage, beta, evaluation.objective, evaluation.constraints, change, evaluation.grayness)
            )
            previous[0] = z.copy()
            last[:] = [evaluation]
            in_stage = sum(it.stage == stage for it in history)
            if len(history) >= budget or (in_stage > 1 and change < continuation.tol):
                raise _Settled(z.copy(), finished=len(history) >= budget)
            return evaluation.linearization

        try:
            if optimizer is Optimizer.OC:
                z = design
                for _ in range(continuation.iterations):
                    evaluate(z)
                    z = _oc_step(last[0], move)
                design = z
            else:
                solution = mma(
                    evaluate,
                    design,
                    np.zeros(n),
                    np.ones(n),
                    conservative=optimizer is Optimizer.GCMMA,
                    max_iterations=continuation.iterations,
                    move=move,
                )
                design = solution.x
        except _Settled as settled:
            design = settled.design
            if settled.finished:
                break
    return TopologyResult(design, chain.forward(design), tuple(history))
