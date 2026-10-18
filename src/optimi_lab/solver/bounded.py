"""Single-objective gradient methods on a box: projected gradient, L-BFGS-B, Newton-CG trust region.

Each minimises ``f`` over ``lo <= x <= hi`` and halts CONVERGED when the projected gradient
``|P(x - grad f) - x|`` falls to *tolerance*. Constraints other than the box are refused here (the
capability match names the penalty transform, or a constrained method).

* :func:`projected_gradient` -- the spectral projected gradient: a Barzilai-Borwein trial length
  along ``-grad f``, searched along the projected arc by Armijo backtracking.
* :func:`lbfgsb` -- the limited-memory BFGS model on a box: the model ``theta I`` updated by the last
  *memory* secant pairs, minimised EXACTLY over the box by :func:`~optimi_lab.operator.bound_qp`
  (where Byrd-Lu-Nocedal approximate that subproblem by a Cauchy point and a subspace step), then a
  strong-Wolfe search capped at the box. A Wolfe failure falls back to Armijo backtracking.
* :func:`newton_cg` -- a trust-region Newton method: Steihaug-CG on the Hessian of the free variables,
  each Hessian-vector product a forward difference of gradients (one evaluation per CG iteration),
  the trial point projected onto the box and judged by the agreement ratio.

Every line search, trust-region update, secant pair and Krylov solve is numerics-lab's.
"""

from __future__ import annotations

import numpy as np
from numerics_lab.line_search import WolfeSearchFailed, armijo, backtracking, strong_wolfe
from numerics_lab.linear.krylov import steihaug_cg
from numerics_lab.quasi_newton import bfgs_update, secant_pair
from numerics_lab.trust_region import agreement, update_radius

from optimi_lab.core.errors import refuse
from optimi_lab.operator import Linearization, bound_qp
from optimi_lab.solver.smooth import Evaluate, Halt, Solution, trial

__all__ = ['lbfgsb', 'newton_cg', 'projected_gradient']

_ARMIJO = 1e-4
#: The trust region's acceptance threshold on the agreement ratio.
_ACCEPT = 1e-4
#: A predicted decrease below this, relative to |f|, is rounding: the model has nothing left to offer.
_ROUNDING = 1e-14


def _gap(point: Linearization, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.max(np.abs(np.clip(point.x - point.gradient, lo, hi) - point.x), initial=0.0))


def _start(evaluate: Evaluate, x0: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Linearization:
    point = evaluate(np.clip(np.asarray(x0, dtype=float), lo, hi))
    if point.c.size or point.h.size:
        refuse(f'a box method takes no constraints; this problem has {point.c.size + point.h.size} -- use sqp')
    return point


def _halt(point: Linearization, box: tuple, iterations: int, halt: Halt, reason: str = '') -> Solution:
    return Solution(halt, point.x, point, iterations, _gap(point, *box), reason=reason)


def _projected_search(evaluate: Evaluate, point: Linearization, d: np.ndarray, lo, hi):
    """Armijo backtracking along the projected arc ``P(x + t d)``; the accepted point or ``None``."""

    def step(t: float) -> Linearization | None:
        return trial(evaluate, np.clip(point.x + t * d, lo, hi))

    def accept(_: float, candidate: Linearization) -> bool:
        return candidate.objective <= point.objective + _ARMIJO * float(point.gradient @ (candidate.x - point.x))

    search = backtracking(step, accept, tries=40)
    return search.trial if search.accepted else None


def projected_gradient(
    evaluate: Evaluate, x0, lo, hi, *, tolerance: float = 1e-8, max_iterations: int = 2000
) -> Solution:
    """The spectral projected gradient method; see the module docstring."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    point = _start(evaluate, x0, lo, hi)
    length = 1.0
    for iteration in range(max_iterations):
        if _gap(point, lo, hi) <= tolerance:
            return _halt(point, (lo, hi), iteration, Halt.CONVERGED)
        accepted = _projected_search(evaluate, point, -length * point.gradient, lo, hi)
        if accepted is None:
            return _halt(point, (lo, hi), iteration, Halt.STALLED, 'no projected step decreased f')
        s, y = accepted.x - point.x, accepted.gradient - point.gradient
        sy = float(s @ y)
        length = float(np.clip(float(s @ s) / sy, 1e-10, 1e10)) if sy > 0 else 1.0
        point = accepted
    return _halt(point, (lo, hi), max_iterations, Halt.ITERATION_LIMIT)


def lbfgsb(
    evaluate: Evaluate, x0, lo, hi, *, memory: int = 10, tolerance: float = 1e-8, max_iterations: int = 500
) -> Solution:
    """Limited-memory BFGS on a box; see the module docstring."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    point = _start(evaluate, x0, lo, hi)
    pairs: list = []
    n = point.x.size
    for iteration in range(max_iterations):
        if _gap(point, lo, hi) <= tolerance:
            return _halt(point, (lo, hi), iteration, Halt.CONVERGED)
        theta = float(pairs[-1].y @ pairs[-1].y) * pairs[-1].rho if pairs else 1.0
        model = theta * np.eye(n)
        for pair in pairs:
            model = bfgs_update(model, pair.s, pair.y)
        d = bound_qp(model, point.gradient, lo - point.x, hi - point.x, x0=np.zeros(n)).x
        slope = float(point.gradient @ d)
        if not slope < 0.0:
            return _halt(point, (lo, hi), iteration, Halt.STALLED, 'the model step is not a descent direction')
        seen: dict[float, Linearization | None] = {}

        def along(t: float, d: np.ndarray = d, seen: dict = seen, base: Linearization = point) -> Linearization | None:
            if t not in seen:
                seen[t] = trial(evaluate, np.clip(base.x + t * d, lo, hi))
            return seen[t]

        def phi_and_slope(t: float, d: np.ndarray = d) -> tuple[float, float]:
            candidate = along(t)
            return (np.inf, 0.0) if candidate is None else (candidate.objective, float(candidate.gradient @ d))

        try:
            accepted = along(strong_wolfe(phi_and_slope, point.objective, slope, step=1.0, max_step=1.0).step)
        except (WolfeSearchFailed, ZeroDivisionError):  # numerics-lab zoom divides by a collapsed bracket
            search = armijo(
                lambda t: None if along(t) is None else along(t).objective, point.objective, slope, c1=_ARMIJO
            )
            accepted = along(search.step) if search.accepted else None
        if accepted is None:
            return _halt(point, (lo, hi), iteration, Halt.STALLED, 'no step along the model direction decreased f')
        pair = secant_pair(accepted.x - point.x, accepted.gradient - point.gradient)
        if pair is not None and pair.rho > 0:
            pairs = [*pairs, pair][-memory:]
        point = accepted
    return _halt(point, (lo, hi), max_iterations, Halt.ITERATION_LIMIT)


def newton_cg(
    evaluate: Evaluate,
    x0,
    lo,
    hi,
    *,
    radius: float = 0.25,
    max_radius: float = 1.0,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> Solution:
    """A trust-region Newton-CG method with finite-difference Hessian-vector products; see the module docstring."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    point = _start(evaluate, x0, lo, hi)
    for iteration in range(max_iterations):
        gap = _gap(point, lo, hi)
        if gap <= tolerance:
            return _halt(point, (lo, hi), iteration, Halt.CONVERGED)
        g = point.gradient
        cauchy, band = (
            np.clip(point.x - g, lo, hi),
            2.0 * min(1e-3, gap),
        )  # Bertsekas: held near a bound it is pushed onto
        free = ~(((point.x <= lo + band) & (g > 0)) | ((point.x >= hi - band) & (g < 0)))

        def hvp(v: np.ndarray, free: np.ndarray = free, g: np.ndarray = g, x: np.ndarray = point.x) -> np.ndarray:
            full = np.zeros_like(x)
            full[free] = v
            size = 1.5e-8 * (1.0 + float(np.linalg.norm(x))) / max(float(np.linalg.norm(full)), 1e-300)
            for sign in (1.0, -1.0):
                shifted = x + sign * size * full
                if np.all(shifted >= lo) and np.all(shifted <= hi):
                    return (sign * (evaluate(shifted).gradient - g) / size)[free]
            refuse('a Hessian-vector product would leave the box on both sides; the box is narrower than the step')
            return None

        step = np.zeros_like(point.x)
        step[free] = steihaug_cg(hvp, g[free], radius, rtol=min(0.5, float(np.sqrt(gap)))).step
        step[~free] = np.clip(cauchy[~free] - point.x[~free], -radius, radius)
        candidate_x = np.clip(point.x + step, lo, hi)
        taken = candidate_x - point.x
        predicted = -(float(g @ taken) + 0.5 * float(taken[free] @ hvp(taken[free]))) if np.any(taken) else 0.0
        if gap <= np.sqrt(tolerance) and predicted <= _ROUNDING * (1.0 + abs(point.objective)):
            reason = 'the model predicts no decrease above rounding'
            return _halt(point, (lo, hi), iteration, Halt.CONVERGED, reason)
        candidate = trial(evaluate, candidate_x)
        actual = -np.inf if candidate is None else point.objective - candidate.objective
        rho = agreement(actual, predicted)
        radius = update_radius(radius, rho, float(np.linalg.norm(taken)), max_radius=max_radius)
        if rho > _ACCEPT and candidate is not None:
            point = candidate
        elif radius < 1e-14:
            return _halt(point, (lo, hi), iteration, Halt.STALLED, 'the trust region collapsed')
    return _halt(point, (lo, hi), max_iterations, Halt.ITERATION_LIMIT)
