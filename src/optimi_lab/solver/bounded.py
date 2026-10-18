"""Single-objective gradient methods on a box: projected gradient, L-BFGS-B, Newton-CG trust region.

Each minimises ``f`` over ``lo <= x <= hi`` and halts CONVERGED when the projected gradient
``|P(x - grad f) - x|`` falls to :func:`~optimi_lab.solver.smooth.kkt_bar` (``rtol`` against the
starting gradient, floored by ``atol``). Constraints other than the box are refused here.

* :func:`projected_gradient` -- the spectral projected gradient: a Barzilai-Borwein trial length
  along ``-grad f``, searched along the projected arc by Armijo backtracking.
* :func:`lbfgsb` -- L-BFGS-B (Byrd, Lu, Nocedal and Zhu 1995): the limited-memory BFGS model, the
  generalised Cauchy point along the projected path (:func:`~optimi_lab.operator.cauchy_point`),
  subspace minimisation over the variables it leaves free (conjugate gradients, truncated to the
  box -- the direct primal method), then a strong-Wolfe search capped at the box with Armijo as the
  fallback. numerics-lab publishes no compact L-BFGS matrix, so the model is ``theta I`` carried
  through the stored pairs by its dense ``bfgs_update`` -- the same matrix, assembled densely.
* :func:`newton_cg` -- a trust-region Newton method: Steihaug-CG on the free variables' Hessian.
  The Hessian-vector product is the caller's ``hvp(x, v)`` -- an oracle's declared product -- or,
  ONLY when asked for by name, ``hvp="finite_difference"``: a forward difference of gradients, one
  evaluation per product. The choice is recorded in the solution's evidence; there is no default.

Every line search, trust-region update, secant pair and Krylov solve is numerics-lab's.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numerics_lab.line_search import WolfeSearchFailed, armijo, backtracking, strong_wolfe
from numerics_lab.linear.krylov import conjugate_gradient, steihaug_cg
from numerics_lab.quasi_newton import bfgs_update, secant_pair
from numerics_lab.trust_region import agreement, update_radius

from optimi_lab.core.errors import refuse
from optimi_lab.operator import Linearization, cauchy_point
from optimi_lab.solver.smooth import ATOL, RTOL, Evaluate, Halt, Solution, kkt_bar, trial

__all__ = ['FINITE_DIFFERENCE', 'lbfgsb', 'newton_cg', 'projected_gradient']

_ARMIJO = 1e-4
#: The trust region's acceptance threshold on the agreement ratio.
_ACCEPT = 1e-4
#: The one name that selects gradient differencing for Newton-CG's Hessian-vector products.
FINITE_DIFFERENCE = 'finite_difference'


def _gap(point: Linearization, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.max(np.abs(np.clip(point.x - point.gradient, lo, hi) - point.x), initial=0.0))


def _start(evaluate: Evaluate, x0: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Linearization:
    point = evaluate(np.clip(np.asarray(x0, dtype=float), lo, hi))
    if point.c.size or point.h.size:
        refuse(f'a box method takes no constraints; this problem has {point.c.size + point.h.size} -- use sqp')
    return point


def _halt(point: Linearization, box: tuple, iterations: int, halt: Halt, reason: str = '', **extra) -> Solution:
    return Solution(halt, point.x, point, iterations, _gap(point, *box), reason=reason, **extra)


def _projected_search(evaluate: Evaluate, point: Linearization, d: np.ndarray, lo, hi):
    """Armijo backtracking along the projected arc ``P(x + t d)``; the accepted point or ``None``."""

    def step(t: float) -> Linearization | None:
        return trial(evaluate, np.clip(point.x + t * d, lo, hi))

    def accept(_: float, candidate: Linearization) -> bool:
        return candidate.objective <= point.objective + _ARMIJO * float(point.gradient @ (candidate.x - point.x))

    search = backtracking(step, accept, tries=40)
    return search.trial if search.accepted else None


def projected_gradient(
    evaluate: Evaluate, x0, lo, hi, *, rtol: float = RTOL, atol: float = ATOL, max_iterations: int = 2000
) -> Solution:
    """The spectral projected gradient method; see the module docstring."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    point = _start(evaluate, x0, lo, hi)
    bar = kkt_bar(point.gradient, rtol=rtol, atol=atol)
    length = 1.0
    for iteration in range(max_iterations):
        if _gap(point, lo, hi) <= bar:
            return _halt(point, (lo, hi), iteration, Halt.CONVERGED)
        accepted = _projected_search(evaluate, point, -length * point.gradient, lo, hi)
        if accepted is None:
            return _halt(point, (lo, hi), iteration, Halt.STALLED, 'no projected step decreased f')
        s, y = accepted.x - point.x, accepted.gradient - point.gradient
        sy = float(s @ y)
        length = float(np.clip(float(s @ s) / sy, 1e-10, 1e10)) if sy > 0 else 1.0
        point = accepted
    return _halt(point, (lo, hi), max_iterations, Halt.ITERATION_LIMIT)


def _subspace_step(point: Linearization, model: np.ndarray, z: np.ndarray, lo, hi) -> np.ndarray:
    """From the Cauchy point ``x + z``: minimise the model over its free variables, truncated to the box."""
    x_c = point.x + z
    free = (x_c > lo) & (x_c < hi)
    if not free.any():
        return z
    reduced = (point.gradient + model @ z)[free]
    step = conjugate_gradient(model[np.ix_(free, free)], -reduced, rtol=1e-12).x
    upper, lower = hi[free] - x_c[free], lo[free] - x_c[free]
    with np.errstate(divide='ignore', invalid='ignore'):
        room = np.where(step > 0, upper / step, np.where(step < 0, lower / step, np.inf))
    alpha = min(1.0, float(np.min(room, initial=np.inf)))
    z = z.copy()
    z[free] += alpha * step
    return z


def lbfgsb(
    evaluate: Evaluate,
    x0,
    lo,
    hi,
    *,
    memory: int = 10,
    rtol: float = RTOL,
    atol: float = ATOL,
    max_iterations: int = 500,
) -> Solution:
    """L-BFGS-B; see the module docstring."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    point = _start(evaluate, x0, lo, hi)
    bar = kkt_bar(point.gradient, rtol=rtol, atol=atol)
    pairs: list = []
    n = point.x.size
    for iteration in range(max_iterations):
        if _gap(point, lo, hi) <= bar:
            return _halt(point, (lo, hi), iteration, Halt.CONVERGED)
        theta = float(pairs[-1].y @ pairs[-1].y) * pairs[-1].rho if pairs else 1.0
        model = theta * np.eye(n)
        for pair in pairs:
            model = bfgs_update(model, pair.s, pair.y)
        z = cauchy_point(point.x, point.gradient, model, lo, hi)
        d = _subspace_step(point, model, z, lo, hi)
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


def _differenced(evaluate: Evaluate, lo, hi) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """``hvp(x, v)`` by a forward difference of gradients, kept inside the box."""

    def hvp(x: np.ndarray, v: np.ndarray) -> np.ndarray:
        size = 1.5e-8 * (1.0 + float(np.linalg.norm(x))) / max(float(np.linalg.norm(v)), 1e-300)
        for sign in (1.0, -1.0):
            shifted = x + sign * size * v
            if np.all(shifted >= lo) and np.all(shifted <= hi):
                return sign * (evaluate(shifted).gradient - evaluate(x).gradient) / size
        refuse('a Hessian-vector product would leave the box on both sides; the box is narrower than the step')
        return None

    return hvp


def newton_cg(
    evaluate: Evaluate,
    x0,
    lo,
    hi,
    *,
    hvp: Callable[[np.ndarray, np.ndarray], np.ndarray] | str | None = None,
    radius: float = 0.25,
    max_radius: float = 1.0,
    rtol: float = RTOL,
    atol: float = ATOL,
    max_iterations: int = 200,
) -> Solution:
    """Trust-region Newton-CG; *hvp* is a declared ``hvp(x, v)`` or the named ``"finite_difference"``."""
    if hvp is None:
        refuse(
            f'newton_cg needs Hessian-vector products: pass hvp(x, v), or hvp={FINITE_DIFFERENCE!r} to difference '
            f'gradients (one evaluation per product)'
        )
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    if isinstance(hvp, str):
        if hvp != FINITE_DIFFERENCE:
            refuse(f'hvp names {hvp!r}; the one named product is {FINITE_DIFFERENCE!r}')
        product, evidence = _differenced(evaluate, lo, hi), (f'hvp:{FINITE_DIFFERENCE}',)
    else:
        product, evidence = hvp, ('hvp:declared',)
    point = _start(evaluate, x0, lo, hi)
    bar = kkt_bar(point.gradient, rtol=rtol, atol=atol)
    for iteration in range(max_iterations):
        gap = _gap(point, lo, hi)
        if gap <= bar:
            return _halt(point, (lo, hi), iteration, Halt.CONVERGED, evidence=evidence)
        g = point.gradient
        cauchy, band = np.clip(point.x - g, lo, hi), 2.0 * min(1e-3, gap)  # Bertsekas: held near a bound
        free = ~(((point.x <= lo + band) & (g > 0)) | ((point.x >= hi - band) & (g < 0)))

        def reduced(v: np.ndarray, free: np.ndarray = free, x: np.ndarray = point.x) -> np.ndarray:
            full = np.zeros_like(x)
            full[free] = v
            return np.asarray(product(x, full), dtype=float)[free]

        step = np.zeros_like(point.x)
        step[free] = steihaug_cg(reduced, g[free], radius, rtol=min(0.5, float(np.sqrt(gap)))).step
        step[~free] = np.clip(cauchy[~free] - point.x[~free], -radius, radius)
        candidate_x = np.clip(point.x + step, lo, hi)
        taken = candidate_x - point.x
        predicted = -(float(g @ taken) + 0.5 * float(taken[free] @ reduced(taken[free]))) if np.any(taken) else 0.0
        candidate = trial(evaluate, candidate_x)
        actual = -np.inf if candidate is None else point.objective - candidate.objective
        rho = agreement(actual, predicted)
        radius = update_radius(radius, rho, float(np.linalg.norm(taken)), max_radius=max_radius)
        if rho > _ACCEPT and candidate is not None:
            point = candidate
        elif radius < 1e-14:
            return _halt(point, (lo, hi), iteration, Halt.STALLED, 'the trust region collapsed', evidence=evidence)
    return _halt(point, (lo, hi), max_iterations, Halt.ITERATION_LIMIT, evidence=evidence)
