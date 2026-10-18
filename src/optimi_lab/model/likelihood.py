"""The maximum-likelihood machinery every Kriging model shares: one Gaussian log-likelihood and one fit.

* :func:`gaussian_log_likelihood` -- ``log N(values; 0, C)`` and its gradient
  ``0.5 tr((alpha alpha^T - C^-1) dC/dtheta_i)`` for each slope ``dC/dtheta_i``; the factorisation
  is numerics-lab's jittered Cholesky, and a matrix it cannot factor has likelihood ``-inf``.
* :func:`minimise` -- limited-memory BFGS: numerics-lab's two-loop recursion for the direction and
  its strong-Wolfe search for the step (a failed search ends the descent at the best point).
  Unconstrained: a bounded hyperparameter is mapped through :class:`Box`.
* :class:`Box` -- ``theta = lo + (hi - lo) sigmoid(u)``, so an unconstrained ``u`` is a bounded
  ``theta``; its slope converts a gradient in ``theta`` to one in ``u``.
* :func:`maximise_likelihood` -- the multi-start fit: the declared start and *restarts* seeded draws
  from the box, keeping the best.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numerics_lab.line_search import WolfeSearchFailed, strong_wolfe
from numerics_lab.linear.cholesky import JitteredCholesky, jittered_cholesky
from numerics_lab.quasi_newton import SecantPair, secant_pair, two_loop
from scipy.linalg import LinAlgError
from scipy.special import expit

__all__ = ['Box', 'Minimum', 'factor', 'gaussian_log_likelihood', 'maximise_likelihood', 'minimise']

_LOG_2PI = float(np.log(2 * np.pi))

Objective = Callable[[np.ndarray], tuple[float, np.ndarray]]


def factor(covariance: np.ndarray) -> JitteredCholesky | None:
    """numerics-lab's jittered Cholesky of *covariance*, or ``None`` when no rung factors it."""
    try:
        return jittered_cholesky(covariance)
    except LinAlgError:
        return None


def gaussian_log_likelihood(covariance: np.ndarray, values: np.ndarray, slopes: np.ndarray) -> tuple[float, np.ndarray]:
    """``log N(values; 0, covariance)`` and its gradient along each of ``slopes (p, n, n)``."""
    chol = factor(covariance)
    if chol is None:
        return -np.inf, np.full(len(slopes), np.nan)
    alpha = chol.solve(values)
    value = -0.5 * values @ alpha - np.sum(np.log(np.diag(chol.lower))) - 0.5 * values.size * _LOG_2PI
    inner = np.outer(alpha, alpha) - chol.solve(np.eye(values.size))
    return float(value), 0.5 * np.einsum('ij,pji->p', inner, slopes)


@dataclass(frozen=True, slots=True, eq=False)
class Minimum:
    """Where the descent stopped, the value there, and the iterations spent."""

    x: np.ndarray
    value: float
    iterations: int


def minimise(
    fun: Objective,
    x0: np.ndarray,
    *,
    gradient_tolerance: float = 1e-8,
    value_tolerance: float = 1e-13,
    max_iterations: int = 1000,
    memory: int = 10,
) -> Minimum:
    """Limited-memory BFGS from *x0* on ``fun(x) -> (value, gradient)``; tolerances in fun's own units."""
    x = np.asarray(x0, dtype=float)
    f, g = fun(x)
    pairs: list[SecantPair] = []
    gamma = 1.0
    for iteration in range(max_iterations):
        if not np.all(np.isfinite(g)) or np.max(np.abs(g), initial=0.0) <= gradient_tolerance:
            return Minimum(x, f, iteration)
        d = -two_loop(g, pairs, lambda v, scale=gamma: scale * v)
        slope = float(g @ d)
        if not slope < 0:
            pairs.clear()
            d, slope = -g, -float(g @ g)
        seen: dict[float, tuple[float, np.ndarray]] = {}

        def along(a: float, x=x, d=d, seen=seen) -> tuple[float, float]:
            value, grad = fun(x + a * d)
            seen[a] = (value, grad)
            return (value if np.isfinite(value) else np.inf), float(grad @ d) if np.all(np.isfinite(grad)) else np.nan

        first = 1.0 if pairs else min(1.0, 1.0 / np.max(np.abs(g)))
        try:
            step = strong_wolfe(along, f, slope, step=first).step
        except (WolfeSearchFailed, ZeroDivisionError):
            return Minimum(x, f, iteration)
        f_new, g_new = seen[step]
        s, y = step * d, g_new - g
        pair = secant_pair(s, y)
        if pair is not None and float(y @ s) > 0:
            pairs = [*pairs[-(memory - 1) :], pair]
            gamma = float(s @ y) / float(y @ y)
        settled = f - f_new <= value_tolerance * max(abs(f), abs(f_new), 1.0)
        x, f, g = x + s, f_new, g_new
        if settled:
            return Minimum(x, f, iteration + 1)
    return Minimum(x, f, max_iterations)


@dataclass(frozen=True, slots=True, eq=False)
class Box:
    """Bounds ``lo < theta < hi`` reached from an unconstrained ``u`` by ``lo + (hi - lo) sigmoid(u)``."""

    lo: np.ndarray
    hi: np.ndarray

    def theta(self, u: np.ndarray) -> np.ndarray:
        return self.lo + (self.hi - self.lo) * expit(u)

    def slope(self, u: np.ndarray) -> np.ndarray:
        s = expit(u)
        return (self.hi - self.lo) * s * (1 - s)

    def u(self, theta: np.ndarray) -> np.ndarray:
        p = np.clip((np.asarray(theta, dtype=float) - self.lo) / (self.hi - self.lo), 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p))


def maximise_likelihood(
    log_likelihood: Objective, start: np.ndarray, box: Box, *, restarts: int, seed: int
) -> tuple[np.ndarray, float]:
    """The best ``(theta, log-likelihood)`` over *start* and *restarts* seeded uniform draws inside *box*."""

    def negative(u: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = log_likelihood(box.theta(u))
        return -value, -gradient * box.slope(u)

    rng = np.random.default_rng(seed)
    starts = [box.u(start)] + [box.u(rng.uniform(box.lo, box.hi)) for _ in range(restarts)]
    best = min((minimise(negative, u0) for u0 in starts), key=lambda m: m.value if np.isfinite(m.value) else np.inf)
    return box.theta(best.x), -best.value
