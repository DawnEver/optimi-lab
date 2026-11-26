"""The strong Wolfe line search: the step a quasi-Newton minimizer takes along its descent direction.

- :func:`strong_wolfe` -- a step satisfying the strong Wolfe conditions (Nocedal and Wright,
  Algorithms 3.5 and 3.6, cubic interpolation in the zoom with a bisection safeguard).

The search asks only for ``phi(step)`` and ``phi'(step)`` along the ray. The backtracking and Armijo
searches a root finder also uses are numerics-lab's (``numerics_lab.line_search``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ['WolfeSearchFailed', 'WolfeStep', 'strong_wolfe']


@dataclass(frozen=True)
class WolfeStep:
    """A step satisfying the strong Wolfe conditions, with ``phi`` and ``phi'`` there.

    ``capped`` marks More and Thuente's ``stpmax`` exit: the step IS ``max_step``, sufficient
    decrease holds there and ``phi`` still falls, so no larger step may be tried and the curvature
    condition is not met. A box-capped method (L-BFGS-B) takes it; an unbounded caller never sees it
    unless it set ``max_step``.
    """

    step: float
    phi: float
    slope: float
    evaluations: int
    capped: bool = False


class WolfeSearchFailed(RuntimeError):
    """No step satisfying the strong Wolfe conditions was found within the evaluation budget."""


def _cubic_minimizer(a: float, fa: float, ga: float, b: float, fb: float, gb: float) -> float | None:  # noqa: PLR0917 -- two bracket ends, each (step, phi, slope)
    """The minimizer of the cubic interpolating ``(a, fa, ga)`` and ``(b, fb, gb)``, or ``None``."""
    d1 = ga + gb - 3.0 * (fa - fb) / (a - b)
    radicand = d1 * d1 - ga * gb
    if radicand < 0.0:
        return None
    d2 = math.copysign(math.sqrt(radicand), b - a)
    denominator = gb - ga + 2.0 * d2
    if denominator == 0.0:
        return None
    return b - (b - a) * (gb + d2 - d1) / denominator


def strong_wolfe(
    phi_and_slope: Callable[[float], tuple[float, float]],
    phi0: float,
    slope0: float,
    *,
    c1: float = 1e-4,
    c2: float = 0.9,
    step: float = 1.0,
    max_step: float = 1e10,
    max_evaluations: int = 30,
) -> WolfeStep:
    """A step ``a`` with ``phi(a) <= phi0 + c1 a phi'(0)`` and ``|phi'(a)| <= c2 |phi'(0)|``.

    Nocedal and Wright, Algorithm 3.5 (bracketing, the step doubling up to ``max_step``) and 3.6
    (zoom: a cubic interpolant kept inside the central 80 % of the bracket, else bisection).
    A step that reaches ``max_step`` still falling is returned ``capped`` (see :class:`WolfeStep`).
    Raises :class:`WolfeSearchFailed` when ``phi'(0)`` is not strictly negative (``-0.0``, ``0.0``,
    positive or NaN alike: there is no descent to search along -- whether the point is converged is
    the caller's question, not the search's), when the budget runs out, or when the zoom bracket
    collapses (no float lies strictly inside it, as at a jump in ``phi``); never returns an uncapped
    step that fails a condition.
    """
    if not slope0 < 0.0:
        msg = f"no descent to search along, phi'(0) = {slope0!r}"
        raise WolfeSearchFailed(msg)
    if not 0.0 < c1 < c2 < 1.0:
        msg = f'the Wolfe constants must satisfy 0 < c1 < c2 < 1, got c1={c1!r} c2={c2!r}'
        raise ValueError(msg)
    evaluations = 0

    def evaluate(a: float) -> tuple[float, float]:
        nonlocal evaluations
        if evaluations >= max_evaluations:
            msg = f'no strong Wolfe step within {max_evaluations} evaluations'
            raise WolfeSearchFailed(msg)
        evaluations += 1
        value, slope = phi_and_slope(a)
        return float(value), float(slope)

    def decrease_holds(a: float, value: float) -> bool:
        return math.isfinite(value) and value <= phi0 + c1 * a * slope0

    def zoom(lo: float, f_lo: float, g_lo: float, hi: float, f_hi: float, g_hi: float) -> WolfeStep:  # noqa: PLR0917 -- two bracket ends, each (step, phi, slope)
        while True:
            left, right = min(lo, hi), max(lo, hi)
            if not left < 0.5 * (left + right) < right:
                msg = f'the zoom bracket collapsed at a={lo!r} after {evaluations} evaluations'
                raise WolfeSearchFailed(msg)
            trial = _cubic_minimizer(lo, f_lo, g_lo, hi, f_hi, g_hi) if math.isfinite(f_hi) else None
            margin = 0.1 * (right - left)
            if trial is None or not left + margin <= trial <= right - margin:
                trial = 0.5 * (lo + hi)
            f, g = evaluate(trial)
            if not decrease_holds(trial, f) or f >= f_lo:
                hi, f_hi, g_hi = trial, f, g
                continue
            if abs(g) <= -c2 * slope0:
                return WolfeStep(trial, f, g, evaluations)
            if g * (hi - lo) >= 0.0:
                hi, f_hi, g_hi = lo, f_lo, g_lo
            lo, f_lo, g_lo = trial, f, g

    previous, f_previous, g_previous = 0.0, float(phi0), float(slope0)
    a = float(step)
    while True:
        f, g = evaluate(a)
        if not decrease_holds(a, f) or (evaluations > 1 and f >= f_previous):
            return zoom(previous, f_previous, g_previous, a, f, g)
        if abs(g) <= -c2 * slope0:
            return WolfeStep(a, f, g, evaluations)
        if g >= 0.0:
            return zoom(a, f, g, previous, f_previous, g_previous)
        if a >= max_step:
            return WolfeStep(a, f, g, evaluations, capped=True)
        previous, f_previous, g_previous = a, f, g
        a = min(2.0 * a, max_step)
