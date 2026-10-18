"""The generalised Cauchy point (Byrd, Lu, Nocedal and Zhu 1995, section 4) of a quadratic model on a box.

Along the projected steepest-descent path ``x(t) = P(x - t g)`` the model ``m(z) = g.z + z.B z / 2``
(``z = x(t) - x``) is piecewise quadratic: each variable leaves the path's direction at its
breakpoint ``t_i``, where it reaches a bound. The generalised Cauchy point is the FIRST local
minimiser of ``m`` along that path, found by walking the segments in breakpoint order and stopping
inside the first one whose slope turns non-negative. It fixes which variables L-BFGS-B holds at a
bound; the subspace step then moves the rest.
"""

from __future__ import annotations

import numpy as np

__all__ = ['cauchy_point']


def cauchy_point(x: np.ndarray, g: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """The step ``z`` from *x* to the generalised Cauchy point of ``g.z + z.B z / 2`` on ``[lo, hi]``."""
    x, g = np.asarray(x, dtype=float), np.asarray(g, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        breaks = np.where(g < 0, (x - hi) / g, np.where(g > 0, (x - lo) / g, np.inf))
    breaks = np.maximum(breaks, 0.0)
    d = np.where(breaks > 0, -g, 0.0)
    z = np.zeros_like(x)
    start = 0.0
    for t in np.unique(breaks[np.isfinite(breaks) & (breaks > 0)]):
        slope = float(g @ d + z @ b @ d)
        if slope >= 0.0:
            return z
        curvature = float(d @ b @ d)
        span = t - start
        if curvature > 0.0 and -slope / curvature < span:
            return z + (-slope / curvature) * d
        z = z + span * d
        reached = breaks == t
        z[reached] = np.where(g[reached] < 0, hi[reached], lo[reached]) - x[reached]
        d[reached] = 0.0
        start = t
    slope, curvature = float(g @ d + z @ b @ d), float(d @ b @ d)
    if slope < 0.0 and curvature > 0.0:
        z = z + (-slope / curvature) * d
    return z
