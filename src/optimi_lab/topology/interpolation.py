"""Material interpolation and smooth aggregation -- the two scalar maps every density formulation reads.

:func:`interpolate` maps a density to ANY property between N ordered material values (two is the usual
solid/void pair; three or more is the ordered multi-material form): within the segment ``[k, k + 1] / (N - 1)``
the local density ``t`` is weighted by the scheme's ``w(t)``,

* SIMP -- ``w = t^p`` (Bendsoe 1989; Bendsoe and Sigmund 1999);
* RAMP -- ``w = t / (1 + q (1 - t))`` (Stolpe and Svanberg 2001), finite slope at ``t = 0``,

so the property is ``v_k + w(t) (v_{k+1} - v_k)``. Nothing here knows WHICH property it is: a caller
interpolates a stiffness, a conductivity, a permeability or a reluctivity, and which one to interpolate
(and in which direction) is the caller's modelling decision.

:func:`p_norm` and :func:`ks` are smooth maxima with exact gradients: the p-norm
``(sum |v|^p)^(1/p)`` (scale-free form) bounds ``max |v|`` from above by at most a factor ``n^(1/p)``;
the Kreisselmeier-Steinhauser ``max + log(sum exp(rho (v - max))) / rho`` by at most ``log(n) / rho``.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import numpy as np

from optimi_lab.core.errors import coerce_enum, refuse

__all__ = ['Interpolation', 'interpolate', 'ks', 'p_norm']


class Interpolation(StrEnum):
    """The interpolation schemes :func:`interpolate` accepts."""

    SIMP = 'simp'
    RAMP = 'ramp'


def interpolate(
    rho: np.ndarray,
    values: Sequence[float | np.ndarray],
    *,
    scheme: Interpolation | str = Interpolation.SIMP,
    penalty: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """``(property, d property / d rho)`` between the ordered material *values*; see the module docstring."""
    scheme = coerce_enum(Interpolation, scheme, 'the interpolation scheme')
    if len(values) < 2:
        refuse(f'interpolation needs at least two material values, got {len(values)}')
    *stacked, rho = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in values], np.asarray(rho, dtype=float))
    stacked = np.stack(stacked)
    segments = len(values) - 1
    k = np.clip(np.floor(rho * segments).astype(int), 0, segments - 1)
    t = rho * segments - k
    if scheme is Interpolation.SIMP:
        w, dw = t**penalty, penalty * t ** (penalty - 1.0)
    else:
        den = 1.0 + penalty * (1.0 - t)
        w, dw = t / den, (1.0 + penalty) / den**2
    low = np.take_along_axis(stacked, k[None], axis=0)[0]
    jump = np.take_along_axis(stacked, k[None] + 1, axis=0)[0] - low
    return low + w * jump, dw * segments * jump


def p_norm(values: np.ndarray, p: float) -> tuple[float, np.ndarray]:
    """``((sum |v|^p)^(1/p), its gradient)``, evaluated scale-free so a large *p* does not overflow."""
    values = np.asarray(values, dtype=float)
    magnitude = np.abs(values)
    top = float(magnitude.max())
    if top == 0.0:
        return 0.0, np.zeros_like(values)
    value = top * float(np.sum((magnitude / top) ** p)) ** (1.0 / p)
    return value, np.sign(values) * (magnitude / value) ** (p - 1.0)


def ks(values: np.ndarray, rho: float) -> tuple[float, np.ndarray]:
    """``(KS aggregate, its gradient)``: a smooth upper bound on ``max v`` within ``log(n) / rho``."""
    values = np.asarray(values, dtype=float)
    top = float(values.max())
    w = np.exp(rho * (values - top))
    return top + float(np.log(w.sum())) / rho, w / w.sum()
