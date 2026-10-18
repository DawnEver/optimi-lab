"""KKT assembly: the first-order data of a smooth problem at a point, and what it says about optimality.

The smooth problem every gradient solver sees is

    minimise f(x)   subject to   h(x) = 0,   c(x) <= 0,   lo <= x <= hi

with multipliers ``y`` for ``h`` and ``z >= 0`` for ``c``; the Lagrangian is ``f + y.h + z.c``. The
bounds carry no explicit multipliers: they are absorbed by PROJECTING the Lagrangian gradient onto
the box, which is zero exactly when bound multipliers exist that make the point stationary.

The linear systems this assembles are solved by numerics-lab (``numerics_lab.linear.kkt``), never here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from optimi_lab.core.errors import refuse

__all__ = ['KKTResidual', 'Linearization', 'damped_secant', 'kkt_residual', 'lagrangian_gradient']


def _rows(values: np.ndarray | None, n: int) -> np.ndarray:
    return np.zeros((0, n)) if values is None else np.atleast_2d(np.asarray(values, dtype=float)).reshape(-1, n)


@dataclass(frozen=True, slots=True, eq=False)
class Linearization:
    """Values and first derivatives at *x*: objectives ``f (k,)``/``df (k, n)``, ``c``/``dc``, ``h``/``dh``.

    ``c`` are inequalities ``c <= 0`` and ``h`` equalities ``h = 0``, each already scaled to a
    dimensionless violation. An absent kind is an empty array, never ``None``.
    """

    x: np.ndarray
    f: np.ndarray
    df: np.ndarray
    c: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dc: np.ndarray | None = None
    h: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dh: np.ndarray | None = None

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float)
        n = x.size
        f = np.atleast_1d(np.asarray(self.f, dtype=float))
        parts = {
            'x': x,
            'f': f,
            'df': _rows(self.df, n),
            'c': np.atleast_1d(np.asarray(self.c, dtype=float)),
            'dc': _rows(self.dc, n),
            'h': np.atleast_1d(np.asarray(self.h, dtype=float)),
            'dh': _rows(self.dh, n),
        }
        for value, slope in (('f', 'df'), ('c', 'dc'), ('h', 'dh')):
            if parts[slope].shape[0] != parts[value].size:
                refuse(f'{slope} has {parts[slope].shape[0]} row(s) for {parts[value].size} value(s) of {value}')
        for name, part in parts.items():
            part.setflags(write=False)
            object.__setattr__(self, name, part)

    @property
    def objective(self) -> float:
        """The single objective; refused when there are several."""
        if self.f.size != 1:
            refuse(f'this point has {self.f.size} objectives; a single-objective method needs one')
        return float(self.f[0])

    @property
    def gradient(self) -> np.ndarray:
        """The gradient of the single objective."""
        self.objective
        return self.df[0]

    @property
    def violation(self) -> float:
        """``||h||_1 + ||max(c, 0)||_1``: the l1 infeasibility."""
        return float(np.sum(np.abs(self.h)) + np.sum(np.maximum(self.c, 0.0)))


def lagrangian_gradient(point: Linearization, y: np.ndarray | None = None, z: np.ndarray | None = None) -> np.ndarray:
    """``df + dh^T y + dc^T z`` of the single objective."""
    gradient = point.gradient.copy()
    if y is not None and y.size:
        gradient += point.dh.T @ y
    if z is not None and z.size:
        gradient += point.dc.T @ z
    return gradient


@dataclass(frozen=True, slots=True)
class KKTResidual:
    """The three first-order conditions, each as a max-norm: zero at a KKT point."""

    stationarity: float
    feasibility: float
    complementarity: float

    @property
    def worst(self) -> float:
        return max(self.stationarity, self.feasibility, self.complementarity)


def kkt_residual(
    point: Linearization,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    y: np.ndarray | None = None,
    z: np.ndarray | None = None,
) -> KKTResidual:
    """Projected stationarity ``|P(x - grad L) - x|``, violation of ``h``/``c``, and ``|z_i c_i|`` with ``z < 0`` counted."""
    gradient = lagrangian_gradient(point, y, z)
    stationarity = float(np.max(np.abs(np.clip(point.x - gradient, lo, hi) - point.x), initial=0.0))
    feasibility = max(float(np.max(np.abs(point.h), initial=0.0)), float(np.max(np.maximum(point.c, 0.0), initial=0.0)))
    complementarity = 0.0
    if z is not None and z.size:
        complementarity = max(float(np.max(np.abs(z * point.c))), float(np.max(np.maximum(-z, 0.0))))
    return KKTResidual(stationarity, feasibility, complementarity)


def damped_secant(b: np.ndarray, s: np.ndarray, y: np.ndarray, *, theta_floor: float = 0.2) -> np.ndarray:
    """Powell's damped ``y`` for a Lagrangian secant pair, so ``s.y >= theta_floor s.B s`` (Nocedal-Wright 18.2).

    The Hessian of a Lagrangian need not be positive definite, so a raw pair can have ``s.y <= 0``;
    the damped pair is what keeps the BFGS model (updated by numerics-lab) positive definite.
    """
    bs = b @ s
    sbs = float(s @ bs)
    sy = float(s @ y)
    if sy >= theta_floor * sbs:
        return y
    theta = (1.0 - theta_floor) * sbs / (sbs - sy)
    return theta * y + (1.0 - theta) * bs
