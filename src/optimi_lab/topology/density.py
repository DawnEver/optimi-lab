"""Design variables -> physical density: tie, FILTER, PROJECT, hold passive cells -- with the exact adjoint.

``z`` (design variables) -> ``x = expand(z)`` (one per cell, :class:`~.domain.DesignDomain`; a passive
``solid``/``void`` cell is no variable and enters at 1/0, so its neighbours filter its declared state) -> FILTER
``x~`` -> PROJECTION ``x^ = (tanh(b eta) + tanh(b (x~ - eta))) / (tanh(b eta) + tanh(b (1 - eta)))``
(smoothed Heaviside; ``b`` raised by continuation) -> passive cells held at 1/0 (zero slope). The
physics reads ``x^``; :meth:`DesignChain.pull_back` carries ``d f / d x^`` back to ``d f / d z``.

Two filters, both LINEAR and both with a radius in the domain's length unit:

* :class:`DensityFilter` -- cone weights ``w_ij = max(0, r - |c_i - c_j|) V_j``, rows normalised
  (Bruns and Tortorelli 2001; Bourdin 2001). *void_padding* normalises every row by the largest row sum,
  as if the domain sat in void (Clausen and Andreassen 2017), so a boundary cell does not filter to 1.
* :class:`HelmholtzFilter` -- the PDE filter ``-r_h^2 lap x~ + x~ = x`` with ``r_h = r / (2 sqrt 3)``
  (Lazarov and Sigmund 2011), discretised on the cell graph by two-point fluxes
  ``face_ij / |c_i - c_j|``, zero-flux on the boundary: ``(M + r_h^2 L) x~ = M x``. Its cost is one
  sparse solve per map, not a neighbour search, so it scales to large 3-D domains.

ROBUST length scale: one filter, projections at ``eta`` eroded / nominal / dilated (:data:`ROBUST_ETAS`;
Wang, Lazarov and Sigmund 2011). Asking the eroded and dilated realizations to perform too gives the
nominal design a minimum solid and void feature set by the radius and the threshold gap.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numerics_lab.linear import solve_linear_system

from optimi_lab.core.errors import refuse
from optimi_lab.topology.domain import DesignDomain

__all__ = [
    'BUILT_READ',
    'NOMINAL_ETA',
    'ROBUST_ETAS',
    'DensityFilter',
    'DesignChain',
    'HelmholtzFilter',
    'Threshold',
    'built',
    'fd_check',
    'grayness',
    'project',
]

#: Eroded / nominal / dilated projection thresholds of the robust formulation.
ROBUST_ETAS = (0.7, 0.5, 0.3)
#: The nominal threshold (the middle robust one): a cell whose physical density exceeds it is BUILT.
NOMINAL_ETA = ROBUST_ETAS[len(ROBUST_ETAS) // 2]
_CHUNK = 2048


def _positive_radius(domain: DesignDomain, radius: float, unit: str) -> float:
    domain.require_unit(unit, 'the filter radius')
    if not radius > 0.0:
        refuse(f'the filter radius must be a positive length, got {radius!r}')
    return float(radius)


class DensityFilter:
    """The cone density filter over a physical *radius* (in *unit*); see the module docstring."""

    def __init__(self, domain: DesignDomain, radius: float, *, unit: str, void_padding: bool = False) -> None:
        radius = _positive_radius(domain, radius, unit)
        c, n = domain.centroids, domain.n_cells
        rows, cols, vals = [], [], []
        for start in range(0, n, _CHUNK):
            block = c[start : start + _CHUNK]
            dist = np.sqrt(((block[:, None, :] - c[None, :, :]) ** 2).sum(axis=2))
            i, j = np.nonzero(dist < radius)
            rows.append(i + start)
            cols.append(j)
            vals.append((radius - dist[i, j]) * domain.measures[j])
        h = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        sums = np.asarray(h.sum(axis=1)).ravel()
        self.domain = domain
        self.matrix = sp.diags(1.0 / (np.full_like(sums, sums.max()) if void_padding else sums)) @ h

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.matrix @ x

    def pull_back(self, d_filtered: np.ndarray) -> np.ndarray:
        return self.matrix.T @ d_filtered


class HelmholtzFilter:
    """The PDE (Helmholtz) filter on the cell graph; *radius* is the equivalent cone radius in *unit*."""

    def __init__(self, domain: DesignDomain, radius: float, *, unit: str) -> None:
        radius = _positive_radius(domain, radius, unit)
        a, b = domain.adjacency.T
        w = domain.faces / np.linalg.norm(domain.centroids[a] - domain.centroids[b], axis=1)
        n = domain.n_cells
        laplacian = sp.coo_matrix(
            (np.concatenate([w, w, -w, -w]), (np.concatenate([a, b, a, b]), np.concatenate([a, b, b, a]))), (n, n)
        )
        self.domain = domain
        self.mass = domain.measures
        self.operator = (sp.diags(self.mass) + (radius / (2.0 * np.sqrt(3.0))) ** 2 * laplacian).tocsc()

    def _solve(self, rhs: np.ndarray) -> np.ndarray:
        return np.asarray(solve_linear_system(self.operator, rhs), dtype=float).reshape(-1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._solve(self.mass * x)

    def pull_back(self, d_filtered: np.ndarray) -> np.ndarray:
        return self.mass * self._solve(d_filtered)  # the operator is symmetric


def project(x: np.ndarray, beta: float, eta: float = NOMINAL_ETA) -> tuple[np.ndarray, np.ndarray]:
    """``(x^, d x^ / d x~)`` of the smoothed Heaviside projection; ``beta = 0`` is the identity."""
    x = np.asarray(x, dtype=float)
    if beta == 0.0:
        return x.copy(), np.ones_like(x)
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    t = np.tanh(beta * (x - eta))
    return (np.tanh(beta * eta) + t) / den, beta * (1.0 - t**2) / den


def built(physical: np.ndarray, eta: float = NOMINAL_ETA) -> np.ndarray:
    """The 0/1 design a physical density builds: the cells above *eta*."""
    return np.asarray(physical, dtype=float) > eta


@dataclass(frozen=True, slots=True)
class Threshold:
    """A smoothed threshold ``H(x; beta, eta)`` (:func:`project`) a measure reads instead of the density.

    Read on the density, gray cells act as material; read on this threshold, a cell that builds to void
    (``x < eta``) acts as void, and the gradient is the measure's chained by ``dH``. ``beta`` is FIXED: no
    continuation stage jumps the response.
    """

    beta: float = 16.0
    eta: float = NOMINAL_ETA

    def __post_init__(self) -> None:
        if not self.beta > 0.0 or not 0.0 < self.eta < 1.0:
            refuse(f'a threshold needs beta > 0 and eta in (0, 1), got {self.beta} and {self.eta}')

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(H(x), dH / dx)``."""
        return project(x, self.beta, self.eta)


#: The BUILT read. At beta 16 a 0.45 cell conducts ~2 % of solid and a 0.55 one ~98 %: a grid joined only by a
#: bridge of 0.45 is refused by the virtual temperature read on it, one of 0.55 met, where the density read is
#: fooled by the 0.45 bridge (``tests/unit/topology/test_built_reads_and_bounds.py``).
BUILT_READ = Threshold()


def grayness(x: np.ndarray) -> float:
    """``4 mean(x (1 - x))``: 0 for a 0/1 design, 1 for all-0.5."""
    x = np.asarray(x, dtype=float)
    return float(4.0 * np.mean(x * (1.0 - x)))


class DesignChain:
    """Tie -> *density_filter* -> one projection per *eta* -> passive cells; ``beta`` is set by continuation."""

    def __init__(
        self, density_filter: DensityFilter | HelmholtzFilter, *, beta: float = 1.0, etas: Sequence[float] = (0.5,)
    ) -> None:
        if not etas or any(not 0.0 < e < 1.0 for e in etas):
            refuse(f'projection thresholds must lie in (0, 1), got {tuple(etas)}')
        self.filter = density_filter
        self.domain = density_filter.domain
        self.beta = float(beta)
        self.etas = tuple(float(e) for e in etas)
        self._slopes: list[np.ndarray] = []

    @property
    def nominal(self) -> int:
        """The index of the nominal realization (the middle threshold)."""
        return len(self.etas) // 2

    def filtered(self, design: np.ndarray) -> np.ndarray:
        """The filtered field at *design*: passive cells enter the filter at their state and leave it there."""
        out = self.filter.forward(self.domain.expand(design))
        out[self.domain.solid], out[self.domain.void] = 1.0, 0.0
        return out

    def forward(self, design: np.ndarray) -> tuple[np.ndarray, ...]:
        """One physical density per threshold, at *design* (the design variables)."""
        filtered = self.filtered(design)
        passive = self.domain.solid | self.domain.void
        out, self._slopes = [], []
        for eta in self.etas:
            value, slope = project(filtered, self.beta, eta)
            value[self.domain.solid], value[self.domain.void] = 1.0, 0.0
            slope[passive] = 0.0
            out.append(value)
            self._slopes.append(slope)
        return tuple(out)

    def pull_back(self, d_physical: Sequence[np.ndarray | None]) -> np.ndarray:
        """``d f / d z`` from one ``d f / d x^`` per realization (``None``: not read), at the last forward point."""
        if len(d_physical) != len(self._slopes):
            refuse(f'{len(d_physical)} sensitivities for {len(self._slopes)} realizations')
        total = np.zeros(self.domain.n_cells)
        for d, slope in zip(d_physical, self._slopes, strict=True):
            if d is not None:
                total += np.asarray(d, dtype=float) * slope
        return self.domain.reduce(self.filter.pull_back(total))


def fd_check(
    f: Callable[[np.ndarray], tuple[float, np.ndarray]], x: np.ndarray, indices: Iterable[int], step: float = 1e-6
) -> list[tuple[int, float, float]]:
    """``(index, analytic, central difference)`` of ``f(x) -> (value, gradient)`` at each of *indices*."""
    x = np.asarray(x, dtype=float)
    analytic = np.asarray(f(x)[1], dtype=float).copy()
    rows = []
    for k in indices:
        plus, minus = x.copy(), x.copy()
        plus[k] += step
        minus[k] -= step
        rows.append((int(k), float(analytic[k]), (f(plus)[0] - f(minus)[0]) / (2.0 * step)))
    return rows
