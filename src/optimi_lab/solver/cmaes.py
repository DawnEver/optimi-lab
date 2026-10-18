"""CMA-ES (Hansen and Ostermeier, 2001; Hansen's 2016 tutorial): the covariance-matrix-adaptation evolution strategy.

``(mu/mu_w, lambda)`` with cumulative step-size adaptation and rank-one plus rank-mu covariance
updates, every constant the tutorial's default. It works in the unit cube: a sample outside it is
CLIPPED before it becomes a design, and the clipped point is what the update learns from, so the
distribution follows the designs that were actually evaluated. Ranking is Deb's rule on ONE score
(``scalarise`` for several objectives): a smaller violation first, then the smaller score.

The solver CONCLUDES (``done``) on the tutorial's stopping rules -- the step below ``tol_x`` in
unit-cube units, the best scores of the last generations within ``tol_fun`` in score units, or the
covariance's condition number past 1e14 -- and names which one in :attr:`stop`. A restart with a
larger population is :func:`optimi_lab.solver.meta.ipop_cmaes` (IPOP, Auger and Hansen, 2005).
"""

import numpy as np

from optimi_lab.core import refuse
from optimi_lab.core.sampling import SampleSpec
from optimi_lab.oracle import Capability
from optimi_lab.problem import Problem
from optimi_lab.solver.population import PopulationSolver, build, require_one_score

__all__ = ['CMAES', 'cmaes']

_MAX_CONDITION = 1e14
"""Covariance condition number past which the distribution has degenerated."""


class CMAES(PopulationSolver):
    """CMA-ES from ``mean`` (unit cube; None: the centre) with step ``sigma`` (unit-cube units)."""

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        population: int | None = None,
        mean: np.ndarray | None = None,
        sigma: float = 0.3,
        tol_x: float = 1e-9,
        tol_fun: float = 1e-12,
        **base: object,
    ) -> None:
        require_one_score(problem, base.get('scalarise'), 'cmaes')
        if 'spec' in base or 'initial' in base:
            refuse('CMA-ES draws its own generations from its distribution; spec= and initial= do not apply')
        n = problem.space.width
        self._n = n
        self._lambda = int(population) if population is not None else 4 + int(3 * np.log(n))
        if self._lambda < 2:
            refuse(f'a CMA-ES population needs at least 2 points, got {self._lambda}')
        if not sigma > 0.0:
            refuse(f'sigma is a step in unit-cube units and must be positive, got {sigma!r}')
        mu = self._lambda // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self._weights = weights / weights.sum()
        self._mu_eff = 1.0 / np.sum(self._weights**2)
        me = self._mu_eff
        self._cc = (4 + me / n) / (n + 4 + 2 * me / n)
        self._cs = (me + 2) / (n + me + 5)
        self._c1 = 2 / ((n + 1.3) ** 2 + me)
        self._cmu = min(1 - self._c1, 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me))
        self._damps = 1 + 2 * max(0.0, np.sqrt((me - 1) / (n + 1)) - 1) + self._cs
        self._chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        self._mean = np.full(n, 0.5) if mean is None else np.asarray(mean, dtype=float).copy()
        self._sigma = float(sigma)
        self._cov = np.eye(n)
        self._basis, self._scale = np.eye(n), np.ones(n)
        self._path_c, self._path_s = np.zeros(n), np.zeros(n)
        self._tol_x, self._tol_fun = tol_x, tol_fun
        self._history: list[float] = []
        self.stop: str | None = None
        self.best: tuple[np.ndarray, float] | None = None
        super().__init__(problem, capability, **base)

    def _first(self, spec: SampleSpec | None) -> np.ndarray:
        del spec
        return self._propose()

    def _propose(self) -> np.ndarray:
        steps = self._rng.standard_normal((self._lambda, self._n)) * self._scale @ self._basis.T
        return np.clip(self._mean + self._sigma * steps, 0.0, 1.0)

    @property
    def done(self) -> bool:
        return self.stop is not None

    def _admit(self, inputs: np.ndarray, values: np.ndarray, violation: np.ndarray, mask: np.ndarray) -> None:
        del mask
        order = np.lexsort((values[:, 0], violation))
        lead = order[0]
        if violation[lead] == 0.0 and (self.best is None or values[lead, 0] < self.best[1]):
            self.best = (inputs[lead].copy(), float(values[lead, 0]))
        chosen = order[: min(len(self._weights), len(order))]
        weights = self._weights[: len(chosen)] / self._weights[: len(chosen)].sum()
        steps = (inputs[chosen] - self._mean) / self._sigma
        old = self._mean
        self._mean = old + self._sigma * weights @ steps
        shift = (self._mean - old) / self._sigma
        inverse_root = self._basis @ np.diag(1.0 / self._scale) @ self._basis.T
        self._path_s = (1 - self._cs) * self._path_s + np.sqrt(self._cs * (2 - self._cs) * self._mu_eff) * (
            inverse_root @ shift
        )
        norm_s = np.linalg.norm(self._path_s)
        stalled = norm_s / np.sqrt(1 - (1 - self._cs) ** 2) < (1.4 + 2 / (self._n + 1)) * self._chi_n
        self._path_c = (1 - self._cc) * self._path_c + stalled * np.sqrt(
            self._cc * (2 - self._cc) * self._mu_eff
        ) * shift
        rank_mu = (steps * weights[:, None]).T @ steps
        correction = (1 - stalled) * self._cc * (2 - self._cc)
        self._cov = (
            (1 - self._c1 - self._cmu) * self._cov
            + self._c1 * (np.outer(self._path_c, self._path_c) + correction * self._cov)
            + self._cmu * rank_mu
        )
        self._sigma *= float(np.exp((self._cs / self._damps) * (norm_s / self._chi_n - 1)))
        self._cov = (self._cov + self._cov.T) / 2
        eigen, self._basis = np.linalg.eigh(self._cov)
        self._scale = np.sqrt(np.maximum(eigen, np.finfo(float).tiny))
        self._history.append(float(values[order[0], 0]))
        self._check_stop(eigen)

    def _check_stop(self, eigen: np.ndarray) -> None:
        span = 10 + int(30 * self._n / self._lambda)
        window = self._history[-span:]
        if self._sigma * self._scale.max() < self._tol_x:
            self.stop = f'step {self._sigma * self._scale.max():.3g} below tol_x {self._tol_x:g} (unit cube)'
        elif len(self._history) >= span and max(window) - min(window) < self._tol_fun:
            self.stop = f'best scores of the last {len(window)} generations within tol_fun {self._tol_fun:g}'
        elif eigen.max() > _MAX_CONDITION * max(eigen.min(), np.finfo(float).tiny):
            self.stop = f'covariance condition past {_MAX_CONDITION:g}'


def cmaes(problem: Problem, capability: Capability, **settings: object) -> CMAES:
    """CMA-ES on one score; ``**settings`` are :class:`CMAES`'s and ``scalarise``/``seed``."""
    return build(CMAES, problem, capability, 'cmaes', **settings)
