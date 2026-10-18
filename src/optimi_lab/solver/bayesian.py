"""D61 -- Bayesian optimisation of one score: a Gaussian process and one acquisition maximised on it.

The loop, the fantasies (D64) and the observed-only reports are :class:`~optimi_lab.solver.model_based.ModelBased`'s.

:func:`bo` takes one score (``scalarise`` for several) and an acquisition from :data:`ACQUISITIONS`:
``log_ei`` (default), ``ei``, ``pi``, ``log_pi``, ``ucb`` (``beta`` default Srinivas et al.'s
``2 log(d t^2 pi^2 / 6 delta)``, delta 0.1) or ``kg`` (knowledge gradient over the training points
and the candidate, Scott et al.'s KGCP). With constraints the EI/PI family is multiplied by the
probability of feasibility (constrained EI), and while no told point is feasible the acquisition IS
that probability; UCB and KG take none.
"""

from typing import Final

import numpy as np

from optimi_lab.core import refuse
from optimi_lab.model import FittedGaussianProcess
from optimi_lab.operator import (
    expected_improvement,
    knowledge_gradient,
    log_expected_improvement,
    log_probability_of_improvement,
    probability_of_improvement,
    upper_confidence_bound,
)
from optimi_lab.oracle import Capability, Result
from optimi_lab.problem import Problem
from optimi_lab.solver.model_based import ModelBased
from optimi_lab.solver.population import Scalarise, build, require_one_score

__all__ = ['ACQUISITIONS', 'BayesianOptimisation', 'bo']

#: The single-score acquisitions :func:`bo` takes.
ACQUISITIONS: Final = ('log_ei', 'ei', 'pi', 'log_pi', 'ucb', 'kg')
_CONSTRAINED: Final = frozenset({'log_ei', 'ei', 'pi', 'log_pi'})


class BayesianOptimisation(ModelBased):
    """D61: one score, one acquisition from :data:`ACQUISITIONS`."""

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        acquisition: str = 'log_ei',
        beta: float | None = None,
        scalarise: Scalarise | None = None,
        **base: object,
    ) -> None:
        require_one_score(problem, scalarise, 'bo')
        if acquisition not in ACQUISITIONS:
            refuse(f'acquisition is one of {list(ACQUISITIONS)}, got {acquisition!r}')
        if problem.constraints and acquisition not in _CONSTRAINED:
            refuse(f'{acquisition} takes no constraint; the constrained acquisitions are {sorted(_CONSTRAINED)}')
        super().__init__(problem, capability, **base)
        self._acquisition, self._beta, self._scalarise = acquisition, beta, scalarise

    def _scores(self) -> np.ndarray:
        values = self._objectives
        return (values if self._scalarise is None else np.asarray(self._scalarise(values), dtype=float)).reshape(-1)

    @property
    def best(self) -> Result | None:
        """The told OK feasible result of lowest score -- an observation, never a prediction (B0-08)."""
        if not self._results:
            return None
        scores = np.where(self._feasible, self._scores(), np.inf)
        return self._results[int(np.argmin(scores))] if np.isfinite(scores.min()) else None

    def _pick(self, pending: np.ndarray) -> np.ndarray:
        model = self._believed(self._fit('score', self._scores()[:, None]), pending)
        log_pf = self._log_feasibility(pending)
        feasible = self._scores()[self._feasible]
        if not len(feasible):
            return self._maximise(log_pf)
        best = float(feasible.min())
        if self._acquisition == 'kg':
            return self._maximise(lambda points: self._knowledge_gradient(model, points))

        def acquisition(points: np.ndarray) -> np.ndarray:
            p = model.predict(points, variance=True)
            mean, std = p.mean[:, 0], np.sqrt(p.variance[:, 0])
            if self._acquisition == 'ucb':
                return upper_confidence_bound(mean, std, self._beta_t())
            log_pf_here = log_pf(points)
            if self._acquisition == 'log_ei':
                return log_expected_improvement(mean, std, best) + log_pf_here
            if self._acquisition == 'log_pi':
                return log_probability_of_improvement(mean, std, best) + log_pf_here
            closed = expected_improvement if self._acquisition == 'ei' else probability_of_improvement
            return closed(mean, std, best) * np.exp(log_pf_here)

        return self._maximise(acquisition)

    def _beta_t(self) -> float:
        if self._beta is not None:
            return self._beta
        t = len(self._x) + 1
        return float(2.0 * np.log(self._width * t**2 * np.pi**2 / (6 * 0.1)))

    @staticmethod
    def _knowledge_gradient(model: FittedGaussianProcess, points: np.ndarray) -> np.ndarray:
        """KGCP: the value of observing each point, over the training points and the point itself."""
        known = model.predict(model.x).mean[:, 0]
        cross = model.posterior_covariance(model.x, points)[0]
        p = model.predict(points, variance=True)
        spread = np.sqrt(p.variance[:, 0] + model.noise[0])
        return np.array([
            knowledge_gradient(-np.append(known, p.mean[j, 0]), np.append(cross[:, j], p.variance[j, 0]) / spread[j])
            for j in range(len(points))
        ])


def bo(problem: Problem, capability: Capability, **settings: object) -> BayesianOptimisation:
    """Bayesian optimisation; ``**settings`` are :class:`BayesianOptimisation`'s and :class:`ModelBased`'s."""
    return build(BayesianOptimisation, problem, capability, 'bo', **settings)
