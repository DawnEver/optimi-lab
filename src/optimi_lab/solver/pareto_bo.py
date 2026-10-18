"""D62-D63 -- model-based multi-objective optimisation: ParEGO, and EHVI / NEHVI on one GP per objective.

The loop, the fantasies (D64) and the observed-only front are :class:`~optimi_lab.solver.model_based.ModelBased`'s.

* :func:`parego` (D62) -- Knowles (2006): every pick draws a Das-Dennis weight, scores the told rows
  by :func:`~optimi_lab.operator.acquisition.parego_scores`, fits one GP to the scores and maximises
  its log EI.
* :func:`ehvi` / :func:`nehvi` (D63) -- one GP per objective; exact EHVI over the told feasible front
  plus the believed pending, or NEHVI (Daulton et al., 2021) over ``samples`` joint draws, the
  pending points carried in its BASELINE rather than believed. ``reference`` (minimised frame)
  defaults to the told maximum plus a tenth of the told range, re-read at each pick.

With constraints each acquisition is multiplied by the probability of feasibility, and while no
told point is feasible the acquisition IS that probability.
"""

from collections.abc import Callable, Sequence

import numpy as np

from optimi_lab.core import refuse
from optimi_lab.model import FittedGaussianProcess
from optimi_lab.model.likelihood import factor
from optimi_lab.operator import (
    das_dennis,
    expected_hypervolume_improvement,
    hypervolume_improvement,
    log_expected_improvement,
    parego_scores,
)
from optimi_lab.oracle import Capability
from optimi_lab.problem import Problem
from optimi_lab.solver.model_based import Acquisition, ModelBased
from optimi_lab.solver.population import build

__all__ = ['HypervolumeImprovement', 'ParEGO', 'ehvi', 'nehvi', 'parego']


class ParEGO(ModelBased):
    """D62: a fresh Das-Dennis weight per pick, a GP on the ParEGO scores, log EI."""

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        rho: float = 0.05,
        divisions: int | None = None,
        **base: object,
    ) -> None:
        m = len(problem.objectives)
        if m < 2:
            refuse('ParEGO scalarises several objectives; one objective is bo()')
        super().__init__(problem, capability, **base)
        self._rho = rho
        self._weights = das_dennis(m, {2: 10, 3: 4}.get(m, 3) if divisions is None else divisions)

    def _pick(self, pending: np.ndarray) -> np.ndarray:
        weight = self._weights[self._rng.integers(len(self._weights))]
        scores = parego_scores(self._objectives, weight, rho=self._rho)
        model = self._believed(self._surrogate.fit(self._inputs, scores[:, None]), pending)
        log_pf = self._log_feasibility(pending)
        if not self._feasible.any():
            return self._maximise(log_pf)
        best = float(scores[self._feasible].min())

        def acquisition(points: np.ndarray) -> np.ndarray:
            p = model.predict(points, variance=True)
            return log_expected_improvement(p.mean[:, 0], np.sqrt(p.variance[:, 0]), best) + log_pf(points)

        return self._maximise(acquisition)


class HypervolumeImprovement(ModelBased):
    """D63: one GP per objective; exact EHVI (``noisy=False``) or NEHVI over ``samples`` joint draws."""

    def __init__(
        self,
        problem: Problem,
        capability: Capability,
        *,
        noisy: bool = False,
        samples: int = 32,
        reference: Sequence[float] | None = None,
        **base: object,
    ) -> None:
        m = len(problem.objectives)
        if m < 2:
            refuse('a hypervolume acquisition needs at least two objectives; one objective is bo()')
        if reference is not None and len(reference) != m:
            refuse(f'reference is one minimised value per objective ({m}), got {len(reference)}')
        if samples < 2:
            refuse(f'NEHVI averages at least 2 joint draws, got {samples}')
        super().__init__(problem, capability, **base)
        self._noisy, self._samples = noisy, samples
        self._reference = None if reference is None else np.asarray(reference, dtype=float)

    def reference(self) -> np.ndarray:
        """The declared reference, else the told maximum plus a tenth of the told range (minimised frame)."""
        if self._reference is not None:
            return self._reference
        values = self._objectives
        span = values.max(axis=0) - values.min(axis=0)
        return values.max(axis=0) + 0.1 * np.where(span > 0, span, 1.0)

    def _pick(self, pending: np.ndarray) -> np.ndarray:
        model = self._fit('objectives', self._objectives)
        log_pf = self._log_feasibility(pending)
        if self._problem.constraints and not self._feasible.any():
            return self._maximise(log_pf)
        reference = self.reference()
        observed = self._objectives[self._feasible]
        if not self._noisy:
            believed = self._believed(model, pending)
            front = np.vstack([observed, believed.predict(pending).mean]) if len(pending) else observed

            def acquisition(points: np.ndarray) -> np.ndarray:
                p = believed.predict(points, variance=True)
                gain = expected_hypervolume_improvement(p.mean, np.sqrt(p.variance), front, reference)
                return gain * np.exp(log_pf(points))

            return self._maximise(acquisition)
        return self._maximise(self._nehvi(model, np.vstack([self._inputs[self._feasible], pending]), reference, log_pf))

    def _nehvi(
        self,
        model: FittedGaussianProcess,
        baseline: np.ndarray,
        reference: np.ndarray,
        log_pf: Callable[[np.ndarray], np.ndarray],
    ) -> Acquisition:
        """NEHVI with the baseline drawn once and each candidate drawn CONDITIONALLY on it (exactly joint)."""
        m, s = len(self._problem.objectives), self._samples
        drawn = model.draw(baseline, self._rng.standard_normal((s, len(baseline), m))) if len(baseline) else None
        own = self._rng.standard_normal((s, m))
        mean_b = model.predict(baseline).mean if len(baseline) else None
        chols = [factor(block) for block in model.posterior_covariance(baseline, baseline)] if len(baseline) else []
        if any(c is None for c in chols):
            refuse('the baseline posterior covariance cannot be factored even with jitter')

        def acquisition(points: np.ndarray) -> np.ndarray:
            p = model.predict(points, variance=True)
            draws = np.repeat(p.mean[None], s, axis=0)
            variance = p.variance.copy()
            if drawn is not None:
                cross = model.posterior_covariance(points, baseline)
                for k, chol in enumerate(chols):
                    gain = chol.solve(cross[k].T).T
                    draws[:, :, k] += (drawn[:, :, k] - mean_b[:, k]) @ gain.T
                    variance[:, k] -= np.sum(gain * cross[k], axis=1)
            draws += np.sqrt(np.maximum(variance, 0.0))[None] * own[:, None, :]
            fronts = drawn if drawn is not None else np.zeros((s, 0, m))
            gains = np.mean(
                [hypervolume_improvement(d, f, reference) for d, f in zip(draws, fronts, strict=True)], axis=0
            )
            return gains * np.exp(log_pf(points))

        return acquisition


def parego(problem: Problem, capability: Capability, **settings: object) -> ParEGO:
    """ParEGO; ``**settings`` are :class:`ParEGO`'s and :class:`ModelBased`'s."""
    return build(ParEGO, problem, capability, 'parego', **settings)


def ehvi(problem: Problem, capability: Capability, **settings: object) -> HypervolumeImprovement:
    """Exact-EHVI Bayesian optimisation; ``**settings`` are :class:`HypervolumeImprovement`'s but ``noisy``."""
    return build(HypervolumeImprovement, problem, capability, 'ehvi', **settings)


def nehvi(problem: Problem, capability: Capability, **settings: object) -> HypervolumeImprovement:
    """NEHVI Bayesian optimisation; ``**settings`` are :class:`HypervolumeImprovement`'s but ``noisy``."""
    return build(HypervolumeImprovement, problem, capability, 'nehvi', noisy=True, **settings)
