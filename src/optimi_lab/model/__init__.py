"""L6 model: surrogates, every one self-implemented on numpy, ``scipy.special`` and numerics-lab.

A surrogate is a frozen declaration of its settings; ``fit`` returns a fitted model whose
``predict(x, variance=, gradient=)`` returns a :class:`Prediction` -- the mean ``(n, k)``, and the
latent variance ``(n, k)`` and mean gradient ``(n, k, d)`` when asked for and declared in ``gives``
(:mod:`optimi_lab.model.protocol`). A prediction is labelled ``Evidence.PREDICTED`` by whoever files
it; it is never an observation.

* response surfaces -- :class:`Polynomial` (+ :class:`Ridge` / :class:`Lasso`), :class:`RadialBasis`,
  :class:`ChaosExpansion`;
* Kriging -- :class:`GaussianProcess` (kernels in :mod:`~optimi_lab.model.kernel`, maximum likelihood,
  closed-form leave-one-out), :class:`GradientEnhancedKriging`, :class:`MultiOutputKriging` (ICM),
  :class:`CoKriging` (AR1 multi-fidelity);
* learners -- :class:`NearestNeighbours`, :class:`RegressionTree`, :class:`Forest` (random forest,
  extra trees, bagging), :class:`MultilayerPerceptron`;
* composition and judgement -- :class:`Ensemble`, :func:`cross_validate`, :func:`score`,
  :func:`calibration`.

scikit-learn is a TEST oracle only (plan R1).
"""

from optimi_lab.model.chaos import ChaosExpansion, FittedChaos, legendre_table
from optimi_lab.model.co_kriging import CoKriging, FittedCoKriging
from optimi_lab.model.ensemble import WEIGHTINGS, Ensemble, FittedEnsemble
from optimi_lab.model.gaussian_process import FittedGaussianProcess, GaussianProcess, log_likelihood
from optimi_lab.model.gradient_enhanced import FittedGradientEnhanced, GradientEnhancedKriging
from optimi_lab.model.kernel import KERNEL_FAMILIES
from optimi_lab.model.multi_output import FittedMultiOutput, MultiOutputKriging
from optimi_lab.model.neighbours import NEIGHBOUR_WEIGHTS, FittedNeighbours, NearestNeighbours
from optimi_lab.model.neural import ACTIVATIONS, MultilayerPerceptron, Network
from optimi_lab.model.polynomial import FittedPolynomial, Lasso, Polynomial, Ridge
from optimi_lab.model.protocol import Fitted, Give, Prediction, Surrogate, query_points, training_data
from optimi_lab.model.radial_basis import RADIAL_KERNELS, FittedRadialBasis, RadialBasis
from optimi_lab.model.registry import SURROGATES, surrogate
from optimi_lab.model.tree import SPLITTERS, FittedForest, FittedTree, Forest, RegressionTree
from optimi_lab.model.validation import (
    COVERAGE_LEVELS,
    SCORES,
    Calibration,
    ScoredFit,
    box_predictions,
    calibration,
    cross_validate,
    folds,
    score,
    scored_fit,
)

__all__ = [
    'ACTIVATIONS',
    'COVERAGE_LEVELS',
    'KERNEL_FAMILIES',
    'NEIGHBOUR_WEIGHTS',
    'RADIAL_KERNELS',
    'SCORES',
    'SPLITTERS',
    'SURROGATES',
    'WEIGHTINGS',
    'Calibration',
    'ChaosExpansion',
    'CoKriging',
    'Ensemble',
    'Fitted',
    'FittedChaos',
    'FittedCoKriging',
    'FittedEnsemble',
    'FittedForest',
    'FittedGaussianProcess',
    'FittedGradientEnhanced',
    'FittedMultiOutput',
    'FittedNeighbours',
    'FittedPolynomial',
    'FittedRadialBasis',
    'FittedTree',
    'Forest',
    'GaussianProcess',
    'Give',
    'GradientEnhancedKriging',
    'Lasso',
    'MultiOutputKriging',
    'MultilayerPerceptron',
    'NearestNeighbours',
    'Network',
    'Polynomial',
    'Prediction',
    'RadialBasis',
    'RegressionTree',
    'Ridge',
    'ScoredFit',
    'Surrogate',
    'box_predictions',
    'calibration',
    'cross_validate',
    'folds',
    'legendre_table',
    'log_likelihood',
    'query_points',
    'score',
    'scored_fit',
    'surrogate',
    'training_data',
]
