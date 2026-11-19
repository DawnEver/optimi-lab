"""Analyses over a finished sample or front: variance-based sensitivity and decision support.

Domain-free like the rest of the package. :mod:`~optimi_lab.analysis.sensitivity` apportions an
output's variance among the declared variables; :mod:`~optimi_lab.analysis.decision` reads a front
in the minimization frame of :mod:`optimi_lab.core.pareto`. :mod:`~optimi_lab.analysis.robust` chooses
robustly from a fixed pool; :mod:`~optimi_lab.analysis.cluster` and
:mod:`~optimi_lab.analysis.selection` label regions and pick across them; :mod:`~optimi_lab.analysis.generations`
refits a run's surrogate per generation. A power law over a sample is learn-lab's ``PowerLaw``.
"""

from optimi_lab.analysis.cluster import kmeans
from optimi_lab.analysis.conflict import conflict_matrix, essential_objectives
from optimi_lab.analysis.decision import effective_dimension, knee_index, monotone_links, topsis
from optimi_lab.analysis.generations import RunHistory, generation_errors, generation_predictions
from optimi_lab.analysis.morris import MorrisEffects, morris
from optimi_lab.analysis.ranking import (
    CORRELATIONS,
    RankedRow,
    SensitivityMeasure,
    SensitivityRanking,
    rank_by_correlation,
    rank_morris,
    rank_sobol,
)
from optimi_lab.analysis.robust import (
    FORMULATIONS,
    Choice,
    box_corners,
    box_draw,
    choose,
    robust_validation,
    thin_candidates,
)
from optimi_lab.analysis.selection import stratified_pick
from optimi_lab.analysis.sensitivity import SobolIndices, main_effect, sobol_indices

__all__ = [
    'CORRELATIONS',
    'FORMULATIONS',
    'Choice',
    'MorrisEffects',
    'RankedRow',
    'RunHistory',
    'SensitivityMeasure',
    'SensitivityRanking',
    'SobolIndices',
    'box_corners',
    'box_draw',
    'choose',
    'conflict_matrix',
    'effective_dimension',
    'essential_objectives',
    'generation_errors',
    'generation_predictions',
    'kmeans',
    'knee_index',
    'main_effect',
    'monotone_links',
    'morris',
    'rank_by_correlation',
    'rank_morris',
    'rank_sobol',
    'robust_validation',
    'sobol_indices',
    'stratified_pick',
    'thin_candidates',
    'topsis',
]
