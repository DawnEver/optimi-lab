"""Analyses over a finished sample or front: variance-based sensitivity and decision support.

Domain-free like the rest of the package. :mod:`~optimi_lab.analysis.sensitivity` apportions an
output's variance among the declared variables; :mod:`~optimi_lab.analysis.decision` reads a front
in the minimization frame of :mod:`optimi_lab.core.pareto`.
"""

from optimi_lab.analysis.conflict import conflict_matrix, essential_objectives
from optimi_lab.analysis.decision import (
    PowerLaw,
    effective_dimension,
    knee_index,
    monotone_links,
    power_law_fit,
    topsis,
)
from optimi_lab.analysis.morris import MorrisEffects, morris
from optimi_lab.analysis.sensitivity import SobolIndices, main_effect, sobol_indices

__all__ = [
    'MorrisEffects',
    'PowerLaw',
    'SobolIndices',
    'conflict_matrix',
    'effective_dimension',
    'essential_objectives',
    'knee_index',
    'main_effect',
    'monotone_links',
    'morris',
    'power_law_fit',
    'sobol_indices',
    'topsis',
]
