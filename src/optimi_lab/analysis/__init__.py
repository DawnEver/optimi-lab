"""Analyses over a finished sample or front: variance-based sensitivity and decision support.

Domain-free like the rest of the package. :mod:`~optimi_lab.analysis.sensitivity` apportions an
output's variance among the declared variables.
"""

from optimi_lab.analysis.sensitivity import SobolIndices, main_effect, sobol_indices

__all__ = ['SobolIndices', 'main_effect', 'sobol_indices']
