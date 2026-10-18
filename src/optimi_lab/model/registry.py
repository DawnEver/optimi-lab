"""The declared L6 surrogates by name, for a caller that names one in data (a TOML table, a CLI flag).

:func:`surrogate` resolves a name and its settings, or refuses naming every declared surrogate. An
ensemble is composed from declarations, so it is not itself a row.
"""

from __future__ import annotations

from typing import Final

from optimi_lab.core.errors import refuse
from optimi_lab.model.chaos import ChaosExpansion
from optimi_lab.model.co_kriging import CoKriging
from optimi_lab.model.gaussian_process import GaussianProcess
from optimi_lab.model.gradient_enhanced import GradientEnhancedKriging
from optimi_lab.model.multi_output import MultiOutputKriging
from optimi_lab.model.neighbours import NearestNeighbours
from optimi_lab.model.neural import MultilayerPerceptron
from optimi_lab.model.polynomial import Polynomial
from optimi_lab.model.radial_basis import RadialBasis
from optimi_lab.model.tree import Forest, RegressionTree

__all__ = ['SURROGATES', 'surrogate']

SURROGATES: Final = {
    'polynomial': Polynomial,
    'radial_basis': RadialBasis,
    'gaussian_process': GaussianProcess,
    'gradient_enhanced_kriging': GradientEnhancedKriging,
    'multi_output_kriging': MultiOutputKriging,
    'co_kriging': CoKriging,
    'nearest_neighbours': NearestNeighbours,
    'regression_tree': RegressionTree,
    'forest': Forest,
    'multilayer_perceptron': MultilayerPerceptron,
    'chaos_expansion': ChaosExpansion,
}


def surrogate(name: object, **settings: object) -> object:
    """The surrogate *name* declares, built with *settings*; a stranger is refused naming the set."""
    if name not in SURROGATES:
        refuse(f'a surrogate is one of {list(SURROGATES)}, got {name!r}')
    return SURROGATES[name](**settings)
