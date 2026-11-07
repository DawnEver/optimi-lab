"""L8 topology: density-based topology optimization over any physics a provider supplies.

The library owns everything that is not physics -- the :class:`DesignDomain`, the design -> physical
density chain (:mod:`~optimi_lab.topology.density`) and the driver; a provider supplies responses and
their sensitivities to the physical density.
"""

from optimi_lab.topology.density import (
    ROBUST_ETAS,
    DensityFilter,
    DesignChain,
    HelmholtzFilter,
    fd_check,
    grayness,
    project,
)
from optimi_lab.topology.domain import DesignDomain
from optimi_lab.topology.interpolation import Scheme, interpolate, ks, p_norm
from optimi_lab.topology.measures import VirtualTemperature, components, feature_sizes, member_widths

__all__ = [
    'ROBUST_ETAS',
    'DensityFilter',
    'DesignChain',
    'DesignDomain',
    'HelmholtzFilter',
    'Scheme',
    'VirtualTemperature',
    'components',
    'fd_check',
    'feature_sizes',
    'grayness',
    'interpolate',
    'ks',
    'member_widths',
    'p_norm',
    'project',
]
