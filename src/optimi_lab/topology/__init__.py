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

__all__ = [
    'ROBUST_ETAS',
    'DensityFilter',
    'DesignChain',
    'DesignDomain',
    'HelmholtzFilter',
    'fd_check',
    'grayness',
    'project',
]
