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
from optimi_lab.topology.domain import DesignDomain, Neighbourhood
from optimi_lab.topology.driver import (
    DEFAULT_CONTINUATION,
    Continuation,
    Iteration,
    Optimizer,
    TopologyResult,
    topology_optimize,
)
from optimi_lab.topology.interpolation import Interpolation, interpolate, ks, p_norm
from optimi_lab.topology.measures import (
    VirtualTemperature,
    component_labels,
    components,
    feature_sizes,
    member_widths,
)
from optimi_lab.topology.provider import PhysicsProvider, ProviderResponse, Role, TopologyPoint, TopologyProblem

__all__ = [
    'DEFAULT_CONTINUATION',
    'ROBUST_ETAS',
    'Continuation',
    'DensityFilter',
    'DesignChain',
    'DesignDomain',
    'HelmholtzFilter',
    'Interpolation',
    'Iteration',
    'Neighbourhood',
    'Optimizer',
    'PhysicsProvider',
    'ProviderResponse',
    'Role',
    'TopologyPoint',
    'TopologyProblem',
    'TopologyResult',
    'VirtualTemperature',
    'component_labels',
    'components',
    'fd_check',
    'feature_sizes',
    'grayness',
    'interpolate',
    'ks',
    'member_widths',
    'p_norm',
    'project',
    'topology_optimize',
]
