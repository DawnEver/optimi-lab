"""L8 topology: density-based topology optimization over any physics a provider supplies.

The library owns everything that is not physics -- the :class:`DesignDomain`, the design -> physical
density chain (:mod:`~optimi_lab.topology.density`) and the driver; a provider supplies responses and
their sensitivities to the physical density.
"""

from optimi_lab.topology.density import (
    BUILT_READ,
    NOMINAL_ETA,
    ROBUST_ETAS,
    DensityFilter,
    DesignChain,
    HelmholtzFilter,
    Threshold,
    built,
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
    P_NORM,
    PENALTY,
    S_MIN,
    VirtualTemperature,
    component_labels,
    components,
    feature_sizes,
    member_widths,
    volume_fraction,
)
from optimi_lab.topology.provider import PhysicsProvider, ProviderResponse, Role, TopologyPoint, TopologyProblem

__all__ = [
    'BUILT_READ',
    'DEFAULT_CONTINUATION',
    'NOMINAL_ETA',
    'PENALTY',
    'P_NORM',
    'ROBUST_ETAS',
    'S_MIN',
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
    'Threshold',
    'TopologyPoint',
    'TopologyProblem',
    'TopologyResult',
    'VirtualTemperature',
    'built',
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
    'volume_fraction',
]
