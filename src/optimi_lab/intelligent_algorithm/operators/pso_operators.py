from abc import ABC, abstractmethod

import numpy as np
from pydantic import field_validator

from optimi_lab.utils.exceptions import ParameterException
from optimi_lab.utils.quantities import BaseModel_with_q
from optimi_lab.utils.variable_space import VariableSpace


class PSOOperatorBase(BaseModel_with_q, ABC):
    """Base class for PSO operators.

    Defines the common interface for various operations used in PSO algorithms.
    """

    @abstractmethod
    def do(self, *args, **kwargs) -> np.ndarray:
        """Execute a PSO operation.

        Returns:
            Resulting array from the operation.

        """
        ...


class PSOPositionUpdate(PSOOperatorBase):
    """PSO position update operator.

    Updates particle positions based on current velocities and enforces variable bounds.
    """

    def do(self, positions: np.ndarray, velocities: np.ndarray, variable_space: VariableSpace) -> np.ndarray:
        """Update particle positions.

        Args:
            positions(np.ndarray): Current particle positions, shape (pop_size, n_var)
            velocities(np.ndarray): Current particle velocities, shape (pop_size, n_var)
            variable_space(VariableSpace): Variable space object

        Returns:
            np.ndarray: Updated particle positions, shape (pop_size, n_var)

        """
        new_positions = positions + velocities
        return np.clip(new_positions, variable_space.lower_bounds, variable_space.upper_bounds)


class PSOVelocityUpdate(PSOOperatorBase):
    """PSO velocity update operator.

    Implements the standard PSO velocity update formula combining inertia, cognitive, and social components.

    Attributes:
        w (float): Inertia weight controlling influence of previous velocity.
        c1 (float): Cognitive learning factor controlling attraction to personal best.
        c2 (float): Social learning factor controlling attraction to global best.
        v_max (float): Maximum velocity limit factor (relative to variable range).

    """

    w: float = 0.4
    c1: float = 2.0
    c2: float = 2.0
    v_max: float = 0.1

    @field_validator('w')
    def check_w(cls, v):
        if not (0.0 <= v <= 1.0):
            msg = 'w (inertia weight) must be between 0 and 1'
            raise ParameterException(msg)
        return v

    @field_validator('c1', 'c2')
    def check_learning_factors(cls, v):
        if v < 0:
            msg = 'learning factors c1 and c2 must be non-negative'
            raise ParameterException(msg)
        return v

    @field_validator('v_max')
    def check_v_max(cls, v):
        if not (0.0 < v <= 1.0):
            msg = 'v_max must be between 0 and 1'
            raise ParameterException(msg)
        return v

    def do(
        self,
        velocities: np.ndarray,
        positions: np.ndarray,
        personal_best_positions: np.ndarray,
        global_best_positions: np.ndarray,
        variable_space: VariableSpace,
    ) -> np.ndarray:
        """Update particle velocities.

        Implements the PSO velocity update formula:
        v(t+1) = w*v(t) + c1*r1*(pbest - x(t)) + c2*r2*(gbest - x(t))

        Args:
            velocities(np.ndarray): Current particle velocities, shape (pop_size, n_var)
            positions(np.ndarray): Current particle positions, shape (pop_size, n_var)
            personal_best_positions(np.ndarray): Personal best positions, shape (pop_size, n_var)
            global_best_positions(np.ndarray): Collection of global best positions, shape (archive_size, n_var)
            variable_space(VariableSpace): Variable space object

        Returns:
            np.ndarray: Updated particle velocities, shape (pop_size, n_var)

        """
        pop_size, n_var = positions.shape

        # Compute velocity limit
        v_max_abs = self.v_max

        # Randomly select one global best position for each particle
        if global_best_positions.size > 0:
            global_indices = np.random.randint(0, global_best_positions.shape[0], pop_size)
            selected_global_best = global_best_positions[global_indices]
        else:
            selected_global_best = positions  # Use current positions when no global best exists

        # Generate random factors
        r1, r2 = np.random.rand(pop_size, n_var), np.random.rand(pop_size, n_var)

        # Three components of the PSO velocity update formula
        inertia_component = self.w * velocities
        cognitive_component = self.c1 * r1 * (personal_best_positions - positions)
        social_component = self.c2 * r2 * (selected_global_best - positions)

        new_velocities = inertia_component + cognitive_component + social_component

        # Apply velocity limits
        return np.clip(new_velocities, -v_max_abs, v_max_abs)
