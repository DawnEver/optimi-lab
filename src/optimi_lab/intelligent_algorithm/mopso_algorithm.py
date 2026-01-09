from collections.abc import Callable

import numpy as np

from optimi_lab.intelligent_algorithm.operators.pso_operators import (
    PSOPositionUpdate,
    PSOVelocityUpdate,
)
from optimi_lab.intelligent_algorithm.operators.selection import ParetoCrowdingSelection
from optimi_lab.intelligent_algorithm.utils import check_dominated
from optimi_lab.utils.variable_space import VariableSpace

from .intelligent_algorithm_base import IntelligentAlgorithmBase


class MOPSO_Algorithm(IntelligentAlgorithmBase):
    """Multi-Objective Particle Swarm Optimization (MOPSO).

    MOPSO extends Particle Swarm Optimization for multi-objective problems.
    It uses non-dominated sorting and an external archive to preserve diversity
    and approximate the Pareto front.

    Reference:
        C. A. Coello Coello and M. S. Lechuga, "MOPSO: a proposal for multiple
        objective particle swarm optimization," Proceedings of the 2002
        Congress on Evolutionary Computation. CEC'02, Honolulu, HI, USA, 2002.

    Attributes:
        _velocity_update_operator (PSOVelocityUpdate): Velocity update operator.
        _position_update_operator (PSOPositionUpdate): Position update operator.
        _selection_operator (ParetoCrowdingSelection): External archive manager / selection operator.

        _velocities (np.ndarray): Particle velocity matrix.
        _personal_best_positions (np.ndarray): Personal best positions.
        _personal_best_fitness (np.ndarray): Personal best fitness values.
        _archive_positions (np.ndarray): External archive positions.
        _archive_fitness (np.ndarray): External archive fitness values.

    """

    _velocity_update_operator: PSOVelocityUpdate
    _position_update_operator: PSOPositionUpdate
    _selection_operator: ParetoCrowdingSelection

    _velocities: np.ndarray
    _personal_best_positions: np.ndarray
    _personal_best_fitness: np.ndarray
    _archive_positions: np.ndarray
    _archive_fitness: np.ndarray

    def __init__(
        self,
        variable_space: VariableSpace,
        n_obj: int,
        pop_size: int,
        max_iter: int,
        object_function: Callable | None = None,
        velocity_update_operator: PSOVelocityUpdate = None,
        position_update_operator: PSOPositionUpdate = None,
        selection_operator: ParetoCrowdingSelection = None,
    ) -> None:
        super().__init__(
            object_function=object_function,
            variable_space=variable_space,
            n_obj=n_obj,
            pop_size=pop_size,
            max_iter=max_iter,
        )

        # Use default operators if not provided
        self._velocity_update_operator = velocity_update_operator or PSOVelocityUpdate()
        self._position_update_operator = position_update_operator or PSOPositionUpdate()
        self._selection_operator = selection_operator or ParetoCrowdingSelection(n_selected=2 * pop_size)

        # Initialize velocity matrix (start from zero velocity)
        self._velocities = np.zeros((self._pop_size, self._n_var))

        # Initialize external archive
        self._archive_positions = np.zeros((0, self._n_var))
        self._archive_fitness = np.zeros((0, self._n_obj))

    def _step(self):
        """Perform a single MOPSO iteration."""
        # If this is the first iteration, initialize personal bests
        if self._id_iter == 1:
            self._personal_best_positions = self._current_inputs.copy()
            self._personal_best_fitness = self._current_outputs.copy()

        # Velocity update
        self._velocities = self._velocity_update_operator.do(
            velocities=self._velocities,
            positions=self._current_inputs,
            personal_best_positions=self._personal_best_positions,
            global_best_positions=self._archive_positions,
            variable_space=self._variable_space,
        )

        # Position update
        new_positions = self._position_update_operator.do(
            positions=self._current_inputs,
            velocities=self._velocities,
            variable_space=self._variable_space,
        )

        # Evaluate new positions
        new_fitness = self._object_function(new_positions)
        # Update personal bests
        # Check whether new solutions dominate personal bests
        new_dominates_best = check_dominated(new_fitness, self._personal_best_fitness)
        # Check whether personal bests dominate new solutions
        best_dominates_new = check_dominated(self._personal_best_fitness, new_fitness)
        # For non-dominated cases, use a small random probability to decide updates
        non_dominated_mask = np.logical_not(new_dominates_best | best_dominates_new)
        random_update_mask = non_dominated_mask & (np.random.rand(self._pop_size) < 0.1)
        # Combined update condition: new solution dominates personal best or
        # random update in non-dominated cases
        update_mask = new_dominates_best | random_update_mask
        self._personal_best_positions[update_mask] = new_positions[update_mask]
        self._personal_best_fitness[update_mask] = new_fitness[update_mask]

        # Update external archive
        combined_inputs = np.vstack([self._archive_positions, new_positions])
        combined_outputs = np.vstack([self._archive_fitness, new_fitness])
        selected_indices = self._selection_operator.do(fitness=combined_outputs)

        self._archive_positions = combined_inputs[selected_indices]
        self._archive_fitness = combined_outputs[selected_indices]

        # Update current population
        self._current_inputs = new_positions
        self._current_outputs = new_fitness
