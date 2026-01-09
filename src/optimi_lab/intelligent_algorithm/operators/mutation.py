from abc import ABC, abstractmethod

import numpy as np
from pydantic import field_validator

from optimi_lab.utils.exceptions import ParameterException
from optimi_lab.utils.quantities import BaseModel_with_q
from optimi_lab.utils.variable_space import VariableSpace


class MutationBase(BaseModel_with_q, ABC):
    """Base class for mutation operators.

    Defines the common interface for all mutation operators.
    """

    @abstractmethod
    def do(self, current_inputs: np.ndarray, best_inputs: np.ndarray, variable_space: VariableSpace) -> np.ndarray:
        """Perform the mutation operation.

        Args:
            current_inputs(np.ndarray): Current input matrix
            best_inputs(np.ndarray): Best individuals input matrix (may be empty)
            variable_space(VariableSpace): Variable space object

        Returns:
            The mutated individuals matrix.

        """
        ...


class PolynomialMutation(MutationBase):
    """Polynomial mutation.

    Real-coded mutation used in genetic algorithms, based on a polynomial distribution.

    Attributes:
        mutation_rate (float): Mutation probability controlling per-gene mutation chance.
        eta_mutation (float): Distribution index controlling mutation strength; larger values concentrate mutations near the original value.

    """

    mutation_rate: float = 0.1
    eta_mutation: float = 20.0

    @field_validator('mutation_rate')
    def check_mutation_rate(cls, v):
        if not (0.0 <= v <= 1.0):
            msg = 'mutation_rate must be between 0 and 1'
            raise ParameterException(msg)
        return v

    @field_validator('eta_mutation')
    def check_eta_mutation(cls, v):
        if v <= 0:
            msg = 'eta_mutation must be positive'
            raise ParameterException(msg)
        return v

    def do(self, current_inputs: np.ndarray, best_inputs: np.ndarray, variable_space: VariableSpace) -> np.ndarray:
        """Perform polynomial mutation.

        Args:
            current_inputs(np.ndarray): Current input matrix, shape (pop_size, n_var)
            best_inputs(np.ndarray): Not used; present for API compatibility
            variable_space: Variable space object containing bounds

        Returns:
            Mutated individuals matrix, shape (pop_size, n_var)

        """
        pop_size, n_var = current_inputs.shape
        offspring = current_inputs.copy()

        # Get variable bounds
        xl, xu = np.array(variable_space.lower_bounds), np.array(variable_space.upper_bounds)

        # Generate mutation mask
        mutation_mask = np.random.rand(pop_size, n_var) < self.mutation_rate

        if not np.any(mutation_mask):
            return offspring

        # Compute mutation
        delta = xu - xl
        y = (offspring - xl) / delta  # normalize to [0,1]

        r = np.random.rand(pop_size, n_var)
        eta_m = self.eta_mutation

        # Compute mutation amount in two cases
        mask1 = (r <= 0.5) & mutation_mask
        mask2 = (r > 0.5) & mutation_mask

        # Case 1: r <= 0.5
        if np.any(mask1):
            xy = 1.0 - y
            val = 2.0 * r + (1.0 - 2.0 * r) * np.power(xy, eta_m + 1.0)
            delta_q = np.power(val, 1.0 / (eta_m + 1.0)) - 1.0
            offspring += delta_q * delta * mask1

        # Case 2: r > 0.5
        if np.any(mask2):
            val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * np.power(y, eta_m + 1.0)
            delta_q = 1.0 - np.power(val, 1.0 / (eta_m + 1.0))
            offspring += delta_q * delta * mask2

        # Enforce boundary constraints
        return np.clip(offspring, xl, xu)


class DifferentialMutation(MutationBase):
    """Differential mutation (Differential Evolution Mutation).

    Mutation used in differential evolution that generates new candidate solutions via difference vectors.

    Attributes:
        sizing_factor (float): Scaling factor controlling the magnitude of the difference vector.

    """

    sizing_factor: float = 0.5

    @field_validator('sizing_factor')
    def check_sizing_factor(cls, v):
        if not (0.0 < v <= 2.0):
            msg = 'sizing_factor must be between 0 and 2'
            raise ParameterException(msg)
        return v

    def do(self, current_inputs: np.ndarray, best_inputs: np.ndarray, variable_space: VariableSpace) -> np.ndarray:
        """Perform differential mutation.

        Implements the DE/rand/1 strategy: V = X_r1 + F * (X_r2 - X_r3)

        Args:
            current_inputs(np.ndarray): Current population matrix, shape (pop_size, n_var)
            best_inputs(np.ndarray): Best individuals matrix, shape (archive_size, n_var)
            variable_space(VariableSpace): Variable space object containing bounds

        Returns:
            Mutated vector matrix, shape (pop_size, n_var)

        """
        pop_size, _ = current_inputs.shape
        xl, xu = variable_space.lower_bounds, variable_space.upper_bounds

        # Select three distinct random individuals for each member
        trial_vectors = np.zeros_like(current_inputs)

        for i in range(pop_size):
            # Select three distinct random indices (exclude current individual)
            candidates = [j for j in range(pop_size) if j != i]
            if len(candidates) < 3:
                # If population too small, use current individual as the base vector
                trial_vectors[i] = current_inputs[i]
                continue

            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

            # Differential mutation: V = X_r1 + F * (X_r2 - X_r3)
            trial_vectors[i] = current_inputs[r1] + self.sizing_factor * (current_inputs[r2] - current_inputs[r3])

        # Enforce boundary constraints
        return np.clip(trial_vectors, xl, xu)
