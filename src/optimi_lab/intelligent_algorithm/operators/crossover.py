from abc import ABC, abstractmethod

import numpy as np
from pydantic import field_validator

from optimi_lab.utils.exceptions import ParameterException
from optimi_lab.utils.quantities import BaseModel_with_q
from optimi_lab.utils.variable_space import VariableSpace


class CrossoverBase(BaseModel_with_q, ABC):
    """Base class for crossover operators.

    Attributes:
        crossover_rate (float): Crossover probability, controls the chance of performing crossover.

    """

    crossover_rate: float = 0.9

    @field_validator('crossover_rate')
    def check_crossover_rate(cls, v):
        if not (0.0 <= v <= 1.0):
            msg = 'crossover_rate must be between 0 and 1'
            raise ParameterException(msg)
        return v

    @abstractmethod
    def do(self, parent1: np.ndarray, parent2: np.ndarray, variable_space: VariableSpace) -> np.ndarray:
        """Perform the crossover operation.

        Args:
            parent1(np.ndarray): Parent 1 matrix
            parent2(np.ndarray): Parent 2 matrix
            variable_space(VariableSpace): Variable space object

        Returns:
            The offspring after crossover.

        """
        ...


class BinomialCrossover(CrossoverBase):
    """Binomial crossover (Binomial Crossover).

    Used in differential evolution crossover. A binomial distribution determines which
    dimensions use the mutation vector.
    """

    def do(self, parent1: np.ndarray, parent2: np.ndarray, variable_space: VariableSpace) -> np.ndarray:
        """Perform binomial crossover.

        Args:
            parent1(np.ndarray): Current population matrix, shape (pop_size, n_var)
            parent2(np.ndarray): Mutation vector matrix, shape (pop_size, n_var)
            variable_space(VariableSpace): Variable space object (not used here)

        Returns:
            np.ndarray: Offspring matrix after crossover, shape (pop_size, n_var)

        """
        offspring = parent1.copy()
        pop_size, n_var = parent2.shape

        # Generate crossover probability matrix
        crossover_mask = np.random.rand(pop_size, n_var) < self.crossover_rate

        # Ensure each individual has at least one dimension from the mutation vector
        j_rand = np.random.randint(0, n_var, pop_size)
        crossover_mask[np.arange(pop_size), j_rand] = True

        # Apply crossover operation
        return np.where(crossover_mask, parent2, offspring)


class SimulatedBinaryCrossover(CrossoverBase):
    """Simulated Binary Crossover (SBX).

    Real-coded crossover used in genetic algorithms that simulates binary crossover behavior.

    Attributes:
        eta_crossover (float): Distribution index controlling offspring distribution concentration; larger values produce offspring closer to parents.

    """

    eta_crossover: float = 20.0

    @field_validator('eta_crossover')
    def check_eta_crossover(cls, v):
        if v < 0:
            msg = 'eta_crossover must be non-negative'
            raise ParameterException(msg)
        return v

    def do(
        self, parent1: np.ndarray, parent2: np.ndarray, variable_space: VariableSpace
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform simulated binary crossover (SBX).

        Args:
            parent1(np.ndarray): Parent 1 matrix, shape (pop_size, n_var)
            parent2(np.ndarray): Parent 2 matrix, shape (pop_size, n_var)
            variable_space(VariableSpace): Variable space object containing bounds

        Returns:
            tuple[np.ndarray, np.ndarray]: (offspring1, offspring2), each of shape (pop_size, n_var)

        """
        pop_size, n_var = parent1.shape
        offspring1, offspring2 = parent1.copy(), parent2.copy()

        # Generate crossover mask
        crossover_mask = np.random.rand(pop_size, n_var) < self.crossover_rate

        if not np.any(crossover_mask):
            return offspring1, offspring2

        # Get bounds
        xl, xu = np.array(variable_space.lower_bounds), np.array(variable_space.upper_bounds)

        # Ensure parent order: p1 <= p2
        swap_mask = parent1 > parent2
        p1 = np.where(swap_mask, parent2, parent1)
        p2 = np.where(swap_mask, parent1, parent2)

        # Avoid division by zero and compute beta values
        delta = p2 - p1
        valid_mask = (delta > 1e-14) & crossover_mask

        if not np.any(valid_mask):
            return offspring1, offspring2

        # Generate random numbers and compute beta
        u = np.random.rand(pop_size, n_var)
        beta = np.ones_like(u)

        # Compute beta distribution parameters
        eta_c = self.eta_crossover
        mask1 = (u <= 0.5) & valid_mask
        mask2 = (u > 0.5) & valid_mask

        beta[mask1] = np.power(2.0 * u[mask1], 1.0 / (eta_c + 1.0))
        beta[mask2] = np.power(1.0 / (2.0 * (1.0 - u[mask2])), 1.0 / (eta_c + 1.0))

        # Compute offspring
        c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
        c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)

        # Apply crossover results and enforce boundary constraints
        offspring1 = np.where(crossover_mask, c1, offspring1)
        offspring2 = np.where(crossover_mask, c2, offspring2)

        offspring1 = np.clip(offspring1, xl, xu)
        offspring2 = np.clip(offspring2, xl, xu)

        return offspring1, offspring2
