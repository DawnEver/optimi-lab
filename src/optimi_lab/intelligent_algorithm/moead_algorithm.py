from collections.abc import Callable

import numpy as np

from optimi_lab.intelligent_algorithm.operators.crossover import SimulatedBinaryCrossover
from optimi_lab.intelligent_algorithm.operators.mutation import PolynomialMutation
from optimi_lab.utils.variable_space import VariableSpace

from .intelligent_algorithm_base import IntelligentAlgorithmBase


def tchebycheff_approach(objectives: np.ndarray, weight: np.ndarray, ideal_point: np.ndarray) -> float:
    """Tchebycheff aggregation function.

    Args:
        objectives (np.ndarray): Objective vector, shape (n_obj,).
        weight (np.ndarray): Weight vector, shape (n_obj,).
        ideal_point (np.ndarray): Ideal point, shape (n_obj,).

    Returns:
        float: Aggregated scalar value.

    """
    return np.max(weight * np.abs(objectives - ideal_point))


def generate_weight_vectors(n_obj: int, pop_size: int) -> np.ndarray:
    """Generate weight vectors.

    Use the Das and Dennis method to generate uniformly distributed weight vectors.

    Args:
        n_obj (int): Number of objectives.
        pop_size (int): Population size.

    Returns:
        np.ndarray: Weight matrix, shape (pop_size, n_obj).

    """
    if n_obj == 2:
        # For 2-objective problems, use simple uniform distribution
        weights = np.linspace(0, 1, pop_size).reshape(-1, 1)
        weights = np.hstack([weights, 1 - weights])
    elif n_obj == 3:
        # For 3-objective problems, use triangular grid
        h = int(np.sqrt(2 * pop_size)) + 1  # Estimate grid size
        i, j = np.meshgrid(np.arange(h + 1), np.arange(h + 1))
        i = i.flatten()
        j = j.flatten()
        mask = i + j <= h
        i = i[mask]
        j = j[mask]
        k = h - i - j
        weights = np.stack([i / h, j / h, k / h], axis=1)
        # If the generated weight vectors are more than needed, randomly sample
        # to select the required number of weight vectors
        indices = np.random.choice(len(weights), pop_size, replace=False)
        weights = weights[indices]
    else:
        # For higher-dimensional problems, use Dirichlet distribution
        weights = np.random.dirichlet(np.ones(n_obj), pop_size)
    return weights


def find_neighbors(weight_vectors: np.ndarray, neighborhood_size: int) -> np.ndarray:
    """Find neighbors for each weight vector.

    Use Euclidean distance to find the nearest neighbors.

    Args:
        weight_vectors (np.ndarray): Weight matrix, shape (pop_size, n_obj).
        neighborhood_size (int): Number of neighbors.

    Returns:
        np.ndarray: Neighbor index matrix, shape (pop_size, neighborhood_size).

    """
    # Compute pairwise distances between weight vectors
    distances = np.linalg.norm(weight_vectors[:, np.newaxis, :] - weight_vectors[np.newaxis, :, :], axis=2)
    # Find nearest neighbors for each weight vector (including itself)
    return np.argsort(distances, axis=1)[:, :neighborhood_size]


class MOEAD_Algorithm(IntelligentAlgorithmBase):
    """Multi-Objective Evolutionary Algorithm based on Decomposition (MOEA/D).

    Transforms a multi-objective problem into multiple scalar subproblems and approximates
    the Pareto front via cooperative optimization. The core idea is divide-and-conquer,
    suitable for high-dimensional multi-objective optimization.
    - Refer to *Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition," in IEEE Transactions on Evolutionary Computation, vol. 11, no. 6, pp. 712-731, Dec. 2007, doi: 10.1109/TEVC.2007.892759.*

    Attributes:
        _neighborhood_size (int): Number of neighbors.
        _neighbor_rate (float): Probability of selecting parents from neighbors.
        _max_replace (int): Maximum number of neighbors to replace.

        _crossover_operator (SimulatedBinaryCrossover): Crossover operator.
        _mutation_operator (PolynomialMutation): Mutation operator.

        _ideal_point (np.ndarray): Ideal point, shape (n_obj,).
        _weight_vectors (np.ndarray): Weight vector matrix, shape (pop_size, n_obj).
        _neighbors (np.ndarray): Neighbor index matrix, shape (pop_size, neighborhood_size).

    """

    _neighborhood_size: int
    _neighbor_rate: float
    _max_replace: int

    _crossover_operator: SimulatedBinaryCrossover
    _mutation_operator: PolynomialMutation

    _ideal_point: np.ndarray
    _weight_vectors: np.ndarray
    _neighbors: np.ndarray

    def __init__(
        self,
        variable_space: VariableSpace,
        n_obj: int,
        pop_size: int,
        max_iter: int,
        object_function: Callable | None = None,
        neighborhood_size: int = 20,
        neighbor_rate: float = 0.9,
        max_replace: int = 2,
        crossover_operator: SimulatedBinaryCrossover = None,
        mutation_operator: PolynomialMutation = None,
    ) -> None:
        super().__init__(
            object_function=object_function,
            variable_space=variable_space,
            n_obj=n_obj,
            pop_size=pop_size,
            max_iter=max_iter,
        )

        self._neighborhood_size = min(neighborhood_size, pop_size)
        self._neighbor_rate = neighbor_rate
        self._max_replace = max_replace
        self._crossover_operator = crossover_operator or SimulatedBinaryCrossover()
        self._mutation_operator = mutation_operator or PolynomialMutation()

        # Initialize weight vectors and neighbor information
        self._weight_vectors = generate_weight_vectors(n_obj=n_obj, pop_size=pop_size)
        self._neighbors = find_neighbors(weight_vectors=self._weight_vectors, neighborhood_size=self._neighborhood_size)
        self._ideal_point = None  # Ideal point initialized to None

    def _perform_crossover(self, parents_inputs: np.ndarray) -> np.ndarray:
        """Perform crossover to generate offspring.

        Args:
            parents_inputs (np.ndarray): Parent pairs, shape (pop_size, 2, n_var).

        Returns:
            np.ndarray: Offspring individuals, shape (pop_size, n_var).

        """
        offspring_list = []
        for i in range(self._pop_size):
            parent1 = parents_inputs[i, 0:1]
            parent2 = parents_inputs[i, 1:2]
            child1, child2 = self._crossover_operator.do(
                parent1=parent1, parent2=parent2, variable_space=self._variable_space
            )
            # Randomly select one of the two children
            offspring = child1[0] if np.random.rand() < 0.5 else child2[0]
            offspring_list.append(offspring)
        # Ensure correct offspring count
        return np.array(offspring_list[: self._pop_size])

    def _step(self):
        """Perform a single MOEA/D iteration."""
        # 1. Parent selection - choose from neighbors or the global population
        use_neighbors = np.random.rand(self._pop_size) < self._neighbor_rate
        parent_indices = np.zeros((self._pop_size, 2), dtype=int)
        for i in range(self._pop_size):
            if use_neighbors[i]:
                mating_pool = self._neighbors[i]
            else:
                mating_pool = np.arange(self._pop_size)
            parent_indices[i] = np.random.choice(mating_pool, 2, replace=False)
        parents_inputs = self._current_inputs[parent_indices]

        # 2. Crossover - pairwise crossover to create offspring
        offspring_inputs = self._perform_crossover(parents_inputs)

        # 3. Mutation
        offspring_inputs = self._mutation_operator.do(
            current_inputs=offspring_inputs, best_inputs=self._pareto_inputs, variable_space=self._variable_space
        )

        # 4. Evaluate offspring
        offspring_outputs = self._object_function(offspring_inputs)

        # 5. Update ideal point
        if self._ideal_point is None:
            self._ideal_point = np.min(offspring_outputs, axis=0)
        else:
            self._ideal_point = np.minimum(self._ideal_point, np.min(offspring_outputs, axis=0))

        # 6. Update neighbor solutions and current inputs/outputs
        self._update_neighbors(offspring_inputs, offspring_outputs)

    def _update_neighbors(self, offspring_inputs: np.ndarray, offspring_outputs: np.ndarray) -> None:
        """Update neighbor solutions.

        Args:
            offspring_inputs (np.ndarray): Offspring individuals, shape (pop_size, n_var).
            offspring_outputs (np.ndarray): Offspring objectives, shape (pop_size, n_obj).

        """
        n_replace = min(self._max_replace, self._neighborhood_size)

        for i in range(self._pop_size):
            # Randomly select neighbors to compare (select up to n_replace*2 neighbors)
            neighbors_to_check = np.random.choice(
                self._neighbors[i], min(self._neighborhood_size, n_replace * 2), replace=False
            )

            neighbor_idxs = neighbors_to_check
            # Broadcast offspring_outputs[i] across the selected neighbors
            offspring_scores = tchebycheff_approach(
                np.repeat(offspring_outputs[i][np.newaxis, :], len(neighbor_idxs), axis=0),
                self._weight_vectors[neighbor_idxs],
                self._ideal_point,
            )
            current_scores = tchebycheff_approach(
                self._current_outputs[neighbor_idxs], self._weight_vectors[neighbor_idxs], self._ideal_point
            )
            # Find neighbor indices where offspring_scores < current_scores
            better_idxs = neighbor_idxs[offspring_scores < current_scores]
            if len(better_idxs) == 0:
                continue
            better_idxs = better_idxs.flatten()
            # Replace up to n_replace neighbors
            if len(better_idxs) > n_replace:
                better_idxs = np.random.choice(better_idxs, n_replace, replace=False)
            # Batch replace
            self._current_inputs[better_idxs] = offspring_inputs[i]
            self._current_outputs[better_idxs] = offspring_outputs[i]
