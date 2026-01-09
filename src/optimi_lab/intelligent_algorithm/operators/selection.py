from abc import ABC, abstractmethod

import numpy as np
from pydantic import field_validator, model_validator

from optimi_lab.intelligent_algorithm.utils import non_dominated_sorting
from optimi_lab.utils.exceptions import ParameterException
from optimi_lab.utils.quantities import BaseModel_with_q


class SelectionBase(BaseModel_with_q, ABC):
    """Base selection operator.

    Defines the common interface for all selection operators.

    Attributes:
        n_selected (int): Number of individuals to select; must be greater than 1.

    """

    n_selected: int

    @field_validator('n_selected')
    def check_n_selected(cls, v):
        if v <= 1:
            msg = 'n_selected must be greater than 1'
            raise ParameterException(msg)
        return v

    @abstractmethod
    def do(self, fitness: np.ndarray) -> np.ndarray:
        """Perform the selection operation.

        Args:
            fitness (np.ndarray): Fitness matrix, shape (pop_size, n_obj) or
                (pop_size + offspring_size, n_obj).

        Returns:
            np.ndarray: Selected indices, shape (n_selected,).

        """
        ...


class RandomSelection(SelectionBase):
    """Random selection operator.

    Used for parent selection in genetic algorithms; randomly chooses the
    specified number of individuals.
    """

    def do(self, fitness: np.ndarray) -> np.ndarray:
        """Randomly select individuals.

        Args:
            fitness (np.ndarray): Fitness matrix, shape (pop_size, n_obj) or
                (pop_size + offspring_size, n_obj).

        Returns:
            np.ndarray: Selected indices, shape (n_selected,).

        """
        pop_size = fitness.shape[0]
        return np.random.choice(pop_size, self.n_selected, replace=False)


class TournamentSelection(SelectionBase):
    """Tournament selection.

    Selects individuals via tournament: compare several candidate individuals
    and pick the best. Useful for parent selection in evolutionary algorithms.

    Attributes:
        tournament_size (int): Tournament size; number of individuals competing.

    """

    tournament_size: int = 2

    @field_validator('tournament_size')
    def check_tournament_size(cls, v):
        if v < 2:
            msg = 'tournament_size must be at least 2'
            raise ParameterException(msg)
        return v

    def do(self, fitness: np.ndarray) -> np.ndarray:
        """Perform tournament selection.

        Args:
            fitness (np.ndarray): Fitness matrix, shape (pop_size, n_obj) or
                (pop_size + offspring_size, n_obj).

        Returns:
            np.ndarray: Selected indices, shape (n_selected,).

        """
        pop_size = fitness.shape[0]
        selected_indices = np.zeros(self.n_selected, dtype=int)

        for i in range(self.n_selected):
            # Randomly choose candidate individuals
            candidates = np.random.choice(pop_size, self.tournament_size, replace=False)
            candidate_fitness = fitness[candidates]

            # Use non-dominated sorting to select the best individual
            fronts = non_dominated_sorting(candidate_fitness, only_first_front=False)
            front_indices = fronts[0]
            if len(front_indices) == 1:
                selected_indices[i] = candidates[front_indices[0]]
            else:
                # Use crowding distance when multiple non-dominated solutions exist
                front_fitness = candidate_fitness[front_indices]
                crowding_distances = calc_crowding_distance(front_fitness)
                best_idx = front_indices[np.argmax(crowding_distances)]
                selected_indices[i] = candidates[best_idx]

        return selected_indices


class ParetoSelection(SelectionBase):
    """Selection based on Pareto fronts.

    Uses non-dominated sorting to prefer individuals from better fronts.
    """

    def do(self, fitness: np.ndarray) -> np.ndarray:
        """Execute Pareto-based selection.

        Args:
            fitness (np.ndarray): Fitness matrix, shape (pop_size, n_obj) or
                (pop_size + offspring_size, n_obj).

        Returns:
            np.ndarray: Selected indices, shape (n_selected,).

        """
        fronts = non_dominated_sorting(fitness, only_first_front=False)
        selected_indices = []

        for front_indices in fronts:
            if len(selected_indices) + len(front_indices) <= self.n_selected:
                selected_indices.extend(front_indices)
            else:
                remaining_slots = self.n_selected - len(selected_indices)
                selected_indices.extend(self._select_from_front(fitness, front_indices, remaining_slots))
                break

        return np.array(selected_indices[: self.n_selected])

    def _select_from_front(self, fitness: np.ndarray, front_indices: list, n_select: int) -> list:
        """Select a specified number of individuals from a front.

        Base implementation returns the first `n_select` indices. Subclasses
        may override this to apply different selection criteria.
        """
        return front_indices[:n_select]


if 1:

    def calc_crowding_distance(outputs: np.ndarray) -> np.ndarray:
        """Calculate crowding distance for diversity preservation in MOO.

        Args:
            outputs (np.ndarray): Objective values of front members, shape
                (n_points, n_objectives).

        Returns:
            np.ndarray: Crowding distances, shape (n_points,).

        Examples:
            >>> import numpy as np
            >>> outputs = np.array([[1, 2], [2, 1], [1, 1], [0, 0]])
            >>> np.allclose(calc_crowding_distance(outputs), [np.inf, np.inf, 1, np.inf])
            True

        """
        n_points, n_objectives = outputs.shape
        distance = np.zeros(n_points)

        if n_points <= 2:
            distance[:] = np.inf  # Boundary points or fewer than 3 points -> infinite
            return distance

        # Compute crowding distance for each objective separately
        for m in range(n_objectives):
            obj = outputs[:, m]
            sorted_indices = np.argsort(obj)
            sorted_obj = obj[sorted_indices]

            # Boundary points set to infinite distance
            distance[sorted_indices[0]] = np.inf
            distance[sorted_indices[-1]] = np.inf

            # Compute objective range to avoid division by zero
            obj_range = sorted_obj[-1] - sorted_obj[0]
            if obj_range == 0:
                continue

            # Compute crowding distance for interior points
            for i in range(1, n_points - 1):
                # if distance[sorted_indices[i]] != np.inf:  # do not overwrite
                # boundary points already set to infinite
                distance[sorted_indices[i]] += (sorted_obj[i + 1] - sorted_obj[i - 1]) / obj_range

        return distance

    class ParetoCrowdingSelection(ParetoSelection):
        """Pareto-front selection with crowding distance.

        Prefer individuals with larger crowding distance within the same front
        to preserve solution diversity.
        """

        def _select_from_front(self, fitness: np.ndarray, front_indices: list, n_select: int) -> list:
            """Select a number of individuals from a front, preferring those
            with larger crowding distances.
            """
            if n_select >= len(front_indices):
                return front_indices

            front_fitness = fitness[front_indices]
            crowding_distances = calc_crowding_distance(front_fitness)
            sorted_indices = np.argsort(-crowding_distances)  # sort in descending order
            return [front_indices[i] for i in sorted_indices[:n_select]]

    def generate_das_dennis_reference_directions(n_objectives: int, n_partitions: int) -> np.ndarray:
        """Generate Das-Dennis reference directions.

        Produce reference points for NSGA-III by uniformly distributing
        reference directions on the unit simplex.

        Args:
            n_objectives (int): Number of objectives.
            n_partitions (int): Number of partitions, controls direction density.

        Returns:
            np.ndarray: Reference directions, shape (n_directions, n_objectives).

        """

        def recursive_generation(current_sum: float, remaining_dims: int, current_point: list):
            if remaining_dims == 1:
                current_point.append(current_sum)
                points.append(current_point.copy())
                current_point.pop()
            else:
                for i in range(int(current_sum * n_partitions) + 1):
                    val = i / n_partitions
                    current_point.append(val)
                    recursive_generation(current_sum - val, remaining_dims - 1, current_point)
                    current_point.pop()

        points = []
        recursive_generation(1.0, n_objectives, [])
        return np.array(points)

    def normalize_objectives(
        objectives: np.ndarray, ideal_point: np.ndarray = None, nadir_point: np.ndarray = None
    ) -> np.ndarray:
        """Normalize objectives to [0, 1].

        Args:
            objectives (np.ndarray): Objective matrix, shape (n_points, n_objectives).
            ideal_point (np.ndarray): Ideal point; if None, use minima.
            nadir_point (np.ndarray): Nadir point; if None, use maxima.

        Returns:
            np.ndarray: Normalized objectives.

        """
        if ideal_point is None:
            ideal_point = np.min(objectives, axis=0)
        if nadir_point is None:
            nadir_point = np.max(objectives, axis=0)

        # Avoid division by zero
        range_vals = nadir_point - ideal_point
        range_vals = np.where(range_vals == 0, 1e-8, range_vals)

        return (objectives - ideal_point) / range_vals

    def associate_to_reference_points(
        normalized_objectives: np.ndarray, reference_directions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Associate solutions to the nearest reference points.

        Compute the perpendicular distance from each normalized solution to
        each reference direction and find the closest reference for each.

        Args:
            normalized_objectives (np.ndarray): Normalized objectives, shape
                (n_points, n_objectives).
            reference_directions (np.ndarray): Reference directions, shape
                (n_directions, n_objectives).

        Returns:
            tuple[np.ndarray, np.ndarray]: (closest_reference_indices, distances)

        """
        # Normalize reference directions
        ref_norms = np.linalg.norm(reference_directions, axis=1, keepdims=True)
        ref_norms = np.where(ref_norms == 0, 1e-8, ref_norms)
        normalized_refs = reference_directions / ref_norms

        # Compute projection lengths of each solution onto each reference direction
        projection_lengths = np.dot(normalized_objectives, normalized_refs.T)

        # Compute projection points
        projections = projection_lengths[:, :, np.newaxis] * normalized_refs[np.newaxis, :, :]

        # Compute perpendicular distances to reference directions
        differences = normalized_objectives[:, np.newaxis, :] - projections
        distances = np.linalg.norm(differences, axis=2)

        # Find the nearest reference direction for each solution
        closest_ref_indices = np.argmin(distances, axis=1)
        closest_distances = np.min(distances, axis=1)

        return closest_ref_indices, closest_distances

    class ParetoRefSelection(ParetoSelection):
        """NSGA-III selection strategy based on reference points.

        Uses Das-Dennis reference directions to maintain population diversity,
        suitable for many-objective optimization.

        Attributes:
            n_objectives (int): Number of objectives.
            n_partitions (int): Number of partitions for reference directions.
            reference_directions (np.ndarray): Reference directions matrix.

        """

        n_objectives: int
        n_partitions: int = 12
        reference_directions: np.ndarray = None

        @field_validator('n_objectives')
        def check_n_objectives(cls, v):
            if v < 2:
                msg = 'n_objectives must be at least 2'
                raise ParameterException(msg)
            return v

        @field_validator('n_partitions')
        def check_n_partitions(cls, v):
            if v < 1:
                msg = 'n_partitions must be at least 1'
                raise ParameterException(msg)
            return v

        @model_validator(mode='after')
        def check_reference_directions(self):
            """Generate default Das-Dennis reference directions if missing."""
            if self.reference_directions is None:
                self.reference_directions = generate_das_dennis_reference_directions(
                    self.n_objectives, self.n_partitions
                )
            if self.reference_directions.shape[1] != self.n_objectives:
                msg = f'reference_directions must have shape (n_directions, {self.n_objectives})'
                raise ParameterException(msg)
            return self

        def _select_from_front(self, fitness: np.ndarray, front_indices: list, n_select: int) -> list:
            """Select individuals from a front based on reference points."""
            if n_select >= len(front_indices):
                return front_indices

            # Normalize fitness of the current front
            front_fitness = fitness[front_indices]
            normalized_fitness = normalize_objectives(front_fitness)

            # Associate to reference points
            ref_indices, distances = associate_to_reference_points(normalized_fitness, self.reference_directions)

            # Count how many individuals are associated with each reference point
            ref_counts = np.bincount(ref_indices, minlength=len(self.reference_directions))

            # NSGA-III selection: prefer solutions associated with reference points
            # that currently have fewer associated individuals
            selected = []
            available_indices = set(range(len(front_indices)))

            for _ in range(n_select):
                # TODO: If no error occurs, remove the following two lines (Aug/22/2025)
                # if not available_indices:
                #     break

                # Find reference points with the minimum associated count
                min_count = np.min(ref_counts)
                candidate_refs = np.where(ref_counts == min_count)[0]

                # Find available individuals associated with these reference points
                valid_candidates = [
                    (i, ref_idx) for ref_idx in candidate_refs for i in available_indices if ref_indices[i] == ref_idx
                ]

                if valid_candidates:
                    # Select the individual with the smallest distance
                    best_idx, best_ref = min(valid_candidates, key=lambda x: distances[x[0]])
                    selected.append(front_indices[best_idx])
                    available_indices.remove(best_idx)
                    ref_counts[best_ref] += 1
                else:
                    # If no valid candidates, randomly choose an available individual
                    idx = available_indices.pop()
                    selected.append(front_indices[idx])

            return selected
