from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from optimi_lab.intelligent_algorithm.utils import non_dominated_sorting
from optimi_lab.utils.logger import log
from optimi_lab.utils.variable_space import VariableSpace


class IntelligentAlgorithmBase(ABC):
    """Base class for all intelligent optimization algorithms.

    Attributes:
        _object_function (Callable): Objective function that accepts inputs and
            returns outputs. It may accept the entire `_current_inputs` and
            return `_current_outputs`; the objective function may implement
            internal parallelization.
        _max_iter (int): Maximum number of iterations.
        _pop_size (int): Population size per iteration.

        _id_iter (int): Iteration counter.
        _variable_space (VariableSpace): Variable space object containing
            variable bounds and names.

        _n_var (int): Number of input variables.
        _n_obj (int): Number of objectives / outputs.
        _all_inputs (np.ndarray): Accumulated inputs across iterations.
        _all_outputs (np.ndarray): Accumulated outputs across iterations.
        _pareto_inputs (np.ndarray): Inputs corresponding to Pareto-optimal solutions.
        _pareto_outputs (np.ndarray): Outputs corresponding to Pareto-optimal solutions.
        _current_inputs (np.ndarray): Current population inputs.
        _current_outputs (np.ndarray): Current population outputs / fitness.

    """

    _object_function: Callable
    _max_iter: int
    _pop_size: int

    _id_iter: int
    _variable_space: VariableSpace

    _n_var: int
    _n_obj: int
    _all_inputs: np.ndarray
    _all_outputs: np.ndarray
    _pareto_inputs: np.ndarray
    _pareto_outputs: np.ndarray
    _current_inputs: np.ndarray
    _current_outputs: np.ndarray

    def __init__(
        self,
        variable_space: VariableSpace,
        n_obj: int,
        pop_size: int,
        max_iter: int,
        object_function: Callable | None = None,
    ) -> None:
        self._object_function = object_function
        self._max_iter = max_iter
        self._variable_space = variable_space
        self._pop_size = pop_size

        self._id_iter = 0
        self._n_var = n_var = len(variable_space.var_name_list)
        self._n_obj = n_obj
        self._all_inputs = np.zeros((0, n_var))
        self._all_outputs = np.zeros((0, n_obj))
        self._pareto_inputs = np.zeros((0, n_var))
        self._pareto_outputs = np.zeros((0, n_obj))

    def minimize(self):
        if self._object_function is None:
            msg = 'The objective function is not defined, please define it first.'
            log(msg=msg, level='ERROR')
            raise AttributeError(msg)

        while self.check_iteration():
            if self._id_iter == 0:
                # Initialize population and compute initial fitness
                self._current_inputs = self._variable_space.var_space_matrix
                self._current_outputs = self._object_function(self._current_inputs)
            else:
                # Perform iteration step
                self._step()

            # Archive current iteration results
            self._all_inputs = np.vstack([self._all_inputs, self._current_inputs])
            self._all_outputs = np.vstack([self._all_outputs, self._current_outputs])
            front_indices = non_dominated_sorting(obj_mat=self._all_outputs, only_first_front=True)
            self._pareto_inputs = self._all_inputs[front_indices]
            self._pareto_outputs = self._all_outputs[front_indices]

            self.callback()
            self.callback()
            # Update iteration counter
            self._id_iter += 1

    @abstractmethod
    def _step(self):
        """Perform a single iteration; subclasses must implement this method."""

    def check_iteration(self) -> bool:
        """Check whether to continue iterating.

        Returns:
            bool: False indicates termination condition met, True to continue.

        """
        # Termination conditions can be customized per algorithm
        return self._id_iter < self._max_iter

    def callback(self):
        """Callback executed after each iteration.

        Example uses: print iteration info, record logs, etc.
        """
        msg = f'Iteration {self._id_iter}/{self._max_iter}'
        log(msg, level='INFO')
