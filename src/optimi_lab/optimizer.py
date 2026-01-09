import pickle
from abc import ABC
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import model_validator

from optimi_lab.intelligent_algorithm.intelligent_algorithm_base import IntelligentAlgorithmBase
from optimi_lab.intelligent_algorithm.utils import non_dominated_sorting
from optimi_lab.object_function.utils import sort_obj_name_list
from optimi_lab.plot import plot, plot_contour, plot_moo
from optimi_lab.surrogate_model.mixture_surrogate_model import MixtureSurrogateModel
from optimi_lab.utils.config import PathData
from optimi_lab.utils.file_io import read_toml, save_toml
from optimi_lab.utils.logger import log, timer
from optimi_lab.utils.multiprocessing import MultiProcessingParameters
from optimi_lab.utils.quantities import BaseModel_with_q
from optimi_lab.utils.variable_space import VariableSpace

PENALTY_VALUE = np.inf  # 2147483647 # 32bit int max value; used to fill unmet objective values
GET_DF_CATEGORIES = ['all', 'pareto']


class Optimizer(BaseModel_with_q, ABC):
    """Base optimizer class providing core structure and helper methods.

    Attributes:
        variable_space (VariableSpace): Variable space object containing bounds and names.
        mp_params (MultiProcessingParameters | dict): Multiprocessing parameters for parallel runs.
        obj_name_list (list[str]): List of objective names.
        base_params_dict_list (list[dict[str,dict]]): List of base parameters per condition.
        obj_func (Callable): Objective function that accepts a variable matrix and returns results.
        intelligent_algorithm (IntelligentAlgorithmBase): Algorithm object used to perform optimization.
        mixture_surrogate_model (MixtureSurrogateModel): Combined surrogate model object.

        use_multiprocessing (bool): Whether to use multiprocessing.
        use_surrogate_model (bool): Whether to use surrogate models.
        log_at_opt (bool): Whether to log during optimization.

        _n_var (int): Number of input variables.
        _n_obj (int): Number of objective outputs.
        _max_obj_flags (list[bool]): Flags indicating which objectives are maximization.
        _obj_name_matrix (list[list[str]]): Objectives grouped per condition.
        _err_value_matrix (list[list[float]]): Penalty values for objectives.

        _all_inputs (np.ndarray): All collected input samples.
        _all_outputs (np.ndarray): All collected outputs.
        _pareto_inputs (np.ndarray): Pareto front input samples.
        _pareto_outputs (np.ndarray): Pareto front outputs.

    """

    variable_space: VariableSpace | None = None
    mp_params: MultiProcessingParameters | dict = {}
    obj_name_list: list[str] = []
    base_params_dict_list: list[dict[str, dict]] = []
    obj_func: Callable = None
    intelligent_algorithm: IntelligentAlgorithmBase | None = None
    mixture_surrogate_model: MixtureSurrogateModel | None = None
    use_multiprocessing: bool = True
    use_surrogate_model: bool = False
    log_at_opt: bool = True

    _n_var: int = None
    _n_obj: int = None
    _max_obj_flags: list[bool] = None
    _obj_name_matrix: list[list[str]] = None
    _err_value_matrix: list[list[float]] = None

    _all_inputs: np.ndarray = None
    _all_outputs: np.ndarray = None
    _pareto_inputs: np.ndarray = None
    _pareto_outputs: np.ndarray = None

    @model_validator(mode='after')
    def init_optimizer(self):
        self._n_obj = len(self.obj_name_list)
        self._n_var = len(self.variable_space.var_name_list)
        self._obj_name_matrix, self._err_value_matrix, self._max_obj_flags = sort_obj_name_list(self.obj_name_list)

        # Ensure the length of base_params_dict_list matches the number of objective-condition groups
        num_oc = len(self._obj_name_matrix)
        len_base_params_dict_list = len(self.base_params_dict_list)
        if len_base_params_dict_list < num_oc:
            while len(self.base_params_dict_list) < num_oc:
                self.base_params_dict_list.append(self.base_params_dict_list[0])
        elif len_base_params_dict_list > num_oc:
            self.base_params_dict_list = self.base_params_dict_list[:num_oc]

        self.init_inputs_outputs()
        return self

    # -------------------
    # Objective function / sampling sweep / optimization algorithm
    # -------------------

    def _obj_func_normalized(self, var_value_matrix: np.ndarray) -> np.ndarray:
        """Normalize objective outputs according to max/min flags.

        This converts all objectives into a minimization form by negating
        objectives that are marked as maximization. The workflow is:
        - `obj_func` returns raw, unnormalized objective values.
        - `_obj_func_normalized()` applies normalization based on max/min flags.
        - The optimizer internally works with minimization objectives.
        - `optimize()` will de-normalize results as needed for output.

        Args:
            var_value_matrix (np.ndarray): Input variable matrix with shape (n_samples, n_vars).

        Returns:
            np.ndarray: Normalized objective matrix with shape (n_samples, n_obj).

        """
        if self.use_surrogate_model:
            if self.mixture_surrogate_model is None:
                msg = 'Surrogate model function is not initialized. Train or load a surrogate model first.'
                log(msg, level='ERROR')
                raise AttributeError(msg)
            obj_value_matrix = self.mixture_surrogate_model.predict(var_value_matrix)
            obj_value_matrix[obj_value_matrix > PENALTY_VALUE] = np.inf
            obj_value_matrix[obj_value_matrix < -PENALTY_VALUE] = -np.inf
        else:
            obj_value_matrix = self.obj_func(var_value_matrix)

        if self.log_at_opt:
            data_array = np.hstack((var_value_matrix, obj_value_matrix))
            header_str = ','.join(self.variable_space.var_name_list + self.obj_name_list)
            data_str = '\n'.join(','.join(str(x) for x in row) for row in data_array)
            msg = f'{header_str}\n{data_str}'
            log(msg, level='INFO')

        for index, flag in enumerate(self._max_obj_flags):
            if flag:
                obj_value_matrix[:, index] = -obj_value_matrix[:, index]

        return obj_value_matrix

    @timer
    def sample_sweep(self) -> tuple[np.ndarray]:
        """Execute sampling sweep and return sampled variables and objectives.

        Returns:
            tuple[np.ndarray]: (var_value_matrix, obj_value_matrix)
                - var_value_matrix: sampled variable values, shape (n_samples, n_var)
                - obj_value_matrix: sampled objective values, shape (n_samples, n_obj)

        """
        var_value_matrix = self.variable_space.var_space_matrix
        obj_value_matrix = self._obj_func_normalized(var_value_matrix)
        # Denormalize output results
        for index, flag in enumerate(self._max_obj_flags):
            if flag:
                obj_value_matrix[:, index] = -obj_value_matrix[:, index]

        self._all_inputs = np.vstack([self._all_inputs, var_value_matrix])
        self._all_outputs = np.vstack([self._all_outputs, obj_value_matrix])
        return var_value_matrix, obj_value_matrix

    @timer
    def optimize(self) -> tuple[np.ndarray]:
        """Run the optimization algorithm and return denormalized results.

        Returns:
            tuple[np.ndarray]: (all_inputs, all_outputs, pareto_inputs, pareto_outputs)
                - all_inputs: aggregated input samples (n_samples, n_var)
                - all_outputs: aggregated output results (n_samples, n_obj)
                - pareto_inputs: Pareto front inputs (n_pareto_samples, n_var)
                - pareto_outputs: Pareto front outputs (n_pareto_samples, n_obj)

        """
        if self.intelligent_algorithm is None:
            msg = 'Intelligent algorithm is not initialized. Set or load an intelligent_algorithm first.'
            log(msg, level='ERROR')
            raise AttributeError(msg)
        self.intelligent_algorithm._object_function = self._obj_func_normalized
        self.intelligent_algorithm.minimize()
        # Denormalize output results
        all_inputs = self.intelligent_algorithm._all_inputs
        all_outputs = self.intelligent_algorithm._all_outputs.copy()

        pareto_inputs = self.intelligent_algorithm._pareto_inputs
        pareto_outputs = self.intelligent_algorithm._pareto_outputs.copy()
        for index, flag in enumerate(self._max_obj_flags):
            if flag:
                all_outputs[:, index] = -all_outputs[:, index]
                pareto_outputs[:, index] = -pareto_outputs[:, index]
        self._pareto_inputs = pareto_inputs
        self._pareto_outputs = pareto_outputs

        self._all_inputs = np.vstack([self._all_inputs, all_inputs])
        self._all_outputs = np.vstack([self._all_outputs, all_outputs])
        return all_inputs, all_outputs, pareto_inputs, pareto_outputs

    @timer
    def train_surrogate_models(self, inputs: np.ndarray = None, outputs: np.ndarray = None) -> None:
        """Train surrogate models using provided inputs/outputs or internal data.

        If `inputs` or `outputs` are None, use internal `_all_inputs` and `_all_outputs`.
        """
        if self.mixture_surrogate_model is None:
            msg = 'Surrogate model function is not initialized. Train or load a surrogate model first.'
            log(msg, level='ERROR')
            raise AttributeError(msg)
        if inputs is None or outputs is None:
            msg = 'No input/output provided; using internal _all_inputs and _all_outputs for training.'
            log(msg, level='DEBUG')
            inputs, outputs = self._all_inputs, self._all_outputs
        outputs[np.isposinf(outputs)] = PENALTY_VALUE
        outputs[np.isneginf(outputs)] = -PENALTY_VALUE
        for surrogate_model in self.mixture_surrogate_model._surrogate_model_pool:
            surrogate_model.train(inputs, outputs)

    def save_surrogate_models(self, file_path: str = PathData.surrogate_model_path):
        """Save surrogate model object to the specified path.

        Args:
            file_path (str): Path to save the surrogate model (default: PathData.surrogate_model_path).

        Raises:
            FileNotFoundError: If the path does not exist or is not writable.

        """
        if self.mixture_surrogate_model is None:
            msg = 'Surrogate model function is not initialized. Train or load a surrogate model first.'
            log(msg, level='ERROR')
            raise AttributeError(msg)
        with Path.open(file_path, 'wb') as f:
            pickle.dump(self.mixture_surrogate_model, f)
        log('Surrogate model function saved successfully.', level='DEBUG')

    def load_surrogate_models(self, file_path: str = PathData.surrogate_model_path):
        """Load surrogate model object from the specified path.

        Args:
            file_path (str): Path to the surrogate model file (default: PathData.surrogate_model_path).

        Raises:
            FileNotFoundError: If the file does not exist or cannot be read.

        """
        with Path.open(file_path, 'rb') as f:
            self.mixture_surrogate_model = pickle.load(f)  # noqa: S301
            log('Surrogate model function loaded successfully.', level='DEBUG')

    def show_surrogate_models(self, only_score: bool = True) -> dict:
        """Return details of surrogate models.

        Args:
            only_score (bool): If True, only include scoring information.

        Returns:
            dict: Mapping from model type to its details or scores.

        """
        surrogate_model_dict = {
            surrogate_model.model_type: surrogate_model.show(only_score=only_score)
            for surrogate_model in self.mixture_surrogate_model._surrogate_model_pool
        }
        surrogate_model_dict['mixture_surrogate_model'] = self.mixture_surrogate_model.show(only_score=only_score)
        return surrogate_model_dict

    # -------------------
    # Data processing functions
    # -------------------
    def init_inputs_outputs(self) -> None:
        """Clear all stored data, including `_all_inputs` and `_all_outputs`."""
        self._all_inputs = np.zeros((0, self._n_var))
        self._all_outputs = np.zeros((0, self._n_obj))
        self._pareto_inputs = np.zeros((0, self._n_var))
        self._pareto_outputs = np.zeros((0, self._n_obj))
        log('Clear data completed.', level='DEBUG')

    def filter_data(self, mask_mat: np.array) -> None:
        """Filter `_all_inputs` and `_all_outputs` using a boolean mask.

        Args:
            mask_mat (np.array): Boolean array where True keeps the row and False removes it.

        """
        # mask_mat is a boolean array: True keeps the row, False removes it
        self._all_outputs = self._all_outputs[mask_mat]
        self._all_inputs = self._all_inputs[mask_mat]
        msg = 'Filter data completed.'
        log(msg, level='DEBUG')

    def get_dataframe(
        self, category: str = 'all', drop_inf: bool = True, do_recalc_pareto: bool = False
    ) -> pd.DataFrame:
        """Return data as a pandas DataFrame for a given category.

        Args:
            category (str): 'all' or 'pareto'. Defaults to 'all'.
            drop_inf (bool): Whether to drop rows containing +/-inf. Defaults to True.
            do_recalc_pareto (bool): Whether to recalculate Pareto front. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing input and output columns.

        """
        category = category.lower()
        if category in ['all']:
            (inputs, outputs) = self._get_all_data(drop_inf=drop_inf)
        elif category in ['pareto']:
            (inputs, outputs) = self._get_pareto_data(drop_inf=drop_inf, do_recalc_pareto=do_recalc_pareto)
        else:
            msg = f'Invalid category: {category}. It should be one of {GET_DF_CATEGORIES}.'
            log(msg, level='ERROR')
            raise NameError(msg)

        if inputs.shape[1] != self._n_var or outputs.shape[1] != self._n_obj:
            df_inputs = pd.DataFrame(columns=self.variable_space.var_name_list)
            df_outputs = pd.DataFrame(columns=self.obj_name_list)
        else:
            df_inputs = pd.DataFrame(inputs, columns=self.variable_space.var_name_list)
            df_outputs = pd.DataFrame(outputs, columns=self.obj_name_list)
        data_frame = pd.concat([df_inputs, df_outputs], axis=1).drop_duplicates()
        log('Get data completed.', level='DEBUG')
        return data_frame

    def _get_all_data(self, drop_inf: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Return all stored inputs and outputs.

        Args:
            drop_inf (bool): Whether to drop rows containing +/-inf. Defaults to True.

        Returns:
            tuple: (all_inputs, all_outputs)

        """
        all_inputs = self._all_inputs.copy()
        all_outputs = self._all_outputs.copy()
        if drop_inf:
            pos_inf_rows = np.any(all_outputs == PENALTY_VALUE, axis=1)
            neg_inf_rows = np.all(all_outputs == -PENALTY_VALUE, axis=1)
            except_rows = pos_inf_rows + neg_inf_rows
            all_inputs = all_inputs[~except_rows]
            all_outputs = all_outputs[~except_rows]

        return all_inputs, all_outputs

    def _get_pareto_data(self, drop_inf: bool = True, do_recalc_pareto: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Return Pareto front inputs and outputs.

        Args:
            drop_inf (bool): Whether to drop rows containing +/-inf. Defaults to True.
            do_recalc_pareto (bool): Whether to recalculate Pareto front. Defaults to False.

        Returns:
            tuple: (pareto_inputs, pareto_outputs)

        """
        pareto_inputs = self._pareto_inputs.copy()
        pareto_outputs = self._pareto_outputs.copy()
        if do_recalc_pareto:
            pareto_inputs, pareto_outputs = self.recalc_pareto()
        elif drop_inf:
            pos_inf_rows = np.any(pareto_outputs == PENALTY_VALUE, axis=1)
            neg_inf_rows = np.all(pareto_outputs == -PENALTY_VALUE, axis=1)
            except_rows = pos_inf_rows + neg_inf_rows
            pareto_inputs = pareto_inputs[~except_rows]
            pareto_outputs = pareto_outputs[~except_rows]

        return pareto_inputs, pareto_outputs

    def recalc_pareto(self, update_self: bool = True) -> None:
        """Recalculate Pareto front and update `_pareto_inputs` and `_pareto_outputs`.

        Args:
            update_self (bool): Whether to update this object's `_pareto_inputs` and
                `_pareto_outputs` attributes. Defaults to True.

        Returns:
            tuple: A tuple containing recalculated Pareto front inputs and outputs:
                - np.ndarray: Recalculated Pareto front input samples
                - np.ndarray: Recalculated Pareto front output values

        """
        inputs, outputs = self._get_all_data()
        # Assume minimization problem; negate objectives that were maximized
        normalized_outputs = outputs.copy()
        for index, flag in enumerate(self._max_obj_flags):
            if flag:
                normalized_outputs[:, index] = -outputs[:, index]
        index = non_dominated_sorting(obj_mat=normalized_outputs, only_first_front=True)
        pareto_inputs = inputs[index]
        pareto_outputs = outputs[index]
        if update_self:
            self._pareto_inputs = pareto_inputs
            self._pareto_outputs = pareto_outputs

        msg = f'Recalculate pareto front is done, now pareto front size is {len(index)}'
        log(msg, level='DEBUG')
        return pareto_inputs, pareto_outputs

    # -------------------
    # Plotting
    # -------------------
    def plot(self, data_frame=None, *args, **kwargs) -> plt:
        if data_frame is None:
            data_frame = self._get_all_data()
        return plot(
            *args,
            var_name_list=self.variable_space.var_name_list,
            obj_name_list=self.obj_name_list,
            data_frame=data_frame,
            **kwargs,
        )

    def plot_moo(self, df_all=None, df_pareto=None, *args, **kwargs) -> plt:
        if df_all is None:
            df_all = self._get_all_data()
        if df_pareto is None:
            df_pareto = self._get_pareto_data()
        return plot_moo(
            *args,
            var_name_list=self.variable_space.var_name_list,
            obj_name_list=self.obj_name_list,
            df_all=df_all,
            df_pareto=df_pareto,
            **kwargs,
        )

    def plot_contour(self, data_frame=None, *args, **kwargs) -> plt:
        if data_frame is None:
            data_frame = self._get_all_data()
        return plot_contour(
            *args,
            var_name_list=self.variable_space.var_name_list,
            obj_name_list=self.obj_name_list,
            data_frame=data_frame,
            **kwargs,
        )

    # -------------------
    # Save and load optimizer configuration
    # -------------------
    def load_optimizer(self, file_path: Path = PathData.optimizer_file_path) -> bool:
        _dict = read_toml(file_path=file_path)

        self.variable_space = VariableSpace.model_validate(_dict['variable_space'])
        self.mp_params = MultiProcessingParameters.model_validate(_dict['mp_params'])
        # self.obj_func = _dict['obj_func']
        # self.intelligent_algorithm = _dict['intelligent_algorithm']
        # self.mixture_surrogate_model = _dict['mixture_surrogate_model']
        self.base_params_dict_list = _dict['base_params_dict_list']
        self.use_multiprocessing = _dict['use_multiprocessing']
        self.use_surrogate_model = _dict['use_surrogate_model']
        self.log_at_opt = _dict['log_at_opt']
        self.obj_name_list = _dict['obj_name_list']

        self._all_inputs = np.array(_dict['all_inputs'])
        self._all_outputs = np.array(_dict['all_outputs'])
        self._pareto_inputs = np.array(_dict['pareto_inputs'])
        self._pareto_outputs = np.array(_dict['pareto_outputs'])
        self.init_optimizer()

        log('Load optimization config is done', level='DEBUG')

    def save_optimizer(self, file_path: Path = PathData.optimizer_file_path):
        _dict = {
            'variable_space': self.variable_space.model_dump(mode='python'),
            'mp_params': self.mp_params.model_dump(mode='json')
            if isinstance(self.mp_params, MultiProcessingParameters)
            else self.mp_params,
            # 'obj_func': self.obj_func,
            # 'intelligent_algorithm': self.intelligent_algorithm,
            # 'mixture_surrogate_model': self.mixture_surrogate_model,
            'use_multiprocessing': self.use_multiprocessing,
            'use_surrogate_model': self.use_surrogate_model,
            'log_at_opt': self.log_at_opt,
            'obj_name_list': self.obj_name_list,
            'all_inputs': self._all_inputs.tolist(),
            'all_outputs': self._all_outputs.tolist(),
            'pareto_inputs': self._pareto_inputs.tolist(),
            'pareto_outputs': self._pareto_outputs.tolist(),
            'base_params_dict_list': self.base_params_dict_list,
        }
        # To avoid errors, place `base_params_dict_list` at the end

        save_toml(file_path, _dict)
        log('Save optimization config is done', level='DEBUG')
