from copy import deepcopy

import numpy as np
from pydantic import field_validator

from optimi_lab.utils.exceptions import ParameterException
from optimi_lab.utils.logger import log

from .bagging import Bagging
from .decision_tree_regression import DecisionTreeRegression
from .extra_regression import ExtraRegression
from .k_nearest_neighbors import KNearestNeighbors
from .kriging import Kriging
from .lasso_regression import LassoRegression
from .mlp_regression import MlpRegression
from .polynomial_regression import PolinomialRegression
from .random_forest import RandomForest
from .ridge_regression import RidgeRegression
from .surrogate_model_base import SurrogateModelBase

WEIGHT_METHODS = ['linear', 'max']

surrogate_model_pool_dict: dict = {
    'pol': PolinomialRegression,
    'knn': KNearestNeighbors,
    'kri': Kriging,
    'rid': RidgeRegression,
    'las': LassoRegression,
    'mlp': MlpRegression,
    'dec': DecisionTreeRegression,
    'extra': ExtraRegression,
    'forest': RandomForest,
    'bag': Bagging,
}
SurrogateModelType = (
    PolinomialRegression
    | KNearestNeighbors
    | Kriging
    | RidgeRegression
    | LassoRegression
    | MlpRegression
    | DecisionTreeRegression
    | ExtraRegression
    | RandomForest
    | Bagging
)

SURROGATE_MODEL_TYPES = list(surrogate_model_pool_dict.keys())


class MixtureSurrogateModel(SurrogateModelBase):
    model_type: str = 'mixture'

    weight_method: str = 'linear'

    _surrogate_model_pool: list[SurrogateModelType | dict] = []

    @field_validator('weight_method')
    def validate_weight_method(cls, v: str) -> str:
        if v not in WEIGHT_METHODS:
            msg = f'weight_method must be one of {WEIGHT_METHODS}, got {v}'
            raise ParameterException(msg)
        return v

    def build_surrogate_model_pool(self, surrogate_model_pool: list[dict]) -> list[SurrogateModelType]:
        for i_model, model_input in enumerate(surrogate_model_pool):
            if isinstance(model_input, dict):
                model_type = model_input.get('model_type', None)

                if model_type is None or model_type not in SURROGATE_MODEL_TYPES:
                    msg = f'surrogate_model_pool must contain valid surrogate model type, got {model_type}'
                    raise ParameterException(msg)

                model_input['var_name_list'] = self.var_name_list
                model_input['obj_name_list'] = self.obj_name_list
                model_input['do_calc_score'] = self.do_calc_score
                model_input['validate_method'] = self.validate_method
                model_input['n_splits'] = self.n_splits
                model_input['score_methods'] = self.score_methods

                surrogate_model_pool[i_model] = surrogate_model_pool_dict[model_type](**model_input)
            elif not issubclass(type(model_input), SurrogateModelBase):
                # Ensure the input is a subclass/instance of SurrogateModelBase
                msg = f'surrogate_model_pool must contain SurrogateModel instances or dicts, got {type(model_input)}'
                raise ParameterException(msg)
        self._surrogate_model_pool = surrogate_model_pool
        return surrogate_model_pool

    def _predict(self, x: np.ndarray) -> np.ndarray:
        """Based on Dempster-Shafer theory"""
        n_model = len(self._surrogate_model_pool)
        score_array = None
        y_pred_array = None
        for i_model in range(n_model):
            model = self._surrogate_model_pool[i_model]
            y_pred = model.predict(x)
            r2 = model._score_dict.get('r2', 1e-5)
            r2 = max(r2, 0)
            mad = model._score_dict.get('mad', 1e-5)
            mae = model._score_dict.get('mae', 1e-5)
            rmse = model._score_dict.get('rmse', 1e-5)
            scores = [r2, mad, mae, rmse]
            if score_array is None or y_pred_array is None:
                y_pred_array = np.zeros((n_model, *y_pred.shape))
                score_array = np.zeros((n_model, len(scores)))
            y_pred_array[i_model] = y_pred
            score_array[i_model] = scores
        if 1:
            weight_array = score_array[:, 0]  # Use r2 only as the weight
        else:
            score_array[:, 1:] = 1 / score_array[:, 1:]  # use reciprocals for mad, mae, rmse
            mass_array = score_array / np.sum(score_array, axis=0)[np.newaxis, :]

            mass_array[np.isnan(mass_array)] = 0
            if 0:
                # BetP(θ_i) = ∑ θ_i∈B m(B) |A ∩ B|/|B| , ∀B ⊆ Θ
                ...
            else:
                # Combine probabilities
                weight_array = np.prod(mass_array, axis=1)
        if np.all(weight_array == 0):
            msg = 'all surrogate models weights are invalid, please check the surrogate models'
            log(msg=msg, level='DEBUG')
            weight_array = np.ones_like(weight_array)

        weight_array = weight_array / np.sum(weight_array)
        weight_array[np.isnan(weight_array)] = 0
        weight_array = weight_array[:, np.newaxis, np.newaxis]
        if self.weight_method == 'linear':
            y = np.sum(y_pred_array * weight_array, axis=0)
        else:
            y = y_pred_array[np.argmax(weight_array)]

        return y

    def _train(self, x: np.ndarray, y: np.ndarray, from_zero: bool = False) -> None:
        if from_zero:
            surrogate_model_pool = deepcopy(self._surrogate_model_pool)
        else:
            surrogate_model_pool = self._surrogate_model_pool
        for model in surrogate_model_pool:
            model.train(x, y)
