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
        """Combine the pool's predictions, weighted by each model's cross-validated R².

        The weight is `r2` alone. A Dempster-Shafer combination (reciprocals of mad/mae/rmse,
        normalised into masses, then multiplied) used to sit in an `if 1:`-guarded `else`
        branch -- unreachable, and itself containing an `if 0:` with a `...` placeholder. A
        branch no input can reach is not a documented alternative; it is a decision nobody
        made, so it is deleted and the rule that runs is stated here and pinned by
        `test_mixture_weights_are_the_cross_validated_r2`.

        Raises:
            AttributeError: If the pool is empty -- there is nothing to weight.

        """
        n_model = len(self._surrogate_model_pool)
        if n_model == 0:
            msg = 'the surrogate model pool is empty, add surrogate models before predicting'
            log(msg=msg, level='ERROR')
            raise AttributeError(msg)
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
        weight_array = score_array[:, 0]  # Use r2 only as the weight
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
