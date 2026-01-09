from abc import ABC

import numpy as np
from pydantic import field_validator
from sklearn.base import clone as clone_estimator
from sklearn.model_selection import KFold

from optimi_lab.surrogate_model.utils import SCORE_METHODS, score_regression
from optimi_lab.utils.exceptions import ParameterException
from optimi_lab.utils.quantities import BaseModel_with_q

VALIDATE_METHODS = ['holdout', 'kfold']


class SurrogateModelBase(BaseModel_with_q, ABC):
    """Surrogate model base class
    Attributes:
        model_type (str): model name
        var_name_list (list[str]): list of variable names
        obj_name_list (list[str]): list of objective names

        do_calc_score (bool): whether to compute scores
        validate_method (str): validation method, 'holdout' or 'kfold'
        n_splits (int): number of splits; only used when `validate_method` is 'kfold'
        score_methods (list[str]): list of scoring methods; defaults to SCORE_METHODS

        _valid (bool): whether the model is valid
        _model: surrogate model instance
        _estimator: sklearn-style estimator
        _score_dict (dict): dictionary of scores containing results for each score method

    """

    model_type: str
    var_name_list: list[str]
    obj_name_list: list[str]

    do_calc_score: bool = True
    validate_method: str = 'kfold'
    n_splits: int = 5
    score_methods: list[str] = SCORE_METHODS

    _valid: bool = True
    _model = None
    _estimator = None
    _score_dict: dict = {}

    @field_validator('validate_method')
    def validate_validate_method(cls, v):
        if v not in VALIDATE_METHODS:
            msg = f'validate method {v} is not supported, should be in {VALIDATE_METHODS}'
            raise ParameterException(msg)
        return v

    @field_validator('n_splits')
    def validate_n_splits(cls, v):
        if v < 2:
            msg = f'n_splits should be at least 2, got {v}'
            raise ParameterException(msg)
        return v

    @field_validator('score_methods')
    def validate_score_methods(cls, v):
        for method in v:
            if method not in SCORE_METHODS:
                msg = f'score method {method} is not supported, should be in {SCORE_METHODS}'
                raise ParameterException(msg)
        return v

    def _pre_process_hook(self, x: np.ndarray, y: np.ndarray = None):
        """Pre-process hook for the input data."""
        return x, y

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x, _ = self._pre_process_hook(x)
        return self._predict(x)

    def _train(self, x: np.ndarray, y: np.ndarray, from_zero: bool = False) -> None:
        if from_zero:
            estimator = clone_estimator(self._estimator)
        else:
            estimator = self._estimator
        self._model = estimator.fit(x, y)

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        x, y = self._pre_process_hook(x, y)
        if self.do_calc_score:
            selection_model = KFold(n_splits=self.n_splits, shuffle=True)
            y_preds = None
            y_tests = None
            validate_method = self.validate_method
            for i_filter, (train_index, test_index) in enumerate(selection_model.split(x)):
                if validate_method == 'holdout':
                    # exit after just one round
                    if i_filter >= 1:
                        break
                elif validate_method == 'kfold':
                    pass
                else:
                    msg = f'validate method {validate_method} is not supported, should be in {VALIDATE_METHODS}'
                    raise KeyError(msg)

                x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]
                self._train(x_train, y_train, from_zero=True)  # retrain for different fold
                y_pred = self._predict(x_test)

                # Concatenate predictions and targets for scoring to save computation
                if y_preds is None:
                    y_preds = y_pred
                else:
                    y_preds = np.concatenate((y_preds, y_pred), axis=0)
                if y_tests is None:
                    y_tests = y_test
                else:
                    y_tests = np.concatenate((y_tests, y_test), axis=0)
            self._score_dict = score_regression(y_preds, y_tests, self.score_methods)
            x_res2train, y_res2train = x_test, y_test
        else:
            x_res2train, y_res2train = x, y
        # Train on full data for final prediction
        self._train(x_res2train, y_res2train, from_zero=False)

    def check_valid(self, var_name_list: list[str], obj_name_list: list[str]) -> bool:
        self._valid = (var_name_list == self.var_name_list) and (obj_name_list == self.obj_name_list)
        return self._valid

    def show(self, only_score: bool = True) -> dict:
        if only_score:
            return self._score_dict
        return {
            'model_type': self.model_type,
            'score': self._score_dict,
            'var_name_list': self.var_name_list,
            'obj_name_list': self.obj_name_list,
            'valid': self._valid,
        }
