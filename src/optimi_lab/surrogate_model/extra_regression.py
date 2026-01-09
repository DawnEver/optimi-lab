from pydantic import model_validator
from sklearn.tree import ExtraTreeRegressor

from optimi_lab.surrogate_model.utils import float_int_1, float_int_2

from .surrogate_model_base import SurrogateModelBase


class ExtraRegression(SurrogateModelBase):
    model_type: str = 'extra'
    criterion: str = 'squared_error'
    splitter: str = 'random'
    min_samples_split: float = 2
    min_samples_leaf: float = 2
    min_weight_fraction_leaf: float = 0
    max_features: float | None = 1
    min_impurity_decrease: float = 0
    max_leaf_nodes: int = 100
    ccp_alpha: float = 0

    @model_validator(mode='after')
    def init_model(self):
        min_samples_split = float_int_2(self.min_samples_split)
        min_samples_leaf = float_int_1(self.min_samples_leaf)
        max_features = float_int_1(self.max_features)
        self._estimator = ExtraTreeRegressor(
            criterion=self.criterion,
            splitter=self.splitter,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=max_features,
            min_impurity_decrease=self.min_impurity_decrease,
            max_leaf_nodes=self.max_leaf_nodes,
            ccp_alpha=self.ccp_alpha,
            # kernel=kernel,
            # n_restarts_optimizer=1
        )
