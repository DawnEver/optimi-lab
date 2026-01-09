from pydantic import model_validator
from sklearn.tree import DecisionTreeRegressor

from optimi_lab.surrogate_model.utils import float_int_1, float_int_2

from .surrogate_model_base import SurrogateModelBase


class DecisionTreeRegression(SurrogateModelBase):
    model_type: str = 'dec'

    splitter: str = 'best'
    criterion: str = 'squared_error'
    min_samples_split: float = 2
    min_samples_leaf: float = 2
    min_weight_fraction_leaf: float = 0
    max_features: float = 1
    max_leaf_nodes: int = 100
    min_impurity_decrease: float = 0
    ccp_alpha: float = 0

    @model_validator(mode='after')
    def init_model(self):
        min_samples_split = float_int_2(self.min_samples_split)
        min_samples_leaf = float_int_1(self.min_samples_leaf)

        max_features = float_int_1(self.max_features)
        self._estimator = DecisionTreeRegressor(
            splitter=self.splitter,
            criterion=self.criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
        )
