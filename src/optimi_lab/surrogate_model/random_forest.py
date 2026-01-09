from pydantic import model_validator
from sklearn.ensemble import RandomForestRegressor

from optimi_lab.surrogate_model.utils import float_int_1, float_int_2

from .surrogate_model_base import SurrogateModelBase


# Random forest regression
class RandomForest(SurrogateModelBase):
    model_type: str = 'forest'

    n_estimators: int = 100
    criterion: str = 'squared_error'
    min_samples_split: float = 2
    min_samples_leaf: float = 2
    min_weight_fraction_leaf: float = 0
    max_features: float = 1
    max_leaf_nodes: int = 100
    min_impurity_decrease: float = 0
    bootstrap: bool = True
    oob_score: bool = False

    @model_validator(mode='after')
    def init_model(self):
        min_samples_split = float_int_2(self.min_samples_split)
        min_samples_leaf = float_int_1(self.min_samples_leaf)
        max_features = float_int_1(self.max_features)

        self._estimator = RandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=-1,
        )
