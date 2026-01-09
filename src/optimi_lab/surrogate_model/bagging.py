from pydantic import model_validator
from sklearn.ensemble import BaggingRegressor

from optimi_lab.surrogate_model.utils import float_int_1

from .surrogate_model_base import SurrogateModelBase


class Bagging(SurrogateModelBase):
    model_type: str = 'bag'

    base_estimator: object = None
    n_estimators: int = 10
    max_samples: float = 1
    max_features: float = 1
    bootstrap: bool = True
    bootstrap_features: bool = False
    oob_score: bool = False
    warm_start: bool = False

    @model_validator(mode='after')
    def init_model(self):
        max_samples = float_int_1(self.max_samples)
        max_features = float_int_1(min(len(self.var_name_list), self.max_features))
        self._estimator = BaggingRegressor(
            estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=self.bootstrap,
            bootstrap_features=self.bootstrap_features,
            oob_score=self.oob_score,
            warm_start=self.warm_start,
            n_jobs=-1,
            verbose=0,
            # kernel=kernel,
            # n_restarts_optimizer=1
        )
        return self
