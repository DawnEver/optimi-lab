from pydantic import model_validator
from sklearn.linear_model import Lasso

from .surrogate_model_base import SurrogateModelBase


class LassoRegression(SurrogateModelBase):
    model_type: str = 'las'

    lasalpha: float = 1
    fit_intercept: bool = True
    precompute: bool = False
    copy_X: bool = True
    lasmax_iter: int = 1000
    lastol: float = 0.0001
    warm_start: bool = False
    positive: bool = False
    selection: str = 'cyclic'

    @model_validator(mode='after')
    def init_model(self):
        self._estimator = Lasso(
            alpha=self.lasalpha,
            fit_intercept=self.fit_intercept,
            precompute=self.precompute,
            copy_X=self.copy_X,
            max_iter=self.lasmax_iter,
            tol=self.lastol,
            warm_start=self.warm_start,
            positive=self.positive,
            selection=self.selection,
        )
