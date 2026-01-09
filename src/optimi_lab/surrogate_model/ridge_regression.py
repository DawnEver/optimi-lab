from pydantic import model_validator
from sklearn.linear_model import Ridge

from .surrogate_model_base import SurrogateModelBase


# Ridge regression
class RidgeRegression(SurrogateModelBase):
    model_type: str = 'rid'

    ridalpha: float = 1
    fit_intercept: bool = True
    copy_X: bool = True
    max_iter: int = 1000
    ridtol: float = 0.0001
    solver: str = 'auto'
    positive: bool = False

    @model_validator(mode='after')
    def init_model(self):
        self._estimator = Ridge(
            alpha=self.ridalpha,
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            max_iter=self.max_iter,
            tol=self.ridtol,
            solver=self.solver,
            positive=self.positive,
        )
