from pydantic import model_validator
from sklearn.gaussian_process import GaussianProcessRegressor

from .surrogate_model_base import SurrogateModelBase


# Kriging method
class Kriging(SurrogateModelBase):
    model_type: str = 'kri'

    kernel: object = None
    krialpha: float = 1e-10
    optimizer: str = 'fmin_l_bfgs_b'
    n_restarts_optimizer: int = 0
    normalize_y: bool = False
    copy_X_train: bool = True

    @model_validator(mode='after')
    def init_model(self):
        self._estimator = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.krialpha,
            optimizer=self.optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=self.normalize_y,
            copy_X_train=self.copy_X_train,
        )
