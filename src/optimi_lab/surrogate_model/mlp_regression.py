from pydantic import model_validator
from sklearn.neural_network import MLPRegressor

from .surrogate_model_base import SurrogateModelBase


# MLP regression
class MlpRegression(SurrogateModelBase):
    model_type: str = 'mlp'

    activation: str = 'relu'
    solver: str = 'adam'
    mlpalpha: float = 0.0001
    batch_size: int | str = 'auto'
    learning_rate: str = 'constant'
    learning_rate_init: float = 0.001
    power_t: float = 0.5
    mlpmax_iter: int = 200
    shuffle: bool = True

    @model_validator(mode='after')
    def init_model(self):
        self._estimator = MLPRegressor(
            activation=self.activation,
            solver=self.solver,
            alpha=self.mlpalpha,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            power_t=self.power_t,
            max_iter=self.mlpmax_iter,
            shuffle=self.shuffle,
        )
