import numpy as np
from pydantic import model_validator
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from .surrogate_model_base import SurrogateModelBase


class PolinomialRegression(SurrogateModelBase):
    model_type: str = 'pol'

    polinomial_order: int = 3
    _pf: PolynomialFeatures = None

    @model_validator(mode='after')
    def init_model(self):
        self._pf = PolynomialFeatures(degree=self.polinomial_order, include_bias=False)
        # include_bias: Defaults to True. If True, a constant term of 1 is included.
        # interaction_only: Defaults to False. If True, only interaction features are produced (no self-combinations).
        self._estimator = LinearRegression()

    def _pre_process_hook(self, x: np.ndarray, y: np.ndarray = None):
        x_expand = self._pf.fit_transform(x)
        return x_expand, y
