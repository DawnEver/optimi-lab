from pydantic import model_validator
from sklearn.neighbors import KNeighborsRegressor

from .surrogate_model_base import SurrogateModelBase


class KNearestNeighbors(SurrogateModelBase):
    model_type: str = 'knn'

    n_neighbors: int = 3
    weights: str = 'uniform'
    algorithm: str = 'auto'
    leaf_size: int = 30
    p: int = 2
    metric: str = 'minkowski'

    @model_validator(mode='after')
    def init_model(self):
        n_neighbors = min(len(self.var_name_list), self.n_neighbors)
        self._estimator = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric,
        )
