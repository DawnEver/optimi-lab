"""Comprehensive tests for all surrogate model modules."""

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression

from optimi_lab.surrogate_model.bagging import Bagging
from optimi_lab.surrogate_model.decision_tree_regression import DecisionTreeRegression
from optimi_lab.surrogate_model.extra_regression import ExtraRegression
from optimi_lab.surrogate_model.k_nearest_neighbors import KNearestNeighbors
from optimi_lab.surrogate_model.kriging import Kriging
from optimi_lab.surrogate_model.lasso_regression import LassoRegression
from optimi_lab.surrogate_model.mixture_surrogate_model import (
    SURROGATE_MODEL_TYPES,
    WEIGHT_METHODS,
    MixtureSurrogateModel,
    surrogate_model_pool_dict,
)
from optimi_lab.surrogate_model.mlp_regression import MlpRegression
from optimi_lab.surrogate_model.polynomial_regression import PolinomialRegression
from optimi_lab.surrogate_model.random_forest import RandomForest
from optimi_lab.surrogate_model.ridge_regression import RidgeRegression
from optimi_lab.surrogate_model.surrogate_model_base import VALIDATE_METHODS, SurrogateModelBase
from optimi_lab.surrogate_model.utils import (
    SCORE_METHODS,
    float_int_1,
    float_int_2,
    score_regression,
)
from optimi_lab.utils.exceptions import ParameterException


# Test Data Fixtures
@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(50, 3) * 10
    y = (X[:, 0] ** 2 + X[:, 1] * X[:, 2] + np.random.normal(0, 0.1, 50)).reshape(-1, 1)
    return X, y


@pytest.fixture
def small_data():
    return np.array([[1, 2, 3], [4, 5, 6]]), np.array([[10], [20]])


@pytest.fixture(params=VALIDATE_METHODS)
def model_params(request):
    validate_method = request.param
    return {
        'var_name_list': ['x1', 'x2', 'x3'],
        'obj_name_list': ['y1'],
        'do_calc_score': False,  # Disable for faster testing
        'validate_method': validate_method,
        'n_splits': 3,
    }


# Mock Model for Base Class Testing
class MockSurrogateModel(SurrogateModelBase):
    model_type: str = 'mock'

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._estimator = LinearRegression()


class TestSurrogateModelBase:
    """Test SurrogateModelBase functionality."""

    def test_validators(self):
        """Test field validators."""
        with pytest.raises(ParameterException):
            MockSurrogateModel(var_name_list=['x'], obj_name_list=['y'], validate_method='invalid')
        with pytest.raises(ParameterException):
            MockSurrogateModel(var_name_list=['x'], obj_name_list=['y'], n_splits=1)
        with pytest.raises(ParameterException):
            MockSurrogateModel(var_name_list=['x'], obj_name_list=['y'], score_methods=['invalid'])

    def test_train_predict(self, model_params, sample_data):
        """Test training and prediction."""
        model = MockSurrogateModel(**model_params)
        X, y = sample_data
        model.train(X, y)
        predictions = model.predict(X[:5])
        assert predictions.shape == (5, 1)

    def test_check_valid(self, model_params):
        """Test validity checking."""
        model = MockSurrogateModel(**model_params)
        assert model.check_valid(['x1', 'x2', 'x3'], ['y1']) is True
        assert model.check_valid(['different'], ['y1']) is False

    def test_show_with_scoring(self, model_params, sample_data):
        """Test show method with scoring enabled."""
        params = model_params.copy()
        params['do_calc_score'] = True
        model = MockSurrogateModel(**params)
        X, y = sample_data
        model.train(X, y)

        score_only = model.show(only_score=True)
        full_info = model.show(only_score=False)

        assert isinstance(score_only, dict)
        assert 'model_type' in full_info
        assert full_info['model_type'] == 'mock'

    def test_invalid_validate_method_during_training(self, model_params, sample_data):
        """Test error handling for invalid validate method during training."""
        model = MockSurrogateModel(**model_params)
        # Manually set an invalid validate method to bypass validator
        model.validate_method = 'invalid_method'
        model.do_calc_score = True

        X, y = sample_data
        with pytest.raises(KeyError, match='validate method invalid_method is not supported'):
            model.train(X, y)


class TestUtils:
    """Test utility functions."""

    def test_score_regression(self):
        """Test regression scoring."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])

        scores = score_regression(y_true, y_pred)
        assert all(method in scores for method in SCORE_METHODS)

        # Test perfect prediction
        perfect_scores = score_regression(y_true, y_true)
        assert perfect_scores['r2'] == 1.0
        assert perfect_scores['mse'] == 0.0

        # Test invalid method
        with pytest.raises(KeyError):
            score_regression(y_true, y_pred, ['invalid'])

    def test_float_validators(self):
        """Test float/int validators."""
        # float_int_1 tests
        assert float_int_1(0.5) == 0.5
        assert isinstance(float_int_1(0.5), float)
        assert float_int_1(2.0) == 2
        assert isinstance(float_int_1(2.0), int)

        with pytest.raises(ParameterException):
            float_int_1(0.0)
        with pytest.raises(ParameterException):
            float_int_1(-1.0)

        # float_int_2 tests
        assert float_int_2(0.5) == 0.5
        assert float_int_2(2.0) == 2

        with pytest.raises(ParameterException):
            float_int_2(0.0)
        with pytest.raises(ParameterException):
            float_int_2(1.5)


class TestAllSurrogateModels:
    """Test all surrogate model implementations."""

    @pytest.mark.parametrize(
        ('model_class', 'extra_params'),
        [
            (PolinomialRegression, {}),
            (KNearestNeighbors, {}),
            (Kriging, {}),
            (RidgeRegression, {}),
            (LassoRegression, {}),
            (MlpRegression, {'mlpmax_iter': 100}),  # Reduce iterations for speed
            (DecisionTreeRegression, {}),
            (RandomForest, {'n_estimators': 5}),  # Reduce estimators for speed
            (Bagging, {'n_estimators': 5}),
            (ExtraRegression, {}),  # ExtraRegression uses single tree, no n_estimators
        ],
    )
    def test_model_basic_functionality(self, model_class, extra_params, model_params, sample_data):
        """Test basic functionality for all models."""
        params = {**model_params, **extra_params}
        model = model_class(**params)
        X, y = sample_data

        # Test training and prediction
        model.train(X, y)
        predictions = model.predict(X[:5])

        # Handle both 1D and 2D prediction outputs
        if predictions.ndim == 1:
            assert predictions.shape == (5,)
        else:
            assert predictions.shape == (5, 1)
        assert not np.any(np.isnan(predictions))
        assert model._model is not None

    def test_polynomial_regression_specific(self, model_params, sample_data):
        """Test PolinomialRegression specific features."""
        model = PolinomialRegression(polinomial_order=2, **model_params)
        X, y = sample_data

        # Test feature expansion
        processed_X, _ = model._pre_process_hook(X, y)
        assert processed_X.shape[1] > X.shape[1]  # More features after expansion

        model.train(X, y)
        predictions = model.predict(X[:5])
        assert predictions.shape == (5, 1)

    def test_knn_specific(self, model_params, sample_data):
        """Test KNearestNeighbors specific features."""
        model = KNearestNeighbors(n_neighbors=5, weights='distance', **model_params)
        X, y = sample_data

        # Check n_neighbors adjustment
        assert model._estimator.n_neighbors == min(5, len(model.var_name_list))

        model.train(X, y)
        predictions = model.predict(X[:5])
        assert predictions.shape == (5, 1)

    def test_kriging_specific(self, model_params, small_data):
        """Test Kriging specific features."""
        model = Kriging(kernel=RBF(1.0), normalize_y=True, **model_params)
        X, y = small_data

        model.train(X, y)
        predictions = model.predict(X)
        # Handle both 1D and 2D outputs
        if predictions.ndim == 1:
            assert predictions.shape == (y.shape[0],)
        else:
            assert predictions.shape == y.shape

    def test_regression_models_specific(self, model_params, sample_data):
        """Test Ridge and Lasso regression specific features."""
        X, y = sample_data

        # Ridge regression
        ridge = RidgeRegression(ridalpha=0.5, solver='svd', **model_params)
        ridge.train(X, y)
        ridge_pred = ridge.predict(X[:5])
        if ridge_pred.ndim == 1:
            assert ridge_pred.shape == (5,)
        else:
            assert ridge_pred.shape == (5, 1)

        # Lasso regression
        lasso = LassoRegression(lasalpha=0.1, selection='random', **model_params)
        lasso.train(X, y)
        lasso_pred = lasso.predict(X[:5])
        if lasso_pred.ndim == 1:
            assert lasso_pred.shape == (5,)
        else:
            assert lasso_pred.shape == (5, 1)

    def test_mlp_specific(self, model_params, sample_data):
        """Test MlpRegression specific features."""
        model = MlpRegression(activation='tanh', solver='lbfgs', mlpmax_iter=1000, **model_params)
        X, y = sample_data

        model.train(X, y)
        predictions = model.predict(X[:5])
        if predictions.ndim == 1:
            assert predictions.shape == (5,)
        else:
            assert predictions.shape == (5, 1)

    def test_tree_models_specific(self, model_params, sample_data):
        """Test tree-based models specific features."""
        X, y = sample_data

        # Decision Tree
        dt = DecisionTreeRegression(criterion='absolute_error', max_leaf_nodes=50, **model_params)
        dt.train(X, y)
        dt_pred = dt.predict(X[:5])
        if dt_pred.ndim == 1:
            assert dt_pred.shape == (5,)
        else:
            assert dt_pred.shape == (5, 1)

        # Random Forest
        rf = RandomForest(n_estimators=5, bootstrap=True, **model_params)
        rf.train(X, y)
        rf_pred = rf.predict(X[:5])
        if rf_pred.ndim == 1:
            assert rf_pred.shape == (5,)
        else:
            assert rf_pred.shape == (5, 1)

        # Extra Trees
        et = ExtraRegression(criterion='absolute_error', **model_params)
        et.train(X, y)
        et_pred = et.predict(X[:5])
        if et_pred.ndim == 1:
            assert et_pred.shape == (5,)
        else:
            assert et_pred.shape == (5, 1)

    def test_bagging_specific(self, model_params, sample_data):
        """Test Bagging specific features."""
        model = Bagging(n_estimators=5, bootstrap=True, oob_score=False, **model_params)
        X, y = sample_data

        model.train(X, y)
        predictions = model.predict(X[:5])
        if predictions.ndim == 1:
            assert predictions.shape == (5,)
        else:
            assert predictions.shape == (5, 1)


class TestMixtureSurrogateModel:
    """Test MixtureSurrogateModel."""

    def test_init_and_validation(self, model_params):
        """Test initialization and validation."""
        model = MixtureSurrogateModel(**model_params)
        assert model.model_type == 'mixture'
        assert model.weight_method == 'linear'

        # Test invalid weight method
        with pytest.raises(ParameterException):
            MixtureSurrogateModel(weight_method='invalid', **model_params)

    def test_constants(self):
        """Test module constants."""
        assert 'linear' in WEIGHT_METHODS
        assert 'max' in WEIGHT_METHODS
        assert len(SURROGATE_MODEL_TYPES) == len(surrogate_model_pool_dict)
        assert 'pol' in SURROGATE_MODEL_TYPES

    def test_build_surrogate_model_pool(self, model_params, sample_data):
        """Test building surrogate model pool."""
        model = MixtureSurrogateModel(**model_params)
        X, y = sample_data

        # Test with dict input
        pool_config = [{'model_type': 'pol', 'polinomial_order': 2}, {'model_type': 'knn', 'n_neighbors': 3}]

        pool = model.build_surrogate_model_pool(pool_config)
        assert len(pool) == 2
        assert all(isinstance(m, SurrogateModelBase) for m in pool)

        # Test training and prediction
        model.train(X, y)
        predictions = model.predict(X[:5])
        assert predictions.shape == (5, 1)

    def test_prediction_methods(self, model_params, sample_data):
        """Test prediction with different weight methods."""
        X, y = sample_data

        pool_config = [{'model_type': 'pol'}, {'model_type': 'knn'}]

        for weight_method in ['linear', 'max']:
            model = MixtureSurrogateModel(weight_method=weight_method, **model_params)
            model.build_surrogate_model_pool(pool_config)
            model.train(X, y)
            predictions = model.predict(X[:5])
            assert predictions.shape == (5, 1)

        # Test if np.all(weight_array == 0):
        for sub_model in model._surrogate_model_pool:
            sub_model._score_dict = {'r2': 0.0, 'mse': 0.0}  # Set scores to zero
        predictions = model.predict(X[:5])

    def test_invalid_model_pool(self, model_params):
        """Test invalid model pool configurations."""
        model = MixtureSurrogateModel(**model_params)

        # Test invalid model type
        with pytest.raises(ParameterException):
            model.build_surrogate_model_pool([{'model_type': 'invalid'}])

        # Test missing model type
        with pytest.raises(ParameterException):
            model.build_surrogate_model_pool([{}])

        # Test invalid object type (not dict or SurrogateModelBase instance)
        with pytest.raises(ParameterException):
            model.build_surrogate_model_pool(['invalid_string'])

    def test_mixture_with_from_zero_training(self, model_params, sample_data):
        """Test mixture model training with from_zero parameter."""
        X, y = sample_data
        model = MixtureSurrogateModel(**model_params)

        pool_config = [{'model_type': 'pol'}]
        model.build_surrogate_model_pool(pool_config)

        # First train normally to set up the models
        model.train(X, y)

        # Test from_zero=True path (this tests the deepcopy branch)
        model._train(X, y, from_zero=True)
        predictions = model.predict(X[:5])
        if predictions.ndim == 1:
            assert predictions.shape == (5,)
        else:
            assert predictions.shape == (5, 1)

        # Test from_zero=False path
        model._train(X, y, from_zero=False)
        predictions = model.predict(X[:5])
        if predictions.ndim == 1:
            assert predictions.shape == (5,)
        else:
            assert predictions.shape == (5, 1)

    def test_edge_cases(self, model_params, sample_data):
        """Test edge cases for mixture model."""
        X, y = sample_data
        model = MixtureSurrogateModel(**model_params)

        # Test with single model
        pool_config = [{'model_type': 'pol'}]
        model.build_surrogate_model_pool(pool_config)
        model.train(X, y)
        predictions = model.predict(X[:5])
        assert predictions.shape == (5, 1)


def test_comprehensive_coverage():
    """Test that all modules are importable and have expected attributes."""
    # Test all model classes have required attributes
    model_classes = [
        PolinomialRegression,
        KNearestNeighbors,
        Kriging,
        RidgeRegression,
        LassoRegression,
        MlpRegression,
        DecisionTreeRegression,
        RandomForest,
        Bagging,
        ExtraRegression,
        MixtureSurrogateModel,
    ]

    # Create dummy parameters for testing
    test_params = {'var_name_list': ['x1'], 'obj_name_list': ['y1'], 'do_calc_score': False}

    for model_class in model_classes:
        # Test that we can create an instance and it has model_type
        instance = model_class(**test_params)
        assert hasattr(instance, 'model_type')
        assert isinstance(instance.model_type, str)

    # Test constants
    assert isinstance(SCORE_METHODS, list)
    assert len(SCORE_METHODS) > 0
    assert isinstance(WEIGHT_METHODS, list)
    assert isinstance(SURROGATE_MODEL_TYPES, list)
    assert isinstance(surrogate_model_pool_dict, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
