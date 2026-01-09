"""Core unit tests for the `Optimizer` class."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from optimi_lab.intelligent_algorithm.mode_algorithm import MODE_Algorithm
from optimi_lab.optimizer import Optimizer
from optimi_lab.surrogate_model.mixture_surrogate_model import MixtureSurrogateModel
from optimi_lab.utils.variable_space import VariableSpace


class TestOptimizer:
    """Tests for the `Optimizer` class."""

    @pytest.fixture
    def variable_space(self) -> VariableSpace:
        """Create a variable space fixture."""
        return VariableSpace(
            var_name_list=['x1', 'x2'],
            lower_bounds=[0.0, 0.0],
            upper_bounds=[1.0, 1.0],
            sample_type='latin_hypercube',
            n_count=10,
        )

    @pytest.fixture
    def simple_obj_func(self) -> Callable:
        """Simple objective function: f1 = x1^2 + x2^2, f2 = (x1-1)^2 + (x2-1)^2, f3 = x1 + x2"""

        def obj_func(x: np.ndarray) -> np.ndarray:
            if x.ndim == 1:
                x = x.reshape(1, -1)
            f1 = x[:, 0] ** 2 + x[:, 1] ** 2
            f2 = (x[:, 0] - 1) ** 2 + (x[:, 1] - 1) ** 2
            f3 = x[:, 0] + x[:, 1]  # Uncomment to add more objective functions if needed
            return np.column_stack([f1, f2, f3])

        return obj_func

    @pytest.fixture(
        params=[
            {'use_multiprocessing': False, 'log_at_opt': False},
            {'use_multiprocessing': True, 'log_at_opt': True},
        ]
    )
    def simple_optimizer(
        self, variable_space: VariableSpace, simple_obj_func: Callable, request: pytest.FixtureRequest
    ) -> Optimizer:
        """Create optimizer instances with multiple configurations."""
        cfg = request.param
        return Optimizer(
            variable_space=variable_space,
            obj_name_list=['f1@oc0@max', 'f2@oc0@min', 'f3@oc3@max'],
            base_params_dict_list=[{}],
            obj_func=simple_obj_func,
            use_multiprocessing=cfg['use_multiprocessing'],
            use_surrogate_model=False,
            log_at_opt=cfg['log_at_opt'],
        )

    def test_optimizer_initialization(self, simple_optimizer: Optimizer):
        """Test optimizer initialization."""
        assert simple_optimizer._n_var == 2
        assert simple_optimizer._n_obj == 3
        assert simple_optimizer.obj_name_list == ['f1@oc0@max', 'f2@oc0@min', 'f3@oc3@max']
        assert simple_optimizer._all_inputs.shape == (0, 2)
        assert simple_optimizer._all_outputs.shape == (0, 3)

        assert np.allclose(simple_optimizer._max_obj_flags, [True, False, True])
        assert simple_optimizer._obj_name_matrix == [['f1', 'f2'], ['f3']]
        assert simple_optimizer._err_value_matrix == [[-np.inf, np.inf], [-np.inf]]

    def test_num_oc(self, variable_space: VariableSpace, simple_obj_func: Callable) -> Optimizer:
        """Test different `num_oc` configurations."""
        Optimizer(
            variable_space=variable_space,
            obj_name_list=['f1@oc0@min', 'f2@oc0@max'],
            base_params_dict_list=[{}, {}],
            obj_func=simple_obj_func,
            use_multiprocessing=True,
            use_surrogate_model=False,
            log_at_opt=True,
        )
        Optimizer(
            variable_space=variable_space,
            obj_name_list=['f1@oc0@min', 'f2@oc1@max'],
            base_params_dict_list=[{}],
            obj_func=simple_obj_func,
            use_multiprocessing=True,
            use_surrogate_model=False,
            log_at_opt=True,
        )

    def test_obj_func_normalized(self, simple_optimizer: Optimizer):
        """Test normalized objective function."""
        # Test inputs
        test_vars = np.array([[0.5, 0.5], [0.0, 1.0]])

        # Call normalized objective function
        results = simple_optimizer._obj_func_normalized(test_vars)
        simple_optimizer.use_surrogate_model = True
        with pytest.raises(
            AttributeError,
            match='Surrogate model function is not initialized. Train or load a surrogate model first',
        ):
            results = simple_optimizer._obj_func_normalized(test_vars)

        # Check output shape
        assert results.shape == (2, 3)

    def test_sample_sweep(self, simple_optimizer: Optimizer):
        """Test sample sweep functionality."""
        var_matrix, obj_matrix = simple_optimizer.sample_sweep()

        # Check output shapes
        assert var_matrix.shape == (10, 2)
        assert obj_matrix.shape == (10, 3)

        # Check variable ranges
        assert np.all(var_matrix >= 0.0)
        assert np.all(var_matrix <= 1.0)

        # Ensure objective values are non-negative (by definition)
        assert np.all(obj_matrix >= 0.0)

        # Check data saved internally
        assert simple_optimizer._all_inputs.shape == (10, 2)
        assert simple_optimizer._all_outputs.shape == (10, 3)

    def test_optimize(self, simple_optimizer: Optimizer):
        """Test optimization flow."""
        with pytest.raises(AttributeError, match='Intelligent algorithm is not initialized'):
            simple_optimizer.optimize()

        intelligent_algorithm = MODE_Algorithm(
            object_function=simple_optimizer.obj_func,
            variable_space=simple_optimizer.variable_space,
            n_obj=simple_optimizer._n_obj,
            pop_size=10,
            max_iter=10,
        )
        simple_optimizer.intelligent_algorithm = intelligent_algorithm
        all_inputs, all_outputs, pareto_inputs, pareto_outputs = simple_optimizer.optimize()

        # Check that data has been added to _all_inputs and _all_outputs
        assert all_inputs.shape[0] > 0
        assert all_outputs.shape[0] > 0

        # Check that Pareto front was computed
        assert pareto_inputs.shape[0] > 0
        assert pareto_outputs.shape[0] > 0

        simple_optimizer.plot(data_frame=simple_optimizer.get_dataframe(category='all'))
        simple_optimizer.plot()
        simple_optimizer.plot_moo(
            df_all=simple_optimizer.get_dataframe(category='all'),
            df_pareto=simple_optimizer.get_dataframe(category='pareto'),
        )
        simple_optimizer.plot_moo()

    def test_surrogate_model_training(self, simple_optimizer: Optimizer, tmp_path: Path):
        """Test surrogate model training and persistence."""
        with pytest.raises(AttributeError, match='Surrogate model function is not initialized'):
            simple_optimizer.train_surrogate_models()
        with pytest.raises(AttributeError, match='Surrogate model function is not initialized'):
            simple_optimizer.save_surrogate_models()

        surrogate_model = MixtureSurrogateModel(
            model_type='mixture',
            var_name_list=simple_optimizer.variable_space.var_name_list,
            obj_name_list=simple_optimizer.obj_name_list,
        )
        surrogate_model.build_surrogate_model_pool(
            surrogate_model_pool=[
                {
                    'model_type': 'pol',
                    'polinomial_order': 3,
                },
                {
                    'model_type': 'knn',
                    'n_neighbors': 3,
                },
            ]
        )
        simple_optimizer.mixture_surrogate_model = surrogate_model

        # Generate training data
        simple_optimizer.sample_sweep()

        # Train surrogate models
        simple_optimizer.train_surrogate_models()
        simple_optimizer.train_surrogate_models(
            inputs=simple_optimizer._all_inputs, outputs=simple_optimizer._all_outputs
        )

        # Check that models were trained successfully
        for model in simple_optimizer.mixture_surrogate_model._surrogate_model_pool:
            assert hasattr(model, '_model')
            assert model._model is not None

        simple_optimizer.save_surrogate_models(tmp_path / 'test_surrogate_models.pkl')
        # Check that the saved model file exists
        assert (tmp_path / 'test_surrogate_models.pkl').exists()
        simple_optimizer.mixture_surrogate_model = None
        simple_optimizer.load_surrogate_models(tmp_path / 'test_surrogate_models.pkl')
        # Check that the loaded models are valid
        for model in simple_optimizer.mixture_surrogate_model._surrogate_model_pool:
            assert hasattr(model, '_model')
            assert model._model is not None
        simple_optimizer.show_surrogate_models()
        simple_optimizer.plot_contour(data_frame=simple_optimizer.get_dataframe(category='all'))
        simple_optimizer.plot_contour()

    def test_surrogate_model_prediction(self, simple_optimizer: Optimizer):
        """Test surrogate model prediction"""
        # Set up surrogate model
        surrogate_model = MixtureSurrogateModel(
            model_type='mixture',
            var_name_list=simple_optimizer.variable_space.var_name_list,
            obj_name_list=simple_optimizer.obj_name_list,
        )
        surrogate_model.build_surrogate_model_pool(
            surrogate_model_pool=[
                {
                    'model_type': 'knn',
                    'n_neighbors': 3,
                }
            ]
        )
        simple_optimizer.mixture_surrogate_model = surrogate_model

        # Generate training data and train
        simple_optimizer.sample_sweep()
        simple_optimizer.train_surrogate_models()

        # Enable surrogate model prediction
        simple_optimizer.use_surrogate_model = True

        # Test predictions
        test_vars = np.array([[0.3, 0.7], [0.8, 0.2]])
        predictions = simple_optimizer._obj_func_normalized(test_vars)

        # Check prediction results
        assert predictions.shape == (2, 3)
        assert not np.any(np.isnan(predictions))

    def test_data_management(self, simple_optimizer: Optimizer):
        """Test data management functionality"""
        # Initial data is empty
        assert simple_optimizer._all_inputs.shape[0] == 0

        # Add data
        simple_optimizer.sample_sweep()
        initial_size = simple_optimizer._all_inputs.shape[0]
        assert initial_size > 0

        # Resample; data should accumulate
        simple_optimizer.sample_sweep()
        assert simple_optimizer._all_inputs.shape[0] == 2 * initial_size

        # Test data filtering
        # Create mask to keep the first half of data
        mask = np.arange(simple_optimizer._all_inputs.shape[0]) < initial_size
        simple_optimizer.filter_data(mask)
        assert simple_optimizer._all_inputs.shape[0] == initial_size

    def test_recalc_pareto(self, simple_optimizer: Optimizer):
        """Test recalculating Pareto front"""
        # First perform sampling
        simple_optimizer.sample_sweep()

        # Recalculate Pareto front
        pareto_inputs, pareto_outputs = simple_optimizer.recalc_pareto(update_self=False)
        pareto_inputs, pareto_outputs = simple_optimizer.recalc_pareto()

        # Check Pareto front
        assert len(pareto_inputs) > 0
        assert len(pareto_outputs) > 0
        assert pareto_inputs.shape[1] == 2
        assert pareto_outputs.shape[1] == 3

        # Check internal state updated
        np.testing.assert_array_equal(simple_optimizer._pareto_inputs, pareto_inputs)
        np.testing.assert_array_equal(simple_optimizer._pareto_outputs, pareto_outputs)

    @pytest.mark.parametrize(('drop_inf', 'do_recalc_pareto'), [(True, True), (True, False), (False, False)])
    def test_get_dataframe(self, simple_optimizer: Optimizer, drop_inf: bool, do_recalc_pareto: bool):
        """Test getting DataFrame functionality"""
        # First perform sampling
        simple_optimizer.sample_sweep()
        simple_optimizer.recalc_pareto()

        # Get all data
        df_all = simple_optimizer.get_dataframe(category='all', drop_inf=drop_inf, do_recalc_pareto=do_recalc_pareto)
        assert len(df_all) > 0
        assert len(df_all.columns) == 5  # 2 variables + 3 objectives

        # Get Pareto data
        df_pareto = simple_optimizer.get_dataframe(
            category='pareto', drop_inf=drop_inf, do_recalc_pareto=do_recalc_pareto
        )
        assert len(df_pareto) > 0
        assert len(df_pareto.columns) == 5
        assert len(df_pareto) <= len(df_all)

        # Test invalid category
        with pytest.raises(NameError):
            simple_optimizer.get_dataframe(category='invalid')

        simple_optimizer._n_var = 0
        df_all = simple_optimizer.get_dataframe(category='all')
        assert len(df_all) == 0

        df_pareto = simple_optimizer.get_dataframe(category='pareto')
        assert len(df_pareto) == 0

    def test_optimizer_save_load(self, simple_optimizer: Optimizer, tmp_path: Path):
        """Test optimizer save/load functionality"""
        # Use pytest temporary path
        optimizer_path = tmp_path / 'test_optimizer_save.toml'
        simple_optimizer.save_optimizer(str(optimizer_path))

        # Create new optimizer instance and load saved state
        new_optimizer = Optimizer(
            variable_space=simple_optimizer.variable_space,
            obj_name_list=simple_optimizer.obj_name_list,
            base_params_dict_list=simple_optimizer.base_params_dict_list,
            obj_func=simple_optimizer.obj_func,
            use_multiprocessing=simple_optimizer.use_multiprocessing,
            log_at_opt=simple_optimizer.log_at_opt,
        )
        new_optimizer.load_optimizer(str(optimizer_path))

        # Check loaded state matches original
        assert np.array_equal(new_optimizer._all_inputs, simple_optimizer._all_inputs)
        assert np.array_equal(new_optimizer._all_outputs, simple_optimizer._all_outputs)


if __name__ == '__main__':
    pytest.main([__file__])
