"""Tests for the surrogate models: one fit per registered key, and the mixture over them."""

import re
from pathlib import Path

import numpy as np
import pytest

import optimi_lab.surrogate_model
from optimi_lab.core import Refusal
from optimi_lab.surrogate_model import (
    REGISTRY,
    SCORE_METHODS,
    SURROGATE_KEYS,
    WEIGHT_METHODS,
    MixtureSurrogate,
    SurrogateModel,
    fraction_or_count,
    score_regression,
)


@pytest.fixture(scope='module')
def smooth_data():
    """A smooth two-column target on a 2-D grid that every registered estimator can be fitted to."""
    axis = np.linspace(0.0, 1.0, 5)
    x = np.array([[a, b] for a in axis for b in axis])
    y = np.column_stack([np.sin(2 * np.pi * x[:, 0]) + x[:, 1] ** 2, np.cos(2 * np.pi * x[:, 1])])
    return x, y


def test_the_registry_declares_the_keys_a_pool_entry_may_name():
    assert tuple(REGISTRY) == SURROGATE_KEYS
    assert set(SURROGATE_KEYS) == {'bag', 'dec', 'extra', 'forest', 'knn', 'kri', 'las', 'mlp', 'pol', 'rid'}


@pytest.mark.parametrize('key', SURROGATE_KEYS)
def test_every_registered_key_fits_and_predicts_the_one_shape(key, smooth_data):
    """``predict`` returns ``(n_points, n_obj)`` for every key, never a flat vector.

    ``PolinomialRegression`` and ``KNearestNeighbors`` returned ``(n, 1)`` where ``RidgeRegression``
    returned ``(n,)``, so the mixture could not stack them and one broadcast error killed a search.
    """
    x, y = smooth_data
    prediction = SurrogateModel(key, n_splits=3).fit(x, y).predict(x[:5])
    assert prediction.shape == (5, y.shape[1])
    assert np.all(np.isfinite(prediction))


def test_an_unregistered_key_is_refused_naming_every_registered_key():
    with pytest.raises(Refusal) as refusal:
        SurrogateModel('polynomial')
    for key in SURROGATE_KEYS:
        assert f"'{key}'" in str(refusal.value)


def test_no_module_in_the_package_imports_pydantic():
    """The regressors are data; the ten validators that disagreed about returning ``self`` are gone.

    An IMPORT scan, not a text scan: the modules' own prose may name what they replaced.
    """
    sources = [
        path.read_text(encoding='utf-8') for path in Path(optimi_lab.surrogate_model.__file__).parent.glob('*.py')
    ]
    assert sources  # a scan that finds nothing is vacuous
    assert not [source for source in sources if re.search(r'^\s*(import|from)\s+pydantic', source, re.MULTILINE)]


def test_the_declared_parameter_is_the_one_the_estimator_gets(smooth_data):
    """The field and the estimator are one value: no name list decides what the estimator receives."""
    x, y = smooth_data
    assert SurrogateModel('knn', n_neighbors=3, n_splits=3).fit(x, y).estimator.n_neighbors == 3
    assert SurrogateModel('bag', max_samples=0.5, n_splits=3).fit(x, y).estimator.max_samples == 0.5
    assert SurrogateModel('dec', min_samples_leaf=3, n_splits=3).fit(x, y).estimator.min_samples_leaf == 3


def test_a_count_and_fraction_parameter_has_one_reading():
    assert fraction_or_count(0.5, 2, 'the split') == 0.5
    assert isinstance(fraction_or_count(0.5, 2, 'the split'), float)
    assert fraction_or_count(1.0, 1, 'the sample') == 1.0  # a float at 1.0 is the whole set, a fraction
    assert fraction_or_count(2, 2, 'the split') == 2
    assert fraction_or_count('sqrt', 1, 'the features') == 'sqrt'  # scikit-learn's own non-numeric readings


@pytest.mark.parametrize(('value', 'floor'), [(1.5, 2), (1, 2), (0.0, 1), (-1, 2), (2.0, 2)])
def test_a_count_and_fraction_parameter_refuses_naming_its_floor(value, floor):
    """One rule and one message: the floor is named, where "Value should be at least 0" was not."""
    with pytest.raises(Refusal) as refusal:
        fraction_or_count(value, floor, 'the split')
    message = str(refusal.value)
    assert f'at least {floor}' in message
    assert 'at least 0' not in message


def test_scoring_reads_the_measured_values_first():
    """The pair was swapped by ``fit``, so the R2 the weights read was measured backwards."""
    measured = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([0.0, 3.0, 3.0, 6.0])
    assert score_regression(measured, predicted)['r2'] == pytest.approx(-0.2)
    with pytest.raises(Refusal) as refusal:
        score_regression(measured, predicted, ['nope'])
    assert "'r2'" in str(refusal.value)


def test_fitting_the_mixture_fits_every_member(smooth_data):
    """The old ``_train`` trained a ``deepcopy`` of the pool and never assigned it back."""
    x, y = smooth_data
    mixture = MixtureSurrogate([{'model_type': 'rid'}, {'model_type': 'knn'}]).fit(x, y)
    for member in mixture.pool:
        assert np.all(np.isfinite(member.predict(x[:3])))


def test_the_mixture_is_not_the_unweighted_mean_of_its_members(smooth_data):
    """The weighting is live: the combination is the R2-weighted mean, and it is not the plain mean."""
    x, y = smooth_data
    mixture = MixtureSurrogate([{'model_type': 'rid'}, {'model_type': 'knn'}]).fit(x, y)
    members = np.stack([member.predict(x) for member in mixture.pool])
    scored = np.maximum([member.scores['r2'] for member in mixture.pool], 0.0)
    weights = np.array(scored) / np.sum(scored)
    assert np.allclose(mixture.predict(x), np.tensordot(weights, members, axes=1))
    assert not np.allclose(mixture.predict(x), members.mean(axis=0))


def test_the_max_weight_method_returns_the_best_member(smooth_data):
    x, y = smooth_data
    mixture = MixtureSurrogate([{'model_type': 'rid'}, {'model_type': 'knn'}], weight_method='max').fit(x, y)
    best = int(np.argmax([member.scores['r2'] for member in mixture.pool]))
    assert np.allclose(mixture.predict(x), mixture.pool[best].predict(x))


def test_a_pool_with_no_weight_left_is_refused_rather_than_averaged(smooth_data):
    """``np.all(weights == 0)`` used to answer with a uniform array: a mean, silently, as a weight."""
    x, y = smooth_data
    mixture = MixtureSurrogate([{'model_type': 'rid'}, {'model_type': 'knn'}]).fit(x, y)
    for member in mixture.pool:
        member._scores = dict.fromkeys(SCORE_METHODS, 0.0)
    with pytest.raises(Refusal, match='uniform mean'):
        mixture.predict(x[:3])


@pytest.mark.parametrize('pool', [None, []])
def test_a_mixture_without_a_pool_is_refused_naming_the_registered_keys(pool):
    with pytest.raises(Refusal) as refusal:
        MixtureSurrogate(pool)
    for key in SURROGATE_KEYS:
        assert f"'{key}'" in str(refusal.value)


def test_the_declared_sets_are_named_by_their_refusals():
    with pytest.raises(Refusal) as weight_method:
        MixtureSurrogate([{'model_type': 'rid'}], weight_method='best')
    for method in WEIGHT_METHODS:
        assert f"'{method}'" in str(weight_method.value)
    with pytest.raises(Refusal, match='model_type'):
        MixtureSurrogate(['rid'])


def test_predicting_before_fitting_is_refused_rather_than_answering_None(smooth_data):
    """``valid`` was True before and after training; scoring is what the mixture weighs, not a mode."""
    x, _ = smooth_data
    with pytest.raises(Refusal, match='has not been fitted'):
        SurrogateModel('rid').predict(x[:2])
    with pytest.raises(Refusal, match='has not been fitted'):
        MixtureSurrogate([{'model_type': 'rid'}]).predict(x[:2])
    model = SurrogateModel('rid', n_splits=3).fit(x, smooth_data[1])
    assert not hasattr(model, 'check_valid') and not hasattr(model, '_valid')
    assert set(model.scores) == set(SCORE_METHODS)
    with pytest.raises(TypeError):
        SurrogateModel('rid', do_calc_score=False).fit(x, smooth_data[1])
