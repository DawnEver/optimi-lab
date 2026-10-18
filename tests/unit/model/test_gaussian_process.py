"""D45-D48 -- GP kernels, maximum-likelihood hyperparameters, closed-form LOO, mean/variance/gradient.

Oracle: scikit-learn's ``GaussianProcessRegressor`` with ``ConstantKernel * K + WhiteKernel``,
``normalize_y=True`` and ``alpha=0`` -- its default ``alpha=1e-10`` is a second, unfitted nugget
that moves the likelihood by ~1e-8 relative (tests only). Our hyperparameter vector is sklearn's ``theta`` in the same
order -- ``log[amplitude, length_1..length_d, noise]`` -- so a likelihood is compared at the SAME
point, and a fit is compared by the likelihood it reached.

Tolerances: likelihoods are RELATIVE (dimensionless); predictions are ABSOLUTE in target units.
"""

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel

from optimi_lab.core import Refusal
from optimi_lab.model import KERNEL_FAMILIES, GaussianProcess, Give, log_likelihood

#: Relative: the same Cholesky of the same matrix.
SAME_POINT = 1e-9
#: Absolute, target units: the same posterior algebra at the same hyperparameters.
PREDICT = 1e-8
#: Relative: our maximised log-likelihood may not fall below sklearn's L-BFGS-B optimum by more.
NOT_WORSE = 1e-6


def _kernel(family: str, d: int):
    lengths = np.ones(d)
    base = {
        'squared_exponential': RBF(lengths),
        'matern12': Matern(lengths, nu=0.5),
        'matern32': Matern(lengths, nu=1.5),
        'matern52': Matern(lengths, nu=2.5),
    }[family]
    return ConstantKernel(1.0) * base + WhiteKernel(1e-2)


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(5)
    x = rng.random((25, 2))
    y = (np.sin(5 * x[:, 0]) + 0.5 * x[:, 1] ** 2 + 0.01 * rng.standard_normal(25))[:, None]
    return x, y, rng.random((10, 2))


def _normalised(y):
    std = np.std(y, axis=0)
    return (y - y.mean(axis=0)) / np.where(std > 0, std, 1.0)


@pytest.mark.parametrize('family', KERNEL_FAMILIES)
def test_the_log_likelihood_and_its_gradient_equal_sklearns_at_the_same_point(data, family) -> None:
    x, y, _ = data
    theta = np.log([1.7, 0.3, 0.8, 1e-2])
    sk = GaussianProcessRegressor(_kernel(family, 2), normalize_y=True, optimizer=None, alpha=0.0).fit(x, y)
    value, gradient = sk.log_marginal_likelihood(theta, eval_gradient=True)
    ours, our_gradient = log_likelihood(family, theta, x, _normalised(y)[:, 0])
    assert abs(ours - value) <= SAME_POINT * abs(value)
    assert np.max(np.abs(our_gradient - gradient)) <= SAME_POINT * max(1.0, np.max(np.abs(gradient)))


@pytest.mark.parametrize('family', KERNEL_FAMILIES)
def test_the_fit_reaches_at_least_sklearns_maximum(data, family) -> None:
    x, y, _ = data
    sk = GaussianProcessRegressor(
        _kernel(family, 2), normalize_y=True, n_restarts_optimizer=4, random_state=0, alpha=0.0
    )
    sk.fit(x, y)
    fitted = GaussianProcess(family, restarts=4).fit(x, y)
    best = sk.log_marginal_likelihood_value_
    assert fitted.log_likelihood[0] >= best - NOT_WORSE * abs(best)


@pytest.mark.parametrize('family', KERNEL_FAMILIES)
def test_mean_and_variance_equal_sklearns_at_the_same_hyperparameters(data, family) -> None:
    x, y, q = data
    fitted = GaussianProcess(family).fit(x, y)
    theta = fitted.hyperparameters[0]
    sk = GaussianProcessRegressor(
        _kernel(family, 2).clone_with_theta(theta), normalize_y=True, optimizer=None, alpha=0.0
    )
    mean, std = sk.fit(x, y).predict(q, return_std=True)
    prediction = fitted.predict(q, variance=True)
    assert np.max(np.abs(prediction.mean[:, 0] - mean)) <= PREDICT
    # sklearn's std includes the white-noise term; ours is the LATENT variance, so the noise is added back.
    noise = np.exp(theta[-1]) * np.std(y) ** 2
    assert np.max(np.abs(prediction.variance[:, 0] + noise - std**2)) <= PREDICT


@pytest.mark.parametrize('family', ['squared_exponential', 'matern32', 'matern52'])
def test_the_mean_gradient_is_the_derivative_of_the_mean(data, family) -> None:
    x, y, q = data
    fitted = GaussianProcess(family).fit(x, y)
    analytic = fitted.predict(q, gradient=True).gradient
    h = 1e-6
    for j in range(2):
        step = np.zeros(2)
        step[j] = h
        central = (fitted.predict(q + step).mean - fitted.predict(q - step).mean) / (2 * h)
        # Absolute, target units per unit input.
        assert np.max(np.abs(analytic[:, :, j] - central)) <= 1e-6


def test_closed_form_leave_one_out_equals_refitting_without_each_point(data) -> None:
    """D47: Dubrule's identities against n brute-force posteriors at the fitted hyperparameters."""
    x, y, _ = data
    fitted = GaussianProcess('squared_exponential').fit(x, y)
    loo = fitted.leave_one_out()
    theta = fitted.hyperparameters[0]
    kernel = _kernel('squared_exponential', 2).clone_with_theta(theta)
    for i in range(0, len(x), 6):
        keep = np.arange(len(x)) != i
        # The normalisation is the full data's, as in the closed form: only the point is left out.
        sk = GaussianProcessRegressor(kernel, optimizer=None, alpha=0.0).fit(x[keep], _normalised(y)[keep])
        mean, std = sk.predict(x[i : i + 1], return_std=True)
        scale = np.std(y)
        assert abs(loo.mean[i, 0] - (mean[0] * scale + y.mean())) <= PREDICT
        assert abs(loo.variance[i, 0] - std[0] ** 2 * scale**2) <= PREDICT


def test_each_output_column_gets_its_own_hyperparameters(data) -> None:
    x, y, _ = data
    two = np.column_stack([y[:, 0], np.cos(9 * x[:, 1])])
    fitted = GaussianProcess('squared_exponential').fit(x, two)
    assert len(fitted.hyperparameters) == 2
    assert not np.allclose(fitted.hyperparameters[0], fitted.hyperparameters[1])
    assert fitted.predict(x, variance=True).variance.shape == (25, 2)


def test_a_gp_gives_variance_and_gradient_and_a_stranger_family_names_the_set() -> None:
    assert GaussianProcess().gives == frozenset({Give.VARIANCE, Give.GRADIENT})
    with pytest.raises(Refusal, match='matern52'):
        GaussianProcess('rbf')


def test_the_fit_beats_a_planted_wrong_hyperparameter_and_loo_flags_a_planted_outlier(data) -> None:
    """Planted controls: a deliberately wrong theta scores lower, and a corrupted value stands out in LOO."""
    x, y, _ = data
    fitted = GaussianProcess('squared_exponential').fit(x, y)
    wrong = fitted.hyperparameters[0] + np.array([0.0, 3.0, 3.0, 0.0])  # lengths 20x too long
    assert log_likelihood('squared_exponential', wrong, x, _normalised(y)[:, 0])[0] < fitted.log_likelihood[0] - 1
    corrupted = y.copy()
    corrupted[7] += 5 * np.std(y)
    loo = GaussianProcess('squared_exponential').fit(x, corrupted).leave_one_out()
    z = np.abs(loo.mean[:, 0] - corrupted[:, 0]) / np.sqrt(loo.variance[:, 0])
    assert int(np.argmax(z)) == 7
