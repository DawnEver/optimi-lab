"""Declared uncertainty rows become the native Uncertainty; a declared risk prints the statistic it is."""

from __future__ import annotations

import pytest

from optimi_lab.core import Refusal
from optimi_lab.operator import Risk
from optimi_lab.problem import DISTRIBUTIONS, Normal, Uncertainty, Uniform


def test_declared_rows_become_parameters_in_order() -> None:
    declared = Uncertainty.declared([('t', 'normal', 20.0, 2.0, 'degC'), ('g', 'uniform', 0.9, 1.1, None)])
    assert declared == Uncertainty((Normal('t', 20.0, 2.0, 'degC'), Uniform('g', 0.9, 1.1)))
    assert sorted(DISTRIBUTIONS) == ['normal', 'uniform']


def test_no_rows_declare_no_uncertainty_and_a_stranger_distribution_is_refused() -> None:
    assert Uncertainty.declared([]) is None
    with pytest.raises(Refusal, match="no distribution 'lognormal'"):
        Uncertainty.declared([('t', 'lognormal', 1.0, 1.0, None)])


@pytest.mark.parametrize(
    ('risk', 'label'),
    [
        (Risk('cvar', alpha=0.9), 'CVaR0.9'),
        (Risk('var', alpha=0.95), 'VaR0.95'),
        (Risk('mean_plus_std', k=2.0), 'mean+2std'),
        (Risk('mean'), 'mean'),
        (Risk('worst_case'), 'worst_case'),
    ],
)
def test_a_risk_labels_the_statistic_it_is(risk: Risk, label: str) -> None:
    assert risk.label == label


def test_a_declared_row_with_absent_alpha_and_k_is_the_constructor_itself() -> None:
    assert Risk('cvar', alpha=0.9, k=None) == Risk('cvar', alpha=0.9)
