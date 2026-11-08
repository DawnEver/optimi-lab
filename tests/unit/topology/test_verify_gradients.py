"""Every response of a provider checked against central differences through the whole design chain."""

from __future__ import annotations

import numpy as np
import pytest

from optimi_lab.core import Refusal
from optimi_lab.topology import (
    DensityFilter,
    DesignChain,
    DesignDomain,
    ProviderResponse,
    Role,
    VirtualTemperature,
    verify_gradients,
    volume_fraction,
)


class _Connected:
    """Objective: the volume fraction; constraint: the virtual temperature sunk at the left column."""

    def __init__(self, *, skew: float = 1.0) -> None:
        self.domain = DesignDomain.grid((6, 5), spacing=1.0, length_unit='mm', solid=np.arange(30) < 5)
        self.measure = VirtualTemperature(self.domain, np.arange(30) < 5)
        self.responses = (
            ProviderResponse('volume', Role.OBJECTIVE),
            ProviderResponse('connectivity', Role.CONSTRAINT, bound=np.log(30.0)),
        )
        self.skew = skew

    def evaluate(self, physical):
        value, slope = self.measure.log_ratio(physical)
        return {'volume': volume_fraction(self.domain, physical), 'connectivity': (value, slope * self.skew)}


def _check(provider, **kw):
    chain = DesignChain(DensityFilter(provider.domain, 1.5, unit='mm'), beta=2.0)
    design = np.random.default_rng(5).uniform(0.3, 0.7, provider.domain.n_design)
    return verify_gradients(provider, chain, design, [0, 4, 11, 24], step=1e-6, rtol=1e-5, **kw)


def test_an_exact_provider_passes_every_response() -> None:
    check = _check(_Connected())
    assert set(check.responses) == {'volume', 'connectivity'}
    assert check.passed
    for response in check.responses.values():
        assert [k for k, _, _ in response.rows] == [0, 4, 11, 24]
        assert response.max_rel_error <= 1e-5


def test_a_wrong_sensitivity_fails_its_own_response_only() -> None:
    """Planted control: a 2 % error in one response's slope fails it, and only it."""
    check = _check(_Connected(skew=1.02))
    assert not check.passed
    assert check.responses['volume'].passed
    assert not check.responses['connectivity'].passed
    assert check.responses['connectivity'].max_rel_error == pytest.approx(0.02 / 1.02, rel=1e-3)


def test_a_check_without_variables_is_refused() -> None:
    with pytest.raises(Refusal, match='variable'):
        _check_empty()


def _check_empty():
    provider = _Connected()
    chain = DesignChain(DensityFilter(provider.domain, 1.5, unit='mm'))
    return verify_gradients(provider, chain, np.full(provider.domain.n_design, 0.5), [])
