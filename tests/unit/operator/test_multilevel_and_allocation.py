"""D123 multilevel Monte Carlo and D125 OCBA replicate allocation.

MLMC is Giles' (2008) test: a European call under geometric Brownian motion, Euler-Maruyama paths,
the coarse path driven by the SUM of the fine path's Brownian increments; the reference is the
closed-form Black-Scholes price. OCBA is Chen et al. (2000); its allocation is checked against the
closed-form ratios computed here, and its purpose against equal allocation by simulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from optimi_lab.core import Refusal
from optimi_lab.operator import mlmc, ocba

S0, STRIKE, RATE, VOL, MATURITY = 1.0, 1.0, 0.05, 0.2, 1.0
#: Black-Scholes price of the call above (currency units of S0).
BLACK_SCHOLES = 0.10450583572185565
#: The MLMC accuracy target, in the same currency units.
EPSILON = 2e-3


def call_corrections(coupled: bool):
    def sample(level: int, n: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        steps = 2**level
        h = MATURITY / steps
        dw = rng.standard_normal((n, steps)) * np.sqrt(h)
        fine = np.full(n, S0)
        for k in range(steps):
            fine = fine * (1 + RATE * h + VOL * dw[:, k])
        payoff = np.exp(-RATE * MATURITY) * np.maximum(fine - STRIKE, 0.0)
        if level == 0:
            return payoff, 1.0
        dwc = dw[:, 0::2] + dw[:, 1::2] if coupled else rng.standard_normal((n, steps // 2)) * np.sqrt(2 * h)
        coarse = np.full(n, S0)
        for k in range(steps // 2):
            coarse = coarse * (1 + RATE * 2 * h + VOL * dwc[:, k])
        return payoff - np.exp(-RATE * MATURITY) * np.maximum(coarse - STRIKE, 0.0), float(steps + steps // 2)

    return sample


def test_d123_mlmc_prices_the_call_within_its_target() -> None:
    found = mlmc(call_corrections(coupled=True), EPSILON, np.random.default_rng(0), min_levels=5)
    assert found.converged and abs(found.estimate.value - BLACK_SCHOLES) <= 3 * EPSILON
    assert found.estimate.std_error <= EPSILON
    assert found.variances[4] < found.variances[1] / 4  # the coupling makes corrections shrink, V_l ~ h_l


def test_d123_planted_control_an_uncoupled_coarse_path_does_not_shrink_its_corrections() -> None:
    coupled = mlmc(call_corrections(coupled=True), 1e-2, np.random.default_rng(1), min_levels=5)
    uncoupled = mlmc(call_corrections(coupled=False), 1e-2, np.random.default_rng(1), min_levels=5)
    assert uncoupled.variances[4] > 10 * coupled.variances[4]
    assert uncoupled.cost > 3 * coupled.cost


def _ratios(means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    best = int(np.argmin(means))
    shares = (stds / (means - means[best] + (np.arange(len(means)) == best))) ** 2
    shares[best] = stds[best] * np.sqrt(np.sum(np.delete(shares, best) ** 2 / np.delete(stds, best) ** 2))
    return shares / shares.sum()


def test_d125_ocba_meets_chens_ratios_and_spends_the_budget() -> None:
    means, stds = np.array([1.0, 1.2, 1.5, 2.0, 3.0]), np.array([1.0, 0.8, 1.0, 1.5, 1.0])
    allocation = ocba(means, stds, 1000)
    assert allocation.sum() == 1000 and allocation.dtype.kind == 'i'
    assert np.max(np.abs(allocation - 1000 * _ratios(means, stds))) <= 1.0


def _pcs(allocate, rng: np.random.Generator, repeats: int = 2000) -> float:
    means = np.array([1.0, 1.3, 1.6, 2.0, 2.5])
    correct = 0
    for _ in range(repeats):
        first = rng.normal(means[:, None], 1.0, (5, 10))
        extra = allocate(first.mean(axis=1), first.std(axis=1, ddof=1), 50)
        totals = first.sum(axis=1) + np.array([rng.normal(m, 1.0, k).sum() for m, k in zip(means, extra, strict=True)])
        correct += int(np.argmin(totals / (10 + extra)) == 0)
    return correct / repeats


def test_d125_ocba_selects_the_best_more_often_than_equal_allocation() -> None:
    """Measured 2026-10-18 at 2000 macro-replications: PCS 0.862 against 0.8265 equal; gated at +0.02."""
    rng = np.random.default_rng(4)
    equal = _pcs(lambda m, _s, budget: np.full(len(m), budget // len(m)), rng)
    allocated = _pcs(ocba, rng)
    assert allocated >= equal + 0.02


def test_d125_planted_control_a_tie_for_best_or_a_zero_spread_is_refused() -> None:
    with pytest.raises(Refusal, match='tie'):
        ocba(np.array([1.0, 1.0, 2.0]), np.ones(3), 30)
    with pytest.raises(Refusal, match='positive'):
        ocba(np.array([1.0, 2.0]), np.array([1.0, 0.0]), 30)
