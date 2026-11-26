"""BFGS (direct, inverse, against the two-loop) and SR1: each update keeps its secant equation, and skips a degenerate pair."""

from __future__ import annotations

import numpy as np
from numerics_lab.quasi_newton import secant_pair, two_loop

from optimi_lab.operator.quasi_newton import bfgs_inverse_update, bfgs_update, sr1_update

_RNG = np.random.default_rng(7)
_N = 5
_G = _RNG.standard_normal((_N, _N))
_SPD = _G @ _G.T + _N * np.eye(_N)


def _pairs(count):
    s = _RNG.standard_normal((count, _N))
    return s, s @ _SPD.T  # y = A s: curvature y . s > 0


def test_bfgs_inverse_update_keeps_the_secant_equation_and_symmetry() -> None:
    s, y = _pairs(1)
    h = bfgs_inverse_update(np.eye(_N), s[0], y[0])
    np.testing.assert_allclose(h @ y[0], s[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h, h.T, atol=1e-14)
    assert np.all(np.linalg.eigvalsh(h) > 0.0)


def test_bfgs_direct_and_inverse_updates_are_inverses() -> None:
    s, y = _pairs(3)
    b, h = np.eye(_N), np.eye(_N)
    for k in range(3):
        b, h = bfgs_update(b, s[k], y[k]), bfgs_inverse_update(h, s[k], y[k])
    np.testing.assert_allclose(b @ h, np.eye(_N), atol=1e-10)


def test_a_negative_curvature_pair_is_skipped() -> None:
    s = np.ones(_N)
    h = np.eye(_N)
    assert bfgs_inverse_update(h, s, -s) is h
    # Planted control: the same pair with positive curvature does update.
    assert bfgs_inverse_update(h, s, s) is not h


def test_the_two_loop_equals_the_dense_bfgs_inverse_from_the_same_h0() -> None:
    s, y = _pairs(4)
    h0 = 0.5
    dense = h0 * np.eye(_N)
    for k in range(4):
        dense = bfgs_inverse_update(dense, s[k], y[k])
    pairs = [secant_pair(s[k], y[k]) for k in range(4)]
    q = _RNG.standard_normal(_N)
    np.testing.assert_allclose(two_loop(q, pairs, lambda v: h0 * v), dense @ q, rtol=1e-10)
    # Planted control: dropping the oldest pair changes the answer.
    assert not np.allclose(two_loop(q, pairs[1:], lambda v: h0 * v), dense @ q, rtol=1e-6)


def test_sr1_keeps_the_secant_equation_and_skips_a_vanishing_denominator() -> None:
    s, y = _pairs(1)
    b = sr1_update(np.eye(_N), s[0], y[0])
    np.testing.assert_allclose(b @ s[0], y[0], rtol=1e-12)
    np.testing.assert_allclose(b, b.T, atol=1e-14)
    # B s = y already: y - B s = 0, so there is nothing to update and B comes back unchanged.
    same = np.eye(_N)
    assert sr1_update(same, s[0], s[0]) is same
