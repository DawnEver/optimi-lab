"""D117 estimators: a sampled quantity is a value WITH its standard error, never a bare number.

* :class:`Estimate` -- ``value``, ``std_error``, ``n`` (the samples it averages);
* :func:`mean_estimate` -- the sample mean, CLT standard error ``s / sqrt(n)``;
* :func:`probability` -- the frequency of a boolean event, binomial standard error;
* :func:`joint_probability` -- the frequency with which EVERY column of an ``(n, k)`` event matrix
  holds. Read off the joint samples, never a product of marginals: two 95 % events hold together
  90.25 % only when independent (B0-03);
* :func:`satisfied` -- the probability that sampled outputs meet a :class:`~optimi_lab.problem.Chance`
  constraint's relation and bound;
* :func:`replicated` -- independent replicate estimates (randomised QMC, batches) pooled into their
  mean, the standard error from their spread.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

import numpy as np

from optimi_lab.core.errors import refuse
from optimi_lab.problem import Chance, Relation

__all__ = ['Estimate', 'joint_probability', 'mean_estimate', 'probability', 'replicated', 'satisfied']

_RELATIONS = {Relation.AT_MOST: operator.le, Relation.AT_LEAST: operator.ge}


@dataclass(frozen=True, slots=True)
class Estimate:
    """A sampled *value*, its *std_error* (same unit), and the *n* samples it averages."""

    value: float
    std_error: float
    n: int


def _at_least_two(n: int, what: str) -> None:
    if n < 2:
        refuse(f'{what} needs at least two samples to carry a standard error, got {n}')


def mean_estimate(samples) -> Estimate:
    samples = np.ravel(np.asarray(samples, dtype=float))
    _at_least_two(samples.size, 'a mean estimate')
    return Estimate(float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(samples.size)), samples.size)


def probability(events) -> Estimate:
    events = np.ravel(np.asarray(events))
    if events.dtype != bool:
        refuse(f'a probability is the frequency of a boolean event, got dtype {events.dtype}')
    _at_least_two(events.size, 'a probability')
    p = float(events.mean())
    return Estimate(p, float(np.sqrt(p * (1.0 - p) / events.size)), events.size)


def joint_probability(events) -> Estimate:
    events = np.asarray(events)
    if events.dtype != bool or events.ndim != 2:
        refuse(f'a joint probability reads a boolean (n, k) event matrix, got {events.dtype} {events.shape}')
    return probability(np.all(events, axis=1))


def satisfied(values, constraint: Chance) -> Estimate:
    """``P(output relation bound)`` from sampled *values* of the constraint's output."""
    return probability(_RELATIONS[constraint.relation](np.asarray(values, dtype=float), constraint.bound))


def replicated(estimates) -> Estimate:
    estimates = list(estimates)
    _at_least_two(len(estimates), 'pooling replicates')
    values = np.array([e.value for e in estimates])
    return Estimate(float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size)), sum(e.n for e in estimates))
