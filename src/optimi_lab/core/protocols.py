"""The three interfaces every algorithm and surrogate in this package is written against.

They are Protocols rather than base classes so that an implementation keeps its own constructor:
an algorithm that must inherit a base class to be usable cannot be a wrapper around somebody
else's, and what a caller needs is the behaviour.

The ask/tell loop is the substance kept from the old package, with its two leaky parts removed:

* **A prediction is not an outcome.** The old loop carried a BOOLEAN FLAG that redirected scoring
  onto predicted values instead of measured ones, so the optimizer reported a Pareto front it had
  never evaluated, and the flag's only trace in the output was a log line. Its spelling is retired
  and registered with its replacement in
  ``tests/architecture/test_a_retired_spelling_stays_retired.py``, which is the one file allowed
  to write it. :class:`Surrogate.predict` returns a plain array: no :class:`Outcome` can be built
  from it, so a prediction cannot be handed to :meth:`Proposer.tell` at all — the shape, rather
  than the flag, is what makes it impossible.
* **Fitting is total.** ``check_valid`` compared name lists after training and its result was
  read by nobody — ``valid`` was True before and after every fit. A model exists only as what
  ``fit`` returned, so there is no validity flag to consult or to forget.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from optimi_lab.core.outcomes import Evaluation

__all__ = ['Evaluator', 'Proposer', 'Surrogate']


@runtime_checkable
class Proposer(Protocol):
    """An algorithm that proposes points and is told what they produced.

    The loop is vectorised in and per-point out: one :meth:`ask` returns a whole batch of shape
    ``(n_points, n_var)``, and one :meth:`tell` hands back a whole batch of outcomes. An
    implementation that runs generations internally does so inside :meth:`tell`.
    """

    def ask(self) -> np.ndarray:
        """Return the next batch of points, shape ``(n_points, n_var)``, at least one point.

        The columns are the variables in the order the :class:`~optimi_lab.core.space.VariableSet`
        declared them, and the bounds are the proposer's to respect: this package clips only
        where an operator needs it, and never silently rewrites what was asked for.
        """
        ...

    def tell(self, evaluation: Evaluation) -> None:
        """Accept what the last :meth:`ask` produced.

        ``evaluation`` holds one row per point that was asked for, with values in the
        MINIMIZATION frame (a maximizing objective arrives negated), so an implementation never
        has to know a direction to rank points. It carries an :class:`~optimi_lab.core.outcomes.
        Outcome` per cell: a point that did not produce a value is visible as such, and
        ``evaluation.complete()`` is the subset an implementation can fit or rank.
        """
        ...


class Evaluator(Protocol):
    """What a caller supplies to score a batch: points in, values and outcomes out."""

    def __call__(self, inputs: np.ndarray) -> Evaluation:
        """Evaluate every row of ``inputs`` (shape ``(n_points, n_var)``).

        Returns:
            Evaluation: Shape ``(n_points, n_obj)``, with ``inputs`` echoed back. A point the
            caller could not evaluate is reported as an outcome, never as a number.

        """
        ...


@runtime_checkable
class Surrogate(Protocol):
    """A fit: training data in, a fitted model out, predictions from that model.

    ``fit`` returning the fitted model is what removes the validity flag — there is no state in
    which a surrogate is "trained but invalid", because the object either exists or ``fit``
    refused to produce it.
    """

    def fit(self, inputs: np.ndarray, values: np.ndarray) -> Surrogate:
        """Fit on ``inputs``/``values`` (shape ``(n_points, n_var)`` / ``(n_points, n_obj)``).

        A row of ``values`` must be a measured one: a point whose outcome was not ``OK`` has no
        value to learn from, so the caller passes ``evaluation.complete()`` and not the batch.

        Returns:
            Surrogate: The fitted model.

        """
        ...

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return predicted values, shape ``(n_points, n_obj)``, for ``inputs``.

        A prediction is a number the caller may use and must label as a prediction. It is
        deliberately NOT an :class:`~optimi_lab.core.outcomes.Evaluation`: nothing in this
        package can turn the return of this method into an :class:`~optimi_lab.core.outcomes.
        Outcome`, so no optimizer can score it as if the points had run.
        """
        ...
