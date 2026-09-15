"""The ask/tell driver: one initial sample, then as many batches as the caller asked for.

THE SAMPLE SIZE AND THE POPULATION SIZE ARE ONE NUMBER. The old loop took its first batch from
``variable_space.var_space_matrix`` — the whole grid, measured 121 rows — and then stepped a
population of ``_pop_size`` = 8, so a run of 4 iterations evaluated ``[121, 8, 8, 8]`` points
while ``_pop_size`` was documented "per iteration". Here the initial batch is drawn first, and
its row count is the ONLY count there is: :func:`optimize` hands it to ``build_proposer``, which
receives ``pop_size`` as its one argument. There is no second number to disagree with it.

Nothing here writes to a file and nothing here reads one. The old ``save_optimizer`` /
``load_optimizer`` pair wrote a TOML state file and re-initialised on load — measured: 7 rows
saved, 0 loaded — so a round trip preserved nothing. The record a run returns is arrays; a
caller that wants it on disk hands those arrays to numpy.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from optimi_lab.core.errors import refuse, require_axis, require_positive
from optimi_lab.core.outcomes import Evaluation
from optimi_lab.core.pareto import non_dominated_sorting
from optimi_lab.core.protocols import Evaluator, Proposer
from optimi_lab.core.sampling import SampleSpec, sample
from optimi_lab.core.space import ObjectiveSet, VariableSet

__all__ = ['Record', 'optimize']


@dataclass(frozen=True, slots=True)
class Record:
    """Every point a run asked for, every value it produced, and what produced none.

    ``space`` and ``objectives`` are the columns of ``evaluation`` (shape ``(n_points, n_obj)``,
    values in the caller's own frame, one :class:`Outcome` per cell). The record is
    self-describing on purpose: the old ``optimize`` returned four bare matrices whose columns
    were named only by the optimizer object the caller still had to hold on to.
    """

    space: VariableSet
    objectives: ObjectiveSet
    evaluation: Evaluation

    def __len__(self) -> int:
        return len(self.evaluation)

    def complete(self) -> Evaluation:
        """The points whose every objective produced a value."""
        return self.evaluation.complete()

    def pareto(self) -> Evaluation:
        """The first Pareto front of the complete points, in the caller's own frame.

        Raises:
            Refusal: If no point produced a value. An empty front is not a result — it is a run
                whose points all failed, which reads identically to a run that found nothing.

        """
        complete = self.evaluation.complete()
        if len(complete) == 0:
            outcomes = sorted({outcome.value for outcome in self.evaluation.point_outcomes()})
            refuse(
                f'no point of the {len(self.evaluation)} asked produced a value, so there is no front; '
                f'the outcomes were {outcomes} — an empty front would read as a run that found nothing'
            )
        front = non_dominated_sorting(self.objectives.to_minimization(complete.values))
        return Evaluation(
            inputs=complete.inputs[front], values=complete.values[front], outcomes=complete.outcomes[front]
        )


def optimize(
    *,
    space: VariableSet,
    objectives: ObjectiveSet,
    evaluate: Evaluator,
    build_proposer: Callable[[int], Proposer],
    spec: SampleSpec | None = None,
    n_batches: int = 4,
    on_batch: Callable[[Record], None] | None = None,
) -> Record:
    """Run a proposer against an evaluator and return everything that was learned.

    Batch 0 is the initial sample; every later batch is one :meth:`Proposer.ask`. The proposer
    is built AFTER the sample, so the population size it is given is the number of rows the
    sampler actually produced — the one number this loop has.

    ``evaluate`` is called with a shape ``(n_points, n_var)`` batch and must return an
    :class:`Evaluation` of exactly those points; ``build_proposer`` is handed ``pop_size`` and
    must return an object with ``ask``/``tell``; ``spec`` says how to draw the initial batch;
    ``n_batches`` counts the batches in total, the initial sample included; ``on_batch`` is
    called with the run so far after every batch is told, replacing the old iteration callback
    that logged through a handler the package had installed on the host process.

    Raises:
        Refusal: If ``n_batches < 1``, if the proposer does not implement ask/tell, if an ask is
            empty or wrongly shaped, or if the evaluator returns anything other than one
            evaluation of exactly the points it was asked about.

    """
    require_positive(n_batches, 'n_batches')
    spec = SampleSpec() if spec is None else spec

    asked = sample(space, spec)
    pop_size = asked.shape[0]
    proposer = build_proposer(pop_size)
    if not isinstance(proposer, Proposer):
        refuse(
            f'build_proposer({pop_size}) returned {type(proposer).__name__}, which does not implement ask() and '
            f'tell(); a proposer is any object with both methods'
        )

    batches: list[Evaluation] = []
    for index_batch in range(n_batches):
        if index_batch > 0:
            asked = _require_ask(proposer.ask(), space, pop_size)
        evaluation = _require_evaluation(evaluate(asked), objectives, asked)
        batches.append(evaluation)
        proposer.tell(evaluation.to_minimization(objectives))
        if on_batch is not None:
            on_batch(Record(space=space, objectives=objectives, evaluation=Evaluation.stack(batches)))
    return Record(space=space, objectives=objectives, evaluation=Evaluation.stack(batches))


def _require_ask(asked: np.ndarray, space: VariableSet, pop_size: int) -> np.ndarray:
    """Return the batch an :meth:`Proposer.ask` produced, or refuse naming what it returned."""
    if asked is None:
        refuse(
            f'ask() returned None; it must return an array of shape (n_points, {space.n_var}) with at least one point'
        )
    asked = require_axis(asked, space.n_var, 1, 'the array returned by ask()')
    if asked.shape[0] < 1:
        refuse(
            f'ask() returned no point; a proposer that proposes nothing ends the run it was built for (pop_size={pop_size})'
        )
    return asked


def _require_evaluation(product: object, objectives: ObjectiveSet, asked: np.ndarray) -> Evaluation:
    """Return the batch an :class:`Evaluator` produced, or refuse naming what it returned."""
    if not isinstance(product, Evaluation):
        refuse(
            f'the evaluator returned {type(product).__name__}, not an Evaluation; a prediction is not an outcome, '
            f'so no array of numbers can be told to a proposer — wrap measured values in Evaluation(inputs, values)'
        )
    require_axis(product.inputs, asked.shape[0], 0, 'the inputs on an Evaluation')
    if not np.array_equal(product.inputs, asked):
        refuse(
            f'the evaluator returned points other than the {asked.shape[0]} it was asked about; an evaluation is read '
            f'row by row against the batch that was requested, so a reordered or replaced batch misnames every value'
        )
    if product.n_obj != objectives.n_obj:
        refuse(
            f'the evaluator returned {product.n_obj} objective column(s), but {objectives.n_obj} were declared: '
            f'{list(objectives.names)}'
        )
    return product
