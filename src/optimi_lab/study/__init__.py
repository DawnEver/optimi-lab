"""L10 study: a declared independent-point study or search, run under one writer over one record.

* :func:`study` runs a :class:`Request`: fixed or sampled points (:class:`FixedDesign`,
  :class:`SampleDesign`), bounded feedback (:class:`FeedbackDesign`), or a :class:`SearchStrategy`
  with optional surrogate rounds (:class:`SurrogateStrategy`) and an :class:`EvaluationPolicy`.
* :func:`search` and :func:`evaluate` are the engine underneath, over a
  :class:`~optimi_lab.policy.ScreeningOracle`; they return a :class:`Searched` archive view.
* The caller supplies a :class:`Publisher`, a :class:`Codec` that files answers in its own record
  format, and binds a search (:class:`SearchBinding`).
"""

from optimi_lab.policy import ScreeningOracle
from optimi_lab.study._lock import study_writer
from optimi_lab.study.engine import Searched, evaluate, opened, search
from optimi_lab.study.protocols import Answer, Codec, Filing, Publisher, SearchBinding
from optimi_lab.study.rounds import searched_study
from optimi_lab.study.strategy import (
    ALGORITHMS,
    ROUND_SEED_STRIDE,
    EvaluationPolicy,
    FeedbackDesign,
    FidelityEvaluation,
    FixedDesign,
    SampleDesign,
    SearchStrategy,
    SurrogateRound,
    SurrogateStrategy,
    algorithm_factory,
    canonical_subject,
)
from optimi_lab.study.study import RECORD_SCHEMA, FixedStudyResult, Request, StudyOutcome, fixed_design, study

__all__ = [
    'ALGORITHMS',
    'RECORD_SCHEMA',
    'ROUND_SEED_STRIDE',
    'Answer',
    'Codec',
    'EvaluationPolicy',
    'FeedbackDesign',
    'FidelityEvaluation',
    'Filing',
    'FixedDesign',
    'FixedStudyResult',
    'Publisher',
    'Request',
    'SampleDesign',
    'ScreeningOracle',
    'SearchBinding',
    'SearchStrategy',
    'Searched',
    'StudyOutcome',
    'SurrogateRound',
    'SurrogateStrategy',
    'algorithm_factory',
    'canonical_subject',
    'evaluate',
    'fixed_design',
    'opened',
    'search',
    'searched_study',
    'study',
    'study_writer',
]
