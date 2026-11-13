"""A solver's keyword settings, read off its class signatures, and the one builder that refuses an unknown one.

Every solver factory builds through :func:`build`, so a misspelt setting is refused naming the settings the
solver does take. :data:`LOOP_ARGUMENTS` are what a variation receives from the evolutionary loop rather
than from its settings.
"""

import inspect

from optimi_lab.core import refuse
from optimi_lab.oracle import Capability
from optimi_lab.problem import Problem

__all__ = ['LOOP_ARGUMENTS', 'build', 'settings_of']

#: Arguments every variation receives from the loop rather than from its settings.
LOOP_ARGUMENTS = frozenset({'population', 'width', 'rng', 'values', 'violation'})


def settings_of(owner: type) -> frozenset[str]:
    """Every keyword setting *owner* takes, its own and each base class's, read off their signatures."""
    names: set[str] = set()
    for cls in owner.__mro__[:-1]:
        parameters = inspect.signature(cls.__init__).parameters.values()
        names |= {p.name for p in parameters if p.kind is p.KEYWORD_ONLY}
    return frozenset(names)


def build[T](owner: type[T], problem: Problem, capability: Capability, what: str, **settings: object) -> T:
    """``owner(problem, capability, **settings)``, refusing a setting it does not take and naming those it does."""
    accepted = settings_of(owner)
    unknown = sorted(set(settings) - accepted)
    if unknown:
        refuse(f'{unknown} are not settings of {what}; the valid settings are {sorted(accepted)}')
    return owner(problem, capability, **settings)
