"""The problem: one space, its objectives and constraints, and what is known about them.

Construction cross-checks the declarations, because each one alone reads fine: a linear constraint
over a variable the space does not have, a chance constraint with no uncertainty to take the chance
over, a structure claim about an output nobody produces. Each is refused here, naming the missing
piece.

:attr:`Problem.identity` is a :class:`lab_commons.outcomes.Identity` of the plain declaration, and
:meth:`Problem.identify` the identity of one design under it -- the key an archive files a result
under. The design is reduced to its canonical form first, so an inactive value never splits it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from lab_commons.outcomes import Identity

from optimi_lab.core.errors import refuse, require_declared_names
from optimi_lab.problem.constraints import BlackBox, Chance, Constraint, Explicit, Linear, Path
from optimi_lab.problem.declarations import Fidelity, Structure, Uncertainty
from optimi_lab.problem.objective import Objective
from optimi_lab.problem.space import Space
from optimi_lab.problem.variables import Integer, Real

__all__ = ['Problem']


def _plain(item: object) -> object:
    """A declaration as identity-ready plain data; a callable is named by its qualified name."""
    if callable(item):
        return f'{item.__module__}.{item.__qualname__}'
    if hasattr(item, '__dataclass_fields__'):
        return {'kind': type(item).__name__} | {
            k: _plain(getattr(item, k)) for k in item.__dataclass_fields__ if not k.startswith('_')
        }
    if isinstance(item, Mapping):
        return {str(k): _plain(v) for k, v in item.items()}
    if isinstance(item, (tuple, list, frozenset, set)):
        values = [_plain(v) for v in item]
        return sorted(values, key=repr) if isinstance(item, (frozenset, set)) else values
    return item


@dataclass(frozen=True, slots=True)
class Problem:
    """Space, objectives, constraints, and the optional structure, uncertainty and fidelity."""

    space: Space
    objectives: tuple[Objective, ...]
    constraints: tuple[Constraint, ...] = ()
    structure: Structure | None = None
    uncertainty: Uncertainty | None = None
    fidelity: Fidelity | None = None
    identity: Identity = field(init=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'objectives', tuple(self.objectives))
        object.__setattr__(self, 'constraints', tuple(self.constraints))
        if not self.objectives:
            refuse('a problem needs at least one objective')
        require_declared_names(
            tuple(o.name for o in self.objectives) + tuple(c.name for c in self.constraints), 'problem'
        )
        numeric = {v.name for v in self.space if isinstance(v, (Real, Integer))}
        for constraint in self.constraints:
            if isinstance(constraint, Linear):
                unknown = sorted(set(constraint.coefficients) - set(self.space.names))
                if unknown:
                    refuse(
                        f'linear constraint {constraint.name!r} names {unknown[0]!r}, which the space does not declare'
                    )
                bad = sorted(set(constraint.coefficients) - numeric)
                if bad:
                    refuse(
                        f'linear constraint {constraint.name!r} weighs {bad}, which are not numeric (Real or Integer)'
                    )
            if isinstance(constraint, Chance) and self.uncertainty is None:
                refuse(
                    f'chance constraint {constraint.name!r} needs an Uncertainty to take the chance over; declare one'
                )
        if self.structure is not None:
            unknown = sorted(set(self.structure.forms) - set(self.outputs) - {c.name for c in self.constraints})
            if unknown:
                refuse(
                    f'the structure describes {unknown[0]!r}, which is not an output of this problem {list(self.outputs)}'
                )
        if self.uncertainty is not None:
            clash = sorted(set(self.uncertainty.names) & set(self.space.names))
            if clash:
                refuse(f'uncertain parameter {clash[0]!r} is also a design variable; one name, one meaning')
        object.__setattr__(self, 'identity', Identity.of(self.to_plain()))

    @property
    def outputs(self) -> tuple[str, ...]:
        """What the oracle must report: every objective, then every oracle-evaluated constraint output."""
        names = [o.name for o in self.objectives]
        for c in self.constraints:
            if isinstance(c, (BlackBox, Chance, Path)) and c.output not in names:
                names.append(c.output)
        return tuple(names)

    @property
    def design_constraints(self) -> tuple[Linear | Explicit, ...]:
        """The constraints the optimiser evaluates itself, without the oracle."""
        return tuple(c for c in self.constraints if isinstance(c, (Linear, Explicit)))

    def to_plain(self) -> dict[str, object]:
        """The whole declaration as plain data."""
        return {
            'space': self.space.to_plain(),
            'objectives': _plain(self.objectives),
            'constraints': _plain(self.constraints),
            'structure': _plain(self.structure),
            'uncertainty': _plain(self.uncertainty),
            'fidelity': _plain(self.fidelity),
        }

    def identify(
        self, design: Mapping[str, object], fidelity: str | None = None, *, extra: Mapping[str, object] | None = None
    ) -> Identity:
        """The identity of *design* (canonical, inactive values dropped) under this problem at *fidelity*.

        *extra* adds context a caller's task depends on -- the information wanted, a replicate, the
        evaluator version -- so that a result of one is never served for another (B0-12).
        """
        if fidelity is not None:
            if self.fidelity is None:
                refuse(f'fidelity {fidelity!r} was named, but this problem declares no Fidelity')
            self.fidelity.index(fidelity)
        context = {'problem': self.identity.key, 'fidelity': fidelity, **(extra or {})}
        return Identity.of(dict(self.space.canonical(design)), context)
