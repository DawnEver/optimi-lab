"""Every declared set in `optimi_lab` REFUSES a stranger, and names the set while doing it.

A library of interchangeable parts is a set of registries: which regressor, which sampler, which
direction. The failure this refuses is not a crash -- it is the SILENT one, where an unrecognised
key falls back to a default and the caller gets a run it never asked for, indistinguishable from
the one it did. The old package's own history is the evidence: its refusal for an unregistered
surrogate key printed neither the key's valid set nor a usable exception type.

So the property is two-part and both parts are checked: the call RAISES `Refusal`, and the message
QUOTES at least one member of the valid set. A refusal that does not say what would have worked
sends the caller to read the source, which is the registry it was supposed to be told about.

THE SETS ARE TAKEN FROM THE PACKAGE, NOT RETYPED HERE. A hand-kept copy of `SURROGATE_KEYS` would
agree with itself forever while a new key arrived unchecked -- that is the defect one layer up.
"""

from __future__ import annotations

from typing import Callable

import pytest

from optimi_lab.core.errors import Refusal, coerce_enum
from optimi_lab.core.sampling import SAMPLE_KINDS, SampleKind, SampleSpec
from optimi_lab.surrogate_model.registry import SURROGATE_KEYS, resolve

#: The stranger. Not a typo of a real key on purpose: a near-miss could be served by a fuzzy match
#: nobody documented, and this must be a value no reasonable set contains.
_STRANGER = 'no_such_member_zzz'

#: `(name, call, valid set)` -- one row per declared set that a caller can name from outside.
_SETS: list[tuple[str, Callable[[str], object], tuple[str, ...]]] = [
    ('surrogate registry', lambda key: resolve(key, 'surrogate'), tuple(SURROGATE_KEYS)),
    ('sample kinds', lambda key: coerce_enum(SampleKind, key, 'sample kind'), tuple(SAMPLE_KINDS)),
]


@pytest.mark.parametrize(('name', 'call', 'valid'), _SETS, ids=[row[0] for row in _SETS])
def test_a_declared_set_refuses_a_stranger_and_names_itself(
    name: str,
    call: Callable[[str], object],
    valid: tuple[str, ...],
) -> None:
    """THE CHECK. An unregistered key raises `Refusal`, and the message quotes the valid set."""
    assert valid, f'the {name} set is empty, so this test would pass over nothing.'
    with pytest.raises(Refusal) as caught:
        call(_STRANGER)
    message = str(caught.value)
    assert any(member in message for member in valid), (
        f'the {name} refused {_STRANGER!r} without naming a single valid member: {message!r}. The set '
        f'that exists to spell the answer is one import away from the refusal that withheld it.'
    )


@pytest.mark.parametrize(('name', 'call', 'valid'), _SETS, ids=[row[0] for row in _SETS])
def test_every_declared_member_still_resolves(
    name: str,
    call: Callable[[str], object],
    valid: tuple[str, ...],
) -> None:
    """The other side of the ratchet: a name the set advertises must actually be servable.

    Without this, the refusal above could be made to pass by refusing EVERYTHING -- a guard that
    says no to all input is not a guard, and a registry whose keys do not resolve is a lie the
    error message repeats to every caller.
    """
    for member in valid:
        assert call(member) is not None, f'the {name} advertises {member!r} and did not serve it.'


def test_the_sample_spec_refuses_a_stranger_kind() -> None:
    """The same property one level up: a stranger reaching a dataclass field is refused there too."""
    with pytest.raises(Refusal):
        SampleSpec(kind=_STRANGER)
