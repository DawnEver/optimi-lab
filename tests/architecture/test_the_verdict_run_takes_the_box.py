"""ONE-BOX-ONE-LOCK, and the reason this repo had on record was refuted by its own entry point.

THE OLD REASON: optimi-lab has "no runner to queue on a box lock", so the rule was declared to have
no subject here. That was a statement about a RUNNER, and it was false about this repo on the day it
was written in the only way that matters: the verdict entry point this repo actually uses --
`.venv/Scripts/python.exe -m lab_commons.dev.verify`, the one line the rules page permits -- calls
`lab_commons.dev.boxwait.hold_the_box` and takes the box's one seat for the whole of ruff plus
pytest. A repo whose only verdict command takes the box IS a party to the rendezvous; it does not
become one by acquiring a runner of its own.

WHY THAT MATTERS RATHER THAN BEING A BOOKKEEPING FIX. The rule's own sentence is that a lock only
one party takes is a TAX on whoever obeys it: the party that takes none runs, the party that takes
one is starved by what it cannot see, and both report clean. This repo was on record as the party
that takes none. It was in fact already taking it -- so what was actually missing was not the lock
but any statement that could NOTICE the lock being dropped.

WHAT IS PROVED HERE, and what is deliberately NOT.

* THE PARTICIPATION IS READ OFF THE ENTRY POINT, not off a comment: the function `run_verify` calls
  is the same object this module imports, so a `verify` that stopped holding the box reds here.
* THE EXCLUSION IS DRIVEN, on PLANTED holders in an ISOLATED broker (`resource_dir=` a tmp path).
  Isolation is not squeamishness: this suite runs INSIDE a `verify` run that already holds the real
  seat, so a case that took the real box would be refused by its own runner, and one that released
  it would hand the box to whoever is queued behind this very process.
* THE PATH, not just the mechanism. The rule says the rendezvous is a PATH -- a repo that agreed on
  the mechanism and named its own pool would measure nothing. So a job admitted under a pool name of
  its OWN is planted and must still be refused by the box-scoped seat, and the records directory is
  asserted to be outside every repository.
* NOT PROVED: that any particular run is holding the box right now. That is a fact about a machine
  at an instant, and a test that asserted it would be reading the box rather than this repo.
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import Final

import pytest
from lab_commons.dev import verify
from lab_commons.dev.boxlock import BOX_POOL, BoxLock
from lab_commons.dev.boxwait import PROGRESS_S, WAIT_S, hold_the_box, holders_line
from lab_commons.resources import BOX_SEATS, Broker, Exhausted

_ROOT: Final = Path(__file__).resolve().parents[2]

#: The longest a queue may stay silent, as a fraction of the wait it reports on. A progress line
#: that arrives once per ceiling is a silent queue with an epilogue.
PROGRESS_SHARE_CEILING: Final = 0.25

#: The pool name a run that opted out of the shared spelling would use. The point of planting it is
#: that it must NOT buy an exemption -- the box-scoped seat is not addressed by name.
_A_POOL_OF_ITS_OWN: Final = 'optimi-lab-private-pool'


def _tmp() -> Path:
    """A fresh records directory, so every plant below contends only with its own case."""
    return Path(tempfile.mkdtemp(prefix='optimi-lab-boxlock-'))


def test_this_repos_verdict_entry_point_takes_the_box() -> None:
    """THE REFUTATION OF THE OLD REASON, read off the code this repo runs to get a verdict.

    Both halves are asserted because either alone is weak: the identity says the name it imported is
    the family's function and not a same-named local, and the source says `run_verify` actually
    calls it rather than merely importing it.
    """
    assert verify.hold_the_box is hold_the_box, (
        'the verify entry point bound some other `hold_the_box`. This repo participates in the box '
        'rendezvous only if the function it calls is the family one.'
    )
    source = inspect.getsource(verify.run_verify)
    assert 'hold_the_box(' in source, (
        'the verdict entry point this repo uses no longer takes the box. That is the party-that-takes-none '
        'state the rule names: it would run through whatever else is on the box, and report clean.'
    )


def test_a_second_run_is_refused_while_one_holds_the_box() -> None:
    """PLANTED CONTENTION, on the real `BoxLock`, and the refusal must NAME who is in the way.

    TWO BROKER INSTANCES OVER ONE RECORDS DIRECTORY, because that is the shape the rule is about: two
    parties that share nothing but the path. One object handing itself a refusal would prove only
    that a dataclass can count.
    """
    directory = _tmp()
    with (
        BoxLock('the-planted-holder', broker=Broker(resource_dir=directory)).held(),
        pytest.raises(Exhausted) as refusal,
        BoxLock('the-run-that-must-queue', broker=Broker(resource_dir=directory)).held(),
    ):
        pytest.fail('two CPU-saturating runs held the box at once; the seat excluded nothing.')
    named = holders_line(refusal.value)
    assert 'the-planted-holder' in named, (
        f'the refusal said {named!r}. A reader has to choose between waiting, stopping the holder and '
        f'giving up, and only the name separates those three.'
    )


def test_the_seat_is_released_when_the_run_ends() -> None:
    """THE OTHER SIDE. A seat nothing can ever take again is a hang, not an exclusion."""
    broker = Broker(resource_dir=_tmp())
    with BoxLock('the-first-run', broker=broker).held():
        pass
    with BoxLock('the-run-after-it', broker=broker).held():
        assert BoxLock.holders(broker=broker), 'a held box reports no holder, so the records say nothing'


def test_a_pool_name_of_its_own_does_not_opt_a_run_out_of_the_box() -> None:
    """THE PATH, WHICH IS THE RULE'S ACTUAL CLAIM: agreeing on the mechanism alone measures nothing.

    A job admitted under a private pool still demands the BOX dimension, whose records are one set
    per box rather than one set per pool -- so it meets a holder it never agreed on a name with.
    """
    directory = _tmp()
    with (
        BoxLock('the-run-under-the-shared-name', broker=Broker(resource_dir=directory)).held(),
        pytest.raises(Exhausted) as refusal,
        Broker(resource_dir=directory).admit(
            _A_POOL_OF_ITS_OWN,
            {BOX_SEATS.name: 1},
            what='the-run-with-its-own-pool',
        ),
    ):
        pytest.fail(
            f'{_A_POOL_OF_ITS_OWN} ran while the box was held. A private pool name bought an exemption, '
            f'which is the unifying-the-mechanism-without-the-path failure the rule names.'
        )
    assert 'the-run-under-the-shared-name' in holders_line(refusal.value)


def test_asking_who_holds_the_box_is_not_taking_it() -> None:
    """A reader that probes by acquiring becomes a WRITER of the state it reports."""
    broker = Broker(resource_dir=_tmp())
    assert BoxLock.holders(broker=broker) == (), 'a fresh records directory already named a holder'
    assert BoxLock.holders(broker=broker) == (), 'asking twice created a phantom holder'
    with BoxLock('the-run-that-really-takes-it', broker=broker).held():
        held = BoxLock.holders(broker=broker)
    assert [holder.what for holder in held] == ['the-run-that-really-takes-it'], (
        f'the read reported {[holder.what for holder in held]} while exactly one run held the box. A reader that '
        f'cannot see a real holder is why a second run would start.'
    )
    assert BoxLock.holders(broker=broker) == (), 'the holder outlived the block that took it'


def test_the_rendezvous_is_outside_every_repository() -> None:
    """A lock named INSIDE one tree is one a sibling checkout cannot find, so it excludes nobody."""
    directory = BoxLock.resource_dir()
    assert not directory.is_relative_to(_ROOT), (
        f'the box records live at {directory}, inside this repo. A rendezvous a sibling repo cannot reach is '
        f'a lock only this repo takes -- the tax the rule exists to remove.'
    )


def test_the_wait_is_bounded_and_narrates_while_it_queues() -> None:
    """Queue on a ceiling and SAY who you are waiting for: the fourth of four answers.

    A ratio rather than two absolute readings, because what makes a queue legible is how often it
    speaks RELATIVE to how long it may wait.
    """
    assert 0 < WAIT_S < float('inf'), f'the wait ceiling is {WAIT_S}; blocking forever renders a crash as a hang'
    assert 0 < PROGRESS_S <= WAIT_S * PROGRESS_SHARE_CEILING, (
        f'the queue speaks every {PROGRESS_S}s against a {WAIT_S}s ceiling. A silent queue and a hung process '
        f'look identical from a terminal.'
    )
    assert inspect.signature(hold_the_box).parameters.keys() >= {'stack', 'what', 'wait_s', 'poll_s'}, (
        'the shared wait stopped taking the ceiling as an argument, so a caller can no longer bound it.'
    )


def test_the_box_pool_is_a_name_rather_than_a_measurement() -> None:
    """FLOOR ON THE READING. An empty pool name would make every plant above address nothing."""
    assert BOX_POOL.strip(), 'the records pool has no name, so nothing above wrote where it claimed to'
    assert BOX_SEATS.name.strip(), 'the box-wide dimension has no name, so the seat cannot be demanded'
