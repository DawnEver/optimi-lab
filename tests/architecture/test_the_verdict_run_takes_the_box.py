"""ONE-BOX-ONE-LOCK in optimi-lab, and the reason on record was refuted by this repo's own entry point.

THE OLD REASON: optimi-lab has "no runner to queue on a box lock", so the rule was declared to have
no subject here. That was a statement about a RUNNER, and it was false about optimi-lab on the day
it was written in the only way that matters: the verdict entry point this repo actually uses --
`.venv/Scripts/python.exe -m lab_commons.dev.verify`, the one line the rules page permits -- calls
`lab_commons.dev.boxwait.hold_the_box` and takes the box's one seat for the whole of ruff plus
pytest. A repo whose only verdict command takes the box IS a party to the rendezvous; it does not
become one by acquiring a runner of its own.

WHY THAT MATTERS RATHER THAN BEING A BOOKKEEPING FIX. The rule's own sentence is that a lock only
one party takes is a TAX on whoever obeys it: the party that takes none runs, the party that takes
one is starved by what it cannot see, and both report clean. optimi-lab was on record as the party
that takes none while already taking it -- so what was missing was never the lock but any statement
that could NOTICE the lock being dropped.

THE ASSERTIONS ARE THE FAMILY'S AS OF 2026-09-19. `boxlock` published the EXCLUSION and `boxwait`
the bounded, narrated WAIT; what stayed forked was the eight-arm test over them, measured at 84.6%
identical to wdg-lab's twin, whose 15.4% was a docstring, ONE pool name, a `tempfile` prefix and
four assertion messages differing only in where the line wraps. `lab_commons.dev.famtests.boxseat`
owns the arms now. WHAT OPTIMI-LAB SUPPLIES, each a keyword with NO DEFAULT upstream:

* `ENTRY_POINT` -- the function optimi-lab really runs for a verdict. Passed rather than assumed,
  which is the behaviour the kit changed: both forks hard-coded `verify.run_verify`, so a repo with
  a runner of its own would have asserted about a function it does not call and read GREEN while its
  real entry point took nothing. optimi-lab has no runner of its own, so the value is unchanged --
  and it is now a value that CAN be wrong rather than a constant nobody could see.
* `A_POOL_OF_ITS_OWN` -- the private pool a run opting out would name. Planted so that it must NOT
  buy an exemption; per-repo because a shared spelling would be the two parties agreeing by accident.
* `_ROOT` -- the checkout the rendezvous must live OUTSIDE of.
* `PROGRESS_SHARE_CEILING` -- how long a queue may stay silent as a FRACTION of the wait it reports
  on. A ratio, because what makes a queue legible is how often it speaks RELATIVE to its ceiling,
  and an argument because a bar closed over in the kit is one repo's policy handed silently to three.

THE OTHER BEHAVIOUR THE KIT CHANGED, and it is the red worth having: both forks asserted
`signature(hold_the_box).parameters.keys() >= {...}`. A `>=` is satisfied by every LONGER signature,
so it could not see a parameter ADDED -- exactly the change that breaks the sibling repos' callers
silently. The kit compares `HOLD_THE_BOX_PARAMETERS` for EQUALITY, both directions red.

NOT PROVED, deliberately: that any particular run is holding the box right now. That is a fact about
a machine at an instant, and a test asserting it would be reading the box rather than this repo.
Every plant below runs in an ISOLATED records directory, because this suite runs INSIDE a `verify`
run holding the real seat.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev import verify
from lab_commons.dev.famtests import boxseat

_ROOT: Final = Path(__file__).resolve().parents[2]

#: The verdict function optimi-lab actually runs -- `python -m lab_commons.dev.verify`, the one line
#: this repo's rules page permits. Named here rather than assumed upstream, so a repo that acquired a
#: runner of its own could not keep asserting about a function it had stopped calling.
ENTRY_POINT: Final = verify.run_verify

#: The longest a queue may stay silent, as a fraction of the wait it reports on. A progress line that
#: arrives once per ceiling is a silent queue with an epilogue.
PROGRESS_SHARE_CEILING: Final = 0.25

#: The pool name a run that opted out of the shared spelling would use. The point of planting it is
#: that it must NOT buy an exemption -- the box-scoped seat is not addressed by name.
A_POOL_OF_ITS_OWN: Final = 'optimi-lab-private-pool'


def test_this_repos_verdict_entry_point_takes_the_box() -> None:
    """THE REFUTATION OF THE OLD REASON, read off the code optimi-lab runs to get a verdict."""
    boxseat.assert_the_verdict_entry_point_takes_the_box(entry_point=ENTRY_POINT)


def test_a_second_run_is_refused_while_one_holds_the_box(tmp_path: Path) -> None:
    """PLANTED CONTENTION on the real `BoxLock`, and the refusal must NAME who is in the way."""
    boxseat.assert_a_second_run_is_refused_while_one_holds_the_box(records=tmp_path)


def test_the_seat_is_released_when_the_run_ends(tmp_path: Path) -> None:
    """THE OTHER SIDE. A seat nothing can ever take again is a hang, not an exclusion."""
    boxseat.assert_the_seat_is_released_when_the_run_ends(records=tmp_path)


def test_a_pool_name_of_its_own_does_not_opt_a_run_out_of_the_box(tmp_path: Path) -> None:
    """THE PATH, WHICH IS THE RULE'S ACTUAL CLAIM: agreeing on the mechanism alone measures nothing."""
    boxseat.assert_a_pool_name_of_its_own_does_not_opt_a_run_out(
        records=tmp_path,
        pool_name=A_POOL_OF_ITS_OWN,
    )


def test_asking_who_holds_the_box_is_not_taking_it(tmp_path: Path) -> None:
    """A reader that probes by acquiring becomes a WRITER of the state it reports."""
    boxseat.assert_asking_who_holds_the_box_is_not_taking_it(records=tmp_path)


def test_the_rendezvous_is_outside_every_repository() -> None:
    """A lock named INSIDE optimi-lab is one a sibling checkout cannot find, so it excludes nobody."""
    boxseat.assert_the_rendezvous_is_outside_this_repository(root=_ROOT)


def test_the_wait_is_bounded_and_narrates_while_it_queues() -> None:
    """Queue on a ceiling, SAY who you are waiting for, and pin the wait's parameters by EQUALITY."""
    boxseat.assert_the_wait_is_bounded_and_narrates(progress_share_ceiling=PROGRESS_SHARE_CEILING)


def test_the_box_pool_is_a_name_rather_than_a_measurement() -> None:
    """FLOOR ON THE READING. An empty pool name would make every plant above address nothing."""
    boxseat.assert_the_box_pool_is_a_name_rather_than_a_measurement()
