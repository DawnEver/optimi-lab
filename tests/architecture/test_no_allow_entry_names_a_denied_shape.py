r"""NO `permissions.allow` ENTRY MAY NAME A COMMAND THE DENY ENGINE REFUSES.

`.claude/settings.json` and `.claude/hooks/deny-rules.json` are two hand-written files that speak
about the same thing -- which commands an agent may issue -- and nothing made them agree. The hook
wins at runtime, so a contradiction is never a functional hazard; it is the dominant defect of this
family instead, A DECLARATION THAT LIES. An `allow` row publicly promises a road the engine refuses,
in a file that is committed and read on every box, and the reader who believes it spends its refusal
budget finding out.

MEASURED 2026-09-17 across the three repos that ship this engine, by driving the engine rather than
comparing patterns by eye: exactly one contradiction existed anywhere, `Bash(pytest *)` in wdg-lab
against `BARE-TEST-INVOCATION`, installed the same day. One row is the whole point -- a one-time
reconciliation of two hand-written files drifts again by next week, and the next drift is somebody
else's afternoon.

THE BODY IS THE FAMILY'S AS OF 2026-09-17, and the census that decided it is worth quoting: this
file and wdg-lab's twin were 324 lines together with 36 differing -- 88.9% identical, ZERO repo
nouns in code. The only lines that were ever this repo's are the sanctioned exit a red is redirected
to and the two planted rows.

THE FIX FOR A RED IS REDIRECTION, NOT DELETION. Ask what the row was trying to permit and permit
THAT: a bare test line becomes the repo's verdict command. An agent must always be left its own
door; sealing the road is the other way to make this file green and it is the wrong one. Delete a
row only when nothing sanctioned exists for its intent, and say so where the deletion lands.

HOW IT IS MEASURED. Every allow row is instantiated into a concrete command and fed to the REAL
engine through `agenthooks.decide`, which runs `deny-commands.js` under node against a real
`PreToolUse` payload and hands back the refusal reason. An allow GLOB against a deny REGEX by eye is
exactly the reasoning this family replaces with a measurement.

A HOLE ALL THREE COPIES HAD IS CLOSED HERE, AND CLOSING IT CAUGHT THIS REPO. Every copy read zero
contradictions over ZERO PROBED ROWS as agreement: a settings file with no `Bash(...)` row at all
made the scan trivially empty and the assertion trivially green, which is the strongest possible
agreement reported between two files, one of which was never consulted. The family `Scan` now
carries `probed`, and `VacuousAllowScan` refuses. That is also why the old `readable and wired` arm
is gone rather than ported: the property itself now refuses an unreadable settings file, a missing
rules file, an uninstalled guard AND an unasked question, each by name.

THE UNASKED QUESTION WAS ASKED, 2026-09-18. `.claude/settings.json` here carried NO `permissions`
block at all -- zero rows, zero probed -- so this guard had been green over an unasked question for
its whole life, unlike the sibling lab, which probes seven rows. It was held as a STRICT XFAIL
rather than a skip precisely so the absence would be a measurement somebody had to answer, and the
answer was the one the remedy above prescribes: ONE row, this repo's own sanctioned exit, the
verdict command that `BARE-TEST-INVOCATION` and `PUSH-NO-VERIFY` both redirect to. The mark is gone
in the same change that added the row, which is what the mark's own reason instructed.

WHY IT WAITED, AND WHY THAT IS NOT A TEST PROPERTY. Widening an agent's own permission file is a
human's call -- an agent that grants itself a road has not measured anything -- so the xfail stood
until the user ruled. The row is deliberately ONE: `wdg-lab` permits seven, and every one of those
is a road somebody argued for. An `allow` list is not a place to pre-pay for roads nobody has
needed yet, and the scan below is only as sharp as the list is honest.

WHAT THIS DOES NOT PROVE. That the agent client honours `permissions.allow` at all -- that is a fact
about a tool's configuration loading, not about this tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from lab_commons.dev.famtests import allowguard

_ROOT: Final = Path(__file__).resolve().parents[2]

#: The sanctioned exit a red is pointed at, and the one this repo actually ships.
_SANCTIONED: Final = './.venv/Scripts/python.exe -m lab_commons.dev.verify'

#: THE PLANTED PAIR, and both are this repo's answers rather than the family's: `deny_rules.py`
#: DROPS a rule whose remedy this repo lacks, so a row denied in one tree is permitted in another.
#: `Bash(pytest *)` is refused here by BARE-TEST-INVOCATION, which this repo ships; `Bash(echo *)`
#: is refused by nothing, which is what catches a matcher that started refusing everything.
_DENIED_ENTRY: Final = 'Bash(pytest *)'
_HARMLESS_ENTRY: Final = 'Bash(echo *)'


def test_no_allow_entry_names_a_command_the_engine_refuses() -> None:
    """THE PROPERTY, and its floors. A red names the row, what it promised, and the refusal it gets.

    Three states red separately: a settings or rules file that could not be read, a guard whose
    wiring is not installed (nothing would be in tension with the permissions block), and a scan
    that probed NOTHING -- an unasked question rather than agreement.
    """
    allowguard.assert_no_allow_contradicts(root=_ROOT, sanctioned=_SANCTIONED)


def test_the_scan_catches_a_planted_contradiction(tmp_path: Path) -> None:
    """THE PLANT, both directions, against THIS repo's own rendered rules.

    Without it this module passes exactly as well against an engine that stopped matching -- the
    state in which every real allow row reads as honest.
    """
    allowguard.assert_the_scan_can_still_see(
        root=_ROOT, denied_entry=_DENIED_ENTRY, harmless_entry=_HARMLESS_ENTRY, scratch=tmp_path
    )


@pytest.mark.parametrize(('entry', 'expected'), allowguard.GLOB_CASES)
def test_a_glob_is_instantiated_at_a_command_position(entry: str, expected: str | None) -> None:
    """The instantiation is the only step between the two files, so it is pinned rather than trusted.

    The cases are the FAMILY's: the `Bash(...)` glob spelling belongs to the agent client, not to any
    repository, and the three forked copies each pinned it with one repo-flavoured example that
    taught the next reader nothing.
    """
    assert allowguard.probe_command(entry) == expected
