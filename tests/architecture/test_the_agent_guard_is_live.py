r"""AGENT-GUARD, for THIS checkout -- and the reason on record was true when written and is not now.

MEASURED 2026-09-17, both halves. Before today this repo had NO `.claude/settings.json`, no
`PreToolUse` matcher and no engine: `lab_commons.dev.agent_guard --repo .` answered NOTHING
DECLARED, exit 2. The adoption file's reason for the gap named the missing piece exactly -- "the
shared deny REGISTRY exists but its ENGINE does not: the engine the registry's own recipe points at
is a `.js` file in another repo" -- and that is what changed underneath it. The engine is now
family code (`lab_commons.dev.agenthooks`, installed by `lab_commons.dev.agent_guard`), so the last
thing keeping this row open is gone and the row closes.

WHY THIS FILE IS THE POINT OF THE EXERCISE rather than a formality on top of it. An installed guard
with no test is one `settings.json` edit away from being absent again, and it goes absent SILENTLY:
nothing about a repo whose hook stopped running looks different from a repo whose hook is passing
everything. That is not hypothetical here -- it is precisely how this repo reached 19 declared
pre-commit hooks with zero installed, which
`tests/architecture/test_the_declared_hooks_are_installed.py` was written to end. This file is the
same shape one layer up, over the hook that governs an AGENT's commands rather than a human's.

THE BODY IS THE FAMILY'S AS OF 2026-09-17, AND THE SIBLING LAB'S STRUCTURE IS THE ONE IT TOOK. The
seven arms below are `lab_commons.dev.famtests.agentguard`; what stayed here is the answers no other
checkout can give -- which declaration file this repo renders from, which rules it may honestly
ship, which commands it refuses, and which exit it must leave open.

THE DENY ARM NOW NAMES THE RULE IT EXPECTS, WHICH IS A STRENGTHENING THIS REPO DID NOT HAVE. Asking
only "was it refused" cannot tell a working rule from one a neighbour is covering for: upstream
plants two real rules whose patterns both match `git push --force`, and with the wrong one first the
weak question answers yes while the rule under test has stopped firing entirely. That is the
count-pin failure in a different spelling, and it is measured rather than argued -- the sibling lab
learned it the day `GIT-NETWORK-VERB` shipped and shadowed both of its push rules.

WHAT IS PROVEN, in arms that fail for different reasons, because a single "is it live" assertion
cannot tell a missing engine from a stale rule file:

1. THE GUARD IS LIVE. `guard_installation` reports GUARDED -- engine present AND lab-commons-shipped
   (it reads the provenance stamp, so a hand-copied file in the same slot is FOREIGN, not
   installed), rules readable, and a `PreToolUse` Bash hook whose command is the engine plus the
   rules file. The failure message carries the per-part report, so a red says WHICH part went.
2. THE COMMITTED RULES ARE THE RENDERED RULES. The JSON in the tree is compared byte-for-byte with
   `render(ADOPTION)`. Without this, the declaration in `scripts/deny_rules.py` and the file the
   engine actually reads are free to drift, and the declaration is the half a reader trusts.
3. EVERY RULE IS ACCOUNTED FOR AND EVERY REMEDY EXISTS. `assert_shippable` refuses a rule this repo
   neither remedies nor declares absent -- the silent gap, which looks identical to a rule nobody
   has read -- and refuses a remedy naming an untracked file. It is driven against the REAL tracked
   set, so "the exit exists" is a claim git answered.
4. THE GUARD ACTUALLY REFUSES, AND ACTUALLY ALLOWS. Engine plus rules are run under node against a
   real `PreToolUse` payload. A hook nobody has seen deny is not yet a hook -- and the other side
   matters just as much, because a rule that denied everything would pass arm 1 and arm 4's first
   half while making the repo unusable. THE SANCTIONED ROW IS THIS REPO'S OWN VERDICT COMMAND: if
   shipping BARE-TEST-INVOCATION ever sealed the only road out of it, this arm is what says so.

THE FLOOR, and it is the arm that keeps the rest from being vacuous: a rules file that rendered
EMPTY would satisfy "committed equals rendered" and "nothing is denied that should not be". So the
shipped set is pinned BY NAME, never by a count -- a count cannot say which rule came off, and the
honest-looking repair when a count disagrees is to edit the digit.

WHAT THIS DOES NOT PROVE. That the agent's client reads this `settings.json` at all. That is a fact
about a tool's configuration loading, not about this tree, and claiming it here would be the
declaration-lie the guard exists to remove.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import pytest
from lab_commons.dev.famtests import agentguard

_ROOT: Final = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(_ROOT / 'scripts'))
from deny_rules import ADOPTION  # noqa: E402

#: The rules this repo ships, BY NAME. Two carry `./.venv/Scripts/python.exe -m
#: lab_commons.dev.verify` as their exit; the other four need nothing from a repo and ship
#: everywhere. `GIT-NETWORK-VERB` and `RAW-PROCESS-KILL` are absent ON RECORD -- see
#: `scripts/deny_rules.py` for the measurement behind each.
#:
#: `GIT-COMMIT-AMEND` ARRIVED WITH THE KIT ON 2026-09-18 and is the case this pin was written for.
#: It is a FAMILY rule this repo neither authored nor configures, so it reached the committed JSON
#: purely by upgrading `lab-commons` -- the one route by which the file the engine obeys can change
#: with nothing in this tree edited. A COUNT PIN WOULD HAVE GONE FROM 5 TO 6 AND SAID NOTHING ABOUT
#: WHICH ROW MOVED, and the honest-looking repair would have been to edit the digit; the named set
#: forced the arrival to be read and typed out. Its subject is `git commit --amend` on a shared
#: lane, which this checkout has no single-agent mode to exempt it from.
_SHIPPED: Final = frozenset({
    'BARE-TEST-INVOCATION',
    'GIT-COMMIT-AMEND',
    'GIT-STASH',
    'PUSH-FORCE',
    'PUSH-NO-VERIFY',
    'WORKTREE-BASE-IS-EXPLICIT',
})

#: A heredoc body written to a FILE is data, not a command, even when it spells a denied shape.
#: Built here rather than inline so the row below stays readable.
_PROSE_ABOUT_A_DENIED_SHAPE: Final = "cat > notes.md <<'MD'\npytest is denied here; use the verdict entry point.\nMD"


def test_node_is_on_path() -> None:
    """NOT A SKIP, AND THAT IS THE POINT. No node means no guard, on this box, for every repo.

    The obvious spelling is `skipif(node is None)`, and it is the wrong one twice over. This repo
    forbids a skip outright -- `test_no_test_is_skipped` refuses one -- and the reason generalises
    exactly here: the condition being skipped on is not "this check does not apply", it is "the
    thing under test cannot run at all". A suite that goes quiet precisely when the engine is
    unrunnable reports green for the state it exists to detect.
    """
    agentguard.assert_node_is_available()


def test_the_guard_is_live() -> None:
    """Engine, rules and wiring are all the installed ones -- and a red names the part that is not."""
    agentguard.assert_the_guard_is_live(root=_ROOT)


def test_the_committed_rules_are_the_rendered_rules() -> None:
    """The file the engine reads IS the declaration in `scripts/deny_rules.py`, byte for byte."""
    agentguard.assert_committed_rules_are_rendered(root=_ROOT, adoption=ADOPTION)


def test_every_rule_is_accounted_for_and_every_remedy_exists() -> None:
    """No silently-dropped rule, no typo'd ID, and no exit naming a file this tree does not track."""
    agentguard.assert_every_rule_is_accounted_for(root=_ROOT, adoption=ADOPTION)


def test_the_shipped_set_is_pinned_by_name() -> None:
    """THE FLOOR. An empty rules file would satisfy every other arm in this module."""
    agentguard.assert_the_shipped_set_is_pinned(root=_ROOT, declared=_SHIPPED)


@pytest.mark.parametrize(
    ('command', 'rule'),
    [
        ('pytest tests/architecture', 'BARE-TEST-INVOCATION'),
        ('python -m pytest -q', 'BARE-TEST-INVOCATION'),
        ('git stash', 'GIT-STASH'),
        # THESE TWO ARE WHERE NAMING THE RULE EARNS ITS KEEP. This repo does NOT ship
        # GIT-NETWORK-VERB, so nothing here shadows the two push rules today -- and the day it does,
        # the sibling lab's measurement says these rows would keep passing under the weaker
        # question while PUSH-FORCE and PUSH-NO-VERIFY stopped firing.
        ('git push origin main --force-with-lease', 'PUSH-FORCE'),
        ('git push --no-verify', 'PUSH-NO-VERIFY'),
        ('git worktree add ../scratch', 'WORKTREE-BASE-IS-EXPLICIT'),
    ],
)
def test_a_forbidden_shape_is_refused_by_the_rule_that_owns_it(command: str, rule: str) -> None:
    """A hook nobody has seen deny is not yet a hook -- and a refusal with no exit is a sealed road."""
    agentguard.assert_refused_by(root=_ROOT, command=command, rule=rule)


@pytest.mark.parametrize(
    'command',
    [
        # THIS REPO'S OWN VERDICT COMMAND, and the row that matters most: it is the exit
        # BARE-TEST-INVOCATION and PUSH-NO-VERIFY both name. A guard that refused it would have
        # sealed the only road out of itself.
        './.venv/Scripts/python.exe -m lab_commons.dev.verify',
        'git push origin main',
        'git worktree add --detach ../scratch 1a2b3c4',
        _PROSE_ABOUT_A_DENIED_SHAPE,
    ],
)
def test_a_sanctioned_shape_is_allowed(command: str) -> None:
    """The other side of the ratchet: a guard that refuses everything passes every arm above."""
    agentguard.assert_sanctioned(root=_ROOT, command=command)
