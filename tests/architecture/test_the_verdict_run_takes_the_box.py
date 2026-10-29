"""ONE-BOX-ONE-LOCK, read off THIS repo's verdict entry point.

THE OLD REASON: optimi-lab has "no runner to queue on a box lock", so the rule was declared to have
no subject here. That was a statement about a RUNNER, and it was false: the one verdict command this
repo uses -- `.venv/Scripts/python.exe -m dev_lab.dev.verify` -- takes the box for its run. A
repo whose only verdict command takes the box IS a party to the rendezvous.

THE FAMILY BODY THAT ASSERTED THE SEAT WAS RETIRED UPSTREAM (lab-commons, 2026-10-13) together with
the lock and wait modules it drove: the machine-wide claim is now `lab_commons.box`, whose ledger the
kit's own suite plants contention on, and `verify` runs its pytest step through
`dev_lab.dev.verifybox.boxed`. What stays this repo's is the reading -- the entry point it cites
must still route the run through the box -- the same shape the sibling lab adopted. The check is by AST over
the installed module source, so a docstring naming `boxed` does not satisfy it, and a planted source
without the call must convict.
"""

from __future__ import annotations

import ast
import inspect

from dev_lab.dev import verify


def _calls_boxed(source: str) -> bool:
    """True when *source* calls ``boxed`` imported from ``dev_lab.dev.verifybox``."""
    tree = ast.parse(source)
    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == 'dev_lab.dev.verifybox'
        and any(alias.name == 'boxed' for alias in node.names)
        for node in ast.walk(tree)
    )
    called = any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'boxed'
        for node in ast.walk(tree)
    )
    return imported and called


def test_the_verdict_entry_point_runs_in_the_box() -> None:
    """THE PROPERTY: the verdict command this repo cites routes its run through the box."""
    assert _calls_boxed(inspect.getsource(verify)), (
        'dev_lab.dev.verify no longer calls verifybox.boxed: the verdict run left the box'
    )


def test_a_planted_entry_point_outside_the_box_convicts() -> None:
    """PLANTED CONTROL: naming `boxed` in a docstring and importing it is not calling it."""
    planted = (
        '"""boxed is only named here."""\nfrom dev_lab.dev.verifybox import boxed\ndef run_verify():\n    return 0\n'
    )
    assert not _calls_boxed(planted)
