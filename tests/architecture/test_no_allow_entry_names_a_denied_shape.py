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

THE FIX FOR A RED IS REDIRECTION, NOT DELETION. Ask what the row was trying to permit and permit
THAT: a bare test line becomes the repo's verdict command. An agent must always be left its own
door; sealing the road is the other way to make this file green and it is the wrong one. Delete a
row only when nothing sanctioned exists for its intent, and say so where the deletion lands.

HOW IT IS MEASURED. Every allow row is instantiated into a concrete command and fed to the REAL
engine through `agenthooks.decide`, which runs `deny-commands.js` and hands back the refusal reason.
An allow GLOB against a deny REGEX by eye is exactly the reasoning this family replaces with a
measurement.

THE FLOOR, because a scan that finds nothing is vacuous rather than green:
- `test_the_settings_file_is_readable_and_wired` -- an absent, unparseable or unwired settings file
  would make the scan trivially empty, and that must read RED rather than green.
- `test_the_scan_catches_a_planted_contradiction` -- THE PLANT. A temp settings file carrying a
  deliberately forbidden allow row must come back as a contradiction, and a harmless row next to it
  must not. Without it this module passes equally well against a matcher that stopped matching.

WHAT THIS DOES NOT PROVE. That the agent client honours `permissions.allow` at all -- that is a fact
about a tool's configuration loading, not about this tree.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final

import pytest
from lab_commons.dev import agenthooks

_ROOT: Final = Path(__file__).resolve().parents[2]

_SETTINGS: Final = _ROOT / '.claude' / 'settings.json'
_RULES: Final = _ROOT / '.claude' / 'hooks' / 'deny-rules.json'

#: The sanctioned exit a red is pointed at, and the one this repo actually ships.
_SANCTIONED: Final = './.venv/Scripts/python.exe -m lab_commons.dev.verify'


def _probe_command(entry: str) -> str | None:
    """Turn one `permissions.allow` row into the concrete command it promises, or None if not Bash.

    A row is a GLOB (`Bash(pytest *)`) and the engine reads TEXT, so the glob is instantiated.
    Leading and trailing `*` are DROPPED rather than substituted -- that is the most permissive
    reading of the row, and it puts the named program at a command POSITION where a
    `matches: command` rule can see it, with no trailing noise a rule might anchor against. An
    INTERIOR `*` becomes one opaque WORD, which is what it stands for.
    """
    match = re.fullmatch(r'Bash\((.*)\)', entry.strip(), flags=re.DOTALL)
    if match is None:
        return None
    return ' '.join(re.sub(r'\*+', ' ARG ', match.group(1).strip().strip('*')).split())


def _allow_entries(settings: Path) -> list[str]:
    return list(json.loads(settings.read_text(encoding='utf-8')).get('permissions', {}).get('allow', []))


def _contradictions(settings: Path) -> dict[str, str]:
    """Every allow row whose own promise the engine refuses, mapped to the refusal it gets."""
    found: dict[str, str] = {}
    for entry in _allow_entries(settings):
        command = _probe_command(entry)
        if not command:
            continue
        reason = agenthooks.decide(command, _RULES, cwd=_ROOT)
        if reason is not None:
            found[entry] = reason
    return found


def test_the_settings_file_is_readable_and_wired() -> None:
    """THE FLOOR. An absent or unwired settings file would make every other arm trivially empty."""
    assert _SETTINGS.is_file(), (
        f'{_SETTINGS} does not exist, so the allow list this module judges is not there to judge'
    )
    assert _RULES.is_file(), f'{_RULES} does not exist, so there is no engine for an allow row to contradict'
    doc = json.loads(_SETTINGS.read_text(encoding='utf-8'))
    wiring = [
        hook.get('command', '')
        for block in doc.get('hooks', {}).get('PreToolUse', [])
        if block.get('matcher') == 'Bash'
        for hook in block.get('hooks', [])
    ]
    assert any('deny-rules.json' in command for command in wiring), (
        f'no PreToolUse Bash hook in {_SETTINGS} names the rules file, so nothing in this repo is in '
        f'tension with the permissions block -- and the guard this module sits beside is gone.'
    )


def test_no_allow_entry_names_a_command_the_engine_refuses() -> None:
    """The assertion. A red names the row, what it promised, and the refusal it actually gets."""
    found = _contradictions(_SETTINGS)
    report = '\n'.join(
        f'  {entry}  ->  promises {_probe_command(entry)!r}, engine says: {reason.splitlines()[0]}'
        for entry, reason in found.items()
    )
    assert not found, (
        f'{len(found)} `permissions.allow` row(s) in {_SETTINGS} name a command the deny engine refuses:\n'
        f'{report}\n'
        f'The hook wins at runtime, so this is a DECLARATION THAT LIES rather than an open road. Fix it by '
        f'REDIRECTION: ask what the row was trying to permit and permit the sanctioned spelling for that '
        f'intent (for a test line, {_SANCTIONED}). Delete a row only when nothing sanctioned exists.'
    )


def test_the_scan_catches_a_planted_contradiction(tmp_path: Path) -> None:
    """THE PLANT. Without it, a matcher that stopped matching would pass this module unchanged."""
    planted = tmp_path / 'settings.json'
    planted.write_text(json.dumps({'permissions': {'allow': ['Bash(pytest *)', 'Bash(echo *)']}}), encoding='utf-8')
    found = _contradictions(planted)
    assert 'Bash(pytest *)' in found, (
        'a deliberately contradictory allow row went UNDETECTED, so a green from this module proves '
        'nothing about the real settings file. The engine, the glob instantiation, or the rule that '
        'refuses a bare test line has moved.'
    )
    assert 'Bash(echo *)' not in found, (
        'a harmless allow row was reported as a contradiction, which would push a reader to rewrite or '
        'delete a row that promises nothing forbidden'
    )


@pytest.mark.parametrize(
    ('entry', 'expected'),
    [
        ('Bash(pytest *)', 'pytest'),
        ('Bash(python *kill-server.py*)', 'python ARG kill-server.py'),
        ('Read(**)', None),
    ],
)
def test_a_glob_is_instantiated_at_a_command_position(entry: str, expected: str | None) -> None:
    """The instantiation is the only step between the two files, so it is pinned rather than trusted."""
    assert _probe_command(entry) == expected
