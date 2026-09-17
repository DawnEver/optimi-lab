"""ONE REPO'S HALF of the shared agent deny registry: the exits optimi-lab offers, and the gaps.

WHY A FILE HERE AT ALL. The STATEMENT of every denied shape is authored once, in
`lab_commons.dev.hooks` -- that half is universal and this repo has no business restating it. What
is NOT universal is the REMEDY: a refusal is only legitimate if the agent it refuses can be told
what to type instead, and the command to type is a fact about one tree. `lab_commons.dev.verify`
resolves here; `scripts/hooks/with-retry.sh` does not. So the two halves have different owners, and
this module is the local one.

THE RULE THAT DECIDES THE CONTENT, and it is the whole reason this file is hand-written rather than
a default set installed by the engine's installer: A RULE WHOSE EXIT DOES NOT EXIST HERE IS NOT
SHIPPED HERE. An agent refused by a rule whose remedy names a file this checkout does not have is
left with disobey or stop, and the first is what actually happens -- so the refusal has bought
nothing and cost the guard its authority. `lab_commons.dev.hook_adoption.deny_rules` DROPS such a
rule rather than softening its reason, and `assert_shippable` refuses a remedy whose file the tree
does not track.

WHAT THIS REPO SHIPS, and why each one is honest here:

* BARE-TEST-INVOCATION and PUSH-NO-VERIFY both need a VERDICT ENTRY POINT, and this repo has
  exactly one -- `python -m lab_commons.dev.verify`, the command that holds the box seat across
  ruff and pytest and writes the `.verify/verify-*.log` a later `env=` comparison cites. It is a
  module in a declared dev dependency rather than a file in this tree, so the remedy carries NO
  `path`: there is nothing for `remedy_gaps` to resolve, and claiming a path that does not exist
  would be the sealed road wearing a signpost.
* GIT-STASH, PUSH-FORCE and WORKTREE-BASE-IS-EXPLICIT need NOTHING from a repo -- their remedy is a
  plain git verb, which every checkout on earth can type. They ship unconditionally and are not
  listed below; `needing()` is what draws that line, and listing them would be bookkeeping with one
  right answer.

WHAT THIS REPO DECLARES ABSENT, on record rather than silently, each measured today:

* GIT-NETWORK-VERB needs a RETRY WRAPPER. There is none in this tree and none in `lab_commons.dev`
  -- the same measurement that keeps NETWORK-RETRY-THEN-REPORT in the adoption file's `_ABSENT`.
  A rule pointing at a wrapper that cannot be run is the exact defect this registry removes.
* RAW-PROCESS-KILL needs a PROCESS-TREE KILLER an agent can TYPE. `lab_commons.dev.bounded` has the
  capability -- `reap_tree` and `kill_process_tree`, both already driven by
  `tests/architecture/test_a_bounded_wait_names_its_remedy.py` -- but it exposes NO command-line
  entry point, so there is no line to put in a refusal. The gap is a CLI, not a capability, and
  saying so is more useful than shipping a rule whose reason would have to trail off.

Usage: ``python scripts/deny_rules.py`` rewrites `.claude/hooks/deny-rules.json` from the registry.
`tests/architecture/test_the_agent_guard_is_live.py` is what refuses the drift between the two.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.hook_adoption import HookAdoption, render
from lab_commons.dev.hooks import Remedy
from lab_commons.log import emit

__all__ = ['ADOPTION', 'RULES_FILE', 'VERIFY', 'main', 'repo_root']

#: The one verdict command this repo has. No `path`: it is a module in a declared dev dependency,
#: not a file in this tree, and a remedy may only claim a path it can actually resolve.
VERIFY: Final = Remedy('verdict-entry-point', './.venv/Scripts/python.exe -m lab_commons.dev.verify')

#: Where the engine reads its rules from, as `lab_commons.dev.agent_guard` spells it.
RULES_FILE: Final = Path('.claude') / 'hooks' / 'deny-rules.json'

ADOPTION: Final = HookAdoption(
    app_name='optimi_lab',
    remedies={'BARE-TEST-INVOCATION': VERIFY, 'PUSH-NO-VERIFY': VERIFY},
    declared_absent=frozenset({'GIT-NETWORK-VERB', 'RAW-PROCESS-KILL'}),
)


def repo_root() -> Path:
    """The root of THIS checkout, from this file's own location rather than the caller's cwd."""
    return Path(__file__).resolve().parents[1]


def main() -> int:
    """Render this repo's shippable rules over the committed file, reporting what it wrote."""
    target = repo_root() / RULES_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(ADOPTION), encoding='utf-8')
    emit(f'wrote {target}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
