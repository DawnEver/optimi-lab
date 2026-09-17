"""The always-loaded rule pages may only SHRINK, and their SUM is capped -- this repo's answers only.

`.claude/rules/*.md` is re-read on every turn, so every line in it is a cost paid on every
request. Nothing measured that cost in this repo until 2026-09-15, which was the whole of the
DOCS-SPLIT gap: the two pages existed and could grow without limit, and mechanism could migrate
into them from the module docstrings that own it.

THE MECHANISM IS NO LONGER HERE. `lab_commons.dev.famtests.rulespages` owns the reading, the
four-movement comparison, the total budget and the vacuity floor; this file declares the four facts
that are about optimi-lab and nothing else -- WHICH FILES ARE PAGES, what each is pinned at, what
their sum may be, and how small a scan may get before it is refused. The kit takes every one of
them as a keyword with NO DEFAULT, precisely so that a repo with no measurement cannot be handed
another repo's and have it reported back as measured.

THE ADOPTION ADDED AN ARM THIS FILE NEVER HAD, and it is the reason the swap is not a wash. A
per-page pin is a LOCAL decision: each of these two numbers is separately defensible and NOTHING
capped their sum, so a third and fourth page arriving at 30 lines each would be four defensible
pins and one 115-line document nobody sized. `CEILING` is that missing half. It is the
`DEBT`/`DEBT_CEILING` shape this repo already uses elsewhere -- the names are the population, the
sum is the budget.

WHAT WENT UPSTREAM WITH THE MECHANISM: both of this file's controls. The four planted movements and
the planted-page scan are `tests/test_famtests_rulespages.py` in lab-commons, driven against the
same functions this file calls, so keeping copies here would be duplicate evidence that cannot
diverge from its original.

THE ADMISSION CRITERION for a line in an always-loaded file, and the way past a pin is never to
raise a number:

1. **NOT ENFORCED.** If a test already refuses it, the prose goes and the test stays. Every guard
   in `tests/architecture/` is a line these pages do not have to carry.
2. **NOT DERIVABLE** from another line in the set -- the pages are one document for deduplication.
3. **NOT LOCAL.** A rule governing one module belongs in that module's docstring, where the code
   consults it. A declaration next to its enforcement cannot drift; prose in a rule page can.
4. **COSTLY IF WRONG.** Without the line the DEFAULT behaviour is wrong, not merely suboptimal.

WIDTH IS SOMEBODY ELSE'S JOB, and deliberately so: `test_injected_doc_width_ceiling.py` scans this
exact corpus through `lab_commons.dev.docwidth`, with its own ceiling, named set and floor. A count
pin is one unit per line however long the line is, so the two pins measure different dimensions --
but measuring width twice would be two statements of one rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.famtests import rulespages

_ROOT: Final = Path(__file__).resolve().parents[2]
_RULES: Final = _ROOT / '.claude' / 'rules'


#: WHICH FILES ARE PAGES, and it is a repo decision rather than an obvious one. This repo WALKS the
#: directory: the cost `.claude/rules/` imposes is paid by whatever is on disk when the turn starts,
#: so an untracked page is a page. lab-commons takes its corpus from `git ls-files` instead, on the
#: opposite and equally good ground that what is not tracked is not the fleet's. Neither default
#: would be right for the other, which is why the kit has none.
def _pages() -> list[Path]:
    """THE CORPUS: every markdown file under `.claude/rules/`, recursively, sorted."""
    return sorted(_RULES.rglob('*.md'))


#: Every always-loaded page, to its line count RE-MEASURED 2026-09-18 and unchanged since
#: 2026-09-15. Each number RATCHETS DOWN: lower it in the commit that shrinks the page, never raise
#: it. Raising one is a claim that a new rule could not be expressed inside the budget AND that no
#: existing line fails a criterion above, and that claim belongs in the commit message where a
#: reader can refuse it.
PINNED: Final[dict[str, int]] = {
    # The domain facts no test catches: what this library refuses to become. 33 lines, and JUDGED
    # rather than recorded -- every line is a refusal the code cannot make for itself, and the
    # longest section is the reference-ladder ruling that has already been mis-applied once.
    '.claude/rules/invariant.md': 33,
    # How work gets done here. 22 lines, and the tightest page in the family at that size.
    '.claude/rules/workflow.md': 22,
}

#: THE TOTAL BUDGET, and it is pinned AT the measurement with ZERO headroom -- 33 + 22 = 55,
#: re-measured 2026-09-18. Headroom is the one thing this arm must not have. Slack in a total is
#: budget nobody argued for, spent by whoever arrives next, and handing it out in advance is the
#: exact failure the ceiling exists to refuse; a zero-headroom ceiling says instead that a new page
#: is paid for out of an existing one. The two arms are then never in tension, because the per-page
#: ratchet already drives a pin DOWN in the edit that shrinks its page, so the sum tracks the pins
#: exactly and the ceiling binds at precisely one moment: when somebody wants a line back.
CEILING: Final = 55

#: A floor on the scan: below this the walk did not reach `.claude/rules/` at all, and an empty page
#: set is indistinguishable from a compressed one. Pinned AT the measured 2 -- this corpus can only
#: grow, so a page that stops being seen is a defect rather than noise.
PAGE_FLOOR: Final = 2


def test_the_rule_pages_hold_their_measured_budget_per_page_and_in_total() -> None:
    """THE CHECK: the floor first, then the named set, then the sum -- the kit orders them.

    The floor comes first because a ratchet over a directory the walk did not enter reports exactly
    what a compressed one reports, so the two later verdicts are only given once the reading is
    known to be a reading.
    """
    measured = rulespages.assert_rules_ratchet(
        pages=_pages(),
        root=_ROOT,
        pinned=PINNED,
        ceiling=CEILING,
        floor=PAGE_FLOOR,
    )
    assert sum(measured.values()) == CEILING, (
        f'the pages sum to {sum(measured.values())} against a {CEILING} ceiling. This repo pins the '
        f'ceiling AT the measurement, so slack here is budget nobody decided to spend: lower '
        f'CEILING in the same edit that lowered the pin which freed it.'
    )
