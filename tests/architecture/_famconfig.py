r"""optimi-lab's DELTA against each family config base -- data, computed by nothing.

The bases live in :mod:`lab_commons.dev._famconfig_rows` and the renderer that turns base plus delta
into a file is :mod:`lab_commons.dev.famconfig`. What belongs HERE is only the part of each artefact
that is a fact about optimi-lab: a `usr/local/` tree, a viztracer `result.json`, a coredumpy
`dumps/`. Everything else is the family's and arrives from the kit, so a fix upstream lands here by
upgrading the dependency rather than by remembering to copy it.

WHY THE DELTA IS HAND-WRITTEN AND NOT `measured_delta` OUTPUT. `measured_delta` is the SURVEY half --
it reads the file back and reports what it would have to declare -- and it strips comments, because
it compares meaningful lines. Handing its result straight back as the declaration would delete every
comment in the artefact on the first render. So the survey SIZED this change (18 added / 2 dropped)
and a human wrote the lines.

THE ONE BEHAVIOUR CHANGE, STATED RATHER THAN SLIPPED IN AS FORMATTING. The base spells five patterns
with a trailing slash -- `**/__pycache__/`, `*.egg-info/`, `.mypy_cache/`, `.pytest_cache/`,
`.ruff_cache/` -- and this repo wrote the first two BARE. A trailing slash matches a DIRECTORY ONLY,
so adopting the base narrows those two rules: a FILE named `__pycache__` or ending in `.egg-info`
stops being ignored. MEASURED over this tree on 2026-09-17 with `git status --ignored` before and
after: the only line that moved was `.gitignore` itself, so no file's tracked-or-ignored status
changed, because no such file exists here. The bare spellings are deliberately NOT carried in the
delta -- the delta renders AFTER the base and gitignore is last-match-wins, so keeping them would
silently reinstate exactly the rule the base replaced.

WHAT THIS REPO DOES NOT HAVE, and it is worth one sentence because the sibling lab does. There is no
`**/.claude/**` block here and no re-inclusions, so the last-match-wins hazard that forces wdg-lab to
keep `__pycache__/` as its FINAL line does not arise: nothing in this delta re-includes anything.
That is a fact about this tree today rather than a guarantee, so the ordering property in the test
module is written to bite if a negation is ever added.

`.pre-commit-config.yaml` IS DECLARED UNADOPTABLE HERE -- see :data:`PRECOMMIT_BLOCKED`. It is the
third family base and this repo takes neither it nor a delta against it; the reason is a property of
the RENDERER rather than of this repo, and the test module keeps the refusal live so that the day it
is lifted, this row reds instead of resting.
"""

from __future__ import annotations

from typing import Final

from lab_commons.dev.famconfig import Delta

__all__ = [
    'DELTAS',
    'EXTRA_HOOK_IDS',
    'GITIGNORE_DELTA',
    'MAKEFILE_DELTA',
    'PRECOMMIT_BLOCKED',
    'REPO',
]

#: How this repo names itself in a rendered file's drop comments.
REPO: Final = 'optimi-lab'

#: 18 content lines were MEASURED by `famconfig.measured_delta`; 16 are declared, because the two
#: bare slash-less spellings are what the base REPLACES rather than what this repo adds.
GITIGNORE_DELTA: Final = Delta(
    repo=REPO,
    added=(
        '## Build and distribution',
        'build/',
        'dist/',
        '**/__version__.py',
        '',
        '## Development tools',
        '## Documentation (pdoc)',
        '**/docs/',
        '## viztracer',
        'result.json',
        '## coredumpy',
        'dumps/',
        '',
        '## Quality and testing',
        'report/',
        'htmlcov/',
        '# Logs written by the family verify entry point (`lab_commons.dev.verify`).',
        '.verify/',
        '',
        '## Runtime and environment',
        '**/log/*.png',
        '**/log/*.jpg',
        '!example.log',
        'usr/local/',
        'output/',
        '',
        '## System and others',
        '**/ignore/',
        '**/.DS_Store',
    ),
    dropped={},
    #: 29 declared lines against 16 measured content lines. The number is the point at which this
    #: stops being a delta: this tree is one Python package with no second language and no agent
    #: state to re-include, so a delta that doubled would be saying the base had stopped fitting.
    ceiling=32,
)

#: The `pre-commit-hooks` ids this repo runs BEYOND the family's eleven, MEASURED 2026-09-17 against
#: the live file. Held as data because it is the subject of :data:`PRECOMMIT_BLOCKED`: these seven
#: are the lines that have nowhere to go under an append-only delta.
EXTRA_HOOK_IDS: Final[tuple[str, ...]] = (
    'check-builtin-literals',
    'check-illegal-windows-names',
    'check-shebang-scripts-are-executable',
    'check-symlinks',
    'check-vcs-permalinks',
    'destroyed-symlinks',
    'fix-byte-order-marker',
    'requirements-txt-fixer',
)

#: WHY THIS REPO TAKES NO DELTA AGAINST `.pre-commit-config.yaml`, and it is a DROP with a reason
#: rather than an omission. MEASURED 2026-09-17 by driving `famconfig.delta_problems` with the real
#: base; the test module re-drives it, so this text cannot outlive the refusal it describes.
#:
#: TWO INDEPENDENT CAUSES, and either alone is enough:
#:
#: 1. A DELTA MAY ONLY APPEND. The eight ids in :data:`EXTRA_HOOK_IDS` belong INSIDE the
#:    `pre-commit-hooks` repo entry the base renders, and an append cannot reach into a rendered
#:    block.
#: 2. `delta_problems` REFUSES ANY DELTA LINE EQUAL TO A BASE LINE, comparing against `base.lines`
#:    rather than `base.content_lines`. In a `.gitignore` a line is a whole statement and that is
#:    exactly right. In YAML the structural lines repeat by construction: `    hooks:`,
#:    `    rev: v6.0.0` and the BLANK separator all recur in any second repo entry, so the workaround
#:    for cause 1 -- a second `pre-commit-hooks` entry, which `pre-commit validate-config` accepts as
#:    VALID, measured -- is refused too, and so is drop-then-redeclare, because a dropped line is
#:    still in `base.lines`.
#:
#: THE PIN PAIR IS A SEPARATE QUESTION AND IS NOT BLOCKED BY THIS. The base takes the newest measured
#: pins and this repo was behind on both; moving to them is its own commit with its own measurement,
#: because a pin bump hidden inside a config-unification change is the shape that makes the next
#: regression unattributable.
PRECOMMIT_BLOCKED: Final[dict[str, str]] = {
    '.pre-commit-config.yaml': (
        'the append-only delta cannot reach inside the rendered pre-commit-hooks entry, where this '
        "repo's eight extra stock ids live; and delta_problems compares against base.lines rather "
        'than base.content_lines, so the structural YAML lines a second repo entry needs are refused '
        'as re-statements. Both are renderer properties, not facts about optimi-lab. Adopt when '
        'lab_commons.dev.famconfig can express a nested addition.'
    ),
}

#: The Makefile base is REQUIRED rather than RENDERED -- target NAMES must be present and the recipes
#: are this repo's. So this repo adds no lines and drops none: MEASURED 2026-09-17, all nine base
#: lines are already satisfied here, including `verify:`'s recipe, which is the family's line
#: verbatim because this repo declares no `slow` marker to tier against.
MAKEFILE_DELTA: Final = Delta(repo=REPO, added=(), dropped={}, ceiling=0)

#: Every artefact this repo declares a delta against, by name. `.pre-commit-config.yaml` is absent
#: BY DECLARATION -- :data:`PRECOMMIT_BLOCKED` holds the reason and the test pins that the two sets
#: together cover every base the kit publishes, so a fourth base cannot arrive unnoticed.
DELTAS: Final[dict[str, Delta]] = {
    '.gitignore': GITIGNORE_DELTA,
    'Makefile': MAKEFILE_DELTA,
}
