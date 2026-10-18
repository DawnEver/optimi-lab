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
`**/.claude/**` block here and no SUBTREE re-inclusion, so the last-match-wins hazard does not arise.
THE QUALIFIER IS LOAD-BEARING AND THIS SENTENCE USED TO READ "no re-inclusions", WHICH IS FALSE: the
delta carries `!example.log`, and it has since the adoption. What makes it harmless is not its
absence but its SHAPE -- `reopenings` reports a negation only when it ends in a wildcard, because a
negation naming ONE FILE re-includes that path and nothing under it and so owes no closure, where a
subtree negation brings back every cache directory a floating base rule excludes at any depth. So the
correct claim is about the shape of this repo's one negation rather than about there being none, and
the blanket version was a declaration asserting a property the file does not have.

THE SHAPE OF THE NEGATION IS A FACT ABOUT THIS TREE TODAY rather than a guarantee, so the ordering
property in the test module is written to bite if a SUBTREE negation is ever added -- and since
2026-09-17 it bites through the kit rather than through anything written here, because
`assert_no_rule_is_reopened` refuses a base directory rule re-stated ABOVE a
subtree negation that re-includes it. The condition is BELOW EVERY RE-INCLUSION IT CLOSES; naming a
particular line as the FINAL one was a sufficient approximation of it that no repo needs any more.

`.pre-commit-config.yaml` IS NO LONGER AN ARTEFACT. Lab-commons 900c5bf retired the pre-commit
framework for `[tool.lab_commons.hooks]` and the kit publishes no YAML base, so its delta, the eight
extra stock-hook ids it anchored and the stage-narrowing pin left with the file (2026-10-18).
"""

from __future__ import annotations

from typing import Final

from lab_commons.dev.famconfig import Delta, SectionDelta

__all__ = [
    'DELTAS',
    'GITATTRIBUTES_DELTA',
    'GITIGNORE_DELTA',
    'MAKEFILE_DELTA',
    'PYPROJECT_SECTION_DELTAS',
    'REPO',
    'RGIGNORE_DELTA',
]

#: How this repo names itself in a rendered file's drop comments.
REPO: Final = 'optimi-lab'

#: What optimi-lab ignores BEYOND the family base, and nothing the base already holds. As of
#: lab-commons 9ce57c3 the base carries `**/__version__.py`, `htmlcov/`, `.verify/`, `output/`,
#: `**/ignore/`, `**/.DS_Store`, `/build/`, `/dist/`, `/dumps/`, `/report/` and the whole `.claude/`
#: section (which ignores `.claude/.rem-state.json`), so those lines left this delta in the same
#: upgrade: a restatement is refused at render time. What remains is a fact about this tree only.
GITIGNORE_DELTA: Final = Delta(
    repo=REPO,
    added=(
        '## Documentation (pdoc)',
        '**/docs/',
        '## viztracer',
        'result.json',
        '',
        '## Runtime and environment',
        '**/log/*.png',
        '**/log/*.jpg',
        '!example.log',
        'usr/local/',
    ),
    dropped={},
    #: Ten declared lines, and the number is the measurement rather than headroom: the next pattern
    #: this repo needs raises it in the same edit and says what it is for.
    ceiling=10,
)

#: The Makefile base is REQUIRED rather than RENDERED -- target NAMES must be present and the recipes
#: are this repo's. So this repo adds no lines and drops none: MEASURED 2026-09-17, all nine base
#: lines are already satisfied here, including `verify:`'s recipe, which is the family's line
#: verbatim because this repo declares no `slow` marker to tier against.
MAKEFILE_DELTA: Final = Delta(repo=REPO, added=(), dropped={}, ceiling=0)

#: The `.gitattributes` base (lab-commons 9ce57c3) carries the family's one line; this repo adds
#: nothing, so the delta is empty and its ceiling is zero.
GITATTRIBUTES_DELTA: Final = Delta(repo=REPO, added=(), dropped={}, ceiling=0)

#: Every artefact this repo declares a delta against, by name. The test pins that this set covers the
#: kit's bases EXACTLY, so a new base cannot arrive unnoticed.
#: The `.rgignore` base (lab-commons 9a5adc9) skips `attic/`; this repo adds nothing.
RGIGNORE_DELTA: Final = Delta(repo=REPO, added=(), dropped={}, ceiling=0)

DELTAS: Final[dict[str, Delta]] = {
    '.gitattributes': GITATTRIBUTES_DELTA,
    '.gitignore': GITIGNORE_DELTA,
    'Makefile': MAKEFILE_DELTA,
    '.rgignore': RGIGNORE_DELTA,
}

#: This repo's delta against each owned `pyproject.toml` table -- the kit's SECTION base, judged by
#: `famconfig.inspect_section`. MEASURED 2026-10-02: ``readme`` and ``dynamic`` are the base's
#: verbatim, and ``testpaths`` adds ONE path, ``src``, so ``--doctest-modules`` collects the package's
#: doctests. It names this tree, which is why it is a delta and not a base line; the ceiling sits ON
#: the measurement. This delta used to live in lab-commons under a neutral name; a fact about this
#: tree belongs in this tree.
PYPROJECT_SECTION_DELTAS: Final[dict[str, SectionDelta]] = {
    '[project]': SectionDelta(repo=REPO, added={}, dropped={}, ceiling=0),
    '[tool.pytest.ini_options]': SectionDelta(repo=REPO, added={'testpaths': ('src',)}, dropped={}, ceiling=1),
}
