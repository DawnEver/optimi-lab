"""PYPROJECT-SECTIONS-ARE-OWNED, for optimi_lab: the kit's section base plus this repo's delta.

`dev_lab.dev.famconfig` owns two `pyproject.toml` tables as a SECTION base -- ``[project]``'s
``readme``/``dynamic`` and ``[tool.pytest.ini_options]``'s ``testpaths`` -- and `inspect_section` reads
the real file against the live base and a repo's declared delta. The delta is
`_famconfig.PYPROJECT_SECTION_DELTAS`; this file is the verdict half.

BOTH SIDES OF THE RATCHET: an entry the file holds that no declaration produced is an undeclared local
waiver, and a base entry gone with no declared drop is the base leaving through the file.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Final

from _famconfig import PYPROJECT_SECTION_DELTAS, REPO
from dev_lab.dev.famconfig import OWNED, PYPROJECT_SECTIONS, inspect_section, section_base, section_problems

_ROOT: Final = Path(__file__).resolve().parents[2]


def _pending() -> set[str]:
    """`[tool.dev_lab.pyproject] pending`: family sections not adopted yet (user ruling 2026-11-24), unjudged."""
    data = tomllib.loads((_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    return set(data['tool']['dev_lab']['pyproject']['pending'])


def test_every_owned_table_has_exactly_one_declared_delta() -> None:
    """A table with no delta is unjudged; a delta for a table the base does not own judges nothing."""
    owned = set(PYPROJECT_SECTIONS) - _pending()
    assert set(PYPROJECT_SECTION_DELTAS) == owned, sorted(set(PYPROJECT_SECTION_DELTAS) ^ owned)


def test_no_declared_delta_is_a_fork_and_every_ceiling_is_its_measurement() -> None:
    """Driven on the DECLARATION alone; headroom nobody chose is how a waiver list stops being a delta."""
    for artifact, delta in PYPROJECT_SECTION_DELTAS.items():
        assert delta.repo == REPO, f'{artifact} declares itself as {delta.repo!r}'
        assert section_problems(section_base(artifact), delta) == ()
        assert sum(len(entries) for entries in delta.added.values()) == delta.ceiling, artifact


def test_this_repo_owns_every_table_the_base_declares() -> None:
    """THE PROPERTY. The real file, through the kit's reader, against the live base and this delta."""
    for artifact, delta in PYPROJECT_SECTION_DELTAS.items():
        report = inspect_section(_ROOT / 'pyproject.toml', section_base(artifact), delta)
        assert report.status == OWNED, f'{artifact} is {report.status}: {report.detail} {report.offending}'
