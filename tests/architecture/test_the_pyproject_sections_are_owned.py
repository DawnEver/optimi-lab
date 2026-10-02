"""PYPROJECT-SECTIONS-ARE-OWNED, for THIS checkout: the kit's section base plus this repo's delta.

`lab_commons.dev.famconfig` owns two `pyproject.toml` tables as a SECTION base -- ``[project]``'s
``readme``/``dynamic`` and ``[tool.pytest.ini_options]``'s ``testpaths`` -- and `inspect_section` reads
the real file against the live base and a repo's declared delta. The delta is
`_famconfig.PYPROJECT_SECTION_DELTAS`; this file is the verdict half.

BOTH SIDES OF THE RATCHET: an entry the file holds that no declaration produced is an undeclared local
waiver, and a base entry gone with no declared drop is the base leaving through the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from _famconfig import PYPROJECT_SECTION_DELTAS, REPO
from lab_commons.dev.famconfig import OWNED, PYPROJECT_SECTIONS, inspect_section, section_base, section_problems

_ROOT: Final = Path(__file__).resolve().parents[2]


def test_every_owned_table_has_exactly_one_declared_delta() -> None:
    """A table with no delta is unjudged; a delta for a table the base does not own judges nothing."""
    assert set(PYPROJECT_SECTION_DELTAS) == set(PYPROJECT_SECTIONS), sorted(
        set(PYPROJECT_SECTION_DELTAS) ^ set(PYPROJECT_SECTIONS)
    )


def test_no_declared_delta_is_a_fork_and_every_ceiling_is_its_measurement() -> None:
    """Driven on the DECLARATION alone; headroom nobody chose is how a waiver list stops being a delta."""
    for artefact, delta in PYPROJECT_SECTION_DELTAS.items():
        assert delta.repo == REPO, f'{artefact} declares itself as {delta.repo!r}'
        assert section_problems(section_base(artefact), delta) == ()
        assert sum(len(entries) for entries in delta.added.values()) == delta.ceiling, artefact


def test_this_repo_owns_every_table_the_base_declares() -> None:
    """THE PROPERTY. The real file, through the kit's reader, against the live base and this delta."""
    for artefact, delta in PYPROJECT_SECTION_DELTAS.items():
        report = inspect_section(_ROOT / 'pyproject.toml', section_base(artefact), delta)
        assert report.status == OWNED, f'{artefact} is {report.status}: {report.detail} {report.offending}'
