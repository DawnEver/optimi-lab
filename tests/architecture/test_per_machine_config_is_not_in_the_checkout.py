"""Per-machine preferences do not live in this checkout (user ruling 2026-10-09).

The family's ONE per-machine file is read by ``lab_commons.config``; a package that has preferences
owns the ``[optimi_lab]`` table there. ``optimi_lab`` is pure numerics and reads NONE: the
``usr/`` tree's two keys lost their readers when ``utils.email_reminder`` (``resend_api_key``) and
the text-editor helper (``text_editor_command``) left. So the tracked seed is deleted rather than
migrated, and a leftover ``usr/local/config.toml`` is refused by name with the remedy -- a file
nobody reads must not look like it configures something.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]

#: The retired per-checkout files: the tracked seed and the gitignored local copy.
RETIRED = ('usr/default/default.config.toml', 'usr/local/config.toml')

REMEDY = (
    'optimi-lab reads no per-machine preferences; per-machine settings belong in the [optimi_lab] '
    'table of the lab-commons machine file (lab_commons.config.config_path()), and this file has no '
    'reader -- delete it'
)


def retired_present(root: Path) -> list[str]:
    """The retired config files that still exist under *root*, each with its remedy."""
    return [f'{root / name} is retired: {REMEDY}' for name in RETIRED if (root / name).is_file()]


def test_the_refusal_names_a_planted_retired_file_and_its_remedy(tmp_path):
    """PLANTED CONTROL: the scan below is only worth something if it sees a file that is there."""
    assert retired_present(tmp_path) == []
    planted = tmp_path / 'usr' / 'local' / 'config.toml'
    planted.parent.mkdir(parents=True)
    planted.write_text("[utils]\ntext_editor_command='start'\n", encoding='utf-8')
    (refusal,) = retired_present(tmp_path)
    assert str(planted) in refusal
    assert '[optimi_lab]' in refusal
    assert 'delete it' in refusal


def test_this_checkout_carries_no_per_machine_config():
    found = retired_present(_ROOT)
    if found:
        pytest.fail('\n'.join(found))
