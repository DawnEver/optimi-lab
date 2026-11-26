"""The dependency-door arms' private box keeps a free-space floor of its own (optimi-lab #221).

`test_the_dependency_door_is_wired.py` points the box at a per-test ledger and declares the test
machine's `[storage] free_floor_gb`; this file proves that declaration is what decides, on the free
disk a hosted Windows runner had (29 GiB) when three arms were refused by the 50 GiB default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from lab_commons.box import BOX_ENV, box_path
from lab_commons.box.disk import verdict
from lab_commons.compute import floor
from lab_commons.config import CONFIG_ENV
from lab_commons.storage import FREE_FLOOR_GB_DEFAULT, free_floor

#: The machine file the dependency-door module's autouse fixture declares for its box.
_TEST_MACHINE: Final = '[storage]\nfree_floor_gb = 0.001\n'
_GIB: Final = 1024**3
#: What the hosted Windows runner's system drive had free when these arms were refused.
_HOSTED_FREE: Final = 29 * _GIB


def _refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: str) -> str | None:
    """What a private box under *machine* says of its volumes on the hosted runner's free disk."""
    config = tmp_path / 'machine' / 'config.toml'
    config.parent.mkdir()
    config.write_text(machine, encoding='utf-8')
    monkeypatch.setenv(CONFIG_ENV, str(config))
    monkeypatch.setenv(BOX_ENV, str(tmp_path / 'box' / 'box.sqlite'))
    monkeypatch.setattr(floor, 'free_bytes', lambda _volume: _HOSTED_FREE)
    return verdict((box_path(), Path.cwd()), free_floor())[0]


def test_the_private_box_is_not_refused_on_a_host_under_the_machine_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _HOSTED_FREE < FREE_FLOOR_GB_DEFAULT * _GIB, 'the planted volume must sit under the default floor'
    assert _refusal(tmp_path, monkeypatch, _TEST_MACHINE) is None


def test_planted_control_the_machine_default_floor_refuses_that_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PLANTED CONTROL: under the default floor, the same reading is the refusal the runner gave."""
    said = _refusal(tmp_path, monkeypatch, f'[storage]\nfree_floor_gb = {FREE_FLOOR_GB_DEFAULT}\n')
    assert said is not None
    assert 'floor' in said
