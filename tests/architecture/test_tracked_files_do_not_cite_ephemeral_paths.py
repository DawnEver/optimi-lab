"""optimi_lab's tracked evidence points at persistent inputs, not machine-local output."""

from pathlib import Path

from dev_lab.dev.famtests import ephemeralrefs

_ROOT = Path(__file__).resolve().parents[2]


def test_tracked_files_cite_persistent_evidence() -> None:
    ephemeralrefs.assert_no_tracked_file_cites_ephemeral(root=_ROOT, floor=250, headroom=30)


def test_the_scanner_still_convicts(tmp_path: Path) -> None:
    ephemeralrefs.assert_the_scanner_still_convicts(tmp_path)
