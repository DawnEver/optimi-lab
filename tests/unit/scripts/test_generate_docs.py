# ruff: noqa: E402
import re
import shutil
import sys
from pathlib import Path

import pytest

script_path = Path(__file__).resolve()
root_path_ = re.search(r'(.*?)(tests[/\\])', str(script_path)).group(1)  # type: ignore [union-attr]
sys.path.append(root_path_)

from scripts import pdoc as pdoc_module
from scripts.pdoc import main as generate_docs


@pytest.mark.parametrize(('modules', 'open_webpage'), [(None, False), (['optimi_lab'], True)])
def test_generate_docs(modules, open_webpage: bool, mocker) -> None:
    """Test pdoc.py.

    `open_webpage=True` used to call the real `webbrowser.open`, popping a browser window on
    every test run -- reported twice by the user as an interruption to normal development, and
    one that gives no feedback back to the test (a human has to be watching to learn anything
    from it). Mocked here instead of deleted: the branch stays covered, and the assertion is
    STRONGER than before -- it now pins the exact `file://...optimi_lab.html` URL, which nothing
    checked previously -- while no window opens.

    Patched at `scripts.pdoc.webbrowser.open`, i.e. where `scripts/pdoc.py` LOOKS IT UP (the
    name `webbrowser` bound inside that module's own namespace by its `import webbrowser`), not
    at `webbrowser.open` on the `webbrowser` module itself -- patching the definition site would
    not affect the reference `scripts.pdoc` already holds.
    """
    # `optimi_lab.__all__` is a flat list of names, so pdoc renders ONE page per documented
    # module and no `docs/optimi_lab/` directory: asserting that directory would assert a
    # nested public surface the package does not have.
    output_dir = Path('./docs')
    if Path.exists(output_dir):
        shutil.rmtree(output_dir)
    mock_open = mocker.patch('scripts.pdoc.webbrowser.open')
    generate_docs(modules=modules, open_webpage=open_webpage)
    assert Path.exists(output_dir)
    assert Path.exists(output_dir / 'optimi_lab.html')
    assert Path.exists(output_dir / 'index.html')
    assert Path.exists(output_dir / 'search.js')
    if open_webpage:
        expected_url = 'file://' + str(pdoc_module.root_path / output_dir / 'optimi_lab.html')
        mock_open.assert_called_once_with(expected_url)
    else:
        mock_open.assert_not_called()


if __name__ == '__main__':
    pytest.main([__file__])
