# ruff: noqa: E402
import importlib
import importlib.metadata
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


def test_the_docs_script_imports_without_the_generated_version_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """The version is the INSTALLED distribution's metadata, never the module setuptools-scm writes at install.

    That file is written by the build into whichever tree ran the install; a cluster tree that REUSES a keyed
    env has none (optimi-lab #26, 2026-10-24: a collection error here interrupted 382 other tests of its item).
    """
    monkeypatch.setitem(sys.modules, 'optimi_lab.__version__', None)  # an import of it now raises
    importlib.reload(pdoc_module)
    assert pdoc_module.__version__


def test_the_docs_script_imports_where_the_project_is_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deps-only env has no `optimi-lab` distribution: the import must not raise, the footer says so.

    The forge's cluster hand runs a keyed env that resolves the dependencies and never installs this
    project (its record's `resolved` list carries no `optimi-lab`), so `version('optimi-lab')` raised
    `PackageNotFoundError` at import and the file did not collect -- on 4bdc1ed and 9995dda alike.
    """

    def _absent(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, 'version', _absent)
    importlib.reload(pdoc_module)
    assert pdoc_module.__version__ == pdoc_module.NOT_INSTALLED
    monkeypatch.undo()
    importlib.reload(pdoc_module)


def test_generate_docs_opens_the_rendered_page(mocker) -> None:
    """`open_webpage=True` opens the exact `file://...optimi_lab.html` URL for an explicit module list.

    The render is `test_generate_docs` below; here the driver is mocked (load ruling 2026-10-30: the
    second real pdoc subprocess rendered the same `['optimi_lab']` site again) and asked only what it
    was given, and the browser is patched where `scripts/pdoc.py` looks it up.
    """
    site = mocker.patch('scripts.pdoc.pdoc_site')
    mock_open = mocker.patch('scripts.pdoc.webbrowser.open')
    generate_docs(modules=['optimi_lab'], open_webpage=True)
    assert site.call_args.args[1:] == (pdoc_module.root_path / 'docs', ['optimi_lab'])
    mock_open.assert_called_once_with('file://' + str(pdoc_module.root_path / 'docs' / 'optimi_lab.html'))


@pytest.mark.parametrize(('modules', 'open_webpage'), [(None, False)])
def test_generate_docs(modules, open_webpage: bool, mocker) -> None:
    """Test pdoc.py -- the ONE real render (`modules=None` resolves to `['optimi_lab']`).

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
    # module and no nested package directory: asserting that directory would assert a
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
