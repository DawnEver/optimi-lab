"""Generate this repo's API documentation. The DRIVER is the family's; the facts below are ours.

WHAT MOVED AND WHY. This file used to carry its own pdoc invocation, its own image-mirroring walk
and its own ``subprocess.run(..., check=False)``. The first two were shared, near-identically, with
``wdg-lab/scripts/docs.py`` and ``motronics-studio/scripts/repo/docs.py`` -- three implementations
of one question -- and the ``check=False`` was OURS ALONE: a pdoc that errored produced an empty
``docs/`` and exit 0, which is indistinguishable from a real build at every downstream point. This
file was also asking pdoc for ``-o`` (write files and exit) AND ``-h``/``-p`` (serve) in one call,
which are mutually exclusive modes -- an error ``check=False`` then made invisible.

So the mechanism is now :mod:`lab_commons.dev.docsite`, which this repo already depends on through
``lab-commons[dev]``, and what remains here is DATA: which package, which logo, which edit URL.
There is no copy of the driver in this tree to drift, and the fix that lands there lands here.

WHAT DID NOT MOVE, and deliberately: ``sys.executable -m pdoc`` (it moved INTO the driver, where all
three repos now get it) and the browser branch below, which stays local so that this repo's own test
keeps patching the name in this repo's own namespace.

Usage: ``python scripts/pdoc.py``.
"""

import webbrowser
from pathlib import Path

from lab_commons.dev.docsite import pdoc_site

script_path = Path(__file__).resolve()
parts = script_path.parts
scripts_index = parts.index('scripts')
root_path = Path(*parts[:scripts_index])

# Ignore E402: module level import not at top of file
from optimi_lab.__version__ import __version__  # noqa: E402

__all__ = ['main']


def main(modules: list[str] | None = None, output_dir: str = 'docs', open_webpage: bool = False) -> None:
    """Generate API documentation.

    ``open_webpage`` DEFAULTS TO FALSE and every test passes it explicitly. A browser that opens
    itself during a test run was reported twice by this repo's user as an interruption to normal
    development, and it gives no feedback back to the test -- a human has to be watching to learn
    anything from it. The branch is kept and mocked (``tests/unit/scripts/test_generate_docs.py``
    patches ``scripts.pdoc.webbrowser.open`` and asserts BOTH the called and the not-called side),
    never defaulted on.
    """
    if modules is None:
        modules = ['optimi_lab']
    pdoc_site(
        root_path,
        root_path / output_dir,
        modules,
        edit_url='optimi-lab=https://github.com/DawnEver/optimi-lab',
        favicon='http://cdn.mingyangbao.site/logo-latest/favicon.ico',
        footer_text=f'Py Project Template v{__version__}',
        logo='http://cdn.mingyangbao.site/logo-latest/MB.svg',
        logo_link='https://baomingyang.site/',
    )

    if open_webpage:
        # Open the generated documentation in the browser
        url = 'file://' + str(root_path / output_dir / (modules[0] + '.html'))
        webbrowser.open(url)


if __name__ == '__main__':
    main(open_webpage=True)
