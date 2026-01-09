"""Generate API documentation using pdoc.
See https://pdoc.dev/docs/pdoc.html.
"""

import shutil
import subprocess
import webbrowser
from pathlib import Path

script_path = Path(__file__).resolve()
parts = script_path.parts
scripts_index = parts.index('scripts')
root_path = Path(*parts[:scripts_index])

# Ignore E402: module level import not at top of file
from optimi_lab.__version__ import __version__  # noqa: E402

__all__ = ['main']


def main(modules: list[str] | None = None, output_dir: str = 'docs', open_webpage: bool = False) -> None:
    """Generate API documentation."""
    if modules is None:
        modules = ['optimi_lab']
    host = 'localhost'
    port = '8080'
    # fmt: off
    pdoc_args = [
        'pdoc',
        *modules,
        '-o', output_dir,
        '-d', 'google',  # Google style
        '--include-undocumented',
        '--edit-url', 'optimi-lab=https://github.com/DawnEver/optimi-lab',
        '--favicon', 'http://cdn.mingyangbao.site/logo-latest/favicon.ico',
        '--footer-text', f'Py Project Template v{__version__}',
        '--logo', 'http://cdn.mingyangbao.site/logo-latest/MB.svg',
        '--logo-link', 'https://baomingyang.site/',
        '--math',
        '--mermaid',
        '--search',
        '--show-source',
        # '-t','scripts/generate_docs/templates',
        '-h', host,
        '-p', port,
        ]
    # fmt: on
    subprocess.run(pdoc_args, check=False)
    # Recursively copy all image files under modules to the corresponding locations in output_dir
    for module in modules:
        module_path = root_path / module
        for file_path in module_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.svg', '.png', '.jpg', '.jpeg']:
                relative_path = file_path.parent.relative_to(module_path)
                dest_dir = root_path / output_dir / module / relative_path
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file = dest_dir / file_path.name
                if not dest_file.exists() or file_path.stat().st_mtime > dest_file.stat().st_mtime:
                    shutil.copy2(file_path, dest_file)

    if open_webpage:
        # Open the generated documentation in the browser
        url = 'file://' + str(root_path / output_dir / (modules[0] + '.html'))
        webbrowser.open(url)


if __name__ == '__main__':
    main(open_webpage=True)
