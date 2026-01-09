# ruff: noqa: E402
import re
import shutil
import sys
from pathlib import Path

import pytest

script_path = Path(__file__).resolve()
root_path_ = re.search(r'(.*?)(tests[/\\])', str(script_path)).group(1)  # type: ignore [union-attr]
sys.path.append(root_path_)

from scripts.pdoc import main as generate_docs


@pytest.mark.parametrize(('modules', 'open_webpage'), [(None, False), (['optimi_lab'], True)])
def test_generate_docs(modules, open_webpage: bool) -> None:
    """Test pdoc.py."""

    output_dir = Path('./docs')
    if Path.exists(output_dir):
        shutil.rmtree(output_dir)
    generate_docs(modules=modules, open_webpage=open_webpage)
    assert Path.exists(output_dir)
    assert Path.exists(output_dir / 'optimi_lab')
    assert Path.exists(output_dir / 'optimi_lab.html')
    assert Path.exists(output_dir / 'index.html')
    assert Path.exists(output_dir / 'search.js')


if __name__ == '__main__':
    pytest.main([__file__])
