"""TOML + filesystem helpers — re-exported from the shared ``lab_commons.file_io``.

The implementation was extracted to the standalone ``lab-commons`` package
(plan-lab-commons-standalone.md §3); optimi-lab has no file-IO of its own, so this module is a
thin re-export, not a fork. Callers keep importing ``optimi_lab.utils.file_io`` unchanged.
"""

from lab_commons.file_io import check_path, dumps, list_files_in_dir, loads, read_toml, save_toml

__all__ = ['check_path', 'dumps', 'list_files_in_dir', 'loads', 'read_toml', 'save_toml']
