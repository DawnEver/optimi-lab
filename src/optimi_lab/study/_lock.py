"""One study writer per record path, released by the OS even after the process is killed."""

from __future__ import annotations

import errno
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

if sys.platform == 'win32':
    import msvcrt
else:
    import fcntl

__all__ = ['study_writer']


def _change_lock(descriptor: int, *, acquire: bool) -> None:
    if sys.platform == 'win32':
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK if acquire else msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(descriptor, (fcntl.LOCK_EX | fcntl.LOCK_NB) if acquire else fcntl.LOCK_UN)


@contextmanager
def study_writer(path: Path) -> Iterator[None]:
    """Refuse a concurrent writer instead of merging independently stale record indexes.

    The ``<record>.lock`` sidecar is never deleted: unlinking it while another process holds it open
    would let a third writer lock a different inode. There is no stale PID lease to reclaim.
    """
    record = Path(path).resolve()
    record.parent.mkdir(parents=True, exist_ok=True)
    with record.with_name(f'{record.name}.lock').open('a+b') as lock:
        if os.fstat(lock.fileno()).st_size == 0:
            lock.write(b'\0')
            lock.flush()
        try:
            _change_lock(lock.fileno(), acquire=True)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise
            msg = f'another study already owns this record: {record}'
            raise OSError(msg) from exc
        try:
            yield
        finally:
            _change_lock(lock.fileno(), acquire=False)
