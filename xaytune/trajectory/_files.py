"""Durable, owner-only, no-overwrite file operations for the local trajectory store.

Every helper either completes durably or raises. "Durably" means the file is
fsynced and so is the directory that names it. Nothing here follows a
symlink: paths are checked with ``lstat`` and opened with ``O_NOFOLLOW``.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import tempfile
from pathlib import Path

DIRECTORY_MODE = 0o700
FILE_MODE = 0o600


class StoreConflictError(RuntimeError):
    """A no-overwrite write found different bytes already in place."""


class UnsafePathError(RuntimeError):
    """A path component is a symlink, or a path is not what it must be."""


def digest_of(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def refuse_symlinks(path: Path, *, below: Path) -> None:
    """Refuse if *below*, or any component of *path* under it, is a symlink.

    Components above *below* are the host's own namespace (``/var`` is a
    symlink on macOS), so they are not judged.
    """
    below = Path(below)
    path = Path(path)
    try:
        relative = path.relative_to(below)
    except ValueError as error:
        raise UnsafePathError("the path is not under its declared directory") from error
    current = below
    for part in ("", *relative.parts):
        current = current / part if part else current
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            return
        if stat.S_ISLNK(mode):
            raise UnsafePathError("a path component is a symlink")


def sync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def make_directory(path: Path, *, exist_ok: bool = True) -> None:
    """Create *path* owner-only, and sync its parent."""
    try:
        os.mkdir(path, DIRECTORY_MODE)
    except FileExistsError:
        if not exist_ok:
            raise
        if stat.S_ISLNK(os.lstat(path).st_mode) or not path.is_dir():
            raise UnsafePathError("an expected directory is not one") from None
        return
    os.chmod(path, DIRECTORY_MODE)
    sync_directory(path.parent)


def read_bytes(path: Path) -> bytes:
    """*path*'s bytes, never through a symlink."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        return handle.read()


def _staged(directory: Path, data: bytes) -> Path:
    fd, name = tempfile.mkstemp(dir=directory, prefix=".staging.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(name, FILE_MODE)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return Path(name)


def write_new(path: Path, data: bytes) -> bool:
    """Write *data* at *path* once, durably, never overwriting.

    Returns ``True`` if this call created the file, ``False`` if identical
    bytes were already there.

    Raises:
        StoreConflictError: If different bytes are already there.
    """
    staged = _staged(path.parent, data)
    try:
        os.link(staged, path)
    except FileExistsError:
        if read_bytes(path) != data:
            raise StoreConflictError("different bytes are already published there") from None
        return False
    finally:
        staged.unlink(missing_ok=True)
    sync_directory(path.parent)
    return True


def replace_atomically(path: Path, data: bytes) -> None:
    """Replace *path* with *data* in one rename, durably."""
    staged = _staged(path.parent, data)
    try:
        os.replace(staged, path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    sync_directory(path.parent)


def create_durably(path: Path) -> None:
    """Create *path* empty, owner-only, and sync both it and the directory that names it.

    Syncing a file makes its bytes durable; only syncing its directory makes
    the file's *name* durable. A write-ahead log whose first entry is synced
    in a file the directory never recorded would vanish after a system crash.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, FILE_MODE)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    sync_directory(path.parent)


def append_durably(path: Path, data: bytes) -> None:
    """Append *data* to *path* and fsync it before returning: a write-ahead point.

    If this call creates the file, the directory is synced too, so the entry
    that names the file is as durable as the bytes in it.
    """
    created = not os.path.lexists(path)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, FILE_MODE)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    if created:
        sync_directory(path.parent)


def truncate_durably(path: Path) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def link_new(source: Path, target: Path) -> bool:
    """Hard-link *source* to *target*, never overwriting; ``False`` if *target* exists."""
    try:
        os.link(source, target, follow_symlinks=False)
    except FileExistsError:
        return False
    except OSError as error:
        if error.errno == errno.EXDEV:
            raise UnsafePathError("the link would cross filesystems") from error
        raise
    sync_directory(target.parent)
    return True


def same_file(left: Path, right: Path) -> bool:
    """Whether *left* and *right* are one file: the same ``st_dev`` and ``st_ino``."""
    try:
        a, b = os.lstat(left), os.lstat(right)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(a.st_mode) or stat.S_ISLNK(b.st_mode):
        return False
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def device_of(path: Path) -> int:
    return os.stat(path).st_dev
