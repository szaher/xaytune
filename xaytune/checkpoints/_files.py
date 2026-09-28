"""Byte integrity only; no optimizer or trainer semantics."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

from xaytune.checkpoints.errors import CheckpointCorruptionError
from xaytune.core.checkpoint import CheckpointFile, CheckpointManifest, checkpoint_path


def files_in(directory: Path, *, committed: bool = False) -> tuple[Path, ...]:
    if not directory.exists() or not stat.S_ISDIR(directory.lstat().st_mode):
        raise CheckpointCorruptionError("checkpoint directory is missing or is a symlink")
    files = []
    for parent, directories, names in os.walk(directory, followlinks=False):
        for name in directories + names:
            path = Path(parent) / name
            mode = path.lstat().st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise CheckpointCorruptionError("checkpoint contains a symlink or special file")
            if stat.S_ISREG(mode):
                relative = path.relative_to(directory).as_posix()
                if committed and relative == "manifest.json":
                    continue
                try:
                    checkpoint_path(relative)
                except ValueError as error:
                    raise CheckpointCorruptionError(str(error)) from error
                files.append(path)
    return tuple(sorted(files))


def describe(path: Path, relative: str) -> CheckpointFile:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return CheckpointFile(path=relative, size_bytes=size, digest="sha256:" + digest.hexdigest())


def verify(directory: Path, manifest: CheckpointManifest) -> None:
    actual = tuple(
        describe(path, path.relative_to(directory).as_posix())
        for path in files_in(directory, committed=True)
    )
    if actual != manifest.files:
        raise CheckpointCorruptionError("checkpoint file set, sizes, or digests disagree")


def sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def make_directory(directory: Path) -> None:
    """Persist each newly created ancestor as well as the final directory."""
    missing = []
    parent = directory
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    for parent in reversed(missing):
        parent.mkdir(exist_ok=True)
        sync_directory(parent.parent)
    if directory.is_symlink() or not directory.is_dir():
        raise CheckpointCorruptionError("store directories cannot be symlinks or files")
