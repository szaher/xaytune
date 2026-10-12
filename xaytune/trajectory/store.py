"""The local trajectory store: one directory per attempt instance (ADR-021 T4, T6, T10).

```text
<trajectory output>/
  attempts/<attempt instance id>/
    open.json              immutable; created first, no-overwrite
    tail.jsonl             open-state: append-only, fsynced at every write-ahead point
    progress.json          open-state: replaced atomically, only ever advances
    chunks/<index>-<hex>.jsonl     evidence: immutable, content-addressed
    blobs/<tier>/<hex>             evidence: this attempt's content only
    result.json            evidence: the attempt's result, linked out at the commit point
    manifest.staged.json   evidence: verified before the commit point
    manifest.json          evidence: the staged manifest, linked into place once
    expiry.jsonl           retention metadata: append-only
    .lock                  held by the live writer
  quarantine/<attempt instance id>/    a contained attempt, owner-only
```

Blobs are stored **per attempt**: two trajectories never share a physical
object, so one's retention can never delete another's content, and a
quarantine holds everything its attempt wrote (T6, T8). Every file is
owner-only, and nothing is read or written through a symlink.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from xaytune.core.immutable import FrozenDomainModel
from xaytune.core.trajectory import ChunkRef, TrajectoryOpenRecord
from xaytune.trajectory._files import (
    UnsafePathError,
    make_directory,
    read_bytes,
    refuse_symlinks,
    replace_atomically,
    sync_directory,
    write_new,
)

__all__ = [
    "AttemptExistsError",
    "AttemptPaths",
    "LocalTrajectoryStore",
    "Tier",
    "TrajectoryProgress",
]

Tier = Literal["ordinary", "restricted"]


class AttemptExistsError(RuntimeError):
    """An attempt instance id is already in use. Ids are never reused (T3)."""


class TrajectoryProgress(FrozenDomainModel):
    """Open-state: the chunks rolled so far. Never covered by a fingerprint."""

    trajectory_id: str
    chunks: tuple[ChunkRef, ...] = ()


@dataclass(frozen=True)
class AttemptPaths:
    """Where one attempt instance's objects live."""

    directory: Path

    @property
    def trajectory_id(self) -> str:
        return self.directory.name

    @property
    def open_record(self) -> Path:
        return self.directory / "open.json"

    @property
    def tail(self) -> Path:
        return self.directory / "tail.jsonl"

    @property
    def progress(self) -> Path:
        return self.directory / "progress.json"

    @property
    def chunks(self) -> Path:
        return self.directory / "chunks"

    @property
    def result(self) -> Path:
        return self.directory / "result.json"

    @property
    def staged(self) -> Path:
        return self.directory / "manifest.staged.json"

    @property
    def manifest(self) -> Path:
        return self.directory / "manifest.json"

    @property
    def expiry(self) -> Path:
        return self.directory / "expiry.jsonl"

    @property
    def lock(self) -> Path:
        return self.directory / ".lock"

    @property
    def incident(self) -> Path:
        return self.directory / "incident.json"

    def blob(self, tier: Tier, digest: str) -> Path:
        return self.directory / "blobs" / tier / digest.removeprefix("sha256:")

    def chunk(self, ref: ChunkRef) -> Path:
        return self.chunks / f"{ref.index:06d}-{ref.digest.removeprefix('sha256:')}.jsonl"


class LocalTrajectoryStore:
    """Attempt directories under one trajectory output directory, on the local host."""

    def __init__(self, root: Path) -> None:
        root = Path(root)
        if not root.is_absolute() or ".." in root.parts:
            raise UnsafePathError("the trajectory output must be an absolute path without '..'")
        self.root = root
        root.parent.mkdir(parents=True, exist_ok=True)
        make_directory(root)
        refuse_symlinks(root, below=root)
        make_directory(root / "attempts")
        make_directory(root / "quarantine")

    @property
    def attempts_directory(self) -> Path:
        return self.root / "attempts"

    @property
    def quarantine_directory(self) -> Path:
        return self.root / "quarantine"

    def attempt(self, trajectory_id: str) -> AttemptPaths:
        paths = AttemptPaths(self.attempts_directory / trajectory_id)
        refuse_symlinks(paths.directory, below=self.root)
        return paths

    def attempts(self) -> tuple[AttemptPaths, ...]:
        """Every attempt directory, in id (creation) order."""
        names = sorted(os.listdir(self.attempts_directory))
        return tuple(self.attempt(name) for name in names if not name.startswith("."))

    def create_attempt(self, trajectory_id: str) -> AttemptPaths:
        paths = self.attempt(trajectory_id)
        try:
            make_directory(paths.directory, exist_ok=False)
        except FileExistsError as error:
            raise AttemptExistsError("that attempt instance id is already in use") from error
        for directory in (
            paths.chunks,
            paths.directory / "blobs",
            paths.directory / "blobs" / "ordinary",
            paths.directory / "blobs" / "restricted",
        ):
            make_directory(directory)
        return paths

    # ---- reading ----------------------------------------------------------------------------

    def read_open_record(self, paths: AttemptPaths) -> TrajectoryOpenRecord | None:
        try:
            return TrajectoryOpenRecord.model_validate_json(read_bytes(paths.open_record))
        except (OSError, ValidationError):
            return None

    def read_progress(self, paths: AttemptPaths) -> TrajectoryProgress:
        try:
            return TrajectoryProgress.model_validate_json(read_bytes(paths.progress))
        except FileNotFoundError:
            return TrajectoryProgress(trajectory_id=paths.trajectory_id)

    def read_tail(self, paths: AttemptPaths) -> bytes:
        try:
            return read_bytes(paths.tail)
        except FileNotFoundError:
            return b""

    def read_chunk(self, paths: AttemptPaths, ref: ChunkRef) -> bytes | None:
        try:
            return read_bytes(paths.chunk(ref))
        except FileNotFoundError:
            return None

    def read_blob(self, paths: AttemptPaths, tier: Tier, digest: str) -> bytes | None:
        try:
            return read_bytes(paths.blob(tier, digest))
        except FileNotFoundError:
            return None

    # ---- writing ----------------------------------------------------------------------------

    def put_blob(self, paths: AttemptPaths, tier: Tier, digest: str, data: bytes) -> None:
        write_new(paths.blob(tier, digest), data)

    def put_chunk(self, paths: AttemptPaths, ref: ChunkRef, data: bytes) -> None:
        write_new(paths.chunk(ref), data)

    def write_progress(self, paths: AttemptPaths, progress: TrajectoryProgress) -> None:
        replace_atomically(paths.progress, progress.model_dump_json().encode("utf-8"))

    def contain(self, paths: AttemptPaths) -> AttemptPaths:
        """Move a whole attempt, blobs included, into the owner-only quarantine (T8)."""
        target = self.quarantine_directory / paths.trajectory_id
        os.rename(paths.directory, target)
        sync_directory(self.attempts_directory)
        sync_directory(self.quarantine_directory)
        return AttemptPaths(target)
