"""Local atomic bundle publication; storage has no training-state semantics."""

from __future__ import annotations

import errno
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from xaytune.checkpoints._files import make_directory, sync_directory, verify
from xaytune.checkpoints.errors import CheckpointCorruptionError
from xaytune.core.checkpoint import CheckpointManifest
from xaytune.core.errors import IdempotencyConflictError
from xaytune.core.ids import CheckpointId
from xaytune.core.refs import CheckpointRef


@dataclass(frozen=True)
class StagingRef:
    checkpoint_id: CheckpointId
    directory: Path
    manifest_digest: str


@dataclass(frozen=True)
class LocalizedCheckpoint:
    reference: CheckpointRef
    directory: Path
    manifest: CheckpointManifest


class CheckpointStore(Protocol):
    async def put_staging(self, source: Path, manifest: CheckpointManifest) -> StagingRef: ...

    async def commit(self, staging: StagingRef) -> CheckpointRef: ...

    async def get(self, reference: CheckpointRef) -> LocalizedCheckpoint: ...

    async def list(self) -> tuple[CheckpointRef, ...]: ...


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate manifest JSON key")
        result[key] = value
    return result


def _manifest(directory: Path) -> CheckpointManifest:
    try:
        path = directory / "manifest.json"
        if path.is_symlink():
            raise ValueError("manifest cannot be a symlink")
        return CheckpointManifest.model_validate(
            json.loads(path.read_text(), object_pairs_hook=_unique_pairs)
        )
    except (OSError, ValueError, ValidationError) as error:
        raise CheckpointCorruptionError("checkpoint manifest is missing or invalid") from error


class LocalCheckpointStore:
    """One trusted local filesystem, same-filesystem staging and atomic rename.

    All files and directories are fsynced before publication, and the parent
    is fsynced before acknowledgement. A crash before publication leaves only
    unlisted staging; a crash after it permits an identical commit retry.
    No retention or deletion API until deletion can be evented and auditable.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._staging = self.root / ".staging"
        self._committed = self.root / "committed"
        make_directory(self._staging)
        make_directory(self._committed)
        sync_directory(self.root)

    async def put_staging(self, source: Path, manifest: CheckpointManifest) -> StagingRef:
        manifest = CheckpointManifest.model_validate_json(manifest.model_dump_json())
        verify(source, manifest)
        directory = Path(tempfile.mkdtemp(prefix="bundle-", dir=self._staging))
        try:
            for file in manifest.files:
                target = directory / file.path
                target.parent.mkdir(parents=True, exist_ok=True)
                with (source / file.path).open("rb") as incoming, target.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
            with (directory / "manifest.json").open("x") as stream:
                stream.write(json.dumps(manifest.model_dump(mode="json"), sort_keys=True))
                stream.flush()
                os.fsync(stream.fileno())
            verify(directory, manifest)
            for parent in sorted(
                (path for path in directory.rglob("*") if path.is_dir()), reverse=True
            ):
                sync_directory(parent)
            sync_directory(directory)
            sync_directory(self._staging)
        except BaseException:
            shutil.rmtree(directory)
            raise
        return StagingRef(manifest.context.checkpoint_id, directory, manifest.manifest_digest)

    async def commit(self, staging: StagingRef) -> CheckpointRef:
        checkpoint_id = CheckpointId.validate(staging.checkpoint_id)
        destination = self._committed / str(checkpoint_id)
        # A caller may be retrying after rename and before acknowledgement.
        if destination.exists():
            reference = self._existing(destination, staging.manifest_digest)
            self._discard_replay(staging)
            return reference
        if staging.directory.parent != self._staging or staging.directory.is_symlink():
            raise CheckpointCorruptionError("staging belongs to another store")
        manifest = _manifest(staging.directory)
        if (
            manifest.context.checkpoint_id != checkpoint_id
            or manifest.manifest_digest != staging.manifest_digest
        ):
            raise CheckpointCorruptionError("staging reference disagrees with its manifest")
        verify(staging.directory, manifest)
        try:
            # No replacement: a published directory is nonempty. Concurrent
            # writers lose this rename, then compare against the winner.
            os.rename(staging.directory, destination)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            reference = self._existing(destination, staging.manifest_digest)
            self._discard_replay(staging)
            return reference
        sync_directory(self._committed)
        sync_directory(self._staging)
        return manifest.reference(destination.as_uri())

    def _discard_replay(self, staging: StagingRef) -> None:
        if staging.directory.parent == self._staging and staging.directory.exists():
            if staging.directory.is_symlink():
                raise CheckpointCorruptionError("staging cannot be a symlink")
            manifest = _manifest(staging.directory)
            if manifest.manifest_digest != staging.manifest_digest:
                raise CheckpointCorruptionError("replay staging digest disagrees")
            verify(staging.directory, manifest)
            shutil.rmtree(staging.directory)
            sync_directory(self._staging)

    def _existing(self, directory: Path, expected_digest: str) -> CheckpointRef:
        manifest = _manifest(directory)
        verify(directory, manifest)
        if manifest.manifest_digest != expected_digest:
            raise IdempotencyConflictError(directory.name, ("manifest",), kind="checkpoint")
        if str(manifest.context.checkpoint_id) != directory.name:
            raise CheckpointCorruptionError("checkpoint ID disagrees with its directory")
        sync_directory(self._committed)
        return manifest.reference(directory.as_uri())

    async def get(self, reference: CheckpointRef) -> LocalizedCheckpoint:
        reference = CheckpointRef.model_validate_json(reference.model_dump_json())
        directory = self._committed / str(reference.id)
        manifest = _manifest(directory)
        verify(directory, manifest)
        expected = manifest.reference(directory.as_uri())
        if reference != expected:
            raise CheckpointCorruptionError(
                "checkpoint reference disagrees with its committed bundle"
            )
        return LocalizedCheckpoint(expected, directory, manifest)

    async def list(self) -> tuple[CheckpointRef, ...]:
        refs = []
        for directory in sorted(self._committed.iterdir()):
            try:
                CheckpointId.validate(directory.name)
            except ValueError as error:
                raise CheckpointCorruptionError("invalid committed checkpoint ID") from error
            manifest = _manifest(directory)
            localized = await self.get(manifest.reference(directory.as_uri()))
            refs.append(localized.reference)
        return tuple(refs)
