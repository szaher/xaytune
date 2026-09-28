"""The initial codec consumes already serialized state, without importing ML.

Trainer adapters remain responsible for capturing state and for applying it.
This codec understands the ADR-012 component layout and verifies every declared
component against bytes; it never deserializes arbitrary pickle or claims that
loading a bundle has applied its state to a trainer.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from xaytune._version import __version__
from xaytune.checkpoints._files import describe, files_in, verify
from xaytune.checkpoints.errors import CheckpointCompatibilityError, CheckpointCorruptionError
from xaytune.core.capabilities import PluginDescriptor
from xaytune.core.checkpoint import (
    CheckpointContext,
    CheckpointManifest,
    RestoreContext,
    checkpoint_path,
    checkpoint_state_refs,
)
from xaytune.core.fingerprint import fingerprint
from xaytune.core.resume import CheckpointStateManifest, DataCursor, ResumeGuarantee


@dataclass(frozen=True)
class CheckpointState:
    source: Path
    optimizer_step: int
    data_cursor: DataCursor | None
    resume_guarantee: ResumeGuarantee
    state_manifest: CheckpointStateManifest


@dataclass(frozen=True)
class RestoredCheckpoint:
    """Validated local encoded components, ready for an adapter to apply."""

    source: Path
    manifest: CheckpointManifest


class CheckpointCodec(Protocol):
    descriptor: PluginDescriptor

    def compatibility_key(self, context: CheckpointContext | RestoreContext) -> str: ...

    def validate(self, manifest: CheckpointManifest) -> None:
        """Validate the encoded layout without decoding or applying state."""
        ...

    async def encode(
        self, state: CheckpointState, destination: Path, context: CheckpointContext
    ) -> CheckpointManifest: ...

    async def decode(
        self, source: Path, manifest: CheckpointManifest, context: RestoreContext
    ) -> RestoredCheckpoint: ...


class SerializedStateCodec:
    """Versioned encoded-component layout, with exact compatibility only."""

    descriptor = PluginDescriptor(
        api_version="xaytune.plugins/v1alpha1",
        name="serialized-state",
        plugin_version="1",
        provider="xaytune",
        xaytune_version=__version__,
    )

    def compatibility_key(self, context: CheckpointContext | RestoreContext) -> str:
        return fingerprint(
            {
                "codec": self.descriptor.name,
                "codec_version": self.descriptor.plugin_version,
                "compatibility": context.compatibility.model_dump(mode="json"),
            }
        )

    def validate(self, manifest: CheckpointManifest) -> None:
        self._validate_components(manifest)

    async def encode(
        self, state: CheckpointState, destination: Path, context: CheckpointContext
    ) -> CheckpointManifest:
        context = CheckpointContext.model_validate_json(context.model_dump_json())
        # Inputs must be quiescent captured state, not a live trainer directory.
        source_files = files_in(state.source)
        destination.mkdir(parents=True, exist_ok=False)
        for source in source_files:
            target = destination / source.relative_to(state.source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        files = tuple(
            describe(path, path.relative_to(destination).as_posix())
            for path in files_in(destination)
        )
        data = {
            "context": context,
            "codec": self.descriptor.name,
            "codec_version": self.descriptor.plugin_version,
            "compatibility_key": self.compatibility_key(context),
            "files": files,
            "optimizer_step": state.optimizer_step,
            "data_cursor": state.data_cursor,
            "resume_guarantee": state.resume_guarantee,
            "state_manifest": state.state_manifest,
        }
        draft = CheckpointManifest.model_construct(_fields_set=set(data), **data)
        manifest = CheckpointManifest.model_validate(
            {**data, "manifest_digest": fingerprint(draft.digest_input())}
        )
        self._validate_components(manifest)
        return manifest

    async def decode(
        self, source: Path, manifest: CheckpointManifest, context: RestoreContext
    ) -> RestoredCheckpoint:
        if (
            manifest.codec != self.descriptor.name
            or manifest.codec_version != self.descriptor.plugin_version
            or manifest.compatibility_key != self.compatibility_key(context)
        ):
            raise CheckpointCompatibilityError("checkpoint codec or state layout is incompatible")
        verify(source, manifest)
        self._validate_components(manifest)
        return RestoredCheckpoint(source=source, manifest=manifest)

    def _validate_components(self, manifest: CheckpointManifest) -> None:
        if manifest.state_manifest.schema_version != "xaytune.checkpoint-state/v1alpha1":
            raise CheckpointCompatibilityError("unsupported checkpoint state schema")
        if manifest.compatibility_key != self.compatibility_key(manifest.context):
            raise CheckpointCorruptionError("compatibility key disagrees with its declaration")
        compatibility = manifest.context.compatibility
        if (
            manifest.state_manifest.optimizer is not None and compatibility.optimizer_layout is None
        ) or (
            manifest.state_manifest.scheduler is not None and compatibility.scheduler_layout is None
        ):
            raise CheckpointCompatibilityError("captured optimizer/scheduler layout is unknown")
        files = {file.path: file for file in manifest.files}
        for ref in checkpoint_state_refs(manifest.state_manifest, manifest.data_cursor):
            try:
                checkpoint_path(ref.uri)
            except ValueError as error:
                raise CheckpointCorruptionError(
                    "state references must point inside the bundle"
                ) from error
            file = files.get(ref.uri)
            if file is None or ref.digest != file.digest:
                raise CheckpointCorruptionError("state reference is missing or has a wrong digest")
            if (
                ref.producer_attempt_id != manifest.context.producer_attempt_id
                or ref.producer_evaluation_id is not None
            ):
                raise CheckpointCorruptionError("state reference names a different producer")
