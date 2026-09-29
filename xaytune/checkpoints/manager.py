"""Coordinate serialization, publication, compatibility and localization only."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast

from xaytune.checkpoints.codec import (
    CHECKPOINT_VALIDATION_API_VERSION,
    CheckpointCodec,
    CheckpointState,
    CheckpointValidationCodec,
    RestoredCheckpoint,
)
from xaytune.checkpoints.errors import CheckpointCompatibilityError, CheckpointCorruptionError
from xaytune.checkpoints.store import CheckpointStore, LocalizedCheckpoint
from xaytune.core.capabilities import require_supported_plugin
from xaytune.core.checkpoint import CheckpointContext, RecordedCheckpoint, RestoreContext
from xaytune.core.fingerprint import fingerprint
from xaytune.core.refs import CheckpointRef


class CheckpointManager:
    def __init__(self, codec: CheckpointCodec, store: CheckpointStore) -> None:
        require_supported_plugin(codec.descriptor)
        self.codec = codec
        self.store = store

    @property
    def supports_validation(self) -> bool:
        """Whether the codec explicitly declares the supported validation API."""
        return (
            self.codec.descriptor.metadata.get("checkpoint_validation_api")
            == CHECKPOINT_VALIDATION_API_VERSION
        )

    async def save(self, state: CheckpointState, context: CheckpointContext) -> CheckpointRef:
        context = CheckpointContext.model_validate_json(context.model_dump_json())
        with tempfile.TemporaryDirectory(prefix="xaytune-checkpoint-") as scratch:
            directory = Path(scratch) / "encoded"
            manifest = await self.codec.encode(state, directory, context)
            if (
                manifest.context != context
                or manifest.codec != self.codec.descriptor.name
                or manifest.codec_version != self.codec.descriptor.plugin_version
                or manifest.compatibility_key != self.codec.compatibility_key(context)
            ):
                raise CheckpointCorruptionError("codec returned conflicting checkpoint provenance")
            staging = await self.store.put_staging(directory, manifest)
            return await self.store.commit(staging)

    async def restore(
        self, reference: CheckpointRef, context: RestoreContext
    ) -> RestoredCheckpoint:
        """Check compatibility before decoding; the caller applies encoded state.

        This does not execute recovery or create/update an attempt. No achieved
        restore guarantee is recorded until an adapter actually applies state.
        """
        context = RestoreContext.model_validate_json(context.model_dump_json())
        localized = await self.store.get(reference)
        return await self._decode(localized, context)

    async def restore_recorded(
        self, record: RecordedCheckpoint, context: RestoreContext
    ) -> RestoredCheckpoint:
        """Bind localized bytes to the durable producer and reported capture."""
        context = RestoreContext.model_validate_json(context.model_dump_json())
        localized = await self._localize_recorded(record, context)
        return await self.codec.decode(localized.directory, localized.manifest, context)

    async def validate_recorded(
        self, record: RecordedCheckpoint, context: RestoreContext
    ) -> LocalizedCheckpoint:
        """Require validation capability; never infer eligibility from v1alpha1.

        Compatibility/corruption failures retain their typed distinction.
        ValueError from layout validation is normalized to corruption. Programmer
        errors propagate; absence of this extension never falls back to decode.
        """
        if not self.supports_validation:
            raise CheckpointCompatibilityError(
                "codec lacks validation-only checkpoint capability at a supported version"
            )
        localized = await self._localize_recorded(record, context)
        try:
            cast(CheckpointValidationCodec, self.codec).validate(localized.manifest)
        except ValueError as error:
            raise CheckpointCorruptionError(
                "codec rejected malformed checkpoint capture"
            ) from error
        return localized

    async def _localize_recorded(
        self, record: RecordedCheckpoint, context: RestoreContext
    ) -> LocalizedCheckpoint:
        """The v1alpha1 recorded-restore provenance/compatibility checks."""
        record = RecordedCheckpoint.model_validate_json(record.model_dump_json())
        context = RestoreContext.model_validate_json(context.model_dump_json())
        localized = await self.store.get(record.payload.checkpoint_ref)
        manifest = localized.manifest
        if (
            str(manifest.context.producer_attempt_id) != record.context.target.id
            or manifest.context.candidate_fingerprint != record.candidate_fingerprint
            or manifest.context.execution_fingerprint != record.execution_fingerprint
            or fingerprint(manifest.committed_payload(localized.reference))
            != fingerprint(record.payload)
        ):
            raise CheckpointCorruptionError(
                "checkpoint bundle disagrees with its durable commit report"
            )
        self._validate_compatibility(localized, context)
        return localized

    async def _decode(
        self, localized: LocalizedCheckpoint, context: RestoreContext
    ) -> RestoredCheckpoint:
        self._validate_compatibility(localized, context)
        return await self.codec.decode(localized.directory, localized.manifest, context)

    def _validate_compatibility(
        self, localized: LocalizedCheckpoint, context: RestoreContext
    ) -> None:
        manifest = localized.manifest
        if (
            manifest.codec != self.codec.descriptor.name
            or manifest.codec_version != self.codec.descriptor.plugin_version
            or manifest.compatibility_key != self.codec.compatibility_key(context)
            or manifest.context.candidate_fingerprint != context.candidate_fingerprint
        ):
            raise CheckpointCompatibilityError(
                "checkpoint scientific identity or state layout differs"
            )
        cursor = manifest.data_cursor
        if cursor is not None and (
            cursor.dataset_fingerprint != context.dataset_fingerprint
            or cursor.ordering_fingerprint != context.ordering_fingerprint
        ):
            raise CheckpointCompatibilityError(
                "checkpoint dataset or ordering differs or is unknown"
            )
        if context.required_guarantee is not None and (
            manifest.resume_guarantee != context.required_guarantee
        ):
            raise CheckpointCompatibilityError(
                "checkpoint does not provide the requested guarantee"
            )
