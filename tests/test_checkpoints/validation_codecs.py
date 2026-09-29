"""A historical v1alpha1 codec and its optional validation extension."""

from pathlib import Path

from xaytune.checkpoints import CheckpointState, RestoredCheckpoint, SerializedStateCodec
from xaytune.core.checkpoint import CheckpointContext, CheckpointManifest, RestoreContext
from xaytune.core.immutable import FrozenDict


class LegacyCodec:
    """Delegate the old ABI only; deliberately has no validate method."""

    descriptor = SerializedStateCodec.descriptor.model_copy(update={"metadata": FrozenDict()})

    def __init__(self) -> None:
        self.delegate = SerializedStateCodec()
        self.delegate.descriptor = self.descriptor
        self.decodes = 0

    def compatibility_key(self, context: CheckpointContext | RestoreContext) -> str:
        return self.delegate.compatibility_key(context)

    async def encode(
        self, state: CheckpointState, destination: Path, context: CheckpointContext
    ) -> CheckpointManifest:
        return await self.delegate.encode(state, destination, context)

    async def decode(
        self, source: Path, manifest: CheckpointManifest, context: RestoreContext
    ) -> RestoredCheckpoint:
        self.decodes += 1
        return await self.delegate.decode(source, manifest, context)


class ValidationCodec(LegacyCodec):
    descriptor = SerializedStateCodec.descriptor

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self.error = error
        self.validations = 0

    def validate(self, manifest: CheckpointManifest) -> None:
        self.validations += 1
        if self.error is not None:
            raise self.error
        self.delegate.validate(manifest)


class UndeclaredValidationCodec(ValidationCodec):
    """Method presence deliberately does not declare the validation contract."""

    descriptor = LegacyCodec.descriptor
