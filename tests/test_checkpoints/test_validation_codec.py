"""The validation extension must not redefine the v1alpha1 save/restore ABI."""

import asyncio

import pytest

from tests.test_checkpoints.helpers import make_bundle
from tests.test_checkpoints.validation_codecs import (
    LegacyCodec,
    UndeclaredValidationCodec,
    ValidationCodec,
)
from xaytune.checkpoints import (
    CHECKPOINT_VALIDATION_API_VERSION,
    CheckpointCompatibilityError,
    CheckpointCorruptionError,
    CheckpointManager,
    LocalCheckpointStore,
)
from xaytune.core.checkpoint import RecordedCheckpoint
from xaytune.core.domain.incident import AttemptContext
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.ids import ExperimentId, ExperimentNodeId, RunId
from xaytune.core.immutable import FrozenDict


def report(manager, reference):
    manifest = asyncio.run(manager.store.get(reference)).manifest
    context = AttemptContext(
        experiment_id=ExperimentId.generate(),
        node_id=ExperimentNodeId.generate(),
        run_id=str(RunId.generate()),
        target=RuntimeOperationTarget(
            kind="training-attempt", id=str(manifest.context.producer_attempt_id)
        ),
    )
    payload = manifest.committed_payload(reference)
    return RecordedCheckpoint(
        context=context,
        candidate_fingerprint=manifest.context.candidate_fingerprint,
        execution_fingerprint=manifest.context.execution_fingerprint,
        payload=payload,
        evidence={
            "stream_generation": 0,
            "sequence": 0,
            "target": context.target.model_dump(mode="json"),
            "payload": {"workload": "training", "data": payload.model_dump(mode="json")},
        },
    )


def test_legacy_v1alpha1_save_restore_and_recorded_restore_remain_supported(tmp_path):
    state, context, consumer = make_bundle(tmp_path / "source")
    codec = LegacyCodec()
    assert codec.descriptor.api_version == "xaytune.plugins/v1alpha1"
    assert "checkpoint_validation_api" not in codec.descriptor.metadata
    manager = CheckpointManager(codec, LocalCheckpointStore(tmp_path / "store"))
    reference = asyncio.run(manager.save(state, context))
    recorded = report(manager, reference)
    restored = asyncio.run(manager.restore(reference, consumer))
    recorded_restore = asyncio.run(manager.restore_recorded(recorded, consumer))
    assert restored == recorded_restore
    assert codec.decodes == 2
    assert not manager.supports_validation
    with pytest.raises(CheckpointCompatibilityError, match="lacks validation-only"):
        asyncio.run(manager.validate_recorded(recorded, consumer))
    assert codec.decodes == 2


def test_validation_capable_codec_inspects_without_decode(tmp_path):
    state, context, consumer = make_bundle(tmp_path / "source")
    codec = ValidationCodec()
    assert (
        codec.descriptor.metadata["checkpoint_validation_api"] == CHECKPOINT_VALIDATION_API_VERSION
    )
    manager = CheckpointManager(codec, LocalCheckpointStore(tmp_path / "store"))
    reference = asyncio.run(manager.save(state, context))
    recorded = report(manager, reference)
    assert manager.supports_validation
    localized = asyncio.run(manager.validate_recorded(recorded, consumer))
    assert localized.reference == reference
    assert codec.validations == 1
    assert codec.decodes == 0


def test_layout_value_error_is_normalized_with_its_cause(tmp_path):
    state, context, consumer = make_bundle(tmp_path / "source")
    failure = ValueError("malformed provider capture")
    codec = ValidationCodec(failure)
    manager = CheckpointManager(codec, LocalCheckpointStore(tmp_path / "store"))
    reference = asyncio.run(manager.save(state, context))
    recorded = report(manager, reference)
    with pytest.raises(CheckpointCorruptionError) as raised:
        asyncio.run(manager.validate_recorded(recorded, consumer))
    assert raised.value.__cause__ is failure
    assert codec.decodes == 0


@pytest.mark.parametrize("declared_version", [None, "xaytune.checkpoint-validation/v99"])
def test_method_presence_without_supported_declaration_is_ineligible(tmp_path, declared_version):
    state, context, consumer = make_bundle(tmp_path / "source")
    codec = UndeclaredValidationCodec()
    if declared_version is not None:
        codec.descriptor = codec.descriptor.model_copy(
            update={"metadata": FrozenDict({"checkpoint_validation_api": declared_version})}
        )
    manager = CheckpointManager(codec, LocalCheckpointStore(tmp_path / "store"))
    reference = asyncio.run(manager.save(state, context))
    assert not manager.supports_validation
    with pytest.raises(CheckpointCompatibilityError, match="supported version"):
        asyncio.run(manager.validate_recorded(report(manager, reference), consumer))
    assert codec.validations == 0
    assert codec.decodes == 0


def test_declared_capability_with_missing_implementation_is_a_programmer_error(tmp_path):
    state, context, consumer = make_bundle(tmp_path / "source")
    codec = LegacyCodec()
    codec.descriptor = ValidationCodec.descriptor
    manager = CheckpointManager(codec, LocalCheckpointStore(tmp_path / "store"))
    reference = asyncio.run(manager.save(state, context))
    with pytest.raises(AttributeError, match="validate"):
        asyncio.run(manager.validate_recorded(report(manager, reference), consumer))
    assert codec.decodes == 0
