"""Value objects referencing things outside the control plane.

These are immutable descriptors, not handles: none of them carries a live
model, dataset, or runtime connection. Keeping them free of runtime objects is
what allows an execution plan to be serialized and handed to another process
(ADR-001).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from xaytune.core.ids import ArtifactId, CheckpointId, EvaluationId, RunAttemptId
from xaytune.core.immutable import FrozenDict

__all__ = [
    "Actor",
    "ActorType",
    "ArtifactKind",
    "ArtifactRef",
    "CheckpointRef",
    "ControllerHostRef",
    "DatasetRef",
    "ModelRef",
    "ResourceUsage",
    "RuntimeRef",
]

ActorType = Literal["human", "rule", "llm_agent", "search_provider", "system"]

ArtifactKind = Literal[
    "model",
    "adapter",
    "checkpoint",
    "tokenizer",
    "metrics",
    "evaluation_report",
    "dataset_snapshot",
    "logs",
    "execution_manifest",
    "provenance_bundle",
]


class _Frozen(BaseModel):
    """Base for immutable value objects.

    ``extra="forbid"`` is deliberate: silently dropping an unknown field would
    lose provenance rather than surface a schema mismatch.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class Actor(_Frozen):
    """Who or what caused a change.

    Attributes:
        type: The kind of actor. An ``llm_agent`` may only ever propose typed
            actions; it never mutates state directly (Invariant E).
        id: Stable identifier for the actor within its type.
        metadata: Free-form provenance, e.g. model name or rule version.
    """

    type: ActorType
    id: str
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class ArtifactRef(_Frozen):
    """A produced artifact and where it came from."""

    id: ArtifactId
    kind: ArtifactKind
    uri: str
    digest: str | None = None
    producer_attempt_id: RunAttemptId | None = None
    producer_evaluation_id: EvaluationId | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class DatasetRef(_Frozen):
    """Immutable dataset identity.

    The fingerprints matter as much as the URI: the same source data processed
    with a different template or tokenizer is a different scientific input, and
    must produce a different training fingerprint (ADR-006).
    """

    uri: str
    revision: str | None = None
    split: str | None = None
    content_digest: str | None = None
    transform_fingerprint: str | None = None
    tokenizer_fingerprint: str | None = None
    template_fingerprint: str | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class ModelRef(_Frozen):
    """Immutable model identity.

    Never holds a loaded model object — core contracts stay serializable.
    """

    uri: str
    revision: str | None = None
    digest: str | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class RuntimeRef(_Frozen):
    """A handle to work submitted to a runtime backend.

    Deliberately generic: the core must not assume Kubernetes identifiers.
    """

    backend: str
    external_id: str
    namespace: str | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class ControllerHostRef(_Frozen):
    """Which controller host owns an experiment (ADR-004)."""

    kind: Literal["embedded", "local_daemon", "remote"]
    id: str | None = None
    address: str | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class CheckpointRef(_Frozen):
    """A committed checkpoint.

    Phase 1 placeholder: the codec/store/manager split and the full
    compatibility key land with the checkpoint subsystem (ADR-009).
    """

    id: CheckpointId
    uri: str
    global_step: int | None = None
    epoch: float | None = None
    digest: str | None = None
    compatibility_key: str | None = None
    created_at: datetime | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class ResourceUsage(_Frozen):
    """What an attempt consumed.

    All fields are optional: on-prem runtimes commonly report GPU-hours with no
    currency cost attached.
    """

    gpu_hours: float | None = None
    cpu_hours: float | None = None
    wall_time_seconds: float | None = None
    tokens: int | None = None
    cost: Decimal | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)
