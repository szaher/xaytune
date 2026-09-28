"""Serializable checkpoint bundle declarations (ADR-009/012).

These describe captured state. They do not capture or apply trainer state, and
a declared resume guarantee is not evidence that a restore has been executed.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from xaytune.core.clock import utc_now
from xaytune.core.domain.incident import AttemptContext
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import CheckpointId, RunAttemptId
from xaytune.core.immutable import FrozenDict, FrozenDomainModel
from xaytune.core.observability import Counter, Name
from xaytune.core.refs import ArtifactRef, CheckpointRef
from xaytune.core.resume import CheckpointStateManifest, DataCursor, ResumeGuarantee
from xaytune.core.telemetry import CheckpointCommittedPayload

Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


def checkpoint_state_refs(
    state: CheckpointStateManifest | None, cursor: DataCursor | None
) -> tuple[ArtifactRef, ...]:
    """Every component backing the declared capture, including worker streams."""
    refs = [] if state is None else [state.model]
    if state is not None:
        refs.extend(
            ref for ref in (state.optimizer, state.scheduler, state.scaler) if ref is not None
        )
        if state.rng is not None:
            refs.extend((state.rng.python, state.rng.numpy, state.rng.torch_cpu))
            for worker in state.rng.workers:
                if worker.accelerator is not None:
                    refs.append(worker.accelerator)
                refs.extend(worker.dataloader)
    if cursor is not None and cursor.sampler_state is not None:
        refs.append(cursor.sampler_state.state_ref)
    return tuple(refs)


def checkpoint_path(value: str) -> str:
    """An exact relative POSIX path, never a URI or a traversal."""
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or str(path) != value
        or any(part in (".", "..") for part in path.parts)
        or any(char in value for char in ("\\", ":", "\x00"))
        or path.parts[0] == "manifest.json"
    ):
        raise ValueError("checkpoint files require canonical relative paths")
    return value


class CheckpointFile(FrozenDomainModel):
    path: str
    size_bytes: Counter
    digest: Digest

    _path = field_validator("path")(checkpoint_path)


class CheckpointCompatibilityKey(FrozenDomainModel):
    """Conservative exact-match compatibility; no implicit resharding.

    The producer supplies state-layout and framework identities. Operational
    batch settings are not compatibility identity; topology and data ordering
    must still match. Unknown compatibility is never a wildcard.
    """

    state_format: Name
    model_fingerprint: Name
    optimizer_layout: Name | None
    scheduler_layout: Name | None
    distributed_strategy: Name
    sharding_scheme: Name
    topology_fingerprint: Name
    framework_versions: FrozenDict
    adapter_fingerprint: Name | None = None
    tokenizer_fingerprint: Name | None = None

    @field_validator("framework_versions")
    @classmethod
    def _versions(cls, value: FrozenDict) -> FrozenDict:
        if not value or any(
            not key.strip() or not isinstance(version, str) or not version.strip()
            for key, version in value.items()
        ):
            raise ValueError("framework versions must be explicit nonempty strings")
        return value


class CheckpointContext(FrozenDomainModel):
    """Stable provenance chosen before serialization; reuse it on a retry."""

    checkpoint_id: CheckpointId
    producer_attempt_id: RunAttemptId
    candidate_fingerprint: Name
    execution_fingerprint: Name
    compatibility: CheckpointCompatibilityKey
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("checkpoint timestamps require a timezone")
        return value


class CheckpointManifest(FrozenDomainModel):
    schema_version: Literal["xaytune.checkpoint/v1alpha1"] = "xaytune.checkpoint/v1alpha1"
    context: CheckpointContext
    codec: Name
    codec_version: Name
    compatibility_key: Digest
    files: tuple[CheckpointFile, ...] = Field(min_length=1)
    optimizer_step: Counter
    data_cursor: DataCursor | None
    resume_guarantee: ResumeGuarantee
    state_manifest: CheckpointStateManifest
    manifest_digest: Digest

    def digest_input(self) -> dict:
        return self.model_dump(mode="json", exclude={"manifest_digest"})

    @model_validator(mode="after")
    def _valid(self) -> CheckpointManifest:
        paths = [file.path for file in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("checkpoint files must be unique and sorted by path")
        if any(a.startswith(b + "/") for a in paths for b in paths if a != b):
            raise ValueError("a checkpoint path cannot be both a file and a directory")
        if self.manifest_digest != fingerprint(self.digest_input()):
            raise ValueError("checkpoint manifest digest disagrees with its contents")
        # Reuse the existing ADR-012 claims validator rather than duplicating it.
        self.committed_payload(self.reference("checkpoint:unlocalized"))
        return self

    def reference(self, uri: str) -> CheckpointRef:
        return CheckpointRef(
            id=self.context.checkpoint_id,
            uri=uri,
            global_step=self.optimizer_step,
            digest=self.manifest_digest,
            compatibility_key=self.compatibility_key,
            created_at=self.context.created_at,
        )

    def committed_payload(self, reference: CheckpointRef) -> CheckpointCommittedPayload:
        return CheckpointCommittedPayload(
            checkpoint_ref=reference,
            optimizer_step=self.optimizer_step,
            data_cursor=self.data_cursor,
            resume_guarantee=self.resume_guarantee,
            artifact_digest=self.manifest_digest,
            state_manifest=self.state_manifest,
        )


class RestoreContext(FrozenDomainModel):
    candidate_fingerprint: Name
    compatibility: CheckpointCompatibilityKey
    dataset_fingerprint: Name | None = None
    ordering_fingerprint: Name | None = None
    required_guarantee: ResumeGuarantee | None = None


class RecordedCheckpoint(FrozenDomainModel):
    """A worker's durable commit report; bytes still require store validation."""

    context: AttemptContext
    candidate_fingerprint: Name
    execution_fingerprint: Name
    payload: CheckpointCommittedPayload
    evidence: FrozenDict
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _report(self) -> RecordedCheckpoint:
        raw = self.evidence
        target = self.context.target
        if target.kind != "training-attempt":
            raise ValueError("only training attempts produce checkpoints")
        for name in ("stream_generation", "sequence"):
            if type(raw.get(name)) is not int or raw[name] < 0:
                raise ValueError("checkpoint report requires an authoritative stream position")
        if raw.get("target") != target.model_dump(mode="json"):
            raise ValueError("checkpoint report target disagrees with its context")
        observation = raw.get("payload") or {}
        if observation.get("workload") != "training":
            raise ValueError("checkpoint report requires training telemetry")
        if fingerprint(observation.get("data")) != fingerprint(
            self.payload.model_dump(mode="json")
        ):
            raise ValueError("checkpoint report evidence disagrees with its payload")
        ref = self.payload.checkpoint_ref
        if ref.digest is None or ref.compatibility_key is None:
            raise ValueError("checkpoint report requires manifest digest and compatibility key")
        return self
