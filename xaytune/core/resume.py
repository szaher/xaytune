"""ADR-012 checkpoint declarations; no sampler, RNG or restore implementation.

Cursors advance on committed consumption, never prefetch. EXACT describes data
continuation, not bitwise arithmetic. Encoded state lives in artifacts, so no
pickle, tensors or raw RNG bytes enter the controller's wire records.
"""

from __future__ import annotations

from enum import Enum

from pydantic import model_validator

from xaytune.core.immutable import FrozenDict, FrozenDomainModel
from xaytune.core.observability import Counter, Name
from xaytune.core.refs import ArtifactRef


class StateRestore(str, Enum):
    FULL = "full"
    MODEL_ONLY = "model-only"


class DataResume(str, Enum):
    EXACT = "exact"
    AT_LEAST_ONCE = "at-least-once"
    EPOCH_BOUNDARY = "epoch-boundary"
    NONE = "none"


class CheckpointBoundary(str, Enum):
    OPTIMIZER_STEP = "optimizer-step"
    MID_ACCUMULATION = "mid-accumulation"


class ResumeGuarantee(FrozenDomainModel):
    state: StateRestore
    data: DataResume
    boundary: CheckpointBoundary


class SamplerState(FrozenDomainModel):
    """Provider/version plus captured sampler and permutation state reference."""

    provider: Name
    version: Name
    state_ref: ArtifactRef


class DataCursor(FrozenDomainModel):
    """Next unconsumed sample in an identified deterministic stream.

    Provider cursor and indexed offset are alternative position encodings;
    neither is a batch index. A cursor without either cannot claim EXACT.
    """

    dataset_fingerprint: Name
    ordering_fingerprint: Name
    epoch: Counter | None = None
    next_sample_offset: Counter | None = None
    sampler_state: SamplerState | None = None
    iterable_cursor: FrozenDict | None = None
    examples_seen: Counter | None = None
    tokens_seen: Counter | None = None

    @model_validator(mode="after")
    def _one_position(self) -> DataCursor:
        if self.next_sample_offset is not None and self.iterable_cursor is not None:
            raise ValueError("indexed and iterable positions are alternative cursor encodings")
        if self.iterable_cursor is not None and not self.iterable_cursor:
            raise ValueError("an iterable cursor must contain a provider position")
        return self


class WorkerRNGState(FrozenDomainModel):
    logical_worker_id: Name
    accelerator: ArtifactRef | None = None
    dataloader: tuple[ArtifactRef, ...] = ()


class RNGState(FrozenDomainModel):
    """Captured streams, not seeds. The producer must enumerate every worker."""

    python: ArtifactRef
    numpy: ArtifactRef
    torch_cpu: ArtifactRef
    workers: tuple[WorkerRNGState, ...] = ()

    @model_validator(mode="after")
    def _unique_workers(self) -> RNGState:
        ids = [worker.logical_worker_id for worker in self.workers]
        if len(ids) != len(set(ids)):
            raise ValueError("RNG logical worker IDs must be unique")
        return self


class CheckpointStateManifest(FrozenDomainModel):
    """Evidence for resume claims, references into the committed checkpoint.

    None means not captured. Empty intervention IDs means captured and no prior
    applications; it differs from None. Workers must include all applicable RNG
    streams. References declare capture; checkpoint codecs verify their contents.
    """

    schema_version: str = "xaytune.checkpoint-state/v1alpha1"
    model: ArtifactRef
    optimizer: ArtifactRef | None = None
    scheduler: ArtifactRef | None = None
    scaler: ArtifactRef | None = None
    rng: RNGState | None = None
    micro_step: Counter
    applied_intervention_application_ids: tuple[Name, ...] | None = None

    @property
    def has_full_state(self) -> bool:
        return all(
            value is not None
            for value in (
                self.optimizer,
                self.scheduler,
                self.scaler,
                self.rng,
                self.applied_intervention_application_ids,
            )
        )
