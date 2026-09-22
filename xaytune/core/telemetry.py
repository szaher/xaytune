"""Typed observation bodies, shared by all trainer/runtime integrations.

These are evidence, not policy decisions or state transitions. The controller
validates and durably records observations before sinks consume domain events.
All utilization and fraction fields use [0, 1], never percentages in [0, 100].
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from xaytune.core.immutable import FrozenDict, FrozenDomainModel
from xaytune.core.observability import (
    Counter,
    Finite,
    Fraction,
    Name,
    NonNegative,
    Positive,
    PositiveCount,
)
from xaytune.core.refs import ArtifactRef, CheckpointRef
from xaytune.core.resume import (
    CheckpointBoundary,
    CheckpointStateManifest,
    DataCursor,
    DataResume,
    ResumeGuarantee,
    StateRestore,
)


class ResourceMetricObserved(FrozenDomainModel):
    type: Literal["ResourceMetricObserved"] = "ResourceMetricObserved"
    worker: str | None = None
    rank: Counter | None = None
    cpu_utilization: Fraction | None = None
    host_memory_used_bytes: Counter | None = None
    host_memory_utilization: Fraction | None = None
    gpu_index: Counter | None = None
    gpu_utilization: Fraction | None = None
    gpu_memory_used_bytes: Counter | None = None
    gpu_memory_reserved_bytes: Counter | None = None
    gpu_power_watts: NonNegative | None = None
    gpu_temperature_celsius: (
        Annotated[float, Field(strict=True, ge=-273.15, allow_inf_nan=False)] | None
    ) = None
    disk_read_bytes: Counter | None = None
    disk_write_bytes: Counter | None = None
    network_rx_bytes: Counter | None = None
    network_tx_bytes: Counter | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class TrainingMetricObserved(FrozenDomainModel):
    type: Literal["TrainingMetricObserved"] = "TrainingMetricObserved"
    optimizer_step: Counter | None = None
    loss: Finite | None = None
    learning_rate: NonNegative | None = None
    gradient_norm: NonNegative | None = None
    weight_norm: NonNegative | None = None
    samples_per_second: NonNegative | None = None
    tokens_per_second: NonNegative | None = None
    optimizer_steps_per_second: NonNegative | None = None
    step_duration_seconds: NonNegative | None = None
    forward_duration_seconds: NonNegative | None = None
    backward_duration_seconds: NonNegative | None = None
    optimizer_duration_seconds: NonNegative | None = None
    examples_seen: Counter | None = None
    tokens_seen: Counter | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class DataMetricObserved(FrozenDomainModel):
    """Interval measurements, not the resume position represented by DataCursor."""

    type: Literal["DataMetricObserved"] = "DataMetricObserved"
    optimizer_step: Counter | None = None
    examples_consumed: Counter | None = None
    tokens_consumed: Counter | None = None
    batch_tokens: Counter | None = None
    padding_fraction: Fraction | None = None
    truncation_fraction: Fraction | None = None
    dataloader_wait_seconds: NonNegative | None = None
    preprocessing_seconds: NonNegative | None = None
    rejected_examples: Counter | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class DistributedMetricObserved(FrozenDomainModel):
    type: Literal["DistributedMetricObserved"] = "DistributedMetricObserved"
    worker: str | None = None
    rank: Counter | None = None
    world_size: PositiveCount | None = None
    step_duration_seconds: NonNegative | None = None
    synchronization_wait_seconds: NonNegative | None = None
    collective_seconds: NonNegative | None = None
    # Slowest worker duration / mean worker duration, hence >= 1.
    straggler_ratio: Annotated[float, Field(strict=True, ge=1, allow_inf_nan=False)] | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)

    @model_validator(mode="after")
    def _rank_in_world(self) -> DistributedMetricObserved:
        if self.rank is not None and self.world_size is not None and self.rank >= self.world_size:
            raise ValueError("rank must be less than world_size")
        return self


class AlignmentMetricObserved(FrozenDomainModel):
    type: Literal["AlignmentMetricObserved"] = "AlignmentMetricObserved"
    optimizer_step: Counter | None = None
    reward_mean: Finite | None = None
    reward_std: NonNegative | None = None
    # Sampled KL and differential entropy estimates may be negative.
    kl_mean: Finite | None = None
    entropy: Finite | None = None
    preference_margin: Finite | None = None
    rollout_count: Counter | None = None
    rollout_success_rate: Fraction | None = None
    rollouts_per_second: NonNegative | None = None
    environment_latency_seconds: NonNegative | None = None
    generation_tokens_per_second: NonNegative | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class WorkerReadyPayload(FrozenDomainModel):
    type: Literal["WorkerReady"] = "WorkerReady"
    pid: PositiveCount | None = None


class HeartbeatPayload(FrozenDomainModel):
    type: Literal["Heartbeat"] = "Heartbeat"
    expected_interval_seconds: Positive
    # Sequence is supplied once, in the envelope, by the supervisor.


class TrainingStartedPayload(FrozenDomainModel):
    type: Literal["TrainingStarted"] = "TrainingStarted"
    optimizer_step: Counter | None = None


class TrainingCompletedPayload(FrozenDomainModel):
    type: Literal["TrainingCompleted"] = "TrainingCompleted"
    optimizer_step: Counter | None = None


class TrainingFailedPayload(FrozenDomainModel):
    type: Literal["TrainingFailed"] = "TrainingFailed"
    reason: Name
    detail: str | None = None


class StepCompletedPayload(FrozenDomainModel):
    type: Literal["StepCompleted"] = "StepCompleted"
    optimizer_step: Counter


class MetricObservedPayload(FrozenDomainModel):
    """Named finite scalar for custom metrics, not an untyped event escape hatch."""

    type: Literal["MetricObserved"] = "MetricObserved"
    name: Name
    value: Finite
    unit: str | None = None
    optimizer_step: Counter | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class CheckpointStartedPayload(FrozenDomainModel):
    type: Literal["CheckpointStarted"] = "CheckpointStarted"
    optimizer_step: Counter


class CheckpointCommittedPayload(FrozenDomainModel):
    type: Literal["CheckpointCommitted"] = "CheckpointCommitted"
    checkpoint_ref: CheckpointRef
    optimizer_step: Counter
    data_cursor: DataCursor | None
    resume_guarantee: ResumeGuarantee
    artifact_digest: Name | None = None
    state_manifest: CheckpointStateManifest | None = None

    @model_validator(mode="after")
    def _claims_have_evidence(self) -> CheckpointCommittedPayload:
        guarantee, state, cursor = self.resume_guarantee, self.state_manifest, self.data_cursor
        if (
            self.checkpoint_ref.global_step is not None
            and self.checkpoint_ref.global_step != self.optimizer_step
        ):
            raise ValueError("checkpoint reference and optimizer_step disagree")
        if (
            self.artifact_digest is not None
            and self.checkpoint_ref.digest is not None
            and self.artifact_digest != self.checkpoint_ref.digest
        ):
            raise ValueError("checkpoint digests disagree")
        if guarantee.state == StateRestore.FULL and (state is None or not state.has_full_state):
            raise ValueError(
                "FULL requires model, optimizer, scheduler, scaler, RNG and intervention state"
            )
        if state is not None:
            at_boundary = guarantee.boundary == CheckpointBoundary.OPTIMIZER_STEP
            if at_boundary != (state.micro_step == 0):
                raise ValueError("checkpoint boundary and micro_step disagree")
        if guarantee.data == DataResume.EXACT:
            if state is None or not state.has_full_state:
                raise ValueError("EXACT requires captured resumable state (ADR-012 AC-19)")
            if guarantee.boundary != CheckpointBoundary.OPTIMIZER_STEP:
                raise ValueError("EXACT requires a closed optimizer-step boundary")
            if (
                cursor is None
                or cursor.sampler_state is None
                or (cursor.next_sample_offset is None and cursor.iterable_cursor is None)
            ):
                raise ValueError(
                    "EXACT requires a consumption cursor and captured sampler/ordering state"
                )
        if (
            guarantee.data in (DataResume.AT_LEAST_ONCE, DataResume.EPOCH_BOUNDARY)
            and cursor is None
        ):
            raise ValueError("a data resume guarantee requires a cursor")
        if (
            guarantee.data == DataResume.EPOCH_BOUNDARY
            and cursor is not None
            and cursor.epoch is None
        ):
            raise ValueError("epoch-boundary resume requires an epoch")
        return self


class ProfilerStartedPayload(FrozenDomainModel):
    type: Literal["ProfilerStarted"] = "ProfilerStarted"
    output_name: Name
    provider: Literal["torch-profiler", "runtime-native"]


class ProfilerCompletedPayload(FrozenDomainModel):
    type: Literal["ProfilerCompleted"] = "ProfilerCompleted"
    output_name: Name


class ProfilerFailedPayload(FrozenDomainModel):
    type: Literal["ProfilerFailed"] = "ProfilerFailed"
    output_name: Name
    reason: Name


class ProfilerArtifactProducedPayload(FrozenDomainModel):
    type: Literal["ProfilerArtifactProduced"] = "ProfilerArtifactProduced"
    output_name: Name
    artifact_ref: ArtifactRef

    @model_validator(mode="after")
    def _profile_artifact(self) -> ProfilerArtifactProducedPayload:
        if self.artifact_ref.kind != "profile":
            raise ValueError("profiler output must reference a profile artifact")
        return self


class ArtifactProducedPayload(FrozenDomainModel):
    type: Literal["ArtifactProduced"] = "ArtifactProduced"
    artifact_ref: ArtifactRef


class IncidentObservedPayload(FrozenDomainModel):
    """Evidence for incident classification; not a new incident category enum."""

    type: Literal["IncidentObserved"] = "IncidentObserved"
    reason: Name
    detail: str | None = None
    exit_code: Annotated[int, Field(strict=True)] | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class GradientOverflowObserved(FrozenDomainModel):
    type: Literal["GradientOverflowObserved"] = "GradientOverflowObserved"
    optimizer_step: Counter | None = None
    loss_scale: Positive | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class OptimizerStepSkipped(FrozenDomainModel):
    type: Literal["OptimizerStepSkipped"] = "OptimizerStepSkipped"
    optimizer_step: Counter | None = None
    reason: Name
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class NumericalInstabilityObserved(FrozenDomainModel):
    """Report nonfinite evidence symbolically, never as a NaN JSON number."""

    type: Literal["NumericalInstabilityObserved"] = "NumericalInstabilityObserved"
    optimizer_step: Counter | None = None
    quantity: Name
    observation: Literal["nan", "positive-infinity", "negative-infinity", "unstable"]
    detail: str | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class EvaluationStartedPayload(FrozenDomainModel):
    type: Literal["EvaluationStarted"] = "EvaluationStarted"


class EvaluationCompletedPayload(FrozenDomainModel):
    type: Literal["EvaluationCompleted"] = "EvaluationCompleted"
    result_ref: ArtifactRef | None = None


class EvaluationFailedPayload(FrozenDomainModel):
    type: Literal["EvaluationFailed"] = "EvaluationFailed"
    reason: Name
    detail: str | None = None


class EvaluationProgressPayload(FrozenDomainModel):
    type: Literal["EvaluationProgress"] = "EvaluationProgress"
    examples_completed: Counter
    examples_total: Counter | None = None

    @model_validator(mode="after")
    def _within_total(self) -> EvaluationProgressPayload:
        if self.examples_total is not None and self.examples_completed > self.examples_total:
            raise ValueError("completed examples exceed total")
        return self


TrainingObservation = Annotated[
    WorkerReadyPayload
    | HeartbeatPayload
    | TrainingStartedPayload
    | TrainingCompletedPayload
    | TrainingFailedPayload
    | StepCompletedPayload
    | MetricObservedPayload
    | TrainingMetricObserved
    | ResourceMetricObserved
    | DataMetricObserved
    | DistributedMetricObserved
    | AlignmentMetricObserved
    | CheckpointStartedPayload
    | CheckpointCommittedPayload
    | ProfilerStartedPayload
    | ProfilerCompletedPayload
    | ProfilerFailedPayload
    | ProfilerArtifactProducedPayload
    | ArtifactProducedPayload
    | IncidentObservedPayload
    | GradientOverflowObserved
    | OptimizerStepSkipped
    | NumericalInstabilityObserved,
    Field(discriminator="type"),
]

EvaluationObservation = Annotated[
    WorkerReadyPayload
    | HeartbeatPayload
    | IncidentObservedPayload
    | ArtifactProducedPayload
    | ResourceMetricObserved
    | EvaluationStartedPayload
    | EvaluationCompletedPayload
    | EvaluationFailedPayload
    | EvaluationProgressPayload
    | MetricObservedPayload,
    Field(discriminator="type"),
]
