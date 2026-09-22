"""Runtime-neutral observation policy and correlation, never executable hooks.

Secret values must never enter plans, telemetry, logs, events, provenance or
request digests. Producers redact before serialization; this declarative policy
contains selectors only, never replacement secrets or a core regex executor.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StrictBool, model_validator

from xaytune.core.ids import (
    ActionId,
    ExperimentId,
    ExperimentNodeId,
    IncidentId,
    RunAttemptId,
    RunId,
)
from xaytune.core.immutable import FrozenDict, FrozenDomainModel

Counter = Annotated[int, Field(strict=True, ge=0)]
PositiveCount = Annotated[int, Field(strict=True, gt=0)]
Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
Positive = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
Fraction = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
Name = Annotated[str, Field(strict=True, min_length=1)]


class TraceContext(FrozenDomainModel):
    """Opaque correlation identifiers; adapters translate them to tracing SDKs."""

    trace_id: Name
    span_id: Name
    trace_flags: str | None = None


class CorrelationContext(FrozenDomainModel):
    """Optional identifiers, not a second source of workload identity."""

    experiment_id: ExperimentId | None = None
    node_id: ExperimentNodeId | None = None
    run_id: RunId | None = None
    attempt_id: RunAttemptId | None = None
    # Evaluation aggregates do not yet expose dedicated public ID types.
    evaluation_run_id: Name | None = None
    evaluation_attempt_id: Name | None = None
    action_id: ActionId | None = None
    incident_id: IncidentId | None = None
    stream_generation: Counter | None = None
    runtime_backend: Name | None = None
    compiler: Name | None = None


class RedactionPolicy(FrozenDomainModel):
    """Selectors only. SecretRef remains the only credential representation."""

    redact_environment_keys: tuple[Name, ...] = ()
    redact_attribute_keys: tuple[Name, ...] = ()
    redact_patterns: tuple[Name, ...] = ()


class TracingSpec(FrozenDomainModel):
    """Sampling intent, not an exporter or SDK configuration."""

    enabled: StrictBool = False
    sample_rate: Fraction = 1.0


class ProfilerSpec(FrozenDomainModel):
    """A requested schedule. Output is a named artifact, never inline trace data."""

    enabled: StrictBool = False
    provider: Literal["torch-profiler", "runtime-native"] | None = None
    wait_steps: Counter = 0
    warmup_steps: Counter = 0
    active_steps: PositiveCount | None = None
    repeat: PositiveCount = 1
    record_shapes: StrictBool = False
    profile_memory: StrictBool = False
    with_stack: StrictBool = False
    output_name: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")] = "profile"

    @model_validator(mode="after")
    def _schedule(self) -> ProfilerSpec:
        if self.enabled and self.provider is None:
            raise ValueError("enabled profiling requires a provider")
        if self.enabled and self.active_steps is None:
            raise ValueError("enabled profiling requires a bounded active_steps window")
        return self


class ObservabilitySpec(FrozenDomainModel):
    """What to observe, separate from TelemetryContract's transport protocol.

    No Python callbacks: local trainer callbacks may emit observations. Changes
    to training semantics require Action → TrainingIntervention; operational
    adaptation requires Action/recovery → ExecutionOverride.
    """

    metric_interval_steps: PositiveCount | None = None
    system_metric_interval_seconds: Positive | None = None
    log_level: Literal["trace", "debug", "info", "warning", "error"] = "info"
    profiler: ProfilerSpec | None = None
    tracing: TracingSpec | None = None
    capture_system_metrics: StrictBool = True
    capture_training_metrics: StrictBool = True
    capture_data_metrics: StrictBool = True
    redaction: RedactionPolicy = Field(default_factory=RedactionPolicy)
    extensions: FrozenDict = Field(default_factory=FrozenDict)
