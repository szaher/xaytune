"""Wire contracts, not collectors, exporters or trainer implementations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import get_args

import pytest
from pydantic import SecretBytes, SecretStr, TypeAdapter, ValidationError

from xaytune.core import telemetry as t
from xaytune.core.execution import (
    CompilerIdentity,
    PythonModuleEntrypoint,
    SecretRef,
    TrainingExecutionSpec,
)
from xaytune.core.ids import ArtifactId, CheckpointId, RunAttemptId
from xaytune.core.immutable import FrozenDict
from xaytune.core.observability import (
    CorrelationContext,
    ObservabilitySpec,
    ProfilerSpec,
    RedactionPolicy,
    TraceContext,
    TracingSpec,
)
from xaytune.core.refs import ArtifactRef, CheckpointRef
from xaytune.core.resume import (
    CheckpointStateManifest,
    DataCursor,
    ResumeGuarantee,
    RNGState,
    SamplerState,
    WorkerRNGState,
)
from xaytune.runtimes import (
    EvaluationEventPayload,
    RuntimeEventEnvelope,
    RuntimeLog,
    TrainingEventPayload,
)


def roundtrip(value):
    assert type(value).model_validate_json(value.model_dump_json()) == value


def artifact(kind="checkpoint_state"):
    return ArtifactRef(id=ArtifactId.generate(), kind=kind, uri="file:///captured-state")


def full_state():
    ref = artifact()
    return CheckpointStateManifest(
        model=ref,
        optimizer=ref,
        scheduler=ref,
        scaler=ref,
        rng=RNGState(
            python=ref,
            numpy=ref,
            torch_cpu=ref,
            workers=(
                WorkerRNGState(logical_worker_id="worker-0", accelerator=ref, dataloader=(ref,)),
            ),
        ),
        micro_step=0,
        applied_intervention_application_ids=(),
    )


def cursor():
    return DataCursor(
        dataset_fingerprint="dataset",
        ordering_fingerprint="order",
        epoch=2,
        next_sample_offset=400,
        examples_seen=400,
        sampler_state=SamplerState(provider="test", version="1", state_ref=artifact()),
    )


def checkpoint(**updates):
    values = dict(
        checkpoint_ref=CheckpointRef(id=CheckpointId.generate(), uri="file:///checkpoint"),
        optimizer_step=100,
        data_cursor=cursor(),
        state_manifest=full_state(),
        resume_guarantee=ResumeGuarantee(state="full", data="exact", boundary="optimizer-step"),
    )
    values.update(updates)
    return t.CheckpointCommittedPayload(**values)


def test_observability_is_policy_not_transport_or_callbacks():
    source = {"nested": [1, {"a": 2}]}
    policy = ObservabilitySpec(
        metric_interval_steps=10,
        system_metric_interval_seconds=0.5,
        profiler=ProfilerSpec(enabled=True, provider="torch-profiler", active_steps=2),
        tracing=TracingSpec(enabled=True, sample_rate=0.5),
        extensions=source,
    )
    source["nested"].append(9)
    assert len(policy.extensions["nested"]) == 2
    roundtrip(policy)
    spec = TrainingExecutionSpec(
        compiler=CompilerIdentity(name="test", version="1"),
        candidate_fingerprint="candidate",
        entrypoint=PythonModuleEntrypoint(module="worker"),
        observability=policy,
    )
    roundtrip(spec)
    assert "callbacks" not in type(spec).model_fields
    for key in ("callbacks", "hooks", "on_step"):
        with pytest.raises(ValidationError):
            spec.model_copy(update={key: []})


@pytest.mark.parametrize("field", ["metric_interval_steps", "system_metric_interval_seconds"])
@pytest.mark.parametrize("value", [-1, 0, True, "1", float("nan"), float("inf")])
def test_invalid_intervals(field, value):
    with pytest.raises(ValidationError):
        ObservabilitySpec(**{field: value})


@pytest.mark.parametrize(
    "values",
    [
        {"enabled": True},
        {"enabled": True, "provider": "torch-profiler"},
        {"wait_steps": -1},
        {"warmup_steps": -1},
        {"active_steps": 0},
        {"active_steps": -1},
        {"repeat": 0},
        {"repeat": True},
        {"output_name": "../trace.json"},
        {"trace_data": "inline"},
        {"enabled": "true"},
    ],
)
def test_profiler_rejects_invalid_schedule_or_inline_output(values):
    with pytest.raises(ValidationError):
        ProfilerSpec(**values)


def test_profiler_disabled_and_bounded_schedules_roundtrip():
    roundtrip(ProfilerSpec())
    roundtrip(ProfilerSpec(enabled=True, provider="runtime-native", active_steps=3, repeat=2))
    ref = artifact("profile")
    roundtrip(t.ProfilerArtifactProducedPayload(output_name="profile", artifact_ref=ref))
    with pytest.raises(ValidationError):
        t.ProfilerArtifactProducedPayload(output_name="profile", artifact_ref=artifact())


METRICS = [
    t.ResourceMetricObserved,
    t.TrainingMetricObserved,
    t.DataMetricObserved,
    t.DistributedMetricObserved,
    t.AlignmentMetricObserved,
]


@pytest.mark.parametrize("model", METRICS)
def test_partial_metrics_and_metadata_are_immutable(model):
    observed = model(metadata={"nested": [1, {"count": 0}]})
    roundtrip(observed)
    with pytest.raises(TypeError):
        observed.metadata["nested"][1]["count"] = 2
    for field in model.model_fields:
        if field not in {"type", "metadata"}:
            assert getattr(observed, field) is None


# Exercise every numeric field, including those added to these models later.
NUMERIC_FIELDS = [
    (model, name)
    for model in METRICS
    for name in model.model_fields
    if name not in {"type", "metadata", "worker"}
]


@pytest.mark.parametrize("model,field", NUMERIC_FIELDS)
@pytest.mark.parametrize("value", [True, "1", float("nan"), float("inf"), float("-inf")])
def test_metrics_reject_coercion_and_nonfinite_numbers(model, field, value):
    with pytest.raises(ValidationError):
        model(**{field: value})


@pytest.mark.parametrize(
    "model,field",
    [
        (m, n)
        for m, n in NUMERIC_FIELDS
        if n
        not in {
            "loss",
            "gpu_temperature_celsius",
            "reward_mean",
            "kl_mean",
            "entropy",
            "preference_margin",
        }
    ],
)
def test_counters_rates_durations_and_norms_reject_negative(model, field):
    with pytest.raises(ValidationError):
        model(**{field: -1})


@pytest.mark.parametrize(
    "model,field",
    [
        (t.ResourceMetricObserved, "cpu_utilization"),
        (t.ResourceMetricObserved, "gpu_utilization"),
        (t.ResourceMetricObserved, "host_memory_utilization"),
        (t.DataMetricObserved, "padding_fraction"),
        (t.DataMetricObserved, "truncation_fraction"),
        (t.AlignmentMetricObserved, "rollout_success_rate"),
    ],
)
def test_fraction_units_are_zero_to_one(model, field):
    for value in (0, 1):
        assert getattr(model(**{field: value}), field) == value
    with pytest.raises(ValidationError):
        model(**{field: 1.01})


def test_zero_is_available_and_signed_metrics_remain_signed():
    value = t.TrainingMetricObserved(loss=0, learning_rate=0, optimizer_step=0)
    assert value.loss == value.learning_rate == value.optimizer_step == 0
    assert value.tokens_seen is None
    roundtrip(value)
    assert t.AlignmentMetricObserved(reward_mean=-2, kl_mean=-0.1, entropy=-1).reward_mean == -2
    with pytest.raises(ValidationError):
        t.DistributedMetricObserved(rank=2, world_size=2)


def test_checkpoint_roundtrips_all_resume_dimensions_and_captured_state():
    value = checkpoint()
    roundtrip(value)
    assert value.data_cursor.next_sample_offset == 400
    assert "micro_batch" not in DataCursor.model_fields
    roundtrip(
        value.model_copy(
            update={
                "resume_guarantee": ResumeGuarantee(
                    state="full", data="at-least-once", boundary="optimizer-step"
                )
            }
        )
    )
    iterable = cursor().model_copy(
        update={
            "next_sample_offset": None,
            "iterable_cursor": {"provider": "stream", "offset": 400},
        }
    )
    roundtrip(checkpoint(data_cursor=iterable))


@pytest.mark.parametrize(
    "missing", ["optimizer", "scheduler", "scaler", "rng", "applied_intervention_application_ids"]
)
def test_full_and_exact_claims_require_captured_state(missing):
    incomplete = full_state().model_copy(update={missing: None})
    with pytest.raises(ValidationError):
        checkpoint(state_manifest=incomplete)


@pytest.mark.parametrize(
    "updates",
    [
        {"state_manifest": None},
        {"data_cursor": None},
        {"resume_guarantee": None},
        {
            "data_cursor": DataCursor(
                dataset_fingerprint="d", ordering_fingerprint="o", next_sample_offset=400
            )
        },
        {
            "resume_guarantee": ResumeGuarantee(
                state="full", data="exact", boundary="mid-accumulation"
            )
        },
    ],
)
def test_impossible_checkpoint_claims(updates):
    with pytest.raises(ValidationError):
        checkpoint(**updates)


def test_checkpoint_requires_guarantee_and_consistent_position():
    raw = checkpoint().model_dump()
    del raw["resume_guarantee"]
    with pytest.raises(ValidationError):
        t.CheckpointCommittedPayload.model_validate(raw)
    with pytest.raises(ValidationError):
        cursor().model_copy(update={"iterable_cursor": {"offset": 1}})
    with pytest.raises(ValidationError):
        checkpoint(state_manifest=full_state().model_copy(update={"micro_step": 1}))
    with pytest.raises(ValidationError):
        checkpoint(
            checkpoint_ref=CheckpointRef(
                id=CheckpointId.generate(), uri="file:///c", global_step=99
            )
        )
    # Legacy/weights-only capture cannot claim a recovered data stream.
    roundtrip(
        checkpoint(
            state_manifest=None,
            data_cursor=None,
            resume_guarantee=ResumeGuarantee(
                state="model-only", data="none", boundary="optimizer-step"
            ),
        )
    )


def examples():
    return [
        t.WorkerReadyPayload(pid=1),
        t.HeartbeatPayload(expected_interval_seconds=5),
        t.TrainingStartedPayload(),
        t.TrainingCompletedPayload(),
        t.TrainingFailedPayload(reason="error"),
        t.StepCompletedPayload(optimizer_step=0),
        t.MetricObservedPayload(name="custom", value=0),
        *(m() for m in METRICS),
        t.CheckpointStartedPayload(optimizer_step=0),
        checkpoint(),
        t.ProfilerStartedPayload(output_name="profile", provider="torch-profiler"),
        t.ProfilerCompletedPayload(output_name="profile"),
        t.ProfilerFailedPayload(output_name="profile", reason="error"),
        t.ProfilerArtifactProducedPayload(output_name="profile", artifact_ref=artifact("profile")),
        t.ArtifactProducedPayload(artifact_ref=artifact()),
        t.IncidentObservedPayload(reason="evidence"),
        t.GradientOverflowObserved(),
        t.OptimizerStepSkipped(reason="overflow"),
        t.NumericalInstabilityObserved(quantity="loss", observation="nan"),
    ]


def test_every_training_variant_roundtrips_and_family_boundary_is_exhaustive():
    adapter = TypeAdapter(t.TrainingObservation)
    values = examples()
    # A new union variant requires an example: no silently untested payloads.
    assert {type(v) for v in values} == set(get_args(get_args(t.TrainingObservation)[0]))
    for value in values:
        assert adapter.validate_json(value.model_dump_json()) == value
        payload = TrainingEventPayload(data=value)
        envelope = RuntimeEventEnvelope(
            target={"kind": "training-attempt", "id": "a"},
            event_id="event",
            sequence=0,
            payload=payload,
        )
        roundtrip(envelope)
        with pytest.raises(ValidationError):
            envelope.model_copy(update={"target": {"kind": "evaluation-attempt", "id": "e"}})
    for value in (
        checkpoint(),
        t.TrainingStartedPayload(),
        t.OptimizerStepSkipped(reason="overflow"),
    ):
        with pytest.raises(ValidationError):
            EvaluationEventPayload.model_validate({"data": value.model_dump()})
    with pytest.raises(ValidationError):
        TrainingEventPayload.model_validate({"data": {"type": "EvaluationStarted"}})


def test_structured_log_and_context_roundtrip():
    context = CorrelationContext(attempt_id=RunAttemptId.generate(), stream_generation=0)
    trace = TraceContext(trace_id="trace", span_id="span", trace_flags="01")
    log = RuntimeLog(
        stream="stderr",
        line="message",
        level="warning",
        logger="worker",
        rank=0,
        emitted_at=datetime.now(timezone.utc),
        trace_context=trace,
        context=context,
        attributes={"nested": [0]},
    )
    roundtrip(log)
    with pytest.raises(TypeError):
        log.attributes["new"] = "value"
    with pytest.raises(ValidationError):
        log.model_copy(update={"rank": -1})
    envelope = RuntimeEventEnvelope(
        target={"kind": "training-attempt", "id": str(context.attempt_id)},
        event_id="event",
        sequence=0,
        context=context,
        trace_context=trace,
        payload=TrainingEventPayload(data=t.WorkerReadyPayload()),
    )
    roundtrip(envelope)
    for bad_context in (
        CorrelationContext(stream_generation=1),
        CorrelationContext(attempt_id=RunAttemptId.generate()),
        CorrelationContext(evaluation_attempt_id="evaluation"),
    ):
        with pytest.raises(ValidationError):
            envelope.model_copy(update={"context": bad_context})


@pytest.mark.parametrize(
    "value", [object(), lambda: None, SecretStr("secret"), SecretBytes(b"secret")]
)
def test_live_values_and_secret_objects_cannot_hide_in_escape_hatches(value):
    for model, field in [
        (ObservabilitySpec, "extensions"),
        (t.TrainingMetricObserved, "metadata"),
        (RuntimeLog, "attributes"),
    ]:
        with pytest.raises(ValueError):
            model(
                **{field: {"nested": [value]}, **({"line": "log"} if model is RuntimeLog else {})}
            )
    with pytest.raises(ValueError):
        TrainingExecutionSpec(
            compiler=CompilerIdentity(name="test", version="1"),
            candidate_fingerprint="candidate",
            entrypoint=PythonModuleEntrypoint(module="worker"),
            config={"nested": [value]},
        )


def test_secret_refs_are_reference_only_and_redaction_has_no_value_slot():
    assert set(SecretRef.model_fields) == {"name", "source", "version"}
    with pytest.raises(ValidationError):
        SecretRef(name="API_KEY", source="env", value="secret")
    policy = RedactionPolicy(
        redact_environment_keys=("API_KEY",),
        redact_attribute_keys=("authorization",),
        redact_patterns=("Bearer .*",),
    )
    roundtrip(policy)
    with pytest.raises(ValidationError):
        policy.model_copy(update={"secret_value": "secret"})
    with pytest.raises(ValueError):
        FrozenDict({"credentials": SecretRef(name="key", source="env")})


def test_event_sink_is_a_protocol_over_committed_domain_events():
    import inspect

    from xaytune.core.sinks import EventSink

    assert inspect.iscoroutinefunction(EventSink.consume)
    assert list(inspect.signature(EventSink.consume).parameters) == ["self", "event"]


def test_all_evaluation_variants_roundtrip():
    values = [
        t.WorkerReadyPayload(),
        t.HeartbeatPayload(expected_interval_seconds=1),
        t.IncidentObservedPayload(reason="evidence"),
        t.ArtifactProducedPayload(artifact_ref=artifact()),
        t.ResourceMetricObserved(),
        t.EvaluationStartedPayload(),
        t.EvaluationCompletedPayload(),
        t.EvaluationFailedPayload(reason="error"),
        t.EvaluationProgressPayload(examples_completed=0),
        t.MetricObservedPayload(name="accuracy", value=0),
    ]
    assert {type(v) for v in values} == set(get_args(get_args(t.EvaluationObservation)[0]))
    for value in values:
        roundtrip(EvaluationEventPayload(data=value))


def test_legacy_complete_events_are_not_silently_dropped(tmp_path):
    """A v1alpha1 envelope is a hole in the stream, not something to skip.

    Same intent as before; ``_read_events`` became the shared
    ``AppendOnlyJsonlReader`` so the launcher could tail worker observations
    with the same primitive. The two properties it asserted are the reader's:
    a complete record that does not validate raises, a partial one waits.
    """
    from pydantic import TypeAdapter

    from xaytune.runtimes import RuntimeEventEnvelope
    from xaytune.runtimes.local.jsonl import AppendOnlyJsonlReader, CorruptRecordError

    adapter = TypeAdapter(RuntimeEventEnvelope)

    path = tmp_path / "events.jsonl"
    path.write_text('{"protocol_version":"xaytune.telemetry/v1alpha1"}\n')
    with pytest.raises(CorruptRecordError):
        AppendOnlyJsonlReader(path, adapter).read_new()

    partial = tmp_path / "partial.jsonl"
    partial.write_text('{"partial":')
    assert AppendOnlyJsonlReader(partial, adapter).read_new() == ()


def test_runtime_refuses_incompatible_telemetry_protocol():
    from xaytune.core.capabilities import PLUGIN_API_VERSIONS, PluginDescriptor
    from xaytune.core.execution import ResolvedExecutionPlan, TelemetryContract
    from xaytune.runtimes.local.runtime import _refuse

    # A supported descriptor, so this isolates the telemetry refusal: ADR-008
    # now refuses a plan whose producer is unidentifiable, and that check runs
    # first.
    descriptor = PluginDescriptor(
        api_version=PLUGIN_API_VERSIONS[0],
        name="test",
        plugin_version="1",
        provider="tests",
        xaytune_version="0.6.0",
    )
    plan = ResolvedExecutionPlan(
        runtime="local",
        target={"kind": "training-attempt", "id": "a"},
        spec=TrainingExecutionSpec(
            compiler=CompilerIdentity(name="test", version="1", descriptor=descriptor),
            candidate_fingerprint="c",
            entrypoint=PythonModuleEntrypoint(module="worker"),
            telemetry=TelemetryContract(protocol_version="xaytune.telemetry/v1alpha1"),
        ),
    )
    assert "v1alpha2" in _refuse(plan)
