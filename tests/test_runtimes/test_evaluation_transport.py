"""An evaluation runs through the same transport as training, as its own workload.

```text
EvaluationExecutionSpec ─┐
TrainingExecutionSpec   ─┴─ ExecutionSpec (wire union, api_version)
                              ↓ ResolvedExecutionPlan(target=evaluation-attempt)
LocalRuntime                  workload-blind: starts the process, checks the producer
launcher                      EvaluationEventPayload only, protocol v1alpha3
```

The runtime does not learn what evaluation is. What is checked here is that
the transport carries it faithfully: the right payload family for the target,
the protocol the plan declared, and a completion that carries its result.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from xaytune.core.capabilities import PLUGIN_API_VERSIONS, PluginDescriptor
from xaytune.core.domain.evaluation import MetricResult
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.execution import (
    CommandEntrypoint,
    CompilerIdentity,
    EvaluationExecutionSpec,
    EvaluatorIdentity,
    ExecutionSpec,
    PythonModuleEntrypoint,
    ResolvedExecutionPlan,
    TelemetryContract,
    TrainingExecutionSpec,
)
from xaytune.core.ids import ArtifactId, OperationId
from xaytune.core.refs import ArtifactRef
from xaytune.core.telemetry import (
    EvaluationCompletedPayload,
    EvaluationStartedPayload,
    TrainingStartedPayload,
)
from xaytune.runtimes import EvaluationEventPayload, RuntimeEventEnvelope, TrainingEventPayload
from xaytune.runtimes.local import LocalRuntime, UnsupportedPlanError

_DESCRIPTOR = PluginDescriptor(
    api_version=PLUGIN_API_VERSIONS[0],
    name="test-evaluator",
    plugin_version="0.1.0",
    provider="tests",
    xaytune_version="0.6.0",
)
_TERMINAL = frozenset({"succeeded", "failed", "cancelled", "unknown"})
_EVALUATION = RuntimeOperationTarget(kind="evaluation-attempt", id="evalattempt_transport")
_TRAINING = RuntimeOperationTarget(kind="training-attempt", id="attempt_transport")

_PRELUDE = (
    "from xaytune.core import telemetry as t\n"
    "from xaytune.core.domain.evaluation import MetricResult\n"
    "from xaytune.runtimes.worker import ObservationWriter\n"
    "writer = ObservationWriter.from_environment()\n"
)


def _subject() -> ArtifactRef:
    return ArtifactRef(id=ArtifactId.generate(), kind="model", uri="/models/m", digest="sha256:m")


def _evaluation_spec(**overrides: object) -> EvaluationExecutionSpec:
    fields: dict[str, object] = {
        "evaluator": EvaluatorIdentity(
            name="test-evaluator", version="0.1.0", descriptor=_DESCRIPTOR
        ),
        "evaluation_fingerprint": "sha256:" + "e" * 64,
        "subject": _subject(),
        "entrypoint": PythonModuleEntrypoint(module="evaluator"),
    }
    fields.update(overrides)
    return EvaluationExecutionSpec(**fields)  # type: ignore[arg-type]


def _training_spec(**overrides: object) -> TrainingExecutionSpec:
    fields: dict[str, object] = {
        "compiler": CompilerIdentity(name="fake", version="0.1.0", descriptor=_DESCRIPTOR),
        "candidate_fingerprint": "sha256:" + "0" * 64,
        "entrypoint": PythonModuleEntrypoint(module="trainer"),
    }
    fields.update(overrides)
    return TrainingExecutionSpec(**fields)  # type: ignore[arg-type]


def _metric() -> MetricResult:
    return MetricResult(name="accuracy", value=0.75, evaluator_name="test-evaluator", seed=7)


# ---- the wire union ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "target"),
    [(_training_spec(), _TRAINING), (_evaluation_spec(), _EVALUATION)],
    ids=["training", "evaluation"],
)
def test_a_plan_round_trips_as_the_workload_it_was(spec: object, target: object) -> None:
    plan = ResolvedExecutionPlan(spec=spec, runtime="local", target=target)  # type: ignore[arg-type]
    restored = ResolvedExecutionPlan.model_validate_json(plan.model_dump_json())
    assert type(restored.spec) is type(spec)
    assert restored.request_digest("submit") == plan.request_digest("submit")


def test_the_union_is_told_apart_by_api_version() -> None:
    adapter: TypeAdapter[object] = TypeAdapter(ExecutionSpec)
    assert isinstance(
        adapter.validate_python(_evaluation_spec().model_dump()), EvaluationExecutionSpec
    )
    assert isinstance(adapter.validate_python(_training_spec().model_dump()), TrainingExecutionSpec)


@pytest.mark.parametrize(
    ("spec", "target"),
    [(_training_spec(), _EVALUATION), (_evaluation_spec(), _TRAINING)],
    ids=["training-spec-for-evaluation", "evaluation-spec-for-training"],
)
def test_a_spec_runs_only_for_its_own_kind_of_attempt(spec: object, target: object) -> None:
    with pytest.raises(ValidationError, match="two statements about the same workload"):
        ResolvedExecutionPlan(spec=spec, runtime="local", target=target)  # type: ignore[arg-type]


def test_an_evaluation_reports_under_the_result_carrying_protocol() -> None:
    assert _evaluation_spec().telemetry.protocol_version == "xaytune.telemetry/v1alpha3"
    with pytest.raises(ValidationError, match="v1alpha3"):
        _evaluation_spec(telemetry=TelemetryContract(protocol_version="xaytune.telemetry/v1alpha2"))


def test_evaluation_left_the_training_request_digest_unchanged() -> None:
    """A recorded training submission must still match after the upgrade.

    ``request_digest`` hashes the whole plan. Had evaluation added a field to
    the training spec or the plan, every training operation recorded before it
    would fail its digest check on re-issue. Pinned to the value ``main``
    produced before this change.
    """
    plan = ResolvedExecutionPlan(
        spec=TrainingExecutionSpec(
            compiler=CompilerIdentity(name="native", version="1.0.0"),
            candidate_fingerprint="sha256:" + "0" * 64,
            entrypoint=PythonModuleEntrypoint(module="xaytune.workers.native"),
            config={"learning_rate": 0.001},
        ),
        runtime="local",
        target=RuntimeOperationTarget(kind="training-attempt", id="attempt_x"),
    )
    assert plan.request_digest("submit") == _PINNED_TRAINING_DIGEST


_PINNED_TRAINING_DIGEST = "sha256:bb3feb32896bf5f54c793ffb2aeeeaec3540398de892eb737cd6ee8f5c3c3813"


# ---- the envelope pairs the completion with its protocol ---------------------------


def _envelope(data: object, *, protocol: str, target: object = _EVALUATION) -> RuntimeEventEnvelope:
    return RuntimeEventEnvelope(
        protocol_version=protocol,  # type: ignore[arg-type]
        event_id="e-0",
        target=target,  # type: ignore[arg-type]
        sequence=0,
        payload=EvaluationEventPayload(data=data),  # type: ignore[arg-type]
    )


def test_a_v1alpha3_completion_carries_its_metrics() -> None:
    envelope = _envelope(
        EvaluationCompletedPayload(metrics=(_metric(),)), protocol="xaytune.telemetry/v1alpha3"
    )
    assert envelope.payload.data.metrics == (_metric(),)  # type: ignore[union-attr]
    with pytest.raises(ValidationError, match="must carry its metrics"):
        _envelope(EvaluationCompletedPayload(), protocol="xaytune.telemetry/v1alpha3")


def test_a_v1alpha2_completion_is_still_read_and_carries_none() -> None:
    _envelope(EvaluationCompletedPayload(), protocol="xaytune.telemetry/v1alpha2")
    with pytest.raises(ValidationError, match="carries no metrics"):
        _envelope(
            EvaluationCompletedPayload(metrics=(_metric(),)),
            protocol="xaytune.telemetry/v1alpha2",
        )


def test_an_empty_result_is_not_a_result() -> None:
    with pytest.raises(ValidationError):
        EvaluationCompletedPayload(metrics=())


@pytest.mark.parametrize(
    ("target", "payload"),
    [
        (_EVALUATION, TrainingEventPayload(data=TrainingStartedPayload())),
        (_TRAINING, EvaluationEventPayload(data=EvaluationStartedPayload())),
    ],
    ids=["training-payload-for-evaluation", "evaluation-payload-for-training"],
)
def test_the_payload_family_must_match_the_target(target: object, payload: object) -> None:
    with pytest.raises(ValidationError, match="telemetry, not"):
        RuntimeEventEnvelope(event_id="e", target=target, sequence=0, payload=payload)  # type: ignore[arg-type]


# ---- LocalRuntime runs it without knowing what it is -------------------------------


def _run(tmp_path: Path, body: str, **overrides: object) -> tuple[object, list]:
    runtime = LocalRuntime(tmp_path / "runtime")
    spec = _evaluation_spec(
        entrypoint=CommandEntrypoint(argv=(sys.executable, "-c", _PRELUDE + body)), **overrides
    )
    plan = ResolvedExecutionPlan(spec=spec, runtime="local", target=_EVALUATION)

    async def scenario() -> tuple[object, list]:
        ref = await runtime.submit_or_get(OperationId.generate(), plan)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 30.0
        status = await runtime.get_status(ref)
        while status.state not in _TERMINAL and loop.time() < deadline:
            await asyncio.sleep(0.02)
            status = await runtime.get_status(ref)
        return status, [event async for event in runtime.watch(ref)]

    try:
        return asyncio.run(scenario())
    finally:
        runtime.close()


def test_an_evaluation_workload_reports_evaluation_telemetry_under_v1alpha3(
    tmp_path: Path,
) -> None:
    status, events = _run(
        tmp_path,
        "writer.write(t.EvaluationStartedPayload())\n"
        "writer.write(t.MetricObservedPayload(name='accuracy', value=0.5))\n"
        "writer.write(t.EvaluationCompletedPayload(metrics=(MetricResult("
        "name='accuracy', value=0.75, evaluator_name='test-evaluator', seed=7),)))\n",
    )

    assert status.state == "succeeded"  # type: ignore[attr-defined]
    assert [event.payload.data.type for event in events] == [
        "WorkerReady",
        "EvaluationStarted",
        "MetricObserved",
        "EvaluationCompleted",
    ]
    assert all(isinstance(event.payload, EvaluationEventPayload) for event in events)
    assert {event.protocol_version for event in events} == {"xaytune.telemetry/v1alpha3"}
    assert {event.target for event in events} == {_EVALUATION}
    assert events[-1].payload.data.metrics == (_metric(),)


def test_an_evaluation_workload_cannot_report_training(tmp_path: Path) -> None:
    """A training observation from an evaluation is refused as an incident, not relayed.

    The reader validates each record against the family the target may carry,
    so the line does not parse as an evaluation observation at all.
    """
    status, events = _run(tmp_path, "writer.write(t.TrainingStartedPayload())\n")

    assert status.state == "succeeded"  # type: ignore[attr-defined]
    types = [event.payload.data.type for event in events]
    assert "TrainingStarted" not in types
    assert [
        event.payload.data.reason
        for event in events
        if event.payload.data.type == "IncidentObserved"
    ] == ["corrupt-observation"]


def test_a_completion_without_its_result_is_refused_on_the_way_in(tmp_path: Path) -> None:
    """Under v1alpha3 the worker cannot report success without saying what it measured."""
    _, events = _run(tmp_path, "writer.write(t.EvaluationCompletedPayload())\n")

    assert "EvaluationCompleted" not in [event.payload.data.type for event in events]
    assert "invalid-observation" in [
        event.payload.data.reason
        for event in events
        if event.payload.data.type == "IncidentObserved"
    ]


def test_an_evaluator_with_no_descriptor_is_refused_by_name(tmp_path: Path) -> None:
    runtime = LocalRuntime(tmp_path / "runtime")
    plan = ResolvedExecutionPlan(
        spec=_evaluation_spec(evaluator=EvaluatorIdentity(name="anonymous", version="0")),
        runtime="local",
        target=_EVALUATION,
    )
    try:
        with pytest.raises(UnsupportedPlanError, match="'anonymous' as its producer"):
            asyncio.run(runtime.submit_or_get(OperationId.generate(), plan))
    finally:
        runtime.close()
