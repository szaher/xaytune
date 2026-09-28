"""Structured evidence determines a diagnosis, never a proposed response."""

from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import ValidationError

from xaytune.core.domain.incident import (
    AttemptContext,
    DetectorProvenance,
    Incident,
    IncidentCandidate,
    IncidentCategory,
)
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.ids import ExperimentId, ExperimentNodeId, RunAttemptId, RunId
from xaytune.core.telemetry import (
    IncidentObservedPayload,
    NumericalInstabilityObserved,
    TrainingFailedPayload,
    WorkerReadyPayload,
)
from xaytune.resilience import IncidentClassifier
from xaytune.runtimes import RuntimeEventEnvelope, TrainingEventPayload


def context() -> AttemptContext:
    return AttemptContext(
        experiment_id=ExperimentId.generate(),
        node_id=ExperimentNodeId.generate(),
        run_id=str(RunId.generate()),
        target=RuntimeOperationTarget(kind="training-attempt", id=str(RunAttemptId.generate())),
    )


def envelope(owner: AttemptContext, signal=None, **changes) -> RuntimeEventEnvelope:
    return RuntimeEventEnvelope(
        event_id="authoritative-event",
        target=owner.target,
        stream_generation=0,
        sequence=3,
        payload=TrainingEventPayload(data=signal or IncidentObservedPayload(reason="cuda-oom")),
        **changes,
    )


def classify(event: RuntimeEventEnvelope, owner: AttemptContext) -> Incident:
    incident = IncidentClassifier().inspect(
        event.payload.data, owner, evidence=event.model_dump(mode="json")
    )
    assert incident is not None
    return incident


@pytest.mark.parametrize(
    ("reason", "category"),
    [
        ("process-failure", IncidentCategory.PROCESS_FAILURE),
        ("nonzero-exit", IncidentCategory.PROCESS_FAILURE),
        ("signalled", IncidentCategory.PROCESS_FAILURE),
        ("spawn-failed", IncidentCategory.PROCESS_FAILURE),
        ("cuda-oom", IncidentCategory.CUDA_OOM),
        ("checkpoint-write-failure", IncidentCategory.CHECKPOINT_WRITE_FAILURE),
        ("checkpoint-corruption", IncidentCategory.CHECKPOINT_CORRUPTION),
        ("checkpoint-incompatible", IncidentCategory.CHECKPOINT_INCOMPATIBLE),
        ("out-of-memory-error", IncidentCategory.UNKNOWN),
        ("runtime-error", IncidentCategory.UNKNOWN),
        ("some-new-failure", IncidentCategory.UNKNOWN),
    ],
)
@pytest.mark.parametrize("payload", [IncidentObservedPayload, TrainingFailedPayload])
def test_structured_reasons_determine_the_category(reason, category, payload) -> None:
    owner = context()
    signal = payload(reason=reason, detail="CUDA out of memory! NaN! checkpoint corrupted!")
    incident = classify(envelope(owner, signal), owner)
    assert incident.category is category
    assert incident.evidence["payload"]["data"] == signal.model_dump(mode="json")
    assert incident.classifier == DetectorProvenance(name="structured-incidents", version="1")
    assert len(incident.detectors) == 4
    if category is not IncidentCategory.UNKNOWN:
        assert len(incident.candidates) == 1
        assert incident.candidates[0].detector in incident.detectors


@pytest.mark.parametrize(
    ("observation", "category"),
    [
        ("nan", IncidentCategory.NUMERICAL_NAN),
        ("positive-infinity", IncidentCategory.NUMERICAL_INF),
        ("negative-infinity", IncidentCategory.NUMERICAL_INF),
        ("unstable", IncidentCategory.UNKNOWN),
    ],
)
def test_nonfinite_evidence_is_symbolic_and_preserved(observation, category) -> None:
    owner = context()
    signal = NumericalInstabilityObserved(
        quantity="loss",
        observation=observation,
        optimizer_step=9,
        metadata={"rank": 2, "evidence": {"samples": [1, 2]}},
    )
    incident = classify(envelope(owner, signal), owner)
    assert incident.category is category
    assert incident.evidence["payload"]["data"]["optimizer_step"] == 9
    assert Incident.model_validate_json(incident.model_dump_json()) == incident
    with pytest.raises(TypeError):
        incident.evidence["payload"]["data"]["metadata"]["rank"] = 7
    with pytest.raises(ValidationError):
        incident.category = IncidentCategory.UNKNOWN


def test_benign_observations_produce_no_incident() -> None:
    owner = context()
    event = envelope(owner, WorkerReadyPayload())
    assert (
        IncidentClassifier().inspect(
            event.payload.data, owner, evidence=event.model_dump(mode="json")
        )
        is None
    )


def test_an_authoritatively_cancelled_process_is_not_a_failure() -> None:
    owner = context()
    signal = IncidentObservedPayload(
        reason="signalled", exit_code=-15, metadata={"cancelled": True}
    )
    event = envelope(owner, signal)
    assert (
        IncidentClassifier().inspect(signal, owner, evidence=event.model_dump(mode="json")) is None
    )


@pytest.mark.parametrize(
    ("reason", "code"),
    [("nonzero-exit", 0), ("nonzero-exit", -15), ("signalled", 1), ("spawn-failed", 1)],
)
def test_contradictory_process_evidence_is_unknown(reason, code) -> None:
    owner = context()
    signal = IncidentObservedPayload(reason=reason, exit_code=code)
    assert classify(envelope(owner, signal), owner).category is IncidentCategory.UNKNOWN


def test_conflicting_diagnoses_are_unknown_regardless_of_order() -> None:
    owner = context()
    classifier = IncidentClassifier()
    candidates = (
        IncidentCandidate(
            category=IncidentCategory.CUDA_OOM,
            detector=DetectorProvenance(name="cuda-oom", version="1"),
            reason="first diagnosis",
        ),
        IncidentCandidate(
            category=IncidentCategory.PROCESS_FAILURE,
            detector=DetectorProvenance(name="process-failure", version="1"),
            reason="second diagnosis",
        ),
    )
    for ordered in (candidates, tuple(reversed(candidates))):
        incident = classifier.classify(
            ordered, owner, evidence=envelope(owner).model_dump(mode="json")
        )
        assert incident.category is IncidentCategory.UNKNOWN
        assert incident.candidates == ordered


def test_signal_cannot_be_classified_against_different_evidence() -> None:
    owner = context()
    with pytest.raises(ValueError, match="signal and evidence disagree"):
        IncidentClassifier().inspect(
            IncidentObservedPayload(reason="process-failure"),
            owner,
            evidence=envelope(owner).model_dump(mode="json"),
        )


def test_signal_and_evidence_comparison_distinguishes_boolean_from_integer() -> None:
    owner = context()
    signal = IncidentObservedPayload(reason="cuda-oom", metadata={"rank": 1})
    evidence = envelope(owner, signal).model_dump(mode="json")
    evidence["payload"]["data"]["metadata"]["rank"] = True
    with pytest.raises(ValueError, match="signal and evidence disagree"):
        IncidentClassifier().inspect(signal, owner, evidence=evidence)


@pytest.mark.parametrize("field", ["sequence", "stream_generation"])
def test_boolean_counters_cannot_alias_integer_observation_keys(field) -> None:
    owner = context()
    event = envelope(owner)
    incident = classify(event, owner)
    evidence = event.model_dump(mode="json")
    evidence[field] = False
    with pytest.raises(ValidationError, match="observation keys disagree"):
        Incident.model_validate({**incident.model_dump(mode="json"), "evidence": evidence})


def test_an_incident_cannot_claim_a_different_workload_family() -> None:
    owner = context()
    incident = classify(envelope(owner), owner)
    evidence = incident.model_dump(mode="json")["evidence"]
    evidence["payload"]["workload"] = "evaluation"
    with pytest.raises(ValidationError, match="another workload family"):
        Incident.model_validate({**incident.model_dump(mode="json"), "evidence": evidence})


def test_observation_key_is_not_a_recovery_signature_or_a_generated_id() -> None:
    owner = context()
    event = envelope(owner)
    first, second = classify(event, owner), classify(event, owner)
    assert first.id != second.id
    assert first.observation_key == second.observation_key
    different_delivery = classify(event.model_copy(update={"sequence": 4}), owner)
    assert first.category == different_delivery.category
    assert first.observation_key != different_delivery.observation_key
    regenerated = event.model_copy(update={"stream_generation": 1})
    assert classify(regenerated, owner).observation_key != first.observation_key
    code = (
        "from xaytune.core.domain.incident import Incident; "
        "import sys; print(Incident.model_validate_json(sys.argv[1]).observation_key)"
    )
    result = subprocess.check_output(
        [sys.executable, "-c", code, first.model_dump_json()], text=True
    )
    assert result.strip() == first.observation_key
