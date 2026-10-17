"""The candidate envelope and the training compatibility boundary (H02, ADR-018 D1, D2, D10).

D10's durable step is a node snapshot that can hold the envelope: version
``"2"``. Training snapshots stay version ``"1"``, byte for byte.

The training literals here are the ones ``test_candidate.py`` already pins --
not new expectations -- so a regression in the envelope cannot hide behind a
freshly computed value.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from tests.test_core.test_candidate import _plain_candidate
from tests.test_core.test_harness import BASE, _harness, _record
from xaytune.core.domain.candidate import CandidateSpec
from xaytune.core.domain.candidate_envelope import (
    AgentHarnessCandidate,
    CandidateEnvelopeError,
    CandidateKind,
    ExperimentCandidate,
    NotATrainingCandidateError,
    TrainingCandidate,
    candidate_of_snapshot,
    read_candidate,
)
from xaytune.core.domain.experiment import CandidateSpecSnapshot, ExperimentNode
from xaytune.core.domain.planning import CandidateProposal, NodeSummary
from xaytune.experiment.spec import ExperimentSpec

V2 = "sha256:730b6faed16f611f7be3ed499a7e72e4703883d6cfff772bfbef7d0351b13fe4"
V1 = "sha256:3c174a88119ffa7ed47693e5854398f10277ec46c2f972de6adfbfc5a73ac4a9"

HISTORICAL = {
    "model": {"model": {"uri": "Qwen/Qwen3-8B"}},
    "data": {"dataset": {"uri": "./support-v4.jsonl"}},
    "training": {"kind": "sft"},
}
"""A training candidate as a record written before H02 stores it: unwrapped, minimal."""


# ---- training, unchanged --------------------------------------------------------------


def test_a_training_candidate_keeps_its_historical_fingerprints() -> None:
    candidate = ExperimentCandidate.training(_plain_candidate())
    assert candidate.kind is CandidateKind.TRAINING
    assert candidate.scientific_fingerprint() == V2
    assert candidate.fingerprints_for_lookup() == (V2, V1)


def test_a_historical_unwrapped_record_reads_as_training_without_rewriting() -> None:
    candidate = read_candidate(HISTORICAL)
    assert candidate.kind is CandidateKind.TRAINING
    assert isinstance(candidate.payload, TrainingCandidate)
    assert candidate.payload.spec == _plain_candidate()
    assert candidate.fingerprints_for_lookup() == (V2, V1)
    full = _plain_candidate().model_dump(mode="json")
    assert read_candidate(full).payload == candidate.payload


def test_the_training_payload_serializes_exactly_as_candidate_spec_does() -> None:
    spec = _plain_candidate()
    dumped = ExperimentCandidate.training(spec).model_dump(mode="json")
    assert dumped == {
        "envelope_version": "xaytune.candidate/v1alpha1",
        "payload": {"kind": "training", "spec": spec.model_dump(mode="json")},
    }
    assert json.dumps(dumped["payload"]["spec"], sort_keys=True) == json.dumps(
        spec.model_dump(mode="json"), sort_keys=True
    )


def test_a_version_one_snapshot_reads_as_training_and_serializes_as_it_always_has() -> None:
    snapshot = CandidateSpecSnapshot.model_validate({"candidate": HISTORICAL, "spec_version": "1"})
    assert candidate_of_snapshot(snapshot).fingerprints_for_lookup() == (V2, V1)
    assert snapshot.training_spec() == _plain_candidate()
    dumped = snapshot.model_dump(mode="json")
    assert set(dumped) == {"candidate", "spec_version", "metadata"}, "no envelope: null"
    of = CandidateSpecSnapshot.of(ExperimentCandidate.training(_plain_candidate()))
    assert of.spec_version == "1" and of.model_dump(mode="json") == dumped


def test_a_version_two_snapshot_holds_a_harness_and_refuses_training_only_work() -> None:
    harness = _harness()
    snapshot = CandidateSpecSnapshot.of(ExperimentCandidate.agent_harness(harness))
    assert snapshot.spec_version == "2"
    dumped = snapshot.model_dump(mode="json")
    assert set(dumped) == {"envelope", "spec_version", "metadata"}
    assert CandidateSpecSnapshot.model_validate(dumped) == snapshot
    assert snapshot.candidate_kind is CandidateKind.AGENT_HARNESS
    assert snapshot.scientific_fingerprint() == harness.harness_fingerprint()
    assert candidate_of_snapshot(snapshot).payload == AgentHarnessCandidate(harness=harness)
    with pytest.raises(NotATrainingCandidateError, match="agent-harness candidate"):
        snapshot.training_spec()
    with pytest.raises(NotATrainingCandidateError):
        _ = snapshot.kind


def test_a_version_two_snapshot_may_hold_training_too_and_reads_the_same() -> None:
    envelope = ExperimentCandidate.training(_plain_candidate())
    snapshot = CandidateSpecSnapshot(envelope=envelope, spec_version="2")
    assert snapshot.training_spec() == _plain_candidate()
    assert snapshot.scientific_fingerprint() == V2


@pytest.mark.parametrize(
    ("record", "error"),
    [
        ({"candidate": HISTORICAL, "spec_version": "3"}, CandidateEnvelopeError),
        ({"spec_version": "1"}, ValidationError),
        ({"spec_version": "2", "candidate": HISTORICAL}, ValidationError),
        (
            {
                "spec_version": "1",
                "candidate": HISTORICAL,
                "envelope": ExperimentCandidate.agent_harness(_harness()).model_dump(mode="json"),
            },
            ValidationError,
        ),
    ],
    ids=["unknown-version", "v1-empty", "v2-with-a-spec", "v1-with-an-envelope"],
)
def test_a_snapshot_of_an_unknown_or_mixed_shape_is_refused(
    record: dict[str, Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        CandidateSpecSnapshot.model_validate(record)


def test_the_planning_and_submission_seams_still_carry_candidate_spec_only() -> None:
    """D10 makes a node able to hold a harness; specs, proposals and planning stay training."""
    assert CandidateSpecSnapshot.model_fields["candidate"].annotation == CandidateSpec | None
    assert CandidateSpecSnapshot.model_fields["envelope"].annotation == ExperimentCandidate | None
    assert ExperimentSpec.model_fields["candidate"].annotation is CandidateSpec
    assert CandidateProposal.model_fields["candidate"].annotation is CandidateSpec
    assert "candidate" in ExperimentNode.model_fields
    assert set(CandidateSpec.model_fields) == {
        "model",
        "data",
        "training",
        "reward",
        "environment",
        "schedule",
        "metadata",
    }
    assert NodeSummary.model_fields["candidate"].annotation is CandidateSpec


# ---- the envelope -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "candidate",
    [
        ExperimentCandidate.training(_plain_candidate()),
        ExperimentCandidate.agent_harness(_harness()),
    ],
)
def test_either_kind_round_trips_deterministically(candidate: ExperimentCandidate) -> None:
    dumped = candidate.model_dump(mode="json")
    again = read_candidate(json.loads(json.dumps(dumped)))
    assert again == candidate
    assert again.model_dump_json() == candidate.model_dump_json()
    assert again.scientific_fingerprint() == candidate.scientific_fingerprint()
    assert again.kind is candidate.kind


def test_a_harness_candidate_has_the_harness_identity_and_nothing_else() -> None:
    candidate = read_candidate(
        ExperimentCandidate.agent_harness(_harness()).model_dump(mode="json")
    )
    assert candidate.kind is CandidateKind.AGENT_HARNESS
    assert isinstance(candidate.payload, AgentHarnessCandidate)
    assert candidate.scientific_fingerprint() == BASE
    assert candidate.fingerprints_for_lookup() == (BASE,)


def _envelope(payload: Any, version: Any = "xaytune.candidate/v1alpha1") -> dict[str, Any]:
    return {"envelope_version": version, "payload": payload}


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (_envelope({"kind": "training", "spec": HISTORICAL}, "xaytune.candidate/v1"), "envelope"),
        (
            _envelope({"kind": "training", "spec": HISTORICAL}, "xaytune.candidate/v2alpha1"),
            "envelope",
        ),
        (_envelope({"kind": "joint", "spec": HISTORICAL}), "kind 'joint'"),
        (_envelope({"spec": HISTORICAL}), "kind None"),
        (_envelope(HISTORICAL), "kind None"),
        # No coercion either way: each payload is read only as the kind it names.
        (_envelope({"kind": "training", "harness": _record()}), "invalid training"),
        (_envelope({"kind": "agent-harness", "spec": HISTORICAL}), "invalid agent-harness"),
        (_envelope({"kind": "agent-harness", "harness": HISTORICAL}), "invalid agent-harness"),
        (_record(), "unwrapped xaytune.agent-harness"),
        ({"model": "gpt"}, "not a historical CandidateSpec"),
        ({}, "not a historical CandidateSpec"),
    ],
)
def test_unknown_versions_kinds_and_ambiguous_records_are_refused(
    record: dict[str, Any], message: str
) -> None:
    with pytest.raises(CandidateEnvelopeError, match=message):
        read_candidate(record)


def test_a_record_that_is_not_a_mapping_is_refused() -> None:
    with pytest.raises(CandidateEnvelopeError, match="mapping"):
        read_candidate(["training"])  # type: ignore[arg-type]


# ---- the version is never supplied (PR #74 review, S3) ---------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "training", "spec": HISTORICAL},
        {"kind": "agent-harness", "harness": _record()},
    ],
)
def test_parsing_an_envelope_never_supplies_a_missing_version(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="envelope_version"):
        ExperimentCandidate.model_validate({"payload": payload})
    with pytest.raises(ValidationError, match="envelope_version"):
        ExperimentCandidate.model_validate_json(json.dumps({"payload": payload}))
    with pytest.raises(CandidateEnvelopeError):
        read_candidate({"payload": payload})


def test_the_constructors_state_the_version_explicitly() -> None:
    for candidate in (
        ExperimentCandidate.training(_plain_candidate()),
        ExperimentCandidate.agent_harness(_harness()),
    ):
        assert candidate.model_dump(mode="json")["envelope_version"] == (
            "xaytune.candidate/v1alpha1"
        )
        assert "envelope_version" in candidate.model_fields_set
