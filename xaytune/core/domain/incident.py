"""Immutable diagnoses of observations, without a recovery decision (PR-017).

The observation key identifies one delivered fact, not a repeating failure
pattern. Recovery-loop signatures and recoverability belong to later PRs.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from xaytune.core.clock import utc_now
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import ExperimentId, ExperimentNodeId, IncidentId
from xaytune.core.immutable import FrozenDict, FrozenDomainModel


class IncidentCategory(str, Enum):
    """The runtime-neutral vocabulary in resilience specification §2."""

    PROCESS_FAILURE = "PROCESS_FAILURE"
    WORKER_FAILURE = "WORKER_FAILURE"
    NODE_FAILURE = "NODE_FAILURE"
    DRIVER_FAILURE = "DRIVER_FAILURE"
    PREEMPTION = "PREEMPTION"
    CUDA_OOM = "CUDA_OOM"
    HOST_OOM = "HOST_OOM"
    DISK_FULL = "DISK_FULL"
    NETWORK_FAILURE = "NETWORK_FAILURE"
    OBJECT_STORE_FAILURE = "OBJECT_STORE_FAILURE"
    CHECKPOINT_WRITE_FAILURE = "CHECKPOINT_WRITE_FAILURE"
    CHECKPOINT_CORRUPTION = "CHECKPOINT_CORRUPTION"
    CHECKPOINT_INCOMPATIBLE = "CHECKPOINT_INCOMPATIBLE"
    DATA_ERROR = "DATA_ERROR"
    DATA_CORRUPTION = "DATA_CORRUPTION"
    NUMERICAL_NAN = "NUMERICAL_NAN"
    NUMERICAL_INF = "NUMERICAL_INF"
    GRADIENT_EXPLOSION = "GRADIENT_EXPLOSION"
    LOSS_DIVERGENCE = "LOSS_DIVERGENCE"
    TRAINING_STALL = "TRAINING_STALL"
    QUALITY_REGRESSION = "QUALITY_REGRESSION"
    REWARD_COLLAPSE = "REWARD_COLLAPSE"
    KL_EXPLOSION = "KL_EXPLOSION"
    TIMEOUT = "TIMEOUT"
    CONFIG_ERROR = "CONFIG_ERROR"
    USER_ERROR = "USER_ERROR"
    UNKNOWN = "UNKNOWN"


class AttemptContext(FrozenDomainModel):
    """An observation's owner, resolved from durable aggregates."""

    experiment_id: ExperimentId
    node_id: ExperimentNodeId
    run_id: str = Field(min_length=1)
    target: RuntimeOperationTarget


class DetectorProvenance(FrozenDomainModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class IncidentCandidate(FrozenDomainModel):
    """One detector's classification of the evidence, not a proposed action."""

    category: IncidentCategory
    detector: DetectorProvenance
    reason: str = Field(min_length=1)


def incident_observation_identity_v1(
    target: RuntimeOperationTarget, generation: int, sequence: int
) -> dict[str, Any]:
    """ADR-014's authoritative delivery key. No timestamps or object identity."""
    return {
        "kind": "incident-observation",
        "identity_version": 1,
        "target": {"kind": target.kind, "id": target.id},
        "stream_generation": generation,
        "sequence": sequence,
    }


class Incident(FrozenDomainModel):
    """What was observed and how it was classified; immutable once recorded.

    ``evidence`` preserves the existing telemetry envelope in full. Different
    deliveries can diagnose the same underlying failure; that does not make
    them the same observation. No recovery signature is implied.
    """

    id: IncidentId = Field(default_factory=IncidentId.generate)
    context: AttemptContext
    stream_generation: int = Field(ge=0, strict=True)
    sequence: int = Field(ge=0, strict=True)
    category: IncidentCategory
    evidence: FrozenDict
    candidates: tuple[IncidentCandidate, ...]
    detectors: tuple[DetectorProvenance, ...]
    classifier: DetectorProvenance
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _consistent(self) -> Incident:
        evidence = self.evidence
        if (
            type(evidence.get("stream_generation")) is not int
            or type(evidence.get("sequence")) is not int
            or evidence.get("target") != self.context.target.model_dump(mode="json")
            or evidence.get("stream_generation") != self.stream_generation
            or evidence.get("sequence") != self.sequence
        ):
            raise ValueError("incident and evidence observation keys disagree")
        workload = "training" if self.context.target.kind == "training-attempt" else "evaluation"
        data_type = evidence["payload"]["data"]["type"]
        if (
            evidence["payload"].get("workload") != workload
            or (workload == "training" and data_type == "EvaluationFailed")
            or (
                workload == "evaluation"
                and data_type != "IncidentObserved"
                and data_type != "EvaluationFailed"
            )
        ):
            raise ValueError("incident evidence belongs to another workload family")
        if not evidence.get("event_id") or evidence["payload"]["data"]["type"] not in (
            "IncidentObserved",
            "TrainingFailed",
            "EvaluationFailed",
            "NumericalInstabilityObserved",
        ):
            raise ValueError("incident evidence is not a failure observation")
        categories = {candidate.category for candidate in self.candidates}
        expected = next(iter(categories)) if len(categories) == 1 else IncidentCategory.UNKNOWN
        if self.category is not expected:
            raise ValueError("incident category disagrees with deterministic classification")
        if any(candidate.detector not in self.detectors for candidate in self.candidates):
            raise ValueError("candidate names a detector absent from the provenance")
        return self

    @property
    def observation_key(self) -> str:
        return fingerprint(
            incident_observation_identity_v1(
                self.context.target, self.stream_generation, self.sequence
            )
        )

    @property
    def evidence_fingerprint(self) -> str:
        return fingerprint(self.evidence)
