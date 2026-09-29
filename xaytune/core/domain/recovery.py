"""Immutable attempt recovery provenance and pure decisions, never execution authority."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from xaytune.core.checkpoint import Digest, RecordedCheckpoint, RestoreContext
from xaytune.core.clock import utc_now
from xaytune.core.domain.incident import AttemptContext, Incident, IncidentCategory
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import IncidentId, RecoveryEpisodeId, RecoveryPlanId
from xaytune.core.immutable import FrozenDict, FrozenDomainModel
from xaytune.core.refs import Actor, CheckpointRef
from xaytune.core.resume import CheckpointBoundary, DataResume, ResumeGuarantee, StateRestore
from xaytune.core.state.status import (
    EvaluationAttemptStatus,
    EvaluationRunStatus,
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
)

COORDINATOR_NAME = "deterministic-recovery"
COORDINATOR_VERSION = "2"


def incident_signature_v1(incident: Incident, candidate_fingerprint: str) -> str:
    """Structured failure identity, independent of delivery/detail/counters."""
    data = incident.evidence["payload"]["data"]
    metadata = data.get("metadata") or {}
    return fingerprint(
        {
            "kind": "recovery-incident-signature/v1",
            "category": incident.category.value,
            "code_location": metadata.get("code_location"),
            "resource_shape": metadata.get("resource_shape"),
            "quantity": data.get("quantity"),
            "candidate_fingerprint": candidate_fingerprint,
        }
    )


def execution_state_fingerprint_v1(attempt: FrozenDict) -> str:
    """Record execution identity; equality is not a generic recovery-loop failure."""
    return fingerprint(
        {
            "kind": "recovery-execution-state/v1",
            "execution_fingerprint": attempt.get("execution_fingerprint"),
            "overrides": [
                {"kind": item["kind"], "values": item["values"], "preserves": item["preserves"]}
                for item in attempt.get("execution_overrides", ())
            ],
        }
    )


class Recoverability(str, Enum):
    RECOVERABLE_SAME_ATTEMPT = "RECOVERABLE_SAME_ATTEMPT"
    RECOVERABLE_NEW_ATTEMPT = "RECOVERABLE_NEW_ATTEMPT"
    RECOVERABLE_WITH_EXECUTION_OVERRIDE = "RECOVERABLE_WITH_EXECUTION_OVERRIDE"
    REQUIRES_NEW_NODE = "REQUIRES_NEW_NODE"
    REQUIRES_HUMAN = "REQUIRES_HUMAN"
    UNRECOVERABLE = "UNRECOVERABLE"
    UNKNOWN = "UNKNOWN"


class RecoveryStrategy(str, Enum):
    FAIL = "FAIL"
    RETRY = "RETRY"
    RESUME = "RESUME"
    ROLLBACK = "ROLLBACK"
    RUNTIME_RECOVER = "RUNTIME_RECOVER"
    EXECUTION_OVERRIDE = "EXECUTION_OVERRIDE"
    NEW_EXPERIMENT_NODE = "NEW_EXPERIMENT_NODE"
    PAUSE_FOR_APPROVAL = "PAUSE_FOR_APPROVAL"


class RecoveryLimits(FrozenDomainModel):
    max_attempts_per_run: int = Field(default=3, ge=1, strict=True)
    max_recoveries_per_experiment: int = Field(default=10, ge=0, strict=True)
    # Number of earlier matching episodes allowed after the initial occurrence.
    max_same_incident_repeats: int = Field(default=2, ge=0, strict=True)


class RecoveryRequest(FrozenDomainModel):
    """Explicit original policy and intended-consumer inputs, captured once."""

    limits: RecoveryLimits = Field(default_factory=RecoveryLimits)
    allow_retry_without_checkpoint: bool = Field(default=False, strict=True)
    restore_context: RestoreContext | None = None


class RecoveryEpisode(FrozenDomainModel):
    """Immutable ownership/provenance; closure and reservations are derived."""

    id: RecoveryEpisodeId = Field(default_factory=RecoveryEpisodeId.generate)
    context: AttemptContext
    attempt_number: int = Field(ge=1, strict=True)
    candidate_fingerprint: str = Field(min_length=1)
    request: RecoveryRequest
    coordinator_name: str = COORDINATOR_NAME
    coordinator_version: str = COORDINATOR_VERSION
    created_by: Actor
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def request_fingerprint(self) -> str:
        return fingerprint(self.request)


class RecoveryEvidenceDisposition(str, Enum):
    ACCEPTED_FOR_DECISION = "ACCEPTED_FOR_DECISION"
    LATE_AFTER_CLOSURE = "LATE_AFTER_CLOSURE"


class RecoveryEpisodeIncident(FrozenDomainModel):
    episode_id: RecoveryEpisodeId
    incident_id: IncidentId
    membership_sequence: int = Field(ge=1, strict=True)
    disposition: RecoveryEvidenceDisposition
    incident_signature: Digest
    signature_version: Literal["recovery-incident-signature/v1"] = "recovery-incident-signature/v1"
    evidence_fingerprint: Digest
    attached_by: Actor
    attached_at: datetime = Field(default_factory=utc_now)


class RecoveryEvidence(FrozenDomainModel):
    """Canonical decision projection; complete raw evidence remains on Incident."""

    incident_id: IncidentId
    membership_sequence: int = Field(ge=1, strict=True)
    signature: Digest
    diagnosis_fingerprint: Digest
    categories: tuple[IncidentCategory, ...]
    stream_generation: int = Field(ge=0, strict=True)
    observation_sequence: int = Field(ge=0, strict=True)
    optimizer_step: int | None = Field(default=None, ge=0, strict=True)
    observed_at: datetime

    @model_validator(mode="after")
    def _categories(self) -> RecoveryEvidence:
        if not self.categories or self.categories != tuple(
            sorted(set(self.categories), key=lambda c: c.value)
        ):
            raise ValueError("evidence diagnoses must be nonempty, sorted and unique")
        return self

    @classmethod
    def from_incident(cls, incident: Incident, sequence: int, candidate: str) -> RecoveryEvidence:
        step = incident.evidence["payload"]["data"].get("optimizer_step")
        categories = {item.category for item in incident.candidates} or {incident.category}
        return cls(
            incident_id=incident.id,
            membership_sequence=sequence,
            signature=incident_signature_v1(incident, candidate),
            diagnosis_fingerprint=fingerprint(incident.model_dump(mode="json")),
            categories=tuple(sorted(categories, key=lambda item: item.value)),
            stream_generation=incident.stream_generation,
            observation_sequence=incident.sequence,
            optimizer_step=step if type(step) is int else None,
            observed_at=incident.created_at,
        )


class RecoveryCheckpointReport(FrozenDomainModel):
    """Small authoritative report binding, not a byte-integrity assertion."""

    checkpoint_ref: CheckpointRef
    context: AttemptContext
    producer_attempt_number: int = Field(ge=1, strict=True)
    report_fingerprint: Digest
    optimizer_step: int = Field(ge=0, strict=True)
    stream_generation: int = Field(ge=0, strict=True)
    observation_sequence: int = Field(ge=0, strict=True)
    reported_at: datetime
    resume_guarantee: ResumeGuarantee
    has_data_cursor: bool = Field(strict=True)

    @classmethod
    def from_record(cls, record: RecordedCheckpoint, number: int) -> RecoveryCheckpointReport:
        return cls(
            checkpoint_ref=record.payload.checkpoint_ref,
            context=record.context,
            producer_attempt_number=number,
            report_fingerprint=fingerprint(record),
            optimizer_step=record.payload.optimizer_step,
            stream_generation=record.evidence["stream_generation"],
            observation_sequence=record.evidence["sequence"],
            reported_at=record.created_at,
            resume_guarantee=record.payload.resume_guarantee,
            has_data_cursor=record.payload.data_cursor is not None,
        )


class RecoveryPredecessor(FrozenDomainModel):
    plan_id: RecoveryPlanId
    sequence: int = Field(ge=1, strict=True)
    accepted_through_sequence: int = Field(ge=1, strict=True)
    accepted_evidence_fingerprint: Digest


class RecoveryRepeatCount(FrozenDomainModel):
    signature: Digest
    prior_matching_episodes: int = Field(ge=0, strict=True)


class RecoveryInputsV1(FrozenDomainModel):
    """Versioned database decision projection, without aggregate payload dumps."""

    schema_version: Literal["xaytune.recovery-inputs/v1alpha1"] = "xaytune.recovery-inputs/v1alpha1"
    context: AttemptContext
    experiment_status: ExperimentStatus
    experiment_revision: int = Field(ge=0, strict=True)
    node_status: ExperimentNodeStatus
    node_revision: int = Field(ge=0, strict=True)
    run_status: RunStatus | EvaluationRunStatus
    run_revision: int = Field(ge=0, strict=True)
    attempt_status: RunAttemptStatus | EvaluationAttemptStatus
    attempt_revision: int = Field(ge=0, strict=True)
    attempt_number: int = Field(ge=1, strict=True)
    actual_attempt_count: int = Field(ge=1, strict=True)
    successor_exists: bool = Field(strict=True)
    pending_other_run_reservations: int = Field(ge=0, strict=True)
    candidate_fingerprint: str = Field(min_length=1)
    execution_state_fingerprint: Digest
    episode_id: RecoveryEpisodeId
    request_fingerprint: Digest
    coordinator_name: str
    coordinator_version: str
    accepted_evidence: tuple[RecoveryEvidence, ...]
    accepted_membership_count: int = Field(ge=1, strict=True)
    accepted_membership_fingerprint: Digest
    predecessor: RecoveryPredecessor | None = None
    experiment_recovery_usage_excluding_target: int = Field(ge=0, strict=True)
    repeat_counts: tuple[RecoveryRepeatCount, ...]
    checkpoint_reports: tuple[RecoveryCheckpointReport, ...] = ()

    @model_validator(mode="after")
    def _canonical(self) -> RecoveryInputsV1:
        evidence = self.accepted_evidence
        if not evidence or tuple(sorted(evidence, key=lambda e: str(e.incident_id))) != evidence:
            raise ValueError("accepted evidence must be nonempty and canonical by incident ID")
        if len({e.incident_id for e in evidence}) != len(evidence):
            raise ValueError("duplicate accepted incident")
        if len({e.membership_sequence for e in evidence}) != len(evidence):
            raise ValueError("duplicate evidence sequence")
        if tuple(c.signature for c in self.repeat_counts) != self.incident_signatures:
            raise ValueError("repeat counts must cover sorted distinct accepted signatures")
        reports = self.checkpoint_reports
        if len({r.checkpoint_ref.id for r in reports}) != len(reports):
            raise ValueError("duplicate checkpoint report")
        if tuple(sorted(reports, key=checkpoint_order, reverse=True)) != reports:
            raise ValueError("checkpoint reports must have canonical newest-first order")
        return self

    @property
    def incident_signatures(self) -> tuple[str, ...]:
        return tuple(sorted({e.signature for e in self.accepted_evidence}))

    @property
    def accepted_evidence_fingerprint(self) -> str:
        return fingerprint(self.accepted_evidence)

    @property
    def accepted_through_sequence(self) -> int:
        return max(e.membership_sequence for e in self.accepted_evidence)


def checkpoint_order(report: RecoveryCheckpointReport) -> tuple[int, int, str]:
    return report.optimizer_step, report.producer_attempt_number, str(report.checkpoint_ref.id)


class CheckpointEligibility(FrozenDomainModel):
    """Trusted coordinator's planning-time inspection; execution must revalidate."""

    checkpoint_ref: CheckpointRef
    report_fingerprint: Digest
    eligible: bool = Field(strict=True)
    reason: str = Field(min_length=1)


class RecoveryDecision(FrozenDomainModel):
    strategy: RecoveryStrategy
    recoverability: Recoverability
    checkpoint_ref: CheckpointRef | None = None
    reason: str = Field(min_length=1)
    requires_approval: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def _consistent(self) -> RecoveryDecision:
        if self.strategy in (RecoveryStrategy.RESUME, RecoveryStrategy.ROLLBACK):
            if self.checkpoint_ref is None:
                raise ValueError("resume/rollback requires an eligible checkpoint")
        elif self.checkpoint_ref is not None:
            raise ValueError("this strategy does not select a checkpoint")
        if self.requires_approval != (self.strategy is RecoveryStrategy.PAUSE_FOR_APPROVAL):
            raise ValueError("pause decisions require approval")
        return self

    @property
    def reserves_recovery(self) -> bool:
        return self.strategy in (RecoveryStrategy.RETRY, RecoveryStrategy.RESUME)


class RecoveryPlan(RecoveryDecision):
    """Append-only episode conclusion. A plan grants no execution authority."""

    id: RecoveryPlanId = Field(default_factory=RecoveryPlanId.generate)
    episode_id: RecoveryEpisodeId
    sequence: int = Field(ge=1, strict=True)
    supersedes_plan_id: RecoveryPlanId | None = None
    accepted_through_sequence: int = Field(ge=1, strict=True)
    accepted_incident_ids: tuple[IncidentId, ...]
    accepted_evidence_fingerprint: Digest
    incident_signatures: tuple[Digest, ...]
    execution_state_fingerprint: Digest
    inputs: RecoveryInputsV1
    checkpoint_eligibility: tuple[CheckpointEligibility, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _bound(self) -> RecoveryPlan:
        inputs = self.inputs
        predecessor = inputs.predecessor
        if (
            self.episode_id != inputs.episode_id
            or self.accepted_through_sequence != inputs.accepted_through_sequence
            or self.accepted_incident_ids != tuple(e.incident_id for e in inputs.accepted_evidence)
            or self.accepted_evidence_fingerprint != inputs.accepted_evidence_fingerprint
            or self.incident_signatures != inputs.incident_signatures
            or self.execution_state_fingerprint != inputs.execution_state_fingerprint
            or self.sequence != (1 if predecessor is None else predecessor.sequence + 1)
            or self.supersedes_plan_id != (None if predecessor is None else predecessor.plan_id)
        ):
            raise ValueError("plan ownership/revision/evidence disagrees with typed inputs")
        if (
            predecessor is not None
            and self.accepted_through_sequence != predecessor.accepted_through_sequence + 1
        ):
            raise ValueError("revision must cover the next accepted evidence extension")
        if self.checkpoint_ref is not None and not any(
            item.eligible and item.checkpoint_ref == self.checkpoint_ref
            for item in self.checkpoint_eligibility
        ):
            raise ValueError("selected checkpoint has no eligibility evidence")
        return self

    @classmethod
    def from_decision(
        cls,
        inputs: RecoveryInputsV1,
        decision: RecoveryDecision,
        eligibility: tuple[CheckpointEligibility, ...],
    ) -> RecoveryPlan:
        previous = inputs.predecessor
        return cls(
            **decision.model_dump(),
            episode_id=inputs.episode_id,
            sequence=1 if previous is None else previous.sequence + 1,
            supersedes_plan_id=None if previous is None else previous.plan_id,
            accepted_through_sequence=inputs.accepted_through_sequence,
            accepted_incident_ids=tuple(e.incident_id for e in inputs.accepted_evidence),
            accepted_evidence_fingerprint=inputs.accepted_evidence_fingerprint,
            incident_signatures=inputs.incident_signatures,
            execution_state_fingerprint=inputs.execution_state_fingerprint,
            inputs=inputs,
            checkpoint_eligibility=eligibility,
        )

    @property
    def input_fingerprint(self) -> str:
        return fingerprint({"inputs": self.inputs, "eligibility": self.checkpoint_eligibility})

    def semantic_fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json", exclude={"id", "created_at"}))


_TRANSIENT = frozenset(
    {
        IncidentCategory.PROCESS_FAILURE,
        IncidentCategory.WORKER_FAILURE,
        IncidentCategory.NODE_FAILURE,
        IncidentCategory.DRIVER_FAILURE,
        IncidentCategory.PREEMPTION,
        IncidentCategory.NETWORK_FAILURE,
        IncidentCategory.OBJECT_STORE_FAILURE,
        IncidentCategory.TIMEOUT,
    }
)
_FATAL = frozenset(
    {
        IncidentCategory.DATA_ERROR,
        IncidentCategory.DATA_CORRUPTION,
        IncidentCategory.CONFIG_ERROR,
        IncidentCategory.USER_ERROR,
    }
)
_NUMERICAL = frozenset(
    {
        IncidentCategory.NUMERICAL_NAN,
        IncidentCategory.NUMERICAL_INF,
        IncidentCategory.GRADIENT_EXPLOSION,
        IncidentCategory.LOSS_DIVERGENCE,
    }
)


class RecoveryRequirement(FrozenDomainModel):
    """Authority requirement, not an executable specialised recovery proposal."""

    recoverability: Recoverability
    specialised_family: str | None = None


def diagnosis_requirement(category: IncidentCategory) -> RecoveryRequirement:
    if category in _FATAL:
        return RecoveryRequirement(recoverability=Recoverability.UNRECOVERABLE)
    if category in _TRANSIENT:
        return RecoveryRequirement(recoverability=Recoverability.RECOVERABLE_NEW_ATTEMPT)
    if category is IncidentCategory.CUDA_OOM:
        return RecoveryRequirement(
            recoverability=Recoverability.RECOVERABLE_WITH_EXECUTION_OVERRIDE,
            specialised_family="adaptive-execution",
        )
    if category in _NUMERICAL:
        return RecoveryRequirement(
            recoverability=Recoverability.REQUIRES_HUMAN,
            specialised_family="numerical-intervention",
        )
    return RecoveryRequirement(recoverability=Recoverability.UNKNOWN)


def _fail(reason: str) -> RecoveryDecision:
    return RecoveryDecision(
        strategy=RecoveryStrategy.FAIL, recoverability=Recoverability.UNRECOVERABLE, reason=reason
    )


def _pause(reason: str) -> RecoveryDecision:
    return RecoveryDecision(
        strategy=RecoveryStrategy.PAUSE_FOR_APPROVAL,
        recoverability=Recoverability.REQUIRES_HUMAN,
        requires_approval=True,
        reason=reason,
    )


def _predecision(inputs: RecoveryInputsV1, request: RecoveryRequest) -> RecoveryDecision | None:
    if fingerprint(request) != inputs.request_fingerprint:
        raise ValueError("request differs from episode planning provenance")
    if inputs.experiment_status in (
        ExperimentStatus.SUCCEEDED,
        ExperimentStatus.FAILED,
        ExperimentStatus.CANCELLED,
        ExperimentStatus.BUDGET_EXHAUSTED,
    ) or (
        inputs.node_status
        in (
            ExperimentNodeStatus.COMPLETED,
            ExperimentNodeStatus.REJECTED,
            ExperimentNodeStatus.CANCELLED,
            ExperimentNodeStatus.FAILED,
        )
    ):
        return _fail("experiment or candidate is terminal")
    if inputs.experiment_status is ExperimentStatus.PAUSED:
        return _pause("experiment is paused; recovery needs review")
    if inputs.run_status in (
        RunStatus.SUCCEEDED,
        RunStatus.CANCELLED,
        EvaluationRunStatus.SUCCEEDED,
        EvaluationRunStatus.CANCELLED,
    ) or (
        inputs.attempt_status
        in (
            RunAttemptStatus.SUCCEEDED,
            RunAttemptStatus.CANCELLED,
            EvaluationAttemptStatus.SUCCEEDED,
            EvaluationAttemptStatus.CANCELLED,
        )
    ):
        return _fail("completed or cancelled work cannot be recovered")
    if inputs.successor_exists:
        return _fail("episode is closed by a successor attempt")
    requirements = [
        diagnosis_requirement(category)
        for evidence in inputs.accepted_evidence
        for category in evidence.categories
    ]
    if any(r.recoverability is Recoverability.UNRECOVERABLE for r in requirements):
        return _fail("structured diagnosis requires corrected data or configuration")
    families = sorted({r.specialised_family for r in requirements if r.specialised_family})
    if len(families) > 1:
        return _pause("conflicting specialised recovery requirements: " + ", ".join(families))
    if families:
        return _pause("specialised recovery required: " + families[0])
    if any(r.recoverability is not Recoverability.RECOVERABLE_NEW_ATTEMPT for r in requirements):
        return _pause("evidence requires unsupported or human-governed recovery")
    if inputs.context.target.kind != "training-attempt":
        return _pause("evaluation recovery requires a workload-specific policy")
    limits = request.limits
    if inputs.actual_attempt_count + inputs.pending_other_run_reservations + 1 > (
        limits.max_attempts_per_run
    ):
        return _fail("maximum attempts per run reached (including planned recoveries)")
    if inputs.experiment_recovery_usage_excluding_target + 1 > limits.max_recoveries_per_experiment:
        return _fail("maximum recoveries per experiment reached")
    if any(
        count.prior_matching_episodes > limits.max_same_incident_repeats
        for count in inputs.repeat_counts
    ):
        return _pause("repeated incident limit reached")
    return None


def checkpoint_report_problem(
    report: RecoveryCheckpointReport, inputs: RecoveryInputsV1
) -> str | None:
    """Database/structured eligibility constraints; this never inspects bytes."""
    if report.context.run_id != inputs.context.run_id:
        return "checkpoint belongs to another logical run"
    for evidence in inputs.accepted_evidence:
        if report.context.target == inputs.context.target:
            if (report.stream_generation, report.observation_sequence) >= (
                evidence.stream_generation,
                evidence.observation_sequence,
            ):
                return "checkpoint was reported at or after the incident"
        elif report.reported_at > evidence.observed_at:
            return "checkpoint was reported after the incident"
        if evidence.optimizer_step is not None and report.optimizer_step > evidence.optimizer_step:
            return "checkpoint is beyond the incident's training position"
    guarantee = report.resume_guarantee
    if (
        guarantee.state is not StateRestore.FULL
        or guarantee.data is not DataResume.EXACT
        or guarantee.boundary is not CheckpointBoundary.OPTIMIZER_STEP
        or not report.has_data_cursor
    ):
        return "generic resume requires FULL/EXACT state and an optimizer-boundary data cursor"
    return None


def recovery_requires_checkpoints(inputs: RecoveryInputsV1, request: RecoveryRequest) -> bool:
    return _predecision(inputs, request) is None


def decide_recovery(
    inputs: RecoveryInputsV1,
    request: RecoveryRequest,
    eligibility: tuple[CheckpointEligibility, ...],
) -> RecoveryDecision:
    """Pure authority-based join of all accepted evidence; no I/O/time/randomness."""
    preliminary = _predecision(inputs, request)
    if preliminary is not None:
        return preliminary
    eligible = next((item for item in eligibility if item.eligible), None)
    if eligible is not None:
        return RecoveryDecision(
            strategy=RecoveryStrategy.RESUME,
            recoverability=Recoverability.RECOVERABLE_NEW_ATTEMPT,
            checkpoint_ref=eligible.checkpoint_ref,
            reason="latest validated compatible FULL/EXACT optimizer-boundary checkpoint",
        )
    if request.allow_retry_without_checkpoint:
        return RecoveryDecision(
            strategy=RecoveryStrategy.RETRY,
            recoverability=Recoverability.RECOVERABLE_NEW_ATTEMPT,
            reason="explicit policy permits restarting without a checkpoint",
        )
    return _pause("no eligible checkpoint; restarting without captured state is not permitted")
