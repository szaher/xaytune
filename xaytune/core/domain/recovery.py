"""Recovery decisions and their inputs. A plan grants no execution authority."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field, model_validator

from xaytune.core.checkpoint import Digest, RestoreContext
from xaytune.core.clock import utc_now
from xaytune.core.domain.incident import AttemptContext, Incident, IncidentCategory
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import IncidentId, RecoveryPlanId
from xaytune.core.immutable import FrozenDict, FrozenDomainModel
from xaytune.core.refs import CheckpointRef


def incident_signature_v1(incident: Incident, candidate_fingerprint: str) -> str:
    """A repeated failure pattern, independent of delivery IDs and timestamps.

    Only structured location/resource evidence participates. Free-form detail
    and changing counters cannot disguise the same failure as a new pattern.
    """
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
    max_same_incident_repeats: int = Field(default=2, ge=0, strict=True)


class RecoveryRequest(FrozenDomainModel):
    """Explicit planning inputs; restarting without captured state is opt-in.

    The restore context must come from the intended consumer, never inferred
    from the checkpoint that happens to be available. v1 requires FULL/EXACT
    at an optimizer boundary even if a caller requests a weaker guarantee.
    """

    limits: RecoveryLimits = Field(default_factory=RecoveryLimits)
    allow_retry_without_checkpoint: bool = Field(default=False, strict=True)
    restore_context: RestoreContext | None = None


class CheckpointEligibility(FrozenDomainModel):
    checkpoint_ref: CheckpointRef
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
    """One durable decision per incident, including the inputs used to decide.

    This PR produces no overrides or scientific mutations. Those future
    proposals must go through Action governance, capability and budget checks.
    A selected checkpoint must be revalidated when execution is implemented.
    """

    id: RecoveryPlanId = Field(default_factory=RecoveryPlanId.generate)
    incident_id: IncidentId
    context: AttemptContext
    incident_signature: Digest
    execution_state_fingerprint: Digest
    input_snapshot: FrozenDict
    request: RecoveryRequest
    checkpoint_eligibility: tuple[CheckpointEligibility, ...] = ()
    coordinator_name: str = "deterministic-recovery"
    coordinator_version: str = "1"
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _selected(self) -> RecoveryPlan:
        if self.checkpoint_ref is not None and not any(
            item.eligible and item.checkpoint_ref == self.checkpoint_ref
            for item in self.checkpoint_eligibility
        ):
            raise ValueError("selected checkpoint has no eligibility evidence")
        return self

    @property
    def input_fingerprint(self) -> str:
        return fingerprint(
            {
                "kind": "recovery-planning-input/v1",
                "snapshot": self.input_snapshot,
                "request": self.request,
                "checkpoints": self.checkpoint_eligibility,
                "coordinator": [self.coordinator_name, self.coordinator_version],
            }
        )

    def semantic_fingerprint(self) -> str:
        """IDs and recording time are bookkeeping, not decision identity."""
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


def decide_recovery(
    snapshot: FrozenDict,
    request: RecoveryRequest,
    eligibility: tuple[CheckpointEligibility, ...],
) -> RecoveryDecision:
    """Pure v1 decision function. No ID generation, time, I/O or randomness."""
    incident = Incident.model_validate(snapshot["incident"])
    attempt = snapshot["attempt"]
    history = snapshot["history"]
    limits = request.limits

    def fail(reason: str) -> RecoveryDecision:
        return RecoveryDecision(
            strategy=RecoveryStrategy.FAIL,
            recoverability=Recoverability.UNRECOVERABLE,
            reason=reason,
        )

    def pause(reason: str) -> RecoveryDecision:
        return RecoveryDecision(
            strategy=RecoveryStrategy.PAUSE_FOR_APPROVAL,
            recoverability=Recoverability.REQUIRES_HUMAN,
            reason=reason,
            requires_approval=True,
        )

    if snapshot["experiment"]["status"] in (
        "succeeded",
        "failed",
        "cancelled",
        "budget_exhausted",
    ) or snapshot["node"]["status"] in ("completed", "rejected", "cancelled", "failed"):
        return fail("experiment or candidate is terminal")
    if snapshot["experiment"]["status"] == "paused":
        return pause("experiment is paused; recovery needs review")
    if snapshot["run"]["status"] in ("succeeded", "cancelled") or attempt["status"] in (
        "succeeded",
        "cancelled",
    ):
        return fail("completed or cancelled work cannot be recovered")
    if attempt["attempt_number"] != max(a["attempt_number"] for a in snapshot["attempts"]):
        return fail("incident belongs to a superseded attempt")
    if incident.context.target.kind != "training-attempt":
        return pause("evaluation recovery requires a workload-specific policy")
    reserved = [item for item in history if item["reserves_recovery"]]
    # A reservation on an older attempt is represented by its successor once
    # that attempt exists. Reservations on the current attempt still need slots.
    pending = sum(item["target_id"] == incident.context.target.id for item in reserved)
    if len(snapshot["attempts"]) + pending >= limits.max_attempts_per_run:
        return fail("maximum attempts per run reached (including planned recoveries)")
    if len(reserved) >= limits.max_recoveries_per_experiment:
        return fail("maximum recoveries per experiment reached")
    signature = incident_signature_v1(incident, snapshot["candidate_fingerprint"])
    repeats = [
        item
        for item in history
        if item["run_id"] == incident.context.run_id and item["signature"] == signature
    ]
    if repeats and len(repeats) >= limits.max_same_incident_repeats:
        return pause("repeated incident limit reached")
    state = execution_state_fingerprint_v1(attempt)
    if any(item["reserves_recovery"] and item["execution_state"] == state for item in repeats):
        return pause("same incident repeated without an execution change; recovery loop refused")
    if incident.category in _FATAL:
        return fail("structured diagnosis requires corrected data or configuration")
    if incident.category is IncidentCategory.CUDA_OOM:
        return pause("CUDA OOM requires adaptive execution-override planning (PR-020)")
    if incident.category in (
        IncidentCategory.NUMERICAL_NAN,
        IncidentCategory.NUMERICAL_INF,
        IncidentCategory.GRADIENT_EXPLOSION,
        IncidentCategory.LOSS_DIVERGENCE,
    ):
        return pause(
            "numerical recovery requires specialised planning and Action governance (PR-021)"
        )
    if incident.category not in _TRANSIENT:
        return pause("no generic recovery policy for this category; specialised planning required")
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
    return pause("no eligible checkpoint; restarting without captured state is not permitted")
