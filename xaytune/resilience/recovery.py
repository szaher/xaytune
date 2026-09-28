"""Deterministic recovery planning; no runtime calls, decode or state application."""

from __future__ import annotations

from collections.abc import Callable

from xaytune.checkpoints.errors import CheckpointCompatibilityError, CheckpointCorruptionError
from xaytune.checkpoints.manager import CheckpointManager
from xaytune.core.checkpoint import RecordedCheckpoint
from xaytune.core.domain.incident import Incident
from xaytune.core.domain.recovery import (
    CheckpointEligibility,
    RecoveryPlan,
    RecoveryRequest,
    decide_recovery,
    execution_state_fingerprint_v1,
    incident_signature_v1,
)
from xaytune.core.errors import IdempotencyConflictError
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import Actor
from xaytune.core.resume import CheckpointBoundary, DataResume, StateRestore
from xaytune.storage.control_plane import ControlPlaneRepository, StaleRecoveryContextError

_ACTOR = Actor(type="rule", id="recovery:deterministic-recovery")


class RecoveryCoordinator:
    """Decide and record once, or reconstruct a missing decision after restart.

    This does not pause/fail aggregates, create Actions/attempts/nodes, reserve
    budget in the execution ledger, or submit work. requires_approval describes
    a future governed action; it does not create an approval request.
    """

    def __init__(
        self,
        repository: ControlPlaneRepository,
        checkpoint_manager: CheckpointManager | None = None,
        *,
        destinations: tuple[str, ...] = (),
    ) -> None:
        self.repository = repository
        self.checkpoint_manager = checkpoint_manager
        self.destinations = destinations

    async def plan(self, incident_id: str, request: RecoveryRequest | None = None) -> RecoveryPlan:
        # A recorded decision is authoritative even when policy/files change.
        # Reconciliation never silently replans it under a new configuration.
        existing = self.repository.recovery_plans.for_incident(incident_id)
        if existing is not None:
            return existing
        request = (
            RecoveryRequest()
            if request is None
            else RecoveryRequest.model_validate_json(request.model_dump_json())
        )
        for _ in range(3):
            snapshot = self.repository.recovery_snapshot(incident_id)
            incident = Incident.model_validate(snapshot["incident"])
            eligibility = await self._checkpoints(snapshot, incident, request)
            decision = decide_recovery(snapshot, request, eligibility)
            plan = RecoveryPlan(
                **decision.model_dump(mode="python"),
                incident_id=incident.id,
                context=incident.context,
                incident_signature=incident_signature_v1(
                    incident, snapshot["candidate_fingerprint"]
                ),
                execution_state_fingerprint=execution_state_fingerprint_v1(snapshot["attempt"]),
                input_snapshot=snapshot,
                request=request,
                checkpoint_eligibility=eligibility,
            )
            # Another process may have committed while bytes were inspected.
            existing = self.repository.recovery_plans.for_incident(incident_id)
            if existing is not None:
                return existing
            try:
                return self.repository.record_recovery_plan(
                    plan, actor=_ACTOR, destinations=self.destinations
                )
            except StaleRecoveryContextError:
                continue
            except IdempotencyConflictError:
                # A different request may have won after the last replay read.
                # Coordinator replay retains that decision; direct recording
                # still refuses changed semantic payloads.
                existing = self.repository.recovery_plans.for_incident(incident_id)
                if existing is not None:
                    return existing
                raise
        raise StaleRecoveryContextError("recovery inputs kept changing; retry reconciliation")

    async def reconcile(
        self,
        experiment_id: str,
        *,
        request_for_incident: Callable[[Incident], RecoveryRequest] | None = None,
    ) -> tuple[RecoveryPlan, ...]:
        """Repair incident→plan crash gaps, retaining all existing decisions.

        Resolve consumer configuration only for missing plans. Persisted incident
        ordering supplies a stable replay order, and each commit is replay-safe.
        """
        self.repository.aggregates.load_experiment(experiment_id)
        plans = []
        for incident in self.repository.incidents.for_experiment(experiment_id):
            existing = self.repository.recovery_plans.for_incident(str(incident.id))
            if existing is not None:
                plans.append(existing)
                continue
            request = None if request_for_incident is None else request_for_incident(incident)
            plans.append(await self.plan(str(incident.id), request))
        return tuple(plans)

    async def _checkpoints(
        self, snapshot: FrozenDict, incident: Incident, request: RecoveryRequest
    ) -> tuple[CheckpointEligibility, ...]:
        attempt_numbers = {a["id"]: a["attempt_number"] for a in snapshot["attempts"]}
        records = [RecordedCheckpoint.model_validate(raw) for raw in snapshot["checkpoints"]]
        records.sort(
            key=lambda r: (
                r.payload.optimizer_step,
                attempt_numbers[r.context.target.id],
                str(r.payload.checkpoint_ref.id),
            ),
            reverse=True,
        )
        results = []
        for record in records:
            reason = await self._checkpoint_problem(record, incident, request)
            results.append(
                CheckpointEligibility(
                    checkpoint_ref=record.payload.checkpoint_ref,
                    eligible=reason is None,
                    reason=reason
                    or "bytes, producer, compatibility and exact continuation validated",
                )
            )
        return tuple(results)

    async def _checkpoint_problem(
        self, record: RecordedCheckpoint, incident: Incident, request: RecoveryRequest
    ) -> str | None:
        if record.context.run_id != incident.context.run_id:
            return "checkpoint belongs to another logical run"
        if record.context.target == incident.context.target:
            if (record.evidence["stream_generation"], record.evidence["sequence"]) >= (
                incident.stream_generation,
                incident.sequence,
            ):
                return "checkpoint was reported at or after the incident"
        elif record.created_at > incident.created_at:
            return "checkpoint was reported after the incident"
        data = incident.evidence["payload"]["data"]
        step = data.get("optimizer_step")
        if type(step) is int and record.payload.optimizer_step > step:
            return "checkpoint is beyond the incident's training position"
        guarantee = record.payload.resume_guarantee
        if (
            guarantee.state is not StateRestore.FULL
            or guarantee.data is not DataResume.EXACT
            or guarantee.boundary is not CheckpointBoundary.OPTIMIZER_STEP
            or record.payload.data_cursor is None
        ):
            return "generic resume requires FULL/EXACT state and an optimizer-boundary data cursor"
        if request.restore_context is None or self.checkpoint_manager is None:
            return "consumer restore context or checkpoint manager is unavailable"
        try:
            await self.checkpoint_manager.validate_recorded(record, request.restore_context)
        except CheckpointCompatibilityError:
            return "checkpoint is incompatible with the intended consumer"
        except (CheckpointCorruptionError, OSError):
            return "checkpoint bytes or provenance are missing or corrupt"
        return None
