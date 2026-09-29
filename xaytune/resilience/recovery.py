"""Decision-only episode coordination. No runtime calls, decode or state application."""

from __future__ import annotations

from collections.abc import Callable

from xaytune.checkpoints.errors import CheckpointCompatibilityError, CheckpointCorruptionError
from xaytune.checkpoints.manager import CheckpointManager
from xaytune.core.domain.incident import Incident
from xaytune.core.domain.recovery import (
    CheckpointEligibility,
    RecoveryInputsV1,
    RecoveryPlan,
    RecoveryRequest,
    checkpoint_report_problem,
    decide_recovery,
)
from xaytune.core.errors import IdempotencyConflictError, XaytuneError
from xaytune.core.fingerprint import fingerprint
from xaytune.core.refs import Actor
from xaytune.storage.control_plane import ControlPlaneRepository, StaleRecoveryContextError
from xaytune.storage.errors import AggregateNotFoundError

_ACTOR = Actor(type="rule", id="recovery:deterministic-recovery")


class RecoveryRequestUnavailableError(XaytuneError):
    """First episode planning requires explicitly reconstructed consumer/policy inputs."""

    def __init__(self, incident_id: str) -> None:
        self.incident_id = incident_id
        super().__init__(
            f"a new recovery episode requires an explicit RecoveryRequest (incident {incident_id})"
        )


class RecoveryEpisodeClosedError(XaytuneError):
    """Historical observation cannot create recovery after execution has moved on."""


class RecoveryCoordinator:
    """Capture original policy once and append conclusions for accepted evidence.

    Recorded complete decisions replay without current configuration or files.
    Membership gaps use stored inputs. This coordinator grants no execution authority.
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
        incident = self.repository.incidents.get(incident_id)
        if incident is None:
            raise AggregateNotFoundError("incident", incident_id)
        episode = self.repository.recovery_episodes.for_attempt(incident.context.target)
        if episode is None:
            if request is None:
                raise RecoveryRequestUnavailableError(incident_id)
            request = RecoveryRequest.model_validate_json(request.model_dump_json())
            episode = self.repository.prepare_recovery_episode(incident_id, request, _ACTOR)
            if self.repository.recovery_target_is_superseded(episode.context):
                raise RecoveryEpisodeClosedError("superseded attempt has no recovery episode")
        # Caller policy is never substituted for an existing episode's original request.
        members = self.repository.recovery_episodes.memberships(str(episode.id))
        # Finite repair of the observed backlog plus at most three stale retries.
        rounds = len(members) + 4
        stale_retries = 0
        for _ in range(rounds):
            owner = self.repository.recovery_episodes.for_attempt(incident.context.target)
            if owner is not None:
                episode = owner
                for observation in self.repository.incidents.for_attempt(episode.context.target):
                    self.repository.attach_recovery_evidence(
                        str(observation.id), actor=_ACTOR, destinations=self.destinations
                    )
                existing = self.repository.recovery_plans.effective_for_episode(str(episode.id))
                if existing is not None and (
                    not self.repository.recovery_episodes.is_open(str(episode.id))
                    or self.repository.recovery_plans.is_effective_and_fresh(str(existing.id))
                ):
                    return existing
            try:
                inputs = self.repository.recovery_snapshot(episode)
                if inputs.successor_exists:
                    frozen = self.repository.recovery_plans.effective_for_episode(str(episode.id))
                    if frozen is not None:
                        return frozen
                    raise RecoveryEpisodeClosedError("superseded attempt has no recovery episode")
                if (
                    inputs.predecessor is not None
                    and inputs.accepted_through_sequence
                    <= inputs.predecessor.accepted_through_sequence
                ):
                    winner = self.repository.recovery_plans.get(str(inputs.predecessor.plan_id))
                    if winner is not None and self.repository.recovery_plans.is_effective_and_fresh(
                        str(winner.id)
                    ):
                        return winner
                    raise StaleRecoveryContextError("another planner repaired the evidence gap")
                eligibility = await self._checkpoints(inputs, episode.request)
                decision = decide_recovery(inputs, episode.request, eligibility)
                plan = RecoveryPlan.from_decision(inputs, decision, eligibility)
                self.repository.record_recovery_plan(
                    plan, actor=_ACTOR, destinations=self.destinations, episode=episode
                )
            except IdempotencyConflictError:
                winner = self.repository.recovery_plans.for_incident(incident_id)
                if winner is None or winner.sequence < plan.sequence:
                    raise
                if not self.repository.recovery_episodes.is_open(
                    str(winner.episode_id)
                ) or self.repository.recovery_plans.is_effective_and_fresh(str(winner.id)):
                    return winner
                stale_retries += 1
                if stale_retries >= 3:
                    break
                continue
            except StaleRecoveryContextError:
                stale_retries += 1
                if stale_retries >= 3:
                    break
                continue
            if self.repository.recovery_plans.is_effective_and_fresh(str(plan.id)):
                return plan
        raise StaleRecoveryContextError("recovery inputs kept changing; retry reconciliation")

    async def reconcile(
        self,
        experiment_id: str,
        *,
        request_for_incident: Callable[[Incident], RecoveryRequest | None] | None = None,
    ) -> tuple[RecoveryPlan, ...]:
        """Repair episodes in typed-run/attempt order; stop at first unresolved gap.

        Later decisions depend on preceding reservations/repeat history; skipping
        an unresolved initial request is unsafe. Reconstruct it and rerun.
        Superseded targets without episodes remain historical Incident audit only.
        """
        self.repository.aggregates.load_experiment(experiment_id)
        groups: dict[tuple[str, str], Incident] = {}
        for incident in self.repository.incidents.for_experiment(experiment_id):
            key = incident.context.target.kind, incident.context.target.id
            groups.setdefault(key, incident)
        ordered = []
        for incident in groups.values():
            context = incident.context
            attempt = (
                self.repository.aggregates.load_attempt(context.target.id)
                if context.target.kind == "training-attempt"
                else self.repository.aggregates.load_evaluation_attempt(context.target.id)
            )
            ordered.append((context.target.kind, context.run_id, attempt.attempt_number, incident))
        plans = []
        for _, _, _, incident in sorted(ordered, key=lambda item: item[:3]):
            episode = self.repository.recovery_episodes.for_attempt(incident.context.target)
            if episode is None:
                if self.repository.recovery_target_is_superseded(incident.context):
                    continue
                request = None if request_for_incident is None else request_for_incident(incident)
                if not isinstance(request, RecoveryRequest):
                    raise RecoveryRequestUnavailableError(str(incident.id))
            else:
                request = None
            plans.append(await self.plan(str(incident.id), request))
        return tuple(plans)

    async def _checkpoints(
        self, inputs: RecoveryInputsV1, request: RecoveryRequest
    ) -> tuple[CheckpointEligibility, ...]:
        results = []
        for report in inputs.checkpoint_reports:
            reason = checkpoint_report_problem(report, inputs)
            if reason is None:
                if request.restore_context is None or self.checkpoint_manager is None:
                    reason = "consumer restore context or checkpoint manager is unavailable"
                elif not self.checkpoint_manager.supports_validation:
                    reason = "codec lacks validation capability"
                else:
                    record = self.repository.checkpoints.get(str(report.checkpoint_ref.id))
                    if record is None or fingerprint(record) != report.report_fingerprint:
                        raise StaleRecoveryContextError("checkpoint report changed during planning")
                    try:
                        await self.checkpoint_manager.validate_recorded(
                            record, request.restore_context
                        )
                    except CheckpointCompatibilityError:
                        reason = "checkpoint incompatible with intended consumer"
                    except (CheckpointCorruptionError, OSError):
                        reason = "checkpoint bytes/provenance missing or corrupt"
            results.append(
                CheckpointEligibility(
                    checkpoint_ref=report.checkpoint_ref,
                    report_fingerprint=report.report_fingerprint,
                    eligible=reason is None,
                    reason=reason
                    or "bytes, producer, compatibility and exact continuation validated",
                )
            )
        return tuple(results)
