"""Domain records → ``xaytune.studio-api/v1`` models (ADR-023 §S5).

Every domain field is either mapped or listed in :data:`OMITTED` with the
reason it is left out, and a test holds the two lists against the domain
models: a new domain field fails that test until someone decides whether
Studio shows it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from enum import Enum
from typing import Any, get_args

from pydantic import BaseModel

from xaytune.core.checkpoint import RecordedCheckpoint
from xaytune.core.domain.action import Action
from xaytune.core.domain.agent_invocation import AgentInvocation
from xaytune.core.domain.budget import BudgetLedgerEntry, BudgetStatus
from xaytune.core.domain.decision import Decision
from xaytune.core.domain.evaluation import (
    EvaluationAttempt,
    EvaluationResult,
    EvaluationRun,
    MetricResult,
)
from xaytune.core.domain.event import DomainEvent
from xaytune.core.domain.experiment import Experiment, ExperimentNode
from xaytune.core.domain.incident import Incident
from xaytune.core.domain.policy import PolicyDecision
from xaytune.core.domain.recovery import RecoveryEpisode, RecoveryPlan
from xaytune.core.domain.run import Run, RunAttempt
from xaytune.core.refs import Actor
from xaytune.storage.leases import ControllerLease
from xaytune.storage.metrics import MetricSeriesSummary, StoredMetricPoint

from . import models as m

__all__ = ["OMITTED", "document"]


OMITTED: dict[type, dict[str, str]] = {
    Experiment: {},
    ExperimentNode: {
        "candidate": "served by the node detail, not in graph items",
        "decision_ids": "decisions are served whole, by node",
        "revision": "concurrency bookkeeping",
    },
    Run: {"revision": "concurrency bookkeeping"},
    RunAttempt: {"revision": "concurrency bookkeeping"},
    EvaluationRun: {"revision": "concurrency bookkeeping"},
    EvaluationAttempt: {"revision": "concurrency bookkeeping"},
    EvaluationResult: {},
    MetricResult: {},
    Decision: {},
    Action: {
        "experiment_id": "always the experiment asked for",
        "policy_decision_id": "the policy decision is served whole",
        "revision": "concurrency bookkeeping",
    },
    PolicyDecision: {
        "action_id": "served on its action",
        "experiment_id": "always the experiment asked for",
        "context": "the full policy snapshot; fingerprinted by input_fingerprint",
    },
    DomainEvent: {"experiment_id": "always the experiment asked for"},
    Incident: {},
    RecoveryEpisode: {},
    BudgetLedgerEntry: {"experiment_id": "always the experiment asked for"},
    AgentInvocation: {
        "intent": "summarised: planner, prompt version, model and fingerprints; no request body",
        "response": "model output body: not served in V1 (§S6)",
        "proposal": "proposal body: identified by proposal_fingerprint",
        "failure": "summarised as failure_kind",
        "revision": "concurrency bookkeeping",
    },
}
"""Domain fields a v1 model leaves out, each with why."""


# ---- documents ---------------------------------------------------------------------------------


def actor(value: Actor) -> m.ActorOut:
    """An actor as ``{type, id}``: its free-form metadata never leaves the server (§S5)."""
    return m.ActorOut(type=value.type, id=value.id)


def document(value: Any) -> Any:
    """*value* as recorded JSON, with every nested actor reduced to ``{type, id}``."""
    if isinstance(value, Actor):
        return actor(value).model_dump(mode="json")
    if isinstance(value, BaseModel):
        dumped = value.model_dump(mode="json", by_alias=True)
        for name, field in type(value).model_fields.items():
            nested = getattr(value, name)
            if _holds_actor(nested):
                key = field.serialization_alias or field.alias or name
                dumped[key] = document(nested)
        return dumped
    if isinstance(value, Mapping):
        if _is_recorded_actor(value):
            return {"type": value["type"], "id": value["id"]}
        return {str(key): document(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [document(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    return value


_ACTOR_TYPES = frozenset(get_args(Actor.model_fields["type"].annotation))
_ACTOR_KEYS = frozenset(Actor.model_fields)


def _is_recorded_actor(value: Mapping[Any, Any]) -> bool:
    """An actor already serialized into a recorded JSON payload (an event's, say).

    Payloads are stored as JSON, so the actor type is gone by the time they
    are read; the shape is what is left: exactly an actor's fields, with an
    actor's type.
    """
    return set(value) == _ACTOR_KEYS and value.get("type") in _ACTOR_TYPES


def _holds_actor(value: Any) -> bool:
    if isinstance(value, Actor):
        return True
    if isinstance(value, BaseModel):
        return any(_holds_actor(getattr(value, name)) for name in type(value).model_fields)
    if isinstance(value, Mapping):
        return any(_holds_actor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_holds_actor(item) for item in value)
    return False


def _optional(value: Any) -> Any:
    return None if value is None else document(value)


def _documents(values: Iterable[Any]) -> list[Any]:
    return [document(value) for value in values]


# ---- mappers -----------------------------------------------------------------------------------


def node_counts(nodes: Iterable[ExperimentNode]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in nodes:
        counts[node.status.value] = counts.get(node.status.value, 0) + 1
    return dict(sorted(counts.items()))


def experiment_summary(
    experiment: Experiment, nodes: Iterable[ExperimentNode]
) -> m.ExperimentSummary:
    return m.ExperimentSummary(
        id=str(experiment.id),
        name=experiment.name,
        status=experiment.status.value,
        objective=document(experiment.objective),
        best_node_id=_str(experiment.best_node_id),
        node_counts=node_counts(nodes),
        created_at=experiment.created_at,
        updated_at=experiment.updated_at,
    )


def dimensions(status: BudgetStatus | None) -> list[m.DimensionOut] | None:
    if status is None:
        return None
    return [
        m.DimensionOut(
            dimension=dimension.dimension.value,
            kind=dimension.kind,
            limit=str(dimension.limit),
            reserved=str(dimension.reserved),
            committed=str(dimension.committed),
            consumed=str(dimension.consumed),
            released=str(dimension.released),
            outstanding=str(dimension.outstanding),
            remaining=str(dimension.remaining),
        )
        for dimension in status.dimensions
    ]


def experiment_out(
    experiment: Experiment,
    nodes: Iterable[ExperimentNode],
    budget_status: BudgetStatus | None,
    *,
    quiescent: bool,
    next_stage: str | None,
) -> m.ExperimentOut:
    return m.ExperimentOut(
        id=str(experiment.id),
        name=experiment.name,
        status=experiment.status.value,
        objective=document(experiment.objective),
        best_node_id=_str(experiment.best_node_id),
        node_counts=node_counts(nodes),
        policy_ref=experiment.policy_ref,
        budget=_optional(experiment.budget),
        budget_status=dimensions(budget_status),
        numerical_recovery=_optional(experiment.numerical_recovery),
        controller_host=m.ControllerHostOut(
            kind=experiment.controller_host.kind, id=experiment.controller_host.id
        ),
        compiler=_optional(experiment.compiler),
        runtime=_optional(experiment.runtime),
        artifact_root=experiment.artifact_root,
        planner=_optional(experiment.planner),
        resilience=_optional(experiment.resilience),
        evaluation=_optional(experiment.evaluation),
        metadata=document(experiment.metadata),
        quiescent=quiescent,
        next_stage=next_stage,
        created_at=experiment.created_at,
        updated_at=experiment.updated_at,
        revision=experiment.revision,
    )


def node_out(
    node: ExperimentNode, decisions: Iterable[Decision], *, best_node_id: str | None
) -> m.NodeOut:
    latest = max(decisions, key=lambda decision: decision.created_at, default=None)
    return m.NodeOut(
        id=str(node.id),
        experiment_id=str(node.experiment_id),
        parent_ids=[str(parent) for parent in node.parent_ids],
        status=node.status.value,
        hypothesis=node.hypothesis,
        reason=node.reason,
        candidate_fingerprint=node.candidate_fingerprint,
        evaluation_cycle=node.evaluation_cycle,
        created_by=actor(node.created_by),
        branch_origin=_optional(node.branch_origin),
        latest_decision=None
        if latest is None
        else m.DecisionRef(
            id=str(latest.id),
            outcome=latest.outcome.value,
            evaluation_cycle=latest.evaluation_cycle,
        ),
        is_best=best_node_id is not None and str(node.id) == best_node_id,
        created_at=node.created_at,
        updated_at=node.updated_at,
    )


def attempt_out(attempt: RunAttempt) -> m.AttemptOut:
    return m.AttemptOut(
        id=str(attempt.id),
        run_id=str(attempt.run_id),
        attempt_number=attempt.attempt_number,
        status=attempt.status.value,
        runtime_ref=_optional(attempt.runtime_ref),
        execution_fingerprint=attempt.execution_fingerprint,
        execution_overrides=_documents(attempt.execution_overrides),
        checkpoint_ref=_optional(attempt.checkpoint_ref),
        artifact_refs=_documents(attempt.artifact_refs),
        incident_ids=[str(incident) for incident in attempt.incident_ids],
        resource_usage=document(attempt.resource_usage),
        started_at=attempt.started_at,
        ended_at=attempt.ended_at,
    )


def run_out(run: Run, attempts: Iterable[RunAttempt]) -> m.RunOut:
    return m.RunOut(
        id=str(run.id),
        node_id=str(run.node_id),
        experiment_id=str(run.experiment_id),
        seed=run.seed,
        seed_origin=_optional(run.seed_origin),
        replicate=run.replicate,
        candidate_fingerprint=run.candidate_fingerprint,
        execution_plan_ref=run.execution_plan_ref,
        status=run.status.value,
        attempts=[attempt_out(attempt) for attempt in attempts],
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


def checkpoint_out(checkpoint: RecordedCheckpoint) -> m.CheckpointOut:
    return m.CheckpointOut(
        context=document(checkpoint.context),
        candidate_fingerprint=checkpoint.candidate_fingerprint,
        execution_fingerprint=checkpoint.execution_fingerprint,
        payload=document(checkpoint.payload),
        evidence=document(checkpoint.evidence),
        created_at=checkpoint.created_at,
    )


def incident_out(incident: Incident) -> m.IncidentOut:
    return m.IncidentOut(
        id=str(incident.id),
        category=incident.category.value,
        stream_generation=incident.stream_generation,
        sequence=incident.sequence,
        context=document(incident.context),
        evidence=document(incident.evidence),
        candidates=_documents(incident.candidates),
        detectors=_documents(incident.detectors),
        classifier=document(incident.classifier),
        created_at=incident.created_at,
    )


def metric_out(metric: MetricResult) -> m.MetricOut:
    return m.MetricOut(
        name=metric.name,
        value=metric.value,
        sample_count=metric.sample_count,
        dataset_ref=_optional(metric.dataset_ref),
        slice=metric.slice,
        evaluator_name=metric.evaluator_name,
        evaluator_version=metric.evaluator_version,
        seed=metric.seed,
        confidence_interval=None
        if metric.confidence_interval is None
        else list(metric.confidence_interval),
        standard_error=metric.standard_error,
        metadata=document(metric.metadata),
    )


def result_out(result: EvaluationResult) -> m.EvaluationResultOut:
    return m.EvaluationResultOut(
        id=str(result.id),
        evaluation_run_id=str(result.evaluation_run_id),
        node_id=str(result.node_id),
        subject=document(result.subject),
        evaluation_fingerprint=result.evaluation_fingerprint,
        metrics=[metric_out(metric) for metric in result.metrics],
        artifacts=_documents(result.artifacts),
        created_at=result.created_at,
    )


def evaluation_attempt_out(attempt: EvaluationAttempt) -> m.EvaluationAttemptOut:
    return m.EvaluationAttemptOut(
        id=str(attempt.id),
        evaluation_run_id=str(attempt.evaluation_run_id),
        attempt_number=attempt.attempt_number,
        status=attempt.status.value,
        runtime_ref=_optional(attempt.runtime_ref),
        started_at=attempt.started_at,
        ended_at=attempt.ended_at,
    )


def evaluation_run_out(run: EvaluationRun, result: EvaluationResult | None) -> m.EvaluationRunOut:
    return m.EvaluationRunOut(
        id=str(run.id),
        experiment_id=str(run.experiment_id),
        node_id=str(run.node_id),
        evaluation_cycle=run.evaluation_cycle,
        spec=document(run.spec),
        subject=document(run.subject),
        evaluation_fingerprint=run.evaluation_fingerprint,
        seed=run.seed,
        replicate=run.replicate,
        status=run.status.value,
        result=None if result is None else result_out(result),
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


def decision_out(decision: Decision) -> m.DecisionOut:
    return m.DecisionOut(
        id=str(decision.id),
        experiment_id=str(decision.experiment_id),
        node_id=str(decision.node_id),
        evaluation_cycle=decision.evaluation_cycle,
        outcome=decision.outcome.value,
        reason=decision.reason,
        evidence=_documents(decision.evidence),
        engine_name=decision.engine_name,
        engine_version=decision.engine_version,
        evaluation_result_ids=[str(result) for result in decision.evaluation_result_ids],
        input_fingerprint=decision.input_fingerprint,
        actor=actor(decision.actor),
        created_at=decision.created_at,
    )


def event_out(event: DomainEvent) -> m.EventOut:
    assert event.sequence is not None  # a stored event always has one
    return m.EventOut(
        sequence=event.sequence,
        id=str(event.id),
        aggregate_type=event.aggregate_type,
        aggregate_id=event.aggregate_id,
        aggregate_revision=event.aggregate_revision,
        event_type=event.event_type,
        schema_version=event.schema_version,
        occurred_at=event.occurred_at,
        actor=actor(event.actor),
        payload=document(event.payload),
    )


def policy_decision_out(decision: PolicyDecision) -> m.PolicyDecisionOut:
    return m.PolicyDecisionOut(
        id=str(decision.id),
        verdict=decision.verdict.value,
        reasons=list(decision.reasons),
        rule_ids=list(decision.rule_ids),
        engine_name=decision.engine_name,
        engine_version=decision.engine_version,
        input_fingerprint=decision.input_fingerprint,
        actor=actor(decision.actor),
        created_at=decision.created_at,
    )


def action_out(
    action: Action, policy: PolicyDecision | None, children: Iterable[Action]
) -> m.ActionOut:
    return m.ActionOut(
        id=str(action.id),
        type=action.type,
        status=action.status.value,
        outcome=None if action.outcome is None else action.outcome.value,
        target=document(action.target),
        proposed_by=actor(action.proposed_by),
        reason=action.reason,
        payload=document(action.payload),
        policy_decision=None if policy is None else policy_decision_out(policy),
        parent_action_id=_str(action.parent_action_id),
        child_action_ids=[str(child.id) for child in children],
        created_at=action.created_at,
        updated_at=action.updated_at,
    )


def ledger_entry_out(entry: BudgetLedgerEntry) -> m.LedgerEntryOut:
    return m.LedgerEntryOut(
        id=entry.id,
        dimension=entry.dimension.value,
        kind=entry.kind.value,
        amount=str(entry.amount),
        subject_kind=entry.subject_kind.value,
        subject_id=entry.subject_id,
        created_at=entry.created_at,
    )


def recovery_episode_out(
    episode: RecoveryEpisode, plan: RecoveryPlan | None
) -> m.RecoveryEpisodeOut:
    return m.RecoveryEpisodeOut(
        id=str(episode.id),
        attempt_number=episode.attempt_number,
        candidate_fingerprint=episode.candidate_fingerprint,
        context=document(episode.context),
        request=document(episode.request),
        coordinator_name=episode.coordinator_name,
        coordinator_version=episode.coordinator_version,
        created_by=actor(episode.created_by),
        effective_plan=_optional(plan),
        created_at=episode.created_at,
    )


def agent_invocation_out(invocation: AgentInvocation) -> m.AgentInvocationOut:
    intent = invocation.intent
    return m.AgentInvocationOut(
        id=str(invocation.id),
        attempt=invocation.attempt,
        status=invocation.status.value,
        planner_kind=intent.planner_kind,
        planner_version=intent.planner_version,
        prompt_version=intent.prompt_version,
        agent_model=document(intent.agent_model),
        context_fingerprint=intent.context_fingerprint,
        request_fingerprint=intent.request_fingerprint,
        proposal_fingerprint=invocation.proposal_fingerprint,
        failure_kind=None if invocation.failure is None else invocation.failure.kind,
        created_at=invocation.created_at,
        settled_at=invocation.settled_at,
    )


def lease_out(lease: ControllerLease, now: datetime) -> m.LeaseOut:
    return m.LeaseOut(
        controller_id=lease.controller_id,
        epoch=lease.epoch,
        heartbeat_at=lease.heartbeat_at,
        lease_expires_at=lease.lease_expires_at,
        live=lease.is_live(now),
    )


def _str(value: Any) -> str | None:
    return None if value is None else str(value)


# ---- metric series (Track B, migration 020) ----------------------------------------------------


def downsample(points: Sequence[StoredMetricPoint], limit: int) -> list[StoredMetricPoint]:
    """At most *limit* of *points*, in stream order, keeping every extreme.

    Splits the series into buckets and keeps each bucket's lowest and highest
    value, plus the first and last point. Deterministic: the same series and
    limit always give the same points. ``limit`` is at least 4.
    """
    if len(points) <= limit:
        return list(points)
    kept = {0, len(points) - 1}
    inner = range(1, len(points) - 1)
    buckets = max(1, (limit - 2) // 2)
    size = -(-len(inner) // buckets)  # ceiling division
    for start in range(1, len(points) - 1, size):
        bucket = range(start, min(start + size, len(points) - 1))
        kept.add(min(bucket, key=lambda index: (points[index].value, index)))
        kept.add(max(bucket, key=lambda index: (points[index].value, -index)))
    return [points[index] for index in sorted(kept)]


def metric_series_out(
    summary: MetricSeriesSummary, points: Sequence[StoredMetricPoint], limit: int
) -> m.MetricSeriesOut:
    kept = downsample(points, limit)
    return m.MetricSeriesOut(
        name=summary.name,
        unit=summary.unit,
        worker=summary.worker,
        rank=summary.rank,
        count=len(points),
        downsampled=len(kept) < len(points),
        points=[
            m.MetricPointOut(
                optimizer_step=point.optimizer_step,
                value=point.value,
                stream_generation=point.stream_generation,
                sequence=point.sequence,
                event_id=point.event_id,
                emitted_at=point.emitted_at,
            )
            for point in kept
        ],
    )
