"""The ``xaytune.studio-api/v1`` wire contract (ADR-023 §S5, §S6).

These are the API's own models, mapped from the domain in
:mod:`xaytune.api.v1.mapping` and never the domain models themselves, so a
domain refactor cannot silently change the wire. Within v1 they only grow.

``Document`` fields carry a recorded spec or payload exactly as recorded:
its shape is the domain's own versioned schema, not this API's (§S5).
Actors are ``{type, id}``; their metadata is never returned.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA = "xaytune.studio-api/v1"

Document = dict[str, Any]
"""A recorded document, passed through as recorded."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _Body(_Model):
    """A top-level response: every body names its contract."""

    # ``schema`` would shadow a BaseModel attribute, hence the alias.
    schema_: Literal["xaytune.studio-api/v1"] = Field(
        default="xaytune.studio-api/v1", serialization_alias="schema", validation_alias="schema"
    )


class ActorOut(_Model):
    type: str
    id: str


class Problem(_Model):
    """``application/problem+json`` (§S8): fixed text, never a value read or sent."""

    type: str
    title: str
    status: int
    detail: str
    correlation_id: str | None = None


# ---- meta --------------------------------------------------------------------------------------


class StoreOut(_Model):
    readable: bool
    reason: str | None
    migration: int | None
    supported_migration: int


class LeaseOut(_Model):
    controller_id: str
    epoch: int
    heartbeat_at: datetime
    lease_expires_at: datetime
    live: bool


class MetaOut(_Body):
    api_version: Literal["v1"] = "v1"
    xaytune_version: str
    store: StoreOut
    latest_event_sequence: int | None
    controller_lease: LeaseOut | None


# ---- experiments -------------------------------------------------------------------------------


class ExperimentSummary(_Model):
    id: str
    name: str
    status: str
    objective: Document
    best_node_id: str | None
    node_counts: dict[str, int]
    created_at: datetime
    updated_at: datetime


class ExperimentPage(_Body):
    items: list[ExperimentSummary]
    next_before: str | None
    """Pass as ``before`` for the next, older page; ``None`` on the last page."""


class DimensionOut(_Model):
    """One limited dimension's balance.

    Amounts are exact decimal strings, as the ledger keeps them.
    """

    dimension: str
    kind: str
    limit: str
    reserved: str
    committed: str
    consumed: str
    released: str
    outstanding: str
    remaining: str


class ControllerHostOut(_Model):
    kind: str
    id: str | None


class ExperimentOut(_Body):
    id: str
    name: str
    status: str
    objective: Document
    best_node_id: str | None
    node_counts: dict[str, int]
    policy_ref: str | None
    budget: Document | None
    budget_status: list[DimensionOut] | None
    numerical_recovery: Document | None
    controller_host: ControllerHostOut
    compiler: Document | None
    runtime: Document | None
    artifact_root: str | None
    planner: Document | None
    resilience: Document | None
    evaluation: Document | None
    metadata: Document
    quiescent: bool
    """Whether all executable work is settled, as the controller's own result reports."""
    next_stage: str | None
    created_at: datetime
    updated_at: datetime
    revision: int


# ---- candidates --------------------------------------------------------------------------------


class DecisionRef(_Model):
    id: str
    outcome: str
    evaluation_cycle: int


class NodeOut(_Model):
    id: str
    experiment_id: str
    parent_ids: list[str]
    status: str
    hypothesis: str | None
    reason: str | None
    candidate_fingerprint: str
    evaluation_cycle: int
    created_by: ActorOut
    branch_origin: Document | None
    latest_decision: DecisionRef | None
    is_best: bool
    created_at: datetime
    updated_at: datetime


class EdgeOut(_Model):
    parent_id: str
    child_id: str


class GraphOut(_Body):
    experiment_id: str
    best_node_id: str | None
    nodes: list[NodeOut]
    edges: list[EdgeOut]


# ---- runs --------------------------------------------------------------------------------------


class AttemptOut(_Model):
    id: str
    run_id: str
    attempt_number: int
    status: str
    runtime_ref: Document | None
    execution_fingerprint: str | None
    execution_overrides: list[Document]
    checkpoint_ref: Document | None
    artifact_refs: list[Document]
    incident_ids: list[str]
    resource_usage: Document
    started_at: datetime | None
    ended_at: datetime | None


class RunOut(_Model):
    id: str
    node_id: str
    experiment_id: str
    seed: int | None
    seed_origin: Document | None
    replicate: int | None
    candidate_fingerprint: str
    execution_plan_ref: str | None
    status: str
    attempts: list[AttemptOut]
    created_at: datetime
    updated_at: datetime


class CheckpointOut(_Model):
    context: Document
    candidate_fingerprint: str
    execution_fingerprint: str
    payload: Document
    evidence: Document
    created_at: datetime


class IncidentOut(_Model):
    id: str
    category: str
    stream_generation: int
    sequence: int
    context: Document
    evidence: Document
    candidates: list[Document]
    detectors: list[Document]
    classifier: Document
    created_at: datetime


class RunDetail(_Body):
    run: RunOut
    checkpoints: list[CheckpointOut]
    interventions: list[Document]
    intervention_applications: list[Document]


class AttemptDetail(_Body):
    attempt: AttemptOut
    incidents: list[IncidentOut]
    checkpoints: list[CheckpointOut]
    directives: list[Document]


# ---- evaluation and decisions ------------------------------------------------------------------


class MetricOut(_Model):
    name: str
    value: float
    sample_count: int | None
    dataset_ref: Document | None
    slice: str | None
    evaluator_name: str
    evaluator_version: str | None
    seed: int | None
    confidence_interval: list[float] | None
    standard_error: float | None
    metadata: Document


class EvaluationResultOut(_Model):
    id: str
    evaluation_run_id: str
    node_id: str
    subject: Document
    evaluation_fingerprint: str
    metrics: list[MetricOut]
    artifacts: list[Document]
    created_at: datetime


class EvaluationAttemptOut(_Model):
    id: str
    evaluation_run_id: str
    attempt_number: int
    status: str
    runtime_ref: Document | None
    started_at: datetime | None
    ended_at: datetime | None


class EvaluationRunOut(_Model):
    id: str
    experiment_id: str
    node_id: str
    evaluation_cycle: int
    spec: Document
    subject: Document
    evaluation_fingerprint: str
    seed: int | None
    replicate: int | None
    status: str
    result: EvaluationResultOut | None
    created_at: datetime
    updated_at: datetime


class EvaluationRunDetail(_Body):
    evaluation_run: EvaluationRunOut
    attempts: list[EvaluationAttemptOut]


class DecisionOut(_Model):
    id: str
    experiment_id: str
    node_id: str
    evaluation_cycle: int
    outcome: str
    reason: str
    evidence: list[Document]
    engine_name: str
    engine_version: str
    evaluation_result_ids: list[str]
    input_fingerprint: str
    actor: ActorOut
    created_at: datetime


class DecisionList(_Body):
    items: list[DecisionOut]


class NodeDetail(_Body):
    node: NodeOut
    candidate: Document
    lineage_paths: list[list[str]]
    runs: list[RunOut]
    evaluation_runs: list[EvaluationRunOut]
    decisions: list[DecisionOut]


# ---- comparison --------------------------------------------------------------------------------


class SpecDifference(_Model):
    path: str
    left: Any
    right: Any
    """``None`` on a side that does not have the path at all."""


class CandidateLineage(_Model):
    """How two candidates relate in the graph. Structural only: it says nothing about
    whether their measurements can be compared (that is :class:`MetricPair`'s)."""

    distinct_candidates: bool
    """``ExperimentGraph.compare``'s ``is_comparable``: two scientific candidates, not
    two records of one candidate fingerprint."""
    same_candidate: bool
    common_ancestor_ids: list[str]
    nearest_common_ancestor_ids: list[str]


class MetricSide(_Model):
    """One candidate's measurement, with what it was measured under."""

    metric: MetricOut
    result_id: str
    evaluation_fingerprint: str
    """The evaluation's identity: evaluator, version and configuration, dataset, slices."""
    unit: str | None
    """The unit the evaluator recorded, or ``None`` if it recorded none."""


class MetricPair(_Model):
    """One metric on both candidates, and whether the two values can be compared.

    ``comparable`` is true only when both sides were measured under the same
    evaluation fingerprint and, where a unit is recorded, the same unit. Only
    then is ``difference`` given; otherwise it is ``None`` and ``reasons`` says
    why. This is a recorded-identity check, not a statistical comparison:
    harness comparisons (ADR-022 C1-C4) are H06's, not this API's.
    """

    evaluator_name: str
    name: str
    slice: str | None
    left: MetricSide | None
    right: MetricSide | None
    comparable: bool
    reasons: list[str]
    difference: float | None
    """``left - right``, only when ``comparable``."""


class CompareOut(_Body):
    left: NodeOut
    right: NodeOut
    lineage: CandidateLineage
    candidate_differences: list[SpecDifference]
    metrics: list[MetricPair]


# ---- activity ----------------------------------------------------------------------------------


class EventOut(_Model):
    sequence: int
    id: str
    aggregate_type: str
    aggregate_id: str
    aggregate_revision: int
    event_type: str
    schema_version: str
    occurred_at: datetime
    actor: ActorOut
    payload: Document


class EventPage(_Body):
    items: list[EventOut]
    next_after: int
    """Pass as ``after`` to read on; the last sequence returned, or *after* if none."""


class PolicyDecisionOut(_Model):
    id: str
    verdict: str
    reasons: list[str]
    rule_ids: list[str]
    engine_name: str
    engine_version: str
    input_fingerprint: str
    actor: ActorOut
    created_at: datetime


class ActionOut(_Model):
    id: str
    type: str
    status: str
    outcome: str | None
    target: Document
    proposed_by: ActorOut
    reason: str
    payload: Document
    policy_decision: PolicyDecisionOut | None
    parent_action_id: str | None
    child_action_ids: list[str]
    created_at: datetime
    updated_at: datetime


class ActionList(_Body):
    items: list[ActionOut]


class LedgerEntryOut(_Model):
    id: str
    dimension: str
    kind: str
    amount: str
    """An exact decimal string."""
    subject_kind: str
    subject_id: str
    created_at: datetime


class BudgetOut(_Body):
    limits: Document | None
    status: list[DimensionOut] | None
    entries: list[LedgerEntryOut]


class RecoveryEpisodeOut(_Model):
    id: str
    attempt_number: int
    candidate_fingerprint: str
    context: Document
    request: Document
    coordinator_name: str
    coordinator_version: str
    created_by: ActorOut
    effective_plan: Document | None
    created_at: datetime


class IncidentList(_Body):
    incidents: list[IncidentOut]
    recovery_episodes: list[RecoveryEpisodeOut]


class AgentInvocationOut(_Model):
    """A planner's model call, summarised: no prompt, request or response body (§S6)."""

    id: str
    attempt: int
    status: str
    planner_kind: str
    planner_version: str
    prompt_version: str
    agent_model: Document
    context_fingerprint: str
    request_fingerprint: str
    proposal_fingerprint: str | None
    failure_kind: str | None
    created_at: datetime
    settled_at: datetime | None


class AgentInvocationList(_Body):
    items: list[AgentInvocationOut]
