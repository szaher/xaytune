"""The ``/api/v1`` routes (ADR-023 §S6). Every route reads; none writes.

Each route reads through one :meth:`StudioStore.snapshot`, so a response is
one consistent state of the record (§S3), and composes the existing stores
(§S4): nothing here holds SQL, and nothing re-derives an answer a store or
the controller's own result already gives.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Iterator
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request, Response

from xaytune._version import __version__
from xaytune.core.clock import utc_now
from xaytune.core.domain.experiment import ExperimentNode
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.ids import ExperimentId
from xaytune.experiment.result import experiment_result
from xaytune.storage.control_plane import ControlPlaneRepository
from xaytune.storage.graph import LineageError
from xaytune.storage.journal import EventJournal
from xaytune.storage.leases import ControllerLeaseStore

from . import mapping
from . import models as m

__all__ = ["EVENT_SEQUENCE_HEADER", "InvalidParameterError", "NotFoundError", "router"]

EVENT_SEQUENCE_HEADER = "X-Xaytune-Event-Sequence"
DEFAULT_LIMIT = 100
MAX_LIMIT = 500

_PROBLEMS: dict[int | str, dict[str, Any]] = {
    404: {"model": m.Problem, "description": "No such record"},
    422: {"model": m.Problem, "description": "A parameter the API cannot use"},
    503: {"model": m.Problem, "description": "The state database cannot be read by this build"},
}

router = APIRouter(prefix="/api/v1", responses=_PROBLEMS)


class NotFoundError(Exception):
    """No record of *kind* with the id asked for. Carries the kind, never the id (§S8)."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


class InvalidParameterError(Exception):
    """A parameter the API cannot use. Carries its name, never its value (§S8)."""

    def __init__(self, parameter: str, reason: str) -> None:
        super().__init__(parameter)
        self.parameter = parameter
        self.reason = reason


def _repository(request: Request, response: Response) -> Iterator[ControlPlaneRepository]:
    with request.app.state.store.snapshot() as repository:
        response.headers[EVENT_SEQUENCE_HEADER] = str(repository.events.latest_sequence())
        yield repository


Repository = Annotated[ControlPlaneRepository, Depends(_repository)]
Limit = Annotated[int, Query(ge=1, le=MAX_LIMIT)]


# ---- meta --------------------------------------------------------------------------------------


@router.get("/meta", response_model=m.MetaOut)
def meta(request: Request, response: Response) -> m.MetaOut:
    """What this API reads, and whether it can: answered even when the store cannot be read."""
    store = request.app.state.store
    status = store.status()
    latest: int | None = None
    lease: m.LeaseOut | None = None
    if status.readable:
        with store.connection() as connection:
            latest = EventJournal(connection).latest_sequence()
            current = ControllerLeaseStore(connection).current()
            lease = None if current is None else mapping.lease_out(current, utc_now())
        response.headers[EVENT_SEQUENCE_HEADER] = str(latest)
    return m.MetaOut(
        xaytune_version=__version__,
        store=m.StoreOut(
            readable=status.readable,
            reason=status.reason,
            migration=status.migration,
            supported_migration=status.supported_migration,
        ),
        latest_event_sequence=latest,
        controller_lease=lease,
    )


# ---- experiments -------------------------------------------------------------------------------


def _encode_cursor(created_at: datetime, experiment_id: str) -> str:
    raw = json.dumps([created_at.isoformat(), experiment_id], separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        created_at, experiment_id = json.loads(base64.urlsafe_b64decode(padded.encode()))
        return datetime.fromisoformat(created_at), str(experiment_id)
    except (ValueError, TypeError, binascii.Error) as error:
        raise InvalidParameterError("before", "not a cursor this API issued") from error


@router.get("/experiments", response_model=m.ExperimentPage)
def list_experiments(
    repository: Repository,
    status: str | None = None,
    before: str | None = None,
    limit: Limit = DEFAULT_LIMIT,
) -> m.ExperimentPage:
    """Experiments, newest first, keyset-paged (§S6)."""
    cursor = None if before is None else _decode_cursor(before)
    experiments = repository.aggregates.list_experiments(status=status, before=cursor, limit=limit)
    items = [
        mapping.experiment_summary(
            experiment, repository.aggregates.nodes_for_experiment(str(experiment.id))
        )
        for experiment in experiments
    ]
    last = experiments[-1] if len(experiments) == limit else None
    return m.ExperimentPage(
        items=items,
        next_before=None if last is None else _encode_cursor(last.created_at, str(last.id)),
    )


def _experiment(repository: ControlPlaneRepository, experiment_id: str) -> Any:
    experiment = repository.aggregates.get_experiment(experiment_id)
    if experiment is None:
        raise NotFoundError("experiment")
    return experiment


@router.get("/experiments/{experiment_id}", response_model=m.ExperimentOut)
def get_experiment(experiment_id: str, repository: Repository) -> m.ExperimentOut:
    experiment = _experiment(repository, experiment_id)
    result = experiment_result(repository, ExperimentId(experiment_id))
    return mapping.experiment_out(
        experiment,
        repository.aggregates.nodes_for_experiment(experiment_id),
        result.budget,
        quiescent=result.quiescent,
        next_stage=result.next_stage,
    )


@router.get("/experiments/{experiment_id}/graph", response_model=m.GraphOut)
def get_graph(experiment_id: str, repository: Repository) -> m.GraphOut:
    experiment = _experiment(repository, experiment_id)
    best = None if experiment.best_node_id is None else str(experiment.best_node_id)
    nodes = repository.aggregates.nodes_for_experiment(experiment_id)
    return m.GraphOut(
        experiment_id=experiment_id,
        best_node_id=best,
        nodes=[_node(repository, node, best) for node in nodes],
        edges=[
            m.EdgeOut(parent_id=str(parent.id), child_id=str(node.id))
            for node in nodes
            for parent in repository.graph.parents(str(node.id))
        ],
    )


def _node(repository: ControlPlaneRepository, node: ExperimentNode, best: str | None) -> m.NodeOut:
    return mapping.node_out(
        node, repository.aggregates.decisions_for_node(str(node.id)), best_node_id=best
    )


@router.get("/experiments/{experiment_id}/events", response_model=m.EventPage)
def get_events(
    experiment_id: str,
    repository: Repository,
    after: Annotated[int, Query(ge=0)] = 0,
    limit: Limit = DEFAULT_LIMIT,
) -> m.EventPage:
    """The experiment's event journal after *after*, by sequence (§S6, §S7)."""
    _experiment(repository, experiment_id)
    events = repository.events.events_for_experiment_after(experiment_id, after, limit=limit)
    items = [mapping.event_out(event) for event in events]
    return m.EventPage(items=items, next_after=items[-1].sequence if items else after)


@router.get("/experiments/{experiment_id}/actions", response_model=m.ActionList)
def get_actions(experiment_id: str, repository: Repository) -> m.ActionList:
    _experiment(repository, experiment_id)
    return m.ActionList(
        items=[
            mapping.action_out(
                action,
                repository.policy.for_action(str(action.id)),
                repository.actions.children(str(action.id)),
            )
            for action in repository.actions.for_experiment(experiment_id)
        ]
    )


@router.get("/experiments/{experiment_id}/decisions", response_model=m.DecisionList)
def get_decisions(experiment_id: str, repository: Repository) -> m.DecisionList:
    _experiment(repository, experiment_id)
    decisions = [
        decision
        for node in repository.aggregates.nodes_for_experiment(experiment_id)
        for decision in repository.aggregates.decisions_for_node(str(node.id))
    ]
    decisions.sort(key=lambda decision: (decision.created_at, str(decision.id)))
    return m.DecisionList(items=[mapping.decision_out(decision) for decision in decisions])


@router.get("/experiments/{experiment_id}/budget", response_model=m.BudgetOut)
def get_budget(experiment_id: str, repository: Repository) -> m.BudgetOut:
    experiment = _experiment(repository, experiment_id)
    return m.BudgetOut(
        limits=None if experiment.budget is None else mapping.document(experiment.budget),
        status=mapping.dimensions(repository.budget_status(experiment_id)),
        entries=[
            mapping.ledger_entry_out(entry) for entry in repository.budget.entries(experiment_id)
        ],
    )


@router.get("/experiments/{experiment_id}/incidents", response_model=m.IncidentList)
def get_incidents(experiment_id: str, repository: Repository) -> m.IncidentList:
    _experiment(repository, experiment_id)
    return m.IncidentList(
        incidents=[
            mapping.incident_out(incident)
            for incident in repository.incidents.for_experiment(experiment_id)
        ],
        recovery_episodes=[
            mapping.recovery_episode_out(
                episode, repository.recovery_plans.effective_for_episode(str(episode.id))
            )
            for episode in repository.recovery_episodes.for_experiment(experiment_id)
        ],
    )


@router.get("/experiments/{experiment_id}/agent-invocations", response_model=m.AgentInvocationList)
def get_agent_invocations(experiment_id: str, repository: Repository) -> m.AgentInvocationList:
    _experiment(repository, experiment_id)
    return m.AgentInvocationList(
        items=[
            mapping.agent_invocation_out(invocation)
            for invocation in repository.agent_invocations.for_experiment(experiment_id)
        ]
    )


# ---- candidates --------------------------------------------------------------------------------


def _load_node(repository: ControlPlaneRepository, node_id: str) -> ExperimentNode:
    node = repository.aggregates.get_node(node_id)
    if node is None:
        raise NotFoundError("node")
    return node


def _best(repository: ControlPlaneRepository, node: ExperimentNode) -> str | None:
    experiment = repository.aggregates.load_experiment(str(node.experiment_id))
    return None if experiment.best_node_id is None else str(experiment.best_node_id)


def _runs(repository: ControlPlaneRepository, node_id: str) -> list[m.RunOut]:
    return [
        mapping.run_out(run, repository.aggregates.attempts_for_run(str(run.id)))
        for run in repository.aggregates.runs_for_node(node_id)
    ]


def _evaluation_runs(repository: ControlPlaneRepository, node_id: str) -> list[m.EvaluationRunOut]:
    return [
        mapping.evaluation_run_out(
            run, repository.aggregates.evaluation_result_for_run(str(run.id))
        )
        for run in repository.aggregates.evaluation_runs_for_node(node_id)
    ]


@router.get("/nodes/{node_id}", response_model=m.NodeDetail)
def get_node(node_id: str, repository: Repository) -> m.NodeDetail:
    node = _load_node(repository, node_id)
    decisions = repository.aggregates.decisions_for_node(node_id)
    return m.NodeDetail(
        node=mapping.node_out(node, decisions, best_node_id=_best(repository, node)),
        candidate=mapping.document(node.candidate),
        lineage_paths=[
            [str(step.id) for step in path] for path in repository.graph.lineage_paths(node_id)
        ],
        runs=_runs(repository, node_id),
        evaluation_runs=_evaluation_runs(repository, node_id),
        decisions=[mapping.decision_out(decision) for decision in decisions],
    )


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict) and value:
        flat: dict[str, Any] = {}
        for key in sorted(value):
            flat.update(_flatten(value[key], f"{prefix}.{key}" if prefix else str(key)))
        return flat
    return {prefix: value}


MetricKey = tuple[str, str, str | None]
"""(evaluator, metric, slice): what a pair of measurements is about."""

ONE_SIDED = "measured on one candidate only"
FINGERPRINTS_DIFFER = (
    "measured under different evaluations (evaluator version or configuration, "
    "dataset or slices differ)"
)
UNITS_DIFFER = "recorded in different units"


def _measurements(
    repository: ControlPlaneRepository, node_id: str
) -> dict[MetricKey, list[tuple[datetime, m.MetricSide]]]:
    """Every recorded measurement of the node, by what it measures, oldest first."""
    found: dict[MetricKey, list[tuple[datetime, m.MetricSide]]] = {}
    results = sorted(
        repository.aggregates.evaluation_results_for_node(node_id),
        key=lambda result: (result.created_at, str(result.id)),
    )
    for result in results:
        for metric in result.metrics:
            unit = metric.metadata.get("unit")
            side = m.MetricSide(
                metric=mapping.metric_out(metric),
                result_id=str(result.id),
                evaluation_fingerprint=result.evaluation_fingerprint,
                unit=unit if isinstance(unit, str) else None,
            )
            key = (metric.evaluator_name, metric.name, metric.slice)
            found.setdefault(key, []).append((result.created_at, side))
    return found


def _pair(
    key: MetricKey,
    left: list[tuple[datetime, m.MetricSide]],
    right: list[tuple[datetime, m.MetricSide]],
) -> m.MetricPair:
    """Pair one metric across the candidates, preferring measurements under one evaluation.

    When both were measured under a shared evaluation fingerprint, the most
    recent such pair is shown; otherwise each side's latest, marked not
    comparable. A recorded-identity check only: the API judges no statistics.
    """
    shared = {side.evaluation_fingerprint for _, side in left} & {
        side.evaluation_fingerprint for _, side in right
    }
    if shared:
        latest_shared = max(
            (moment, side.evaluation_fingerprint)
            for moment, side in (*left, *right)
            if side.evaluation_fingerprint in shared
        )[1]
        left = [entry for entry in left if entry[1].evaluation_fingerprint == latest_shared]
        right = [entry for entry in right if entry[1].evaluation_fingerprint == latest_shared]
    left_side = left[-1][1] if left else None
    right_side = right[-1][1] if right else None
    reasons: list[str] = []
    if left_side is None or right_side is None:
        reasons.append(ONE_SIDED)
    else:
        if left_side.evaluation_fingerprint != right_side.evaluation_fingerprint:
            reasons.append(FINGERPRINTS_DIFFER)
        if left_side.unit != right_side.unit:
            reasons.append(UNITS_DIFFER)
    comparable = not reasons
    return m.MetricPair(
        evaluator_name=key[0],
        name=key[1],
        slice=key[2],
        left=left_side,
        right=right_side,
        comparable=comparable,
        reasons=reasons,
        difference=left_side.metric.value - right_side.metric.value
        if comparable and left_side is not None and right_side is not None
        else None,
    )


@router.get("/compare", response_model=m.CompareOut)
def compare(
    repository: Repository,
    left: Annotated[str, Query(min_length=1)],
    right: Annotated[str, Query(min_length=1)],
) -> m.CompareOut:
    """Two candidates side by side: lineage, spec difference and metrics (§S6).

    Lineage is structural. Whether two measurements can be compared is
    decided per metric, from their recorded evaluation identity.
    """
    left_node = _load_node(repository, left)
    right_node = _load_node(repository, right)
    if left_node.experiment_id != right_node.experiment_id:
        raise InvalidParameterError("right", "the candidates belong to different experiments")
    try:
        comparison = repository.graph.compare(left, right)
    except LineageError as error:
        raise InvalidParameterError("right", "the candidates cannot be compared") from error
    best = _best(repository, left_node)
    left_spec = _flatten(mapping.document(left_node.candidate))
    right_spec = _flatten(mapping.document(right_node.candidate))
    left_metrics = _measurements(repository, left)
    right_metrics = _measurements(repository, right)
    return m.CompareOut(
        left=_node(repository, left_node, best),
        right=_node(repository, right_node, best),
        lineage=m.CandidateLineage(
            distinct_candidates=comparison.is_comparable,
            same_candidate=comparison.same_candidate,
            common_ancestor_ids=[str(node.id) for node in comparison.common_ancestors],
            nearest_common_ancestor_ids=[
                str(node.id) for node in comparison.nearest_common_ancestors
            ],
        ),
        candidate_differences=[
            m.SpecDifference(path=path, left=left_spec.get(path), right=right_spec.get(path))
            for path in sorted(left_spec.keys() | right_spec.keys())
            if left_spec.get(path, _ABSENT) != right_spec.get(path, _ABSENT)
        ],
        metrics=[
            _pair(key, left_metrics.get(key, []), right_metrics.get(key, []))
            for key in sorted(
                left_metrics.keys() | right_metrics.keys(),
                key=lambda key: (key[0], key[1], key[2] or ""),
            )
        ],
    )


_ABSENT = object()


# ---- runs and evaluations ----------------------------------------------------------------------


@router.get("/runs/{run_id}", response_model=m.RunDetail)
def get_run(run_id: str, repository: Repository) -> m.RunDetail:
    run = repository.aggregates.get_run(run_id)
    if run is None:
        raise NotFoundError("run")
    attempts = repository.aggregates.attempts_for_run(run_id)
    interventions = repository.training_interventions.for_run(run_id)
    return m.RunDetail(
        run=mapping.run_out(run, attempts),
        checkpoints=[
            mapping.checkpoint_out(checkpoint)
            for attempt in attempts
            for checkpoint in repository.checkpoints.for_attempt(str(attempt.id))
        ],
        interventions=[mapping.document(intervention) for intervention in interventions],
        intervention_applications=[
            mapping.document(application)
            for application in repository.intervention_applications.for_run(run_id)
        ],
    )


@router.get("/attempts/{attempt_id}", response_model=m.AttemptDetail)
def get_attempt(attempt_id: str, repository: Repository) -> m.AttemptDetail:
    attempt = repository.aggregates.get_attempt(attempt_id)
    if attempt is None:
        raise NotFoundError("attempt")
    incidents = repository.incidents.for_attempt(
        RuntimeOperationTarget(kind="training-attempt", id=attempt_id)
    )
    return m.AttemptDetail(
        attempt=mapping.attempt_out(attempt),
        incidents=[mapping.incident_out(incident) for incident in incidents],
        checkpoints=[
            mapping.checkpoint_out(checkpoint)
            for checkpoint in repository.checkpoints.for_attempt(attempt_id)
        ],
        directives=[
            mapping.document(directive)
            for directive in repository.intervention_directives.for_attempt(attempt_id)
        ],
    )


@router.get("/evaluation-runs/{evaluation_run_id}", response_model=m.EvaluationRunDetail)
def get_evaluation_run(evaluation_run_id: str, repository: Repository) -> m.EvaluationRunDetail:
    run = repository.aggregates.get_evaluation_run(evaluation_run_id)
    if run is None:
        raise NotFoundError("evaluation run")
    return m.EvaluationRunDetail(
        evaluation_run=mapping.evaluation_run_out(
            run, repository.aggregates.evaluation_result_for_run(evaluation_run_id)
        ),
        attempts=[
            mapping.evaluation_attempt_out(attempt)
            for attempt in repository.aggregates.evaluation_attempts_for_run(evaluation_run_id)
        ],
    )
