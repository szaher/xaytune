"""Evaluation's durable record: runs, attempts, results, cycles (ADR-015, PR-013).

```text
begin_evaluation_cycle        node EVALUATING (cycle n) + its runs    one commit
create_evaluation_attempt_with_submit_intent      attempt + INTENDED   one commit
hold_evaluation_completion    pending completion + cursor at it       one commit
record_evaluation_result      result + attempt + run SUCCEEDED        one commit
reconcile_evaluating_node     WAITING | DECIDING | STALLED, over cycle n only
```
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from tests.test_storage.conftest import make_experiment, make_node
from xaytune.core import (
    Actor,
    ArtifactId,
    EvaluationAttempt,
    EvaluationAttemptId,
    EvaluationId,
    EvaluationResult,
    EvaluationRun,
    EvaluationRunId,
    EvaluationSpec,
    EvaluatorSpec,
    MetricResult,
)
from xaytune.core.domain.action import ActionTarget
from xaytune.core.errors import InvalidTransitionError
from xaytune.core.ids import OperationId, RunAttemptId
from xaytune.core.refs import ArtifactRef
from xaytune.core.state.status import (
    EvaluationAttemptStatus,
    EvaluationRunStatus,
    ExperimentNodeStatus,
    ExperimentStatus,
)
from xaytune.storage import connect, write_transaction
from xaytune.storage.control_plane import (
    ControlPlaneRepository,
    EvaluationReconciliation,
    ProvenanceError,
    UnknownOperationTargetError,
)
from xaytune.storage.errors import StorageError
from xaytune.storage.journal import IdempotencyConflictError

_ACTOR = Actor(type="system", id="test")
_SPEC = EvaluationSpec(evaluator=EvaluatorSpec(name="exact-match", version="1.0.0"))


@pytest.fixture
def repo(connection: sqlite3.Connection) -> ControlPlaneRepository:
    return ControlPlaneRepository(connection)


@pytest.fixture
def node(repo: ControlPlaneRepository) -> Any:
    """A trained node: ACTIVE, ready to be evaluated."""
    experiment = repo.create_experiment(make_experiment(), actor=_ACTOR)
    repo.transition_experiment(
        experiment.id,
        expected_revision=0,
        new_status=ExperimentStatus.ACTIVE,
        actor=_ACTOR,
    )
    created = repo.create_node(make_node(experiment), actor=_ACTOR)
    current = created
    for status in (
        ExperimentNodeStatus.PLANNED,
        ExperimentNodeStatus.READY,
        ExperimentNodeStatus.ACTIVE,
    ):
        current = repo.transition_node(
            current.id, expected_revision=current.revision, new_status=status, actor=_ACTOR
        )
    return current


_SUBJECT = ArtifactRef(id=ArtifactId.generate(), kind="model", uri="/m", digest="sha256:m")


def _run(node: Any, *, cycle: int | None = None, seed: int = 7) -> EvaluationRun:
    return EvaluationRun(
        id=EvaluationRunId.generate(),
        experiment_id=node.experiment_id,
        node_id=node.id,
        evaluation_cycle=cycle if cycle is not None else node.evaluation_cycle + 1,
        spec=_SPEC,
        subject=_SUBJECT,
        evaluation_fingerprint=_SPEC.evaluation_fingerprint(),
        seed=seed,
        replicate=1,
    )


def _begin(repo: ControlPlaneRepository, node: Any, *runs: EvaluationRun) -> Any:
    moved, _ = repo.begin_evaluation_cycle(
        node.id, expected_revision=node.revision, runs=runs, actor=_ACTOR
    )
    return moved


def _attempt(repo: ControlPlaneRepository, run: EvaluationRun, number: int = 1) -> Any:
    active = repo.aggregates.load_evaluation_run(str(run.id))
    if active.status is EvaluationRunStatus.CREATED:
        repo.transition_evaluation_run(
            run.id,
            expected_revision=active.revision,
            new_status=EvaluationRunStatus.ACTIVE,
            actor=_ACTOR,
        )
    attempt, operation = repo.create_evaluation_attempt_with_submit_intent(
        EvaluationAttempt(
            id=EvaluationAttemptId.generate(), evaluation_run_id=run.id, attempt_number=number
        ),
        request_digest="sha256:request",
        actor=_ACTOR,
    )
    return attempt, operation


def _running(repo: ControlPlaneRepository, attempt: Any) -> Any:
    for status in (
        EvaluationAttemptStatus.QUEUED,
        EvaluationAttemptStatus.STARTING,
        EvaluationAttemptStatus.RUNNING,
    ):
        attempt = repo.transition_evaluation_attempt(
            attempt.id, expected_revision=attempt.revision, new_status=status, actor=_ACTOR
        )
    return attempt


def _result(run: EvaluationRun, **overrides: object) -> EvaluationResult:
    fields: dict[str, object] = {
        "id": EvaluationId.generate(),
        "evaluation_run_id": run.id,
        "node_id": run.node_id,
        "subject": run.subject,
        "evaluation_fingerprint": run.evaluation_fingerprint,
        "metrics": (_metric(run),),
    }
    fields.update(overrides)
    return EvaluationResult(**fields)  # type: ignore[arg-type]


def _metric(run: EvaluationRun, **overrides: object) -> MetricResult:
    """A metric measured by the run's evaluator, at its version, with its seed."""
    fields: dict[str, object] = {
        "name": "accuracy",
        "value": 0.8,
        "evaluator_name": run.spec.evaluator.name,
        "evaluator_version": run.spec.evaluator.version,
        "seed": run.seed,
    }
    fields.update(overrides)
    return MetricResult(**fields)  # type: ignore[arg-type]


def _succeed(repo: ControlPlaneRepository, run: EvaluationRun) -> EvaluationResult:
    attempt, _ = _attempt(repo, run)
    attempt = _running(repo, attempt)
    return repo.record_evaluation_result(
        attempt.id, _result(run), expected_revision=attempt.revision, actor=_ACTOR
    )


# ---- persistence ------------------------------------------------------------------


def test_a_cycle_and_its_runs_commit_together(repo: ControlPlaneRepository, node: Any) -> None:
    run = _run(node)
    moved = _begin(repo, node, run)

    assert moved.status is ExperimentNodeStatus.EVALUATING
    assert moved.evaluation_cycle == 1
    assert repo.aggregates.evaluation_runs_for_node(str(node.id), cycle=1) == (run,)


def test_a_run_of_another_cycle_refuses_the_whole_cycle(
    repo: ControlPlaneRepository, node: Any
) -> None:
    with pytest.raises(StorageError, match="cycle 5"):
        _begin(repo, node, _run(node), _run(node, cycle=5))

    assert repo.aggregates.load_node(str(node.id)).status is ExperimentNodeStatus.ACTIVE
    assert repo.aggregates.evaluation_runs_for_node(str(node.id)) == ()


def test_a_run_and_its_attempt_round_trip(repo: ControlPlaneRepository, node: Any) -> None:
    run = _run(node)
    _begin(repo, node, run)
    attempt, operation = _attempt(repo, run)

    assert repo.aggregates.load_evaluation_attempt(str(attempt.id)) == attempt
    assert operation.state == "intended"
    assert operation.target.kind == "evaluation-attempt"
    assert repo.operations.for_target("evaluation-attempt", str(attempt.id)) == (operation,)


def test_attempt_numbers_are_unique_within_a_run(repo: ControlPlaneRepository, node: Any) -> None:
    run = _run(node)
    _begin(repo, node, run)
    _attempt(repo, run, number=1)

    with pytest.raises(sqlite3.IntegrityError):
        _attempt(repo, run, number=1)


def test_submission_intent_is_get_or_create(repo: ControlPlaneRepository, node: Any) -> None:
    run = _run(node)
    _begin(repo, node, run)
    repo.transition_evaluation_run(
        run.id, expected_revision=0, new_status=EvaluationRunStatus.ACTIVE, actor=_ACTOR
    )
    attempt = EvaluationAttempt(
        id=EvaluationAttemptId.generate(), evaluation_run_id=run.id, attempt_number=1
    )
    operation_id = OperationId.generate()
    first = repo.create_evaluation_attempt_with_submit_intent(
        attempt, request_digest="sha256:a", actor=_ACTOR, operation_id=operation_id
    )
    again = repo.create_evaluation_attempt_with_submit_intent(
        attempt, request_digest="sha256:a", actor=_ACTOR, operation_id=operation_id
    )
    assert again == first

    other = EvaluationAttempt(
        id=EvaluationAttemptId.generate(), evaluation_run_id=run.id, attempt_number=1
    )
    with pytest.raises(IdempotencyConflictError):
        repo.create_evaluation_attempt_with_submit_intent(
            other, request_digest="sha256:a", actor=_ACTOR, operation_id=operation_id
        )


def test_the_telemetry_cursor_is_durable_and_only_moves_forward(
    repo: ControlPlaneRepository, node: Any, db_path: Any
) -> None:
    run = _run(node)
    _begin(repo, node, run)
    attempt, _ = _attempt(repo, run)
    position = repo.aggregates.telemetry_position

    assert position(str(attempt.id), kind="evaluation-attempt") == (0, -1)
    queued = repo.transition_evaluation_attempt(
        attempt.id,
        expected_revision=attempt.revision,
        new_status=EvaluationAttemptStatus.QUEUED,
        actor=_ACTOR,
        telemetry_position=(0, 3),
    )
    repo.transition_evaluation_attempt(
        attempt.id,
        expected_revision=queued.revision,
        new_status=EvaluationAttemptStatus.STARTING,
        actor=_ACTOR,
        telemetry_position=(0, 1),
    )

    reopened = ControlPlaneRepository(connect(db_path))
    assert reopened.aggregates.telemetry_position(str(attempt.id), kind="evaluation-attempt") == (
        0,
        3,
    )


def test_a_dead_evaluation_stream_degrades_without_a_new_attempt(
    repo: ControlPlaneRepository, node: Any
) -> None:
    run = _run(node)
    _begin(repo, node, run)
    attempt, _ = _attempt(repo, run)

    assert repo.record_telemetry_degraded(
        attempt.id, reason="supervisor gone", actor=_ACTOR, kind="evaluation-attempt"
    ) == (1, -1)
    assert repo.aggregates.evaluation_attempts_for_run(str(run.id)) == (attempt,)


# ---- a result, and its success, are one fact ----------------------------------------


def test_a_result_and_its_success_commit_together(repo: ControlPlaneRepository, node: Any) -> None:
    run = _run(node)
    _begin(repo, node, run)
    attempt, _ = _attempt(repo, run)
    attempt = _running(repo, attempt)

    result = repo.record_evaluation_result(
        attempt.id,
        _result(run),
        expected_revision=attempt.revision,
        actor=_ACTOR,
        telemetry_position=(0, 9),
    )

    assert repo.aggregates.evaluation_result_for_run(str(run.id)) == result
    assert (
        repo.aggregates.load_evaluation_attempt(str(attempt.id)).status
        is EvaluationAttemptStatus.SUCCEEDED
    )
    assert repo.aggregates.load_evaluation_run(str(run.id)).status is EvaluationRunStatus.SUCCEEDED
    assert repo.aggregates.telemetry_position(str(attempt.id), kind="evaluation-attempt") == (0, 9)


@pytest.mark.parametrize(
    "status_change",
    ["run", "attempt"],
)
def test_nothing_succeeds_without_its_result(
    repo: ControlPlaneRepository, node: Any, status_change: str
) -> None:
    run = _run(node)
    _begin(repo, node, run)
    attempt, _ = _attempt(repo, run)
    attempt = _running(repo, attempt)

    with pytest.raises(StorageError, match="succeeds only with its result"):
        if status_change == "run":
            repo.transition_evaluation_run(
                run.id,
                expected_revision=1,
                new_status=EvaluationRunStatus.SUCCEEDED,
                actor=_ACTOR,
            )
        else:
            repo.transition_evaluation_attempt(
                attempt.id,
                expected_revision=attempt.revision,
                new_status=EvaluationAttemptStatus.SUCCEEDED,
                actor=_ACTOR,
            )


_OTHER_SUBJECT = ArtifactRef(id=ArtifactId.generate(), kind="model", uri="/other")
_SAME_BYTES = ArtifactRef(id=ArtifactId.generate(), kind="model", uri="/m", digest="sha256:m")


@pytest.mark.parametrize(
    "disagreement",
    [
        lambda run: {"subject": _OTHER_SUBJECT},
        lambda run: {"subject": _SAME_BYTES},
        lambda run: {"evaluation_fingerprint": "sha256:" + "f" * 64},
        lambda run: {"metrics": (_metric(run, seed=99),)},
        lambda run: {"metrics": (_metric(run, seed=None),)},
        lambda run: {"metrics": (_metric(run, evaluator_name="some-other-evaluator"),)},
        lambda run: {"metrics": (_metric(run, evaluator_version="9.0"),)},
        lambda run: {
            "metrics": (
                _metric(
                    run, evaluator_name="some-other-evaluator", evaluator_version="9.0", seed=None
                ),
            )
        },
        lambda run: {
            "artifacts": (
                ArtifactRef(
                    id=ArtifactId.generate(),
                    kind="evaluation_report",
                    uri="/r",
                    producer_evaluation_id=EvaluationId.generate(),
                ),
            )
        },
        lambda run: {
            "artifacts": (
                ArtifactRef(
                    id=ArtifactId.generate(),
                    kind="evaluation_report",
                    uri="/r",
                    producer_attempt_id=RunAttemptId.generate(),
                ),
            )
        },
    ],
    ids=[
        "subject",
        "subject-same-digest",
        "fingerprint",
        "seed",
        "seed-omitted",
        "evaluator",
        "evaluator-version",
        "all-three-drifted",
        "report-names-another-result",
        "report-names-a-training-attempt",
    ],
)
def test_a_result_that_disagrees_with_its_run_is_refused(
    repo: ControlPlaneRepository, node: Any, disagreement: Any
) -> None:
    """Every field that says where a result came from must agree with its run."""
    run = _run(node)
    _begin(repo, node, run)
    attempt, _ = _attempt(repo, run)
    attempt = _running(repo, attempt)

    with pytest.raises(ProvenanceError):
        repo.record_evaluation_result(
            attempt.id,
            _result(run, **disagreement(run)),
            expected_revision=attempt.revision,
            actor=_ACTOR,
        )

    assert repo.aggregates.evaluation_result_for_run(str(run.id)) is None
    assert (
        repo.aggregates.load_evaluation_attempt(str(attempt.id)).status
        is EvaluationAttemptStatus.RUNNING
    ), "nothing half-written"


def test_the_database_itself_refuses_misattributed_results(
    repo: ControlPlaneRepository, node: Any, connection: sqlite3.Connection
) -> None:
    """Not only the repository: a writer that skipped its checks is refused too."""
    run = _run(node)
    _begin(repo, node, run)

    with pytest.raises(sqlite3.IntegrityError, match="provenance"):
        with write_transaction(connection):
            repo.aggregates._insert_evaluation_result(
                _result(run, evaluation_fingerprint="sha256:" + "f" * 64)
            )


def test_the_database_refuses_another_artifact_with_the_same_bytes(
    repo: ControlPlaneRepository, node: Any, connection: sqlite3.Connection
) -> None:
    """A shared digest is not a shared identity: the run evaluated artifact A, not B."""
    run = _run(node)
    _begin(repo, node, run)
    assert _SAME_BYTES.digest == run.subject.digest and _SAME_BYTES.id != run.subject.id

    with pytest.raises(sqlite3.IntegrityError, match="provenance"):
        with write_transaction(connection):
            repo.aggregates._insert_evaluation_result(_result(run, subject=_SAME_BYTES))


def test_a_completion_is_held_once_with_its_cursor(repo: ControlPlaneRepository, node: Any) -> None:
    """Held durably at its position; a replay or a second completion does not replace it."""
    from xaytune.core.telemetry import EvaluationCompletedPayload

    run = _run(node)
    _begin(repo, node, run)
    attempt, _ = _attempt(repo, run)
    first = EvaluationCompletedPayload(metrics=(_metric(run),))
    second = EvaluationCompletedPayload(metrics=(_metric(run, value=0.1),))

    repo.hold_evaluation_completion(attempt.id, first, telemetry_position=(0, 3), actor=_ACTOR)
    repo.hold_evaluation_completion(attempt.id, second, telemetry_position=(0, 4), actor=_ACTOR)

    assert repo.aggregates.pending_completion(str(attempt.id)) == (first, (0, 3))
    assert repo.aggregates.telemetry_position(str(attempt.id), kind="evaluation-attempt") == (0, 3)
    repo.record_telemetry_degraded(
        attempt.id, reason="stream lost", actor=_ACTOR, kind="evaluation-attempt"
    )
    assert repo.aggregates.pending_completion(str(attempt.id)) == (first, (0, 3)), (
        "a new generation does not lose it"
    )


def test_a_result_is_never_edited(
    repo: ControlPlaneRepository, node: Any, connection: sqlite3.Connection
) -> None:
    run = _run(node)
    _begin(repo, node, run)
    _succeed(repo, run)

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with write_transaction(connection):
            connection.execute("UPDATE evaluation_results SET payload_json = '{}'")


def test_a_running_attempt_is_needed_to_succeed(repo: ControlPlaneRepository, node: Any) -> None:
    run = _run(node)
    _begin(repo, node, run)
    attempt, _ = _attempt(repo, run)

    with pytest.raises(InvalidTransitionError):
        repo.record_evaluation_result(
            attempt.id, _result(run), expected_revision=attempt.revision, actor=_ACTOR
        )


# ---- the node, reconciled over its current cycle ------------------------------------


def test_a_live_run_keeps_the_node_evaluating(repo: ControlPlaneRepository, node: Any) -> None:
    run = _run(node)
    _begin(repo, node, run)
    _attempt(repo, run)

    assert repo.reconcile_evaluating_node(node.id, actor=_ACTOR) is EvaluationReconciliation.WAITING
    assert repo.aggregates.load_node(str(node.id)).status is ExperimentNodeStatus.EVALUATING


def test_every_run_succeeded_with_its_result_moves_the_node_to_deciding(
    repo: ControlPlaneRepository, node: Any
) -> None:
    """ADR-015 §5 case B: the lag a crash leaves is repaired, not reported."""
    first, second = _run(node), _run(node, seed=8)
    _begin(repo, node, first, second)
    _succeed(repo, first)
    _succeed(repo, second)

    assert (
        repo.reconcile_evaluating_node(node.id, actor=_ACTOR) is EvaluationReconciliation.DECIDING
    )
    assert repo.aggregates.load_node(str(node.id)).status is ExperimentNodeStatus.DECIDING


def _stalls(repo: ControlPlaneRepository, node: Any) -> list:
    return [
        event
        for event in repo.events.events_for_aggregate(str(node.id))
        if event.event_type == "EvaluationStalled"
    ]


def test_a_run_that_ended_without_a_result_stalls_the_cycle_once(
    repo: ControlPlaneRepository, node: Any
) -> None:
    run = _run(node)
    _begin(repo, node, run)
    attempt, _ = _attempt(repo, run)
    repo.transition_evaluation_attempt(
        attempt.id,
        expected_revision=attempt.revision,
        new_status=EvaluationAttemptStatus.FAILED,
        actor=_ACTOR,
    )
    repo.transition_evaluation_run(
        run.id, expected_revision=1, new_status=EvaluationRunStatus.FAILED, actor=_ACTOR
    )

    for _ in range(2):
        assert (
            repo.reconcile_evaluating_node(node.id, actor=_ACTOR)
            is EvaluationReconciliation.STALLED
        )
    (stalled,) = _stalls(repo, node)
    assert list(stalled.payload["runs"]) == [str(run.id)]
    assert repo.aggregates.load_node(str(node.id)).status is ExperimentNodeStatus.EVALUATING


def test_a_cycle_with_no_runs_stalls(repo: ControlPlaneRepository, node: Any) -> None:
    _begin(repo, node)

    assert repo.reconcile_evaluating_node(node.id, actor=_ACTOR) is EvaluationReconciliation.STALLED
    (stalled,) = _stalls(repo, node)
    assert "no evaluation runs" in stalled.payload["reason"]


def test_an_earlier_cycle_cannot_satisfy_the_current_one(
    repo: ControlPlaneRepository, node: Any
) -> None:
    """The first round's results are history, not the second round's answer."""
    first = _run(node)
    _begin(repo, node, first)
    _succeed(repo, first)
    assert (
        repo.reconcile_evaluating_node(node.id, actor=_ACTOR) is EvaluationReconciliation.DECIDING
    )

    deciding = repo.aggregates.load_node(str(node.id))
    active = repo.transition_node(
        node.id,
        expected_revision=deciding.revision,
        new_status=ExperimentNodeStatus.ACTIVE,
        actor=_ACTOR,
    )
    second_round = _begin(repo, active)

    assert second_round.evaluation_cycle == 2
    assert repo.reconcile_evaluating_node(node.id, actor=_ACTOR) is EvaluationReconciliation.STALLED
    assert repo.aggregates.load_node(str(node.id)).status is ExperimentNodeStatus.EVALUATING


# ---- cancellation uses the same path as training --------------------------------------


def test_an_evaluation_attempt_is_cancelled_by_action_and_operation(
    repo: ControlPlaneRepository, node: Any
) -> None:
    """ADR-015 AC-7: a cancel-attempt Action and its operation, in one commit."""
    run = _run(node)
    _begin(repo, node, run)
    attempt, _ = _attempt(repo, run)

    action, operation = repo.request_cancellation(
        ActionTarget(kind="evaluation-attempt", id=str(attempt.id)),
        reason="no longer needed",
        actor=_ACTOR,
        request_digest="sha256:cancel",
    )

    assert action.type == "cancel-attempt"
    assert operation is not None and operation.target.kind == "evaluation-attempt"


def test_a_cancellation_names_an_evaluation_attempt_that_exists(
    repo: ControlPlaneRepository,
) -> None:
    with pytest.raises(UnknownOperationTargetError):
        repo.request_cancellation(
            ActionTarget(kind="evaluation-attempt", id=str(EvaluationAttemptId.generate())),
            reason="x",
            actor=_ACTOR,
            request_digest="sha256:cancel",
        )


def test_cancelling_the_experiment_waits_for_its_evaluations(
    repo: ControlPlaneRepository, node: Any
) -> None:
    """CANCELLED means no owned workload runs -- and an evaluation is one."""
    run = _run(node)
    _begin(repo, node, run)
    attempt, _ = _attempt(repo, run)

    parent, children = repo.request_experiment_cancellation(
        node.experiment_id, reason="stop", actor=_ACTOR
    )
    ((child, operation),) = children
    assert child.target.kind == "evaluation-attempt"
    assert operation is not None
    assert any(
        op.target.kind == "evaluation-attempt"
        for op in repo.unsettled_work(str(node.experiment_id))[0]
    )

    repo.reconcile_experiment_cancellation(parent.id, actor=_ACTOR)
    assert repo.aggregates.load_experiment(str(node.experiment_id)).status is (
        ExperimentStatus.ACTIVE
    ), "the evaluation is still live"

    repo.confirm_operation(operation.id, expected_revision=operation.revision, actor=_ACTOR)
    repo.transition_evaluation_attempt(
        attempt.id,
        expected_revision=attempt.revision,
        new_status=EvaluationAttemptStatus.CANCELLED,
        actor=_ACTOR,
    )
    repo.reconcile_experiment_cancellation(parent.id, actor=_ACTOR)

    assert (
        repo.aggregates.load_experiment(str(node.experiment_id)).status
        is ExperimentStatus.CANCELLED
    )
    assert repo.aggregates.load_evaluation_run(str(run.id)).status is EvaluationRunStatus.CANCELLED
