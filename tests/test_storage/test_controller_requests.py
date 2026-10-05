"""The daemon mailbox and atomic admission, at the repository (ADR-004 §3-§4; PR-027).

```text
record_controller_request    get-or-create by the client's id; another request
                             under the same id is a conflict
admit_experiment             experiment, root node, first run, first attempt,
                             reservations, INTENDED submit -- and the request's
                             ACCEPTED -- in one commit, or nothing
migration 016                what a request asks is immutable; state moves
                             forward along its kind's edges only
migration 018 (PR-029)       cancel, propose-action, approve-action and
                             reject-action, each PENDING → COMPLETED | FAILED,
                             with exactly its payload; the rebuilt table keeps
                             every earlier row and rule; controller rests
```
"""

from __future__ import annotations

import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from tests.test_storage.conftest import make_attempt, make_experiment, make_node, make_run
from xaytune.core.domain.action import ActionTarget
from xaytune.core.domain.actions import RejectCandidate, action_from_spec
from xaytune.core.domain.budget import BudgetDimension
from xaytune.core.domain.controller_request import ControllerRequest, ControllerRequestState
from xaytune.core.domain.experiment import Experiment
from xaytune.core.domain.objective import BudgetSpec
from xaytune.core.errors import DomainError, InvalidTransitionError
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import ExperimentId
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import Actor, ControllerHostRef
from xaytune.core.sqlite import connect
from xaytune.core.state.status import (
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
)
from xaytune.storage import (
    ControllerLeaseHeldError,
    ControllerLeaseStore,
    ControlPlaneRepository,
    IdempotencyConflictError,
    NoLiveLeaseFence,
    StorageError,
)
from xaytune.storage.control_plane import AdmissionRefusedError
from xaytune.storage.migrations import MIGRATIONS_DIR, migrate

_ACTOR = Actor(type="system", id="test")
_SPEC = FrozenDict({"name": "a spec", "seed": 7})
# A move the schema allows, so only the change beside it is what is refused.
_LEGAL_MOVE = ", state = 'failed', error_json = '{}', revision = revision + 1"


@pytest.fixture
def repository(connection: sqlite3.Connection) -> ControlPlaneRepository:
    return ControlPlaneRepository(connection)


def _submission(experiment_id: ExperimentId | None = None, *, budget: BudgetSpec | None = None):
    experiment = make_experiment()
    if experiment_id is not None or budget is not None:
        fields = experiment.model_dump()
        fields["id"] = experiment_id or experiment.id
        fields["budget"] = budget
        experiment = type(experiment)(**fields)
    node = make_node(experiment)
    run = make_run(node)
    return experiment, node, run, make_attempt(run)


def _counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "experiments",
            "experiment_nodes",
            "runs",
            "run_attempts",
            "runtime_operations",
            "budget_ledger",
        )
    }


# ---- the request -------------------------------------------------------------------


def test_a_request_is_recorded_once_by_its_id(repository: ControlPlaneRepository) -> None:
    request = ControllerRequest.submit(_SPEC)
    assert repository.record_controller_request(request) == request
    assert repository.record_controller_request(request) == request
    assert repository.controller_requests.unfinished() == (request,)


@pytest.mark.parametrize("change", ["payload", "experiment", "kind"])
def test_another_request_under_the_same_id_is_a_conflict(
    repository: ControlPlaneRepository, change: str
) -> None:
    request = ControllerRequest.submit(_SPEC)
    repository.record_controller_request(request)
    if change == "payload":
        other = ControllerRequest.submit(
            FrozenDict({"name": "another"}),
            experiment_id=request.experiment_id,
            request_id=request.id,
        )
    elif change == "experiment":
        other = ControllerRequest.submit(_SPEC, request_id=request.id)
    else:
        other = ControllerRequest.attach(request.experiment_id, request_id=request.id)
    with pytest.raises(IdempotencyConflictError):
        repository.record_controller_request(other)
    assert repository.controller_requests.get(str(request.id)) == request


def test_a_request_digest_must_describe_its_payload() -> None:
    request = ControllerRequest.submit(_SPEC)
    fields = request.model_dump()
    fields["payload"] = {"name": "tampered"}
    with pytest.raises(DomainError, match="payload_digest"):
        ControllerRequest(**fields)


def test_only_a_pending_submit_fails_and_only_an_accepted_one_completes(
    repository: ControlPlaneRepository,
) -> None:
    request = repository.record_controller_request(ControllerRequest.submit(_SPEC))
    with pytest.raises(InvalidTransitionError):
        repository.complete_controller_request(request.id, expected_revision=0)
    failed = repository.fail_controller_request(
        request.id, expected_revision=0, error=FrozenDict({"type": "X", "message": "no"})
    )
    assert failed.state is ControllerRequestState.FAILED
    assert repository.controller_requests.unfinished() == ()


def test_an_attach_completes_from_pending(repository: ControlPlaneRepository) -> None:
    request = repository.record_controller_request(
        ControllerRequest.attach(ExperimentId.generate())
    )
    done = repository.complete_controller_request(request.id, expected_revision=0)
    assert done.state is ControllerRequestState.COMPLETED


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        (
            "UPDATE controller_requests SET payload_json = '{}'" + _LEGAL_MOVE,
            "immutable",
        ),
        (
            "UPDATE controller_requests SET experiment_id = 'exp_other'" + _LEGAL_MOVE,
            "immutable",
        ),
        (
            "UPDATE controller_requests SET state = 'completed', revision = revision + 1",
            "forward",
        ),
        ("UPDATE controller_requests SET state = 'accepted', revision = revision + 1", "ACCEPTED"),
        ("DELETE FROM controller_requests", "permanent"),
    ],
)
def test_the_schema_holds_the_request_contract(
    repository: ControlPlaneRepository,
    connection: sqlite3.Connection,
    statement: str,
    message: str,
) -> None:
    repository.record_controller_request(ControllerRequest.submit(_SPEC))
    with pytest.raises(sqlite3.IntegrityError, match=message):
        connection.execute(statement)


def test_the_schema_records_requests_pending(connection: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="PENDING"):
        connection.execute(
            "INSERT INTO controller_requests VALUES "
            "('creq_x', 'attach', 'completed', 0, 'exp_x', '{}', 'd', NULL, 't', 't')"
        )


# ---- admission ------------------------------------------------------------------------


def test_admission_records_the_whole_submission_and_accepts_its_request(
    repository: ControlPlaneRepository, connection: sqlite3.Connection
) -> None:
    request = repository.record_controller_request(ControllerRequest.submit(_SPEC))
    experiment, node, run, attempt = _submission(request.experiment_id)

    admission = repository.admit_experiment(
        experiment,
        node,
        run,
        attempt,
        request_digest="sha256:r",
        actor=_ACTOR,
        request_id=request.id,
        submitted_digest=request.payload_digest,
    )

    assert admission is not None and admission.operation is not None
    assert admission.experiment.status is ExperimentStatus.ACTIVE
    assert admission.node.status is ExperimentNodeStatus.ACTIVE
    assert admission.run is not None and admission.run.status is RunStatus.ACTIVE
    assert admission.attempt is not None and admission.attempt.status is RunAttemptStatus.CREATED
    assert admission.operation.state == "intended"
    assert admission.operation.target.id == str(attempt.id)
    accepted = repository.controller_requests.get(str(request.id))
    assert accepted is not None and accepted.state is ControllerRequestState.ACCEPTED
    assert _counts(connection) == {
        "experiments": 1,
        "experiment_nodes": 1,
        "runs": 1,
        "run_attempts": 1,
        "runtime_operations": 1,
        "budget_ledger": 0,
    }


def test_a_request_no_longer_pending_admits_nothing(
    repository: ControlPlaneRepository, connection: sqlite3.Connection
) -> None:
    request = repository.record_controller_request(ControllerRequest.submit(_SPEC))
    first = _submission(request.experiment_id)
    repository.admit_experiment(
        *first,
        request_digest="sha256:r",
        actor=_ACTOR,
        request_id=request.id,
        submitted_digest=request.payload_digest,
    )
    before = _counts(connection)

    again = repository.admit_experiment(
        *_submission(request.experiment_id),
        request_digest="sha256:r",
        actor=_ACTOR,
        request_id=request.id,
        submitted_digest=request.payload_digest,
    )

    assert again is None
    assert _counts(connection) == before


@pytest.mark.parametrize(
    "problem", ["another-experiment", "attach", "missing", "taken", "another-spec"]
)
def test_a_refused_admission_writes_nothing_and_leaves_the_request_pending(
    repository: ControlPlaneRepository, connection: sqlite3.Connection, problem: str
) -> None:
    from xaytune.core.ids import ControllerRequestId
    from xaytune.storage import AggregateNotFoundError

    request = repository.record_controller_request(ControllerRequest.submit(_SPEC))
    request_id = request.id
    submission = _submission(request.experiment_id)
    expected: type[Exception] = AdmissionRefusedError
    digest = request.payload_digest
    if problem == "another-spec":
        # The daemon processing request A hands over a submission derived from
        # spec B: the request must not become ACCEPTED for it.
        digest = ControllerRequest.submit(FrozenDict({"name": "spec B"})).payload_digest
    elif problem == "another-experiment":
        submission = _submission()
    elif problem == "attach":
        attach = repository.record_controller_request(
            ControllerRequest.attach(request.experiment_id)
        )
        request_id = attach.id
    elif problem == "missing":
        request_id = ControllerRequestId.generate()
        expected = AggregateNotFoundError
    else:
        repository.create_experiment(submission[0], actor=_ACTOR)
    before = _counts(connection)

    with pytest.raises(expected):
        repository.admit_experiment(
            *submission,
            request_digest="sha256:r",
            actor=_ACTOR,
            request_id=request_id,
            submitted_digest=digest,
        )

    assert _counts(connection) == before
    stored = repository.controller_requests.get(str(request.id))
    assert stored is not None and stored.state is ControllerRequestState.PENDING


def test_a_failure_inside_the_admission_rolls_all_of_it_back(
    repository: ControlPlaneRepository,
    connection: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The request's ACCEPTED is the last write: a failure before it undoes everything."""
    request = repository.record_controller_request(ControllerRequest.submit(_SPEC))

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("the process dies mid-transaction")

    monkeypatch.setattr(repository, "_insert_training_attempt_with_intent", boom)
    with pytest.raises(RuntimeError):
        repository.admit_experiment(
            *_submission(request.experiment_id),
            request_digest="sha256:r",
            actor=_ACTOR,
            request_id=request.id,
            submitted_digest=request.payload_digest,
        )

    assert set(_counts(connection).values()) == {0}
    stored = repository.controller_requests.get(str(request.id))
    assert stored is not None and stored.state is ControllerRequestState.PENDING


def test_no_run_left_admits_the_experiment_budget_exhausted(
    repository: ControlPlaneRepository, connection: sqlite3.Connection
) -> None:
    request = repository.record_controller_request(ControllerRequest.submit(_SPEC))
    submission = _submission(request.experiment_id, budget=BudgetSpec(max_runs=0))

    admission = repository.admit_experiment(
        *submission,
        request_digest="sha256:r",
        actor=_ACTOR,
        request_id=request.id,
        submitted_digest=request.payload_digest,
    )

    assert admission is not None
    assert admission.experiment.status is ExperimentStatus.BUDGET_EXHAUSTED
    assert admission.run is admission.attempt is admission.operation is None
    assert admission.exhausted and "runs" in admission.exhausted[0]
    counts = _counts(connection)
    assert (counts["runs"], counts["run_attempts"], counts["runtime_operations"]) == (0, 0, 0)
    stored = repository.controller_requests.get(str(request.id))
    assert stored is not None and stored.state is ControllerRequestState.ACCEPTED
    status = repository.budget_status(admission.experiment.id)
    assert status is not None
    runs = status.of(BudgetDimension.RUNS)
    assert runs is not None and runs.consumed == 0


def test_admission_refuses_anything_but_one_fresh_submission(
    repository: ControlPlaneRepository, connection: sqlite3.Connection, tmp_path: Path
) -> None:
    experiment, node, run, attempt = _submission()
    other = make_experiment()
    with pytest.raises(StorageError, match="not the experiment's"):
        repository.admit_experiment(
            experiment,
            make_node(other),
            run,
            attempt,
            request_digest="sha256:r",
            actor=_ACTOR,
        )
    with pytest.raises(StorageError, match="first"):
        repository.admit_experiment(
            experiment,
            node,
            run,
            make_attempt(run, attempt_number=2),
            request_digest="sha256:r",
            actor=_ACTOR,
        )
    assert set(_counts(connection).values()) == {0}


def test_a_request_cannot_be_admitted_without_the_submitted_specs_digest(
    repository: ControlPlaneRepository, connection: sqlite3.Connection
) -> None:
    request = repository.record_controller_request(ControllerRequest.submit(_SPEC))
    with pytest.raises(ValueError, match="digest"):
        repository.admit_experiment(
            *_submission(request.experiment_id),
            request_digest="sha256:r",
            actor=_ACTOR,
            request_id=request.id,
        )
    assert set(_counts(connection).values()) == {0}


# ---- what a daemon is responsible for (PR-028) ------------------------------------------


def _experiment_at(
    repository: ControlPlaneRepository, host: str, *statuses: ExperimentStatus
) -> ExperimentId:
    experiment = Experiment.model_validate(
        {
            **make_experiment().model_dump(mode="python"),
            "controller_host": ControllerHostRef(kind=host, id=f"{host}-1"),  # type: ignore[arg-type]
        }
    )
    repository.create_experiment(experiment, actor=_ACTOR)
    revision = 0
    for status in statuses:
        repository.transition_experiment(
            experiment.id, expected_revision=revision, new_status=status, actor=_ACTOR
        )
        revision += 1
    return experiment.id


def _attached(repository: ControlPlaneRepository, experiment_id: ExperimentId, *, done: bool):
    request = repository.record_controller_request(ControllerRequest.attach(experiment_id))
    if done:
        repository.complete_controller_request(request.id, expected_revision=0)
    return request


def test_a_daemon_is_responsible_for_what_it_admitted_or_adopted_and_has_not_ended(
    repository: ControlPlaneRepository,
) -> None:
    ACTIVE, PAUSED = ExperimentStatus.ACTIVE, ExperimentStatus.PAUSED  # noqa: N806
    admitted = _experiment_at(repository, "local_daemon", ACTIVE)
    paused = _experiment_at(repository, "local_daemon", ACTIVE, PAUSED)
    created = _experiment_at(repository, "local_daemon")
    ended = _experiment_at(repository, "local_daemon", ACTIVE, ExperimentStatus.CANCELLED)
    embedded = _experiment_at(repository, "embedded", ACTIVE)
    adopted = _experiment_at(repository, "embedded", ACTIVE)
    asked = _experiment_at(repository, "embedded", ACTIVE)
    both = _experiment_at(repository, "local_daemon", ACTIVE)
    _attached(repository, adopted, done=True)
    _attached(repository, asked, done=False)
    _attached(repository, both, done=True)
    _attached(repository, both, done=True)
    ended_adopted = _experiment_at(repository, "embedded", ACTIVE, ExperimentStatus.FAILED)
    _attached(repository, ended_adopted, done=True)

    owned = repository.daemon_responsibilities()

    assert owned == (admitted, paused, created, adopted, both), "oldest first, each once"
    assert embedded not in owned, "an embedded experiment nobody attached stays its host's"
    assert asked not in owned, "an attach not yet completed is not adoption"
    assert ended not in owned and ended_adopted not in owned


def test_completing_a_request_of_any_kind_but_submit_is_adoption(
    repository: ControlPlaneRepository,
) -> None:
    """A recorded cancellation, proposal or approval is attached before it is carried on."""
    cancelled = _experiment_at(repository, "embedded", ExperimentStatus.ACTIVE)
    asked = _experiment_at(repository, "embedded", ExperimentStatus.ACTIVE)
    refused = _experiment_at(repository, "embedded", ExperimentStatus.ACTIVE)
    for experiment_id, outcome in ((cancelled, "complete"), (asked, None), (refused, "fail")):
        request = repository.record_controller_request(
            ControllerRequest.cancel(experiment_id, reason="stop")
        )
        if outcome == "complete":
            repository.complete_controller_request(request.id, expected_revision=0)
        elif outcome == "fail":
            repository.fail_controller_request(
                request.id, expected_revision=0, error=FrozenDict({"type": "X", "message": "no"})
            )

    assert repository.daemon_responsibilities() == (cancelled,)


# ---- the mutation kinds (PR-029) -------------------------------------------------------


_HUMAN = Actor(type="human", id="ana")


def _action_kinds(experiment_id: ExperimentId) -> list[ControllerRequest]:
    action = action_from_spec(
        RejectCandidate(target=ActionTarget(kind="node", id="node_1")),
        experiment_id=experiment_id,
        proposed_by=_HUMAN,
        reason="off target",
    )
    return [
        ControllerRequest.cancel(experiment_id, reason="stop"),
        ControllerRequest.propose(action),
        ControllerRequest.resolve(
            "approve-action", action.id, experiment_id, approver=_HUMAN, reason="agreed"
        ),
        ControllerRequest.resolve(
            "reject-action", action.id, experiment_id, approver=_HUMAN, reason="no"
        ),
    ]


@pytest.mark.parametrize("index", range(4), ids=["cancel", "propose", "approve", "reject"])
def test_a_mutation_request_completes_or_fails_from_pending_and_is_never_accepted(
    repository: ControlPlaneRepository, connection: sqlite3.Connection, index: int
) -> None:
    experiment_id = _experiment_at(repository, "local_daemon", ExperimentStatus.ACTIVE)
    completed, failed = (
        repository.record_controller_request(_action_kinds(experiment_id)[index]) for _ in range(2)
    )
    assert completed.action_id is not None
    assert repository.controller_requests.get(str(completed.id)) == completed, "round-trips"
    with pytest.raises(InvalidTransitionError):
        completed.with_state(ControllerRequestState.ACCEPTED)
    with pytest.raises(sqlite3.IntegrityError, match="forward"):
        connection.execute(
            "UPDATE controller_requests SET state = 'accepted', revision = 1 WHERE id = ?",
            (str(completed.id),),
        )
    done = repository.complete_controller_request(completed.id, expected_revision=0)
    assert done.state is ControllerRequestState.COMPLETED
    refused = repository.fail_controller_request(
        failed.id, expected_revision=0, error=FrozenDict({"type": "X", "message": "no"})
    )
    assert refused.state is ControllerRequestState.FAILED


def test_a_mutation_request_carries_exactly_its_payload() -> None:
    request = ControllerRequest.cancel(ExperimentId.generate(), reason="stop")
    assert request.payload["reason"] == "stop"
    assert request.action_id is not None, "minted by the client, before it is sent"
    fields = request.model_dump()
    fields["payload"] = {"reason": "stop"}
    fields["payload_digest"] = fingerprint(FrozenDict(fields["payload"]))
    with pytest.raises(DomainError, match="action_id, reason"):
        ControllerRequest(**fields)
    assert ControllerRequest.attach(ExperimentId.generate()).action_id is None


def test_the_rebuilt_mailbox_keeps_every_earlier_request_and_rule(tmp_path: Path) -> None:
    """Migration 018 rebuilds controller_requests to admit the new kinds, losing nothing."""
    earlier = tmp_path / "migrations"
    earlier.mkdir()
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if int(path.name[:3]) <= 17:
            (earlier / path.name).write_text(path.read_text())
    connection = connect(tmp_path / "state.db")
    migrate(connection, earlier)
    old = ControlPlaneRepository(connection)
    submitted = old.record_controller_request(ControllerRequest.submit(_SPEC))
    attached = old.record_controller_request(ControllerRequest.attach(ExperimentId.generate()))
    old.complete_controller_request(attached.id, expected_revision=0)
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        connection.execute(
            "INSERT INTO controller_requests VALUES "
            "('creq_y', 'cancel', 'pending', 0, 'exp_x', '{}', 'd', NULL, 't', 't')"
        )

    assert migrate(connection)[0] == 18  # and any later migrations
    repository = ControlPlaneRepository(connection)
    assert repository.controller_requests.get(str(submitted.id)) == submitted
    assert repository.controller_requests.get(str(attached.id)).state is (
        ControllerRequestState.COMPLETED
    )
    assert repository.controller_requests.unfinished() == (submitted,)
    cancel = repository.record_controller_request(
        ControllerRequest.cancel(ExperimentId.generate(), reason="stop")
    )
    assert repository.controller_requests.get(str(cancel.id)) == cancel
    for statement, message in (
        ("UPDATE controller_requests SET payload_json = '{}'" + _LEGAL_MOVE, "immutable"),
        ("UPDATE controller_requests SET state = 'completed', revision = 7", "forward"),
        ("DELETE FROM controller_requests", "permanent"),
    ):
        with pytest.raises(sqlite3.IntegrityError, match=message):
            connection.execute(statement)
    with pytest.raises(sqlite3.IntegrityError, match="PENDING"):
        connection.execute(
            "INSERT INTO controller_requests VALUES "
            "('creq_z', 'cancel', 'completed', 0, 'exp_x', '{}', 'd', NULL, 't', 't')"
        )
    indexes = {
        row["name"]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE tbl_name = 'controller_requests' "
            "AND type = 'index' AND name LIKE 'idx_%'"
        )
    }
    assert indexes == {"idx_controller_requests_unfinished", "idx_controller_requests_experiment"}
    connection.close()


# ---- controller rests (PR-029) -----------------------------------------------------------


def test_a_rest_is_stamped_with_the_experiments_latest_event(
    repository: ControlPlaneRepository,
) -> None:
    experiment_id = _experiment_at(repository, "local_daemon", ExperimentStatus.ACTIVE)
    latest = repository.events.latest_sequence_for_experiment(str(experiment_id))
    assert latest > 0

    rest = repository.record_controller_rest(experiment_id, controller_id="daemon-1")
    assert (rest.sequence, rest.escalation) == (latest, None)
    assert repository.controller_requests.rest(str(experiment_id)) == rest
    assert repository.record_controller_rest(experiment_id, controller_id="daemon-1") == rest, (
        "the same rest again rewrites nothing"
    )

    repository.transition_experiment(
        experiment_id, expected_revision=1, new_status=ExperimentStatus.PAUSED, actor=_ACTOR
    )
    assert repository.events.latest_sequence_for_experiment(str(experiment_id)) > rest.sequence
    escalated = repository.record_controller_rest(
        experiment_id,
        controller_id="daemon-1",
        escalation={"type": "ReconciliationEscalatedError", "message": "unknown outcome"},
    )
    assert escalated.sequence > rest.sequence
    assert repository.controller_requests.rest(str(experiment_id)) == escalated


def test_a_rest_is_a_fenced_controller_write(
    repository: ControlPlaneRepository, connection: sqlite3.Connection
) -> None:
    experiment_id = _experiment_at(repository, "local_daemon", ExperimentStatus.ACTIVE)
    ControllerLeaseStore(connection).acquire("daemon-1", timedelta(seconds=30))
    embedded = ControlPlaneRepository(connection, fence=NoLiveLeaseFence())
    with pytest.raises(ControllerLeaseHeldError):
        embedded.record_controller_rest(experiment_id, controller_id="embedded")
    assert repository.controller_requests.rest(str(experiment_id)) is None
