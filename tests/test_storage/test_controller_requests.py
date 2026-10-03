"""The daemon mailbox and atomic admission, at the repository (ADR-004 §3-§4; PR-027).

```text
record_controller_request    get-or-create by the client's id; another request
                             under the same id is a conflict
admit_experiment             experiment, root node, first run, first attempt,
                             reservations, INTENDED submit -- and the request's
                             ACCEPTED -- in one commit, or nothing
migration 016                what a request asks is immutable; state moves
                             forward along its kind's edges only
```
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.test_storage.conftest import make_attempt, make_experiment, make_node, make_run
from xaytune.core.domain.budget import BudgetDimension
from xaytune.core.domain.controller_request import ControllerRequest, ControllerRequestState
from xaytune.core.domain.objective import BudgetSpec
from xaytune.core.errors import DomainError, InvalidTransitionError
from xaytune.core.ids import ExperimentId
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import Actor
from xaytune.core.state.status import (
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
)
from xaytune.storage import ControlPlaneRepository, IdempotencyConflictError, StorageError
from xaytune.storage.control_plane import AdmissionRefusedError

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
