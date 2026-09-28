"""Recovery decisions survive crashes and contention without executing effects."""

from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import sys

import pytest
from pydantic import ValidationError

from tests.test_checkpoints.helpers import make_bundle
from tests.test_resilience.test_incidents import classify, envelope
from xaytune.checkpoints import CheckpointManager, LocalCheckpointStore, SerializedStateCodec
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.domain.recovery import (
    Recoverability,
    RecoveryLimits,
    RecoveryPlan,
    RecoveryRequest,
    RecoveryStrategy,
    decide_recovery,
    execution_state_fingerprint_v1,
    incident_signature_v1,
)
from xaytune.core.ids import CheckpointId, RecoveryPlanId
from xaytune.core.refs import Actor
from xaytune.core.telemetry import IncidentObservedPayload, NumericalInstabilityObserved
from xaytune.resilience.recovery import RecoveryCoordinator
from xaytune.storage import ControlPlaneRepository, connect, write_transaction
from xaytune.storage.control_plane import ProvenanceError, StaleRecoveryContextError
from xaytune.storage.errors import AggregateNotFoundError
from xaytune.storage.journal import IdempotencyConflictError

from .conftest import make_attempt, make_run
from .test_checkpoints import record, setup_bundle

ACTOR = Actor(type="system", id="recovery-test")


def incident(repo, attempt, sequence=9, signal=None):
    owner = repo.incident_context(
        RuntimeOperationTarget(kind="training-attempt", id=str(attempt.id))
    )
    event = envelope(owner, signal or IncidentObservedPayload(reason="process-failure"))
    event = event.model_copy(update={"sequence": sequence})
    return repo.record_incident(classify(event, owner), actor=ACTOR)


def prepare(connection, seeded, tmp_path):
    repo, attempt, event = setup_bundle(connection, seeded, tmp_path)
    report = record(repo, attempt, event)
    store = LocalCheckpointStore(tmp_path / "bundles")
    localized = asyncio.run(store.get(report.payload.checkpoint_ref))
    from xaytune.core.checkpoint import RestoreContext

    request = RecoveryRequest(
        restore_context=RestoreContext(
            candidate_fingerprint=report.candidate_fingerprint,
            compatibility=localized.manifest.context.compatibility,
            dataset_fingerprint=localized.manifest.data_cursor.dataset_fingerprint,
            ordering_fingerprint=localized.manifest.data_cursor.ordering_fingerprint,
            required_guarantee=localized.manifest.resume_guarantee,
        )
    )
    coordinator = RecoveryCoordinator(repo, CheckpointManager(SerializedStateCodec(), store))
    return repo, attempt, report, request, coordinator


def draft(repo, observed, request):
    snapshot = repo.recovery_snapshot(str(observed.id))
    decision = decide_recovery(snapshot, request, ())
    return RecoveryPlan(
        **decision.model_dump(),
        incident_id=observed.id,
        context=observed.context,
        incident_signature=incident_signature_v1(observed, snapshot["candidate_fingerprint"]),
        execution_state_fingerprint=execution_state_fingerprint_v1(snapshot["attempt"]),
        input_snapshot=snapshot,
        request=request,
    )


def test_resume_is_deterministic_and_durable_without_decoding_or_effects(
    connection, seeded, tmp_path, db_path, monkeypatch
):
    repo, attempt, report, request, coordinator = prepare(connection, seeded, tmp_path)
    observed = incident(repo, attempt)

    async def forbidden_decode(*args):
        pytest.fail("planning must not decode trainer state")

    monkeypatch.setattr(coordinator.checkpoint_manager.codec, "decode", forbidden_decode)
    coordinator.destinations = ("audit",)
    before_attempt = repo.aggregates.load_attempt(str(attempt.id))
    before_cursor = repo.aggregates.telemetry_position(str(attempt.id))
    plan = asyncio.run(coordinator.plan(str(observed.id), request))
    assert plan.strategy is RecoveryStrategy.RESUME
    assert plan.recoverability is Recoverability.RECOVERABLE_NEW_ATTEMPT
    assert plan.checkpoint_ref == report.payload.checkpoint_ref
    assert RecoveryPlan.model_validate_json(plan.model_dump_json()) == plan
    code = """
import sys
from xaytune.core.domain.recovery import RecoveryPlan, decide_recovery
plan = RecoveryPlan.model_validate_json(sys.argv[1])
decision = decide_recovery(plan.input_snapshot, plan.request, plan.checkpoint_eligibility)
print(decision.model_dump_json())
print(plan.input_fingerprint)
"""
    output = subprocess.check_output(
        [sys.executable, "-c", code, plan.model_dump_json()], text=True
    ).splitlines()
    assert output == [
        plan.model_dump_json(
            include={"strategy", "recoverability", "checkpoint_ref", "reason", "requires_approval"}
        ),
        plan.input_fingerprint,
    ]
    reopened = connect(db_path)
    try:
        # Replay needs neither a codec nor consumer config, even if files vanish.
        restarted = RecoveryCoordinator(ControlPlaneRepository(reopened))
        assert asyncio.run(restarted.plan(str(observed.id))) == plan
        assert asyncio.run(restarted.reconcile(str(observed.context.experiment_id))) == (plan,)
    finally:
        reopened.close()
    assert repo.aggregates.load_attempt(str(attempt.id)) == before_attempt
    assert repo.aggregates.telemetry_position(str(attempt.id)) == before_cursor
    assert connection.execute("SELECT COUNT(*) FROM recovery_plans").fetchone()[0] == 1
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM events WHERE event_type = 'RecoveryPlanned'"
        ).fetchone()[0]
        == 1
    )
    assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 1
    for table, count in (
        ("actions", 0),
        ("runtime_operations", 0),
        ("run_attempts", 1),
        ("experiment_nodes", 1),
        ("budget_ledger", 0),
    ):
        assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == count


@pytest.mark.parametrize(
    "signal",
    [
        IncidentObservedPayload(reason="cuda-oom"),
        NumericalInstabilityObserved(quantity="loss", observation="nan", optimizer_step=100),
        IncidentObservedPayload(reason="checkpoint-corruption"),
        IncidentObservedPayload(reason="unknown-failure"),
    ],
)
def test_specialised_and_unknown_failures_pause_without_algorithms(
    connection, seeded, tmp_path, signal
):
    repo, attempt, _, request, coordinator = prepare(connection, seeded, tmp_path)
    observed = incident(repo, attempt, signal=signal)
    plan = asyncio.run(coordinator.plan(str(observed.id), request))
    assert plan.strategy is RecoveryStrategy.PAUSE_FOR_APPROVAL
    assert plan.requires_approval
    assert plan.checkpoint_ref is None


@pytest.mark.parametrize("allow", [False, True])
def test_no_checkpoint_requires_explicit_restart_policy(connection, seeded, allow):
    repo = ControlPlaneRepository(connection)
    observed = incident(repo, seeded["attempt"])
    plan = asyncio.run(
        RecoveryCoordinator(repo).plan(
            str(observed.id), RecoveryRequest(allow_retry_without_checkpoint=allow)
        )
    )
    assert plan.strategy is (
        RecoveryStrategy.RETRY if allow else RecoveryStrategy.PAUSE_FOR_APPROVAL
    )


@pytest.mark.parametrize("failure", ["corrupt", "missing", "incompatible", "no-context", "future"])
def test_bad_checkpoints_are_ineligible(connection, seeded, tmp_path, failure):
    repo, attempt, report, request, coordinator = prepare(connection, seeded, tmp_path)
    if failure == "corrupt":
        (
            tmp_path
            / "bundles"
            / "committed"
            / str(report.payload.checkpoint_ref.id)
            / "model.json"
        ).write_text("tampered")
    elif failure == "missing":
        (
            tmp_path
            / "bundles"
            / "committed"
            / str(report.payload.checkpoint_ref.id)
            / "model.json"
        ).unlink()
    elif failure == "incompatible":
        request = request.model_copy(
            update={
                "restore_context": request.restore_context.model_copy(
                    update={"ordering_fingerprint": "another-order"}
                )
            }
        )
    elif failure == "no-context":
        request = RecoveryRequest()
    observed = incident(repo, attempt, sequence=2 if failure == "future" else 9)
    plan = asyncio.run(coordinator.plan(str(observed.id), request))
    assert plan.strategy is RecoveryStrategy.PAUSE_FOR_APPROVAL
    assert not plan.checkpoint_eligibility[0].eligible


def add_checkpoint(repo, attempt, manager, tmp_path, source, step, sequence, corrupt=False):
    state, context, _ = make_bundle(
        tmp_path / source,
        attempt_id=attempt.id,
        candidate=repo.aggregates.load_run(str(attempt.run_id)).candidate_fingerprint,
    )
    from dataclasses import replace

    from xaytune.runtimes import RuntimeEventEnvelope, TrainingEventPayload

    ref = asyncio.run(manager.save(replace(state, optimizer_step=step), context))
    manifest = asyncio.run(manager.store.get(ref)).manifest
    event = RuntimeEventEnvelope(
        event_id=source,
        sequence=sequence,
        target=RuntimeOperationTarget(kind="training-attempt", id=str(attempt.id)),
        payload=TrainingEventPayload(data=manifest.committed_payload(ref)),
    )
    result = record(repo, attempt, event)
    if corrupt:
        (tmp_path / "bundles" / "committed" / str(ref.id) / "model.json").write_text("bad")
    return result


def test_selection_falls_back_from_corrupt_newest_and_ignores_other_runs(
    connection, seeded, tmp_path
):
    repo, attempt, original, request, coordinator = prepare(connection, seeded, tmp_path)
    newer = add_checkpoint(
        repo, attempt, coordinator.checkpoint_manager, tmp_path, "newer", 101, 4, corrupt=True
    )
    other_run = make_run(seeded["node"])
    other = make_attempt(other_run)
    other = type(other).model_validate(
        {**other.model_dump(mode="json"), "execution_fingerprint": "execution-a"}
    )
    with write_transaction(connection):
        repo.aggregates._insert_run(other_run)
        repo.aggregates._insert_attempt(other)
    add_checkpoint(repo, other, coordinator.checkpoint_manager, tmp_path, "other", 999, 4)
    observed = incident(repo, attempt)
    plan = asyncio.run(coordinator.plan(str(observed.id), request))
    assert plan.checkpoint_ref == original.payload.checkpoint_ref
    assert [r.checkpoint_ref for r in plan.checkpoint_eligibility] == [
        newer.payload.checkpoint_ref,
        original.payload.checkpoint_ref,
    ]


@pytest.mark.parametrize(
    "limits",
    [
        RecoveryLimits(max_attempts_per_run=1),
        RecoveryLimits(max_recoveries_per_experiment=0),
    ],
)
def test_limits_refuse_recovery(connection, seeded, limits):
    repo = ControlPlaneRepository(connection)
    observed = incident(repo, seeded["attempt"])
    plan = asyncio.run(
        RecoveryCoordinator(repo).plan(
            str(observed.id), RecoveryRequest(limits=limits, allow_retry_without_checkpoint=True)
        )
    )
    assert plan.strategy is RecoveryStrategy.FAIL


def test_same_incident_and_execution_cannot_loop(connection, seeded):
    repo = ControlPlaneRepository(connection)
    coordinator = RecoveryCoordinator(repo)
    request = RecoveryRequest(allow_retry_without_checkpoint=True)
    first = incident(repo, seeded["attempt"])
    initial = asyncio.run(coordinator.plan(str(first.id), request))
    second = incident(repo, seeded["attempt"], sequence=10)
    repeated = asyncio.run(coordinator.plan(str(second.id), request))
    assert first.observation_key != second.observation_key
    assert initial.incident_signature == repeated.incident_signature
    assert repeated.strategy is RecoveryStrategy.PAUSE_FOR_APPROVAL
    assert "loop" in repeated.reason


def test_pending_plans_reserve_attempt_and_experiment_limits(connection, seeded):
    repo = ControlPlaneRepository(connection)
    coordinator = RecoveryCoordinator(repo)
    request = RecoveryRequest(
        limits=RecoveryLimits(max_attempts_per_run=2), allow_retry_without_checkpoint=True
    )
    first = incident(repo, seeded["attempt"])
    assert asyncio.run(coordinator.plan(str(first.id), request)).strategy is RecoveryStrategy.RETRY
    # Different pattern would pass loop protection, but cannot claim a third slot.
    second = incident(
        repo,
        seeded["attempt"],
        sequence=10,
        signal=IncidentObservedPayload(
            reason="process-failure", metadata={"code_location": "elsewhere"}
        ),
    )
    assert asyncio.run(coordinator.plan(str(second.id), request)).strategy is RecoveryStrategy.FAIL


def test_identical_replay_returns_original_and_changed_decision_conflicts(connection, seeded):
    repo = ControlPlaneRepository(connection)
    observed = incident(repo, seeded["attempt"])
    plan = draft(repo, observed, RecoveryRequest(allow_retry_without_checkpoint=True))
    recorded = repo.record_recovery_plan(plan, actor=ACTOR)
    replay = plan.model_copy(update={"id": RecoveryPlanId.generate()})
    assert repo.record_recovery_plan(replay, actor=ACTOR) == recorded
    changed = plan.model_copy(update={"reason": "different decision"})
    with pytest.raises(IdempotencyConflictError):
        repo.record_recovery_plan(changed, actor=ACTOR)


def test_plan_event_and_outbox_roll_back_together(connection, seeded, monkeypatch):
    repo = ControlPlaneRepository(connection)
    observed = incident(repo, seeded["attempt"])
    plan = draft(repo, observed, RecoveryRequest(allow_retry_without_checkpoint=True))
    original = repo._emit

    def crash(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt("crash before commit")

    monkeypatch.setattr(repo, "_emit", crash)
    with pytest.raises(KeyboardInterrupt):
        repo.record_recovery_plan(plan, actor=ACTOR, destinations=("audit",))
    assert repo.recovery_plans.for_incident(str(observed.id)) is None
    assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 0
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM events WHERE event_type = 'RecoveryPlanned'"
        ).fetchone()[0]
        == 0
    )
    monkeypatch.setattr(repo, "_emit", original)
    assert repo.record_recovery_plan(plan, actor=ACTOR) == plan


def test_stale_snapshot_cannot_bypass_a_reserved_limit(connection, seeded):
    repo = ControlPlaneRepository(connection)
    request = RecoveryRequest(
        limits=RecoveryLimits(max_recoveries_per_experiment=1), allow_retry_without_checkpoint=True
    )
    first = incident(repo, seeded["attempt"])
    second = incident(repo, seeded["attempt"], sequence=10)
    left, right = draft(repo, first, request), draft(repo, second, request)
    repo.record_recovery_plan(left, actor=ACTOR)
    with pytest.raises(StaleRecoveryContextError):
        repo.record_recovery_plan(right, actor=ACTOR)
    assert asyncio.run(RecoveryCoordinator(repo).plan(str(second.id), request)).strategy is (
        RecoveryStrategy.FAIL
    )


@pytest.mark.parametrize("field", ["context", "incident_signature", "execution_state_fingerprint"])
def test_recovery_provenance_is_validated(connection, seeded, field):
    repo = ControlPlaneRepository(connection)
    observed = incident(repo, seeded["attempt"])
    plan = draft(repo, observed, RecoveryRequest())
    value = (
        plan.context.model_copy(update={"run_id": "another-run"})
        if field == "context"
        else "sha256:" + "0" * 64
    )
    with pytest.raises(ProvenanceError):
        repo.record_recovery_plan(plan.model_copy(update={field: value}), actor=ACTOR)


@pytest.mark.parametrize("operation", ["UPDATE", "DELETE"])
def test_recovery_plans_are_append_only_in_sql(connection, seeded, operation):
    repo = ControlPlaneRepository(connection)
    observed = incident(repo, seeded["attempt"])
    asyncio.run(RecoveryCoordinator(repo).plan(str(observed.id)))
    sql = (
        "DELETE FROM recovery_plans"
        if operation == "DELETE"
        else "UPDATE recovery_plans SET run_id = 'another-run'"
    )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"), write_transaction(connection):
        connection.execute(sql)


def test_reconciliation_repairs_an_incident_without_a_plan_after_process_exit(
    connection, seeded, db_path
):
    repo = ControlPlaneRepository(connection)
    observed = incident(repo, seeded["attempt"])
    code = """
import asyncio, sys
from xaytune.resilience.recovery import RecoveryCoordinator
from xaytune.core.domain.recovery import RecoveryRequest
from xaytune.storage import ControlPlaneRepository, connect
repo = ControlPlaneRepository(connect(sys.argv[1]))
plans = asyncio.run(RecoveryCoordinator(repo).reconcile(sys.argv[2],
    request_for_incident=lambda _: RecoveryRequest(allow_retry_without_checkpoint=True)))
print(plans[0].model_dump_json())
"""
    output = subprocess.check_output(
        [sys.executable, "-c", code, str(db_path), str(observed.context.experiment_id)], text=True
    )
    recovered = RecoveryPlan.model_validate_json(output)
    assert recovered.strategy is RecoveryStrategy.RETRY
    assert asyncio.run(
        RecoveryCoordinator(repo).reconcile(str(observed.context.experiment_id))
    ) == (recovered,)


def test_concurrent_processes_record_one_plan_and_event(connection, seeded, db_path):
    repo = ControlPlaneRepository(connection)
    observed = incident(repo, seeded["attempt"])
    code = """
import asyncio, sys
from xaytune.resilience.recovery import RecoveryCoordinator
from xaytune.core.domain.recovery import RecoveryRequest
from xaytune.storage import ControlPlaneRepository, connect
repo = ControlPlaneRepository(connect(sys.argv[1]))
plan = asyncio.run(RecoveryCoordinator(repo, destinations=('audit',)).plan(
    sys.argv[2], RecoveryRequest(allow_retry_without_checkpoint=True)))
print(plan.model_dump_json())
"""
    children = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(db_path), str(observed.id)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(3)
    ]
    outputs = []
    for child in children:
        output, error = child.communicate(timeout=20)
        assert child.returncode == 0, error
        outputs.append(RecoveryPlan.model_validate_json(output))
    assert all(plan == outputs[0] for plan in outputs)
    assert connection.execute("SELECT COUNT(*) FROM recovery_plans").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 1


def test_unknown_incident_is_an_error(connection):
    with pytest.raises(AggregateNotFoundError):
        asyncio.run(RecoveryCoordinator(ControlPlaneRepository(connection)).plan("missing"))


@pytest.mark.parametrize("value", [-1, True, 1.0])
def test_recovery_limits_reject_invalid_counters(value):
    with pytest.raises(ValidationError):
        RecoveryLimits(max_recoveries_per_experiment=value)


def test_signature_ignores_delivery_and_detail_but_tracks_structured_shape(connection, seeded):
    repo = ControlPlaneRepository(connection)
    first = incident(repo, seeded["attempt"])
    changed_detail = incident(
        repo,
        seeded["attempt"],
        sequence=10,
        signal=IncidentObservedPayload(reason="process-failure", detail="new"),
    )
    different_shape = incident(
        repo,
        seeded["attempt"],
        sequence=11,
        signal=IncidentObservedPayload(
            reason="process-failure", metadata={"resource_shape": {"cpu": 2}}
        ),
    )
    candidate = seeded["run"].candidate_fingerprint
    assert incident_signature_v1(first, candidate) == incident_signature_v1(
        changed_detail, candidate
    )
    assert incident_signature_v1(first, candidate) != incident_signature_v1(
        different_shape, candidate
    )


def test_prior_attempt_checkpoint_is_eligible_and_future_attempt_is_not_recovered(
    connection, seeded, tmp_path
):
    repo, attempt, report, request, coordinator = prepare(connection, seeded, tmp_path)
    second = make_attempt(seeded["run"], attempt_number=2)
    with write_transaction(connection):
        repo.aggregates._insert_attempt(second)
    current_incident = incident(repo, second)
    plan = asyncio.run(coordinator.plan(str(current_incident.id), request))
    assert plan.strategy is RecoveryStrategy.RESUME
    assert plan.checkpoint_ref == report.payload.checkpoint_ref
    old_incident = incident(repo, attempt, sequence=10)
    obsolete = asyncio.run(coordinator.plan(str(old_incident.id), request))
    assert obsolete.strategy is RecoveryStrategy.FAIL
    assert "superseded" in obsolete.reason


@pytest.mark.parametrize(
    "state,data,boundary",
    [
        ("model-only", "exact", "optimizer-step"),
        ("full", "at-least-once", "optimizer-step"),
        ("full", "at-least-once", "mid-accumulation"),
    ],
)
def test_weak_capture_never_becomes_exact_recovery(
    connection, seeded, tmp_path, state, data, boundary
):
    from dataclasses import replace

    from xaytune.core.resume import ResumeGuarantee
    from xaytune.runtimes import RuntimeEventEnvelope, TrainingEventPayload

    repo = ControlPlaneRepository(connection)
    attempt = seeded["attempt"]
    attempt = type(attempt).model_validate(
        {**attempt.model_dump(mode="json"), "execution_fingerprint": "execution-a"}
    )
    with write_transaction(connection):
        connection.execute(
            "UPDATE run_attempts SET payload_json = ? WHERE id = ?",
            (attempt.model_dump_json(), str(attempt.id)),
        )
    capture, context, restore = make_bundle(
        tmp_path / "source", attempt_id=attempt.id, candidate=seeded["run"].candidate_fingerprint
    )
    guarantee = ResumeGuarantee(state=state, data=data, boundary=boundary)
    if boundary == "mid-accumulation":
        capture = replace(
            capture, state_manifest=capture.state_manifest.model_copy(update={"micro_step": 1})
        )
    capture = replace(capture, resume_guarantee=guarantee)
    store = LocalCheckpointStore(tmp_path / "bundles")
    manager = CheckpointManager(SerializedStateCodec(), store)
    ref = asyncio.run(manager.save(capture, context))
    manifest = asyncio.run(store.get(ref)).manifest
    event = RuntimeEventEnvelope(
        event_id="weak-checkpoint",
        sequence=3,
        target=RuntimeOperationTarget(kind="training-attempt", id=str(attempt.id)),
        payload=TrainingEventPayload(data=manifest.committed_payload(ref)),
    )
    record(repo, attempt, event)
    observed = incident(repo, attempt)
    plan = asyncio.run(
        RecoveryCoordinator(repo, manager).plan(
            str(observed.id),
            RecoveryRequest(
                restore_context=restore.model_copy(update={"required_guarantee": guarantee})
            ),
        )
    )
    assert plan.strategy is RecoveryStrategy.PAUSE_FOR_APPROVAL
    assert not plan.checkpoint_eligibility[0].eligible


@pytest.mark.parametrize("repeat_limit", [0, 1])
def test_repeat_limit_survives_an_execution_change(connection, seeded, repeat_limit):
    repo = ControlPlaneRepository(connection)
    request = RecoveryRequest(
        limits=RecoveryLimits(max_same_incident_repeats=repeat_limit),
        allow_retry_without_checkpoint=True,
    )
    first = incident(repo, seeded["attempt"])
    asyncio.run(RecoveryCoordinator(repo).plan(str(first.id), request))
    second = make_attempt(seeded["run"], attempt_number=2)
    second = type(second).model_validate(
        {**second.model_dump(mode="json"), "execution_fingerprint": "changed-execution"}
    )
    with write_transaction(connection):
        repo.aggregates._insert_attempt(second)
    observed = incident(repo, second)
    plan = asyncio.run(RecoveryCoordinator(repo).plan(str(observed.id), request))
    assert plan.strategy is RecoveryStrategy.PAUSE_FOR_APPROVAL
    assert "repeated incident limit" in plan.reason


@pytest.mark.parametrize("checkpoint_id", ["unknown", "forged"])
def test_eligibility_cannot_name_an_unrecorded_or_forged_reference(
    connection, seeded, tmp_path, checkpoint_id
):
    repo, attempt, _, request, coordinator = prepare(connection, seeded, tmp_path)
    observed = incident(repo, attempt)
    # Produce a legitimate plan, then target a fresh incident so replay does not mask forgery.
    plan = asyncio.run(coordinator.plan(str(observed.id), request))
    next_incident = incident(repo, attempt, sequence=10)
    snapshot = repo.recovery_snapshot(str(next_incident.id))
    ref = plan.checkpoint_ref.model_copy(
        update=(
            {"id": CheckpointId.generate()} if checkpoint_id == "unknown" else {"uri": "forged:uri"}
        )
    )
    eligibility = plan.checkpoint_eligibility[0].model_copy(update={"checkpoint_ref": ref})
    forged = plan.model_copy(
        update={
            "id": RecoveryPlanId.generate(),
            "incident_id": next_incident.id,
            "input_snapshot": snapshot,
            "checkpoint_ref": ref,
            "checkpoint_eligibility": (eligibility,),
        }
    )
    with pytest.raises(ProvenanceError, match="eligibility"):
        repo.record_recovery_plan(forged, actor=ACTOR)


@pytest.mark.parametrize("boundary", ["before-commit", "after-commit"])
def test_hard_process_exit_reconciles_plan_transaction(connection, seeded, db_path, boundary):
    repo = ControlPlaneRepository(connection)
    observed = incident(repo, seeded["attempt"])
    code = """
import asyncio, os, sys
from xaytune.resilience.recovery import RecoveryCoordinator
from xaytune.core.domain.recovery import RecoveryRequest
from xaytune.storage import ControlPlaneRepository, connect
repo = ControlPlaneRepository(connect(sys.argv[1]))
if sys.argv[3] == 'before-commit':
    emit = repo._emit
    def crash(*args, **kwargs):
        emit(*args, **kwargs)
        os._exit(71)
    repo._emit = crash
asyncio.run(RecoveryCoordinator(repo, destinations=('audit',)).plan(
    sys.argv[2], RecoveryRequest(allow_retry_without_checkpoint=True)))
os._exit(72)
"""
    child = subprocess.run(
        [sys.executable, "-c", code, str(db_path), str(observed.id), boundary],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert child.returncode == (71 if boundary == "before-commit" else 72), child.stderr
    assert len(repo.recovery_plans.for_experiment(str(observed.context.experiment_id))) == (
        0 if boundary == "before-commit" else 1
    )
    recovered = asyncio.run(
        RecoveryCoordinator(repo, destinations=("audit",)).reconcile(
            str(observed.context.experiment_id),
            request_for_incident=lambda _: RecoveryRequest(allow_retry_without_checkpoint=True),
        )
    )
    assert len(recovered) == 1
    assert recovered[0].strategy is RecoveryStrategy.RETRY
    assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 1


def test_concurrent_distinct_incidents_cannot_bypass_experiment_limit(
    connection, seeded, db_path, tmp_path
):
    repo = ControlPlaneRepository(connection)
    first = incident(repo, seeded["attempt"])
    second = incident(
        repo,
        seeded["attempt"],
        sequence=10,
        signal=IncidentObservedPayload(
            reason="process-failure", metadata={"code_location": "elsewhere"}
        ),
    )
    code = """
import asyncio, pathlib, sys, time
from xaytune.resilience.recovery import RecoveryCoordinator
from xaytune.core.domain.recovery import RecoveryLimits, RecoveryRequest
from xaytune.storage import ControlPlaneRepository, connect
repo = ControlPlaneRepository(connect(sys.argv[1]))
snapshot = repo.recovery_snapshot
first_read = True
def synchronised_snapshot(incident_id):
    global first_read
    result = snapshot(incident_id)
    if first_read:
        first_read = False
        root = pathlib.Path(sys.argv[3])
        (root / ('ready-' + sys.argv[4])).touch()
        deadline = time.monotonic() + 10
        while len(list(root.glob('ready-*'))) < 2:
            if time.monotonic() > deadline:
                raise RuntimeError('snapshot barrier timed out')
            time.sleep(0.01)
    return result
repo.recovery_snapshot = synchronised_snapshot
plan = asyncio.run(RecoveryCoordinator(repo).plan(sys.argv[2], RecoveryRequest(
    limits=RecoveryLimits(max_recoveries_per_experiment=1),
    allow_retry_without_checkpoint=True)))
print(plan.strategy.value)
"""
    children = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(db_path), str(observed.id), str(tmp_path), str(index)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index, observed in enumerate((first, second))
    ]
    strategies = []
    for child in children:
        output, error = child.communicate(timeout=20)
        assert child.returncode == 0, error
        strategies.append(output.strip())
    assert sorted(strategies) == ["FAIL", "RETRY"]
    assert len(repo.recovery_plans.for_experiment(str(first.context.experiment_id))) == 2


def test_reconciliation_never_resolves_configuration_for_recorded_plans(connection, seeded):
    repo = ControlPlaneRepository(connection)
    observed = incident(repo, seeded["attempt"])
    coordinator = RecoveryCoordinator(repo)
    original = asyncio.run(coordinator.plan(str(observed.id)))

    def unavailable_configuration(_):
        pytest.fail("recorded decisions must not require current consumer configuration")

    plans = asyncio.run(
        coordinator.reconcile(
            str(observed.context.experiment_id), request_for_incident=unavailable_configuration
        )
    )
    assert plans == (original,)


def test_plan_id_cannot_be_reused_for_another_incident(connection, seeded):
    repo = ControlPlaneRepository(connection)
    first = incident(repo, seeded["attempt"])
    original = asyncio.run(RecoveryCoordinator(repo).plan(str(first.id)))
    second = incident(repo, seeded["attempt"], sequence=10)
    next_plan = draft(repo, second, RecoveryRequest()).model_copy(update={"id": original.id})
    with pytest.raises(IdempotencyConflictError):
        repo.record_recovery_plan(next_plan, actor=ACTOR)


def test_coordinator_returns_winning_request_at_the_recording_race(connection, seeded, monkeypatch):
    repo = ControlPlaneRepository(connection)
    observed = incident(repo, seeded["attempt"])
    competitor = draft(repo, observed, RecoveryRequest())
    record_plan = repo.record_recovery_plan

    def competing_record(plan, **kwargs):
        record_plan(competitor, actor=ACTOR)
        return record_plan(plan, **kwargs)

    monkeypatch.setattr(repo, "record_recovery_plan", competing_record)
    winner = asyncio.run(
        RecoveryCoordinator(repo).plan(
            str(observed.id), RecoveryRequest(allow_retry_without_checkpoint=True)
        )
    )
    assert winner == competitor
    assert winner.strategy is RecoveryStrategy.PAUSE_FOR_APPROVAL
