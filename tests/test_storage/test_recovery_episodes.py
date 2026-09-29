"""Episode ownership, authority arbitration, immutable history and restart safety."""

from __future__ import annotations

import asyncio
import itertools
import sqlite3
import subprocess
import sys
import time

import pytest
from pydantic import ValidationError

from tests.test_resilience.test_incidents import classify, envelope
from tests.test_storage.conftest import make_attempt, make_run
from tests.test_storage.test_recovery import _EPISODES, ACTOR, draft, prepare, record_plan
from xaytune.core.domain.incident import DetectorProvenance, IncidentCandidate, IncidentCategory
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.domain.recovery import (
    RecoveryEpisode,
    RecoveryEvidenceDisposition,
    RecoveryInputsV1,
    RecoveryLimits,
    RecoveryRequest,
    RecoveryStrategy,
    decide_recovery,
)
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import RecoveryEpisodeId, RecoveryPlanId
from xaytune.core.state.status import RunAttemptStatus
from xaytune.core.telemetry import IncidentObservedPayload
from xaytune.resilience.recovery import RecoveryCoordinator, RecoveryRequestUnavailableError
from xaytune.storage import ControlPlaneRepository, connect, write_transaction
from xaytune.storage.control_plane import ProvenanceError, StaleRecoveryContextError
from xaytune.storage.journal import IdempotencyConflictError


def incident(repo, attempt, sequence=9, signal=None):
    """Explicit test diagnoses exercise categories without extending PR-017 detectors."""
    owner = repo.incident_context(
        RuntimeOperationTarget(kind="training-attempt", id=str(attempt.id))
    )
    data = signal or IncidentObservedPayload(reason="process-failure")
    event = envelope(owner, data).model_copy(update={"sequence": sequence})
    proposed = classify(event, owner)
    category = IncidentCategory(data.reason.upper().replace("-", "_"))
    detector = DetectorProvenance(name="test-structured", version="1")
    proposed = proposed.model_copy(
        update={
            "category": category,
            "candidates": (
                IncidentCandidate(category=category, detector=detector, reason=data.reason),
            ),
            "detectors": (detector,),
            "classifier": detector,
        }
    )
    return repo.record_incident(proposed, actor=ACTOR)


def decide(repo, observation, request=None):
    return asyncio.run(RecoveryCoordinator(repo).plan(str(observation.id), request))


def another_run(repo, seeded):
    run = make_run(seeded["node"])
    repo.create_run(run, actor=ACTOR)
    attempt = make_attempt(run)
    with write_transaction(repo._connection):
        repo.aggregates._insert_attempt(attempt)
    return run, attempt


def successor(repo, run, number):
    attempt = make_attempt(run, attempt_number=number)
    with write_transaction(repo._connection):
        repo.aggregates._insert_attempt(attempt)
    return attempt


def signal(reason):
    return IncidentObservedPayload(reason=reason)


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("process-failure", "worker-failure", RecoveryStrategy.RETRY),
        ("cuda-oom", "process-failure", RecoveryStrategy.PAUSE_FOR_APPROVAL),
        ("config-error", "process-failure", RecoveryStrategy.FAIL),
        ("cuda-oom", "numerical-nan", RecoveryStrategy.PAUSE_FOR_APPROVAL),
        ("preemption", "preemption", RecoveryStrategy.RETRY),
    ],
)
def test_arbitration_is_independent_of_arrival_order(connection, seeded, left, right, expected):
    repo = ControlPlaneRepository(connection)
    request = RecoveryRequest(allow_retry_without_checkpoint=True)
    results = []
    for order in ((left, right), (right, left)):
        _, attempt = another_run(repo, seeded)
        first = incident(repo, attempt, signal=signal(order[0]))
        initial = decide(repo, first, request)
        second = incident(repo, attempt, sequence=10, signal=signal(order[1]))
        assert not repo.recovery_plans.is_effective_and_fresh(str(initial.id))
        effective = decide(repo, second)
        assert effective.strategy is expected
        assert effective.sequence == 2
        assert effective.supersedes_plan_id == initial.id
        assert set(effective.accepted_incident_ids) == {first.id, second.id}
        assert repo.recovery_plans.get(str(initial.id)) == initial
        assert repo.recovery_plans.effective_for_episode(str(initial.episode_id)) == effective
        results.append((effective.strategy, effective.reason))
    assert results[0] == results[1]
    if {left, right} == {"cuda-oom", "numerical-nan"}:
        assert "conflicting specialised" in results[0][1]


def test_duplicate_observation_and_signatures_have_distinct_semantics(connection, seeded):
    repo = ControlPlaneRepository(connection)
    request = RecoveryRequest(allow_retry_without_checkpoint=True)
    first = incident(repo, seeded["attempt"])
    initial = decide(repo, first, request)
    replay = repo.record_incident(first, actor=ACTOR, destinations=("audit",))
    assert replay == first
    assert len(repo.recovery_episodes.memberships(str(initial.episode_id))) == 1
    assert decide(repo, replay) == initial
    second = incident(repo, seeded["attempt"], sequence=10)
    third = incident(repo, seeded["attempt"], sequence=11)
    assert not repo.recovery_plans.is_effective_and_fresh(str(initial.id))
    latest = decide(repo, third)
    history = repo.recovery_plans.revisions_for_episode(str(latest.episode_id))
    assert [p.sequence for p in history] == [1, 2, 3]
    assert [len(p.accepted_incident_ids) for p in history] == [1, 2, 3]
    assert all(p.strategy is RecoveryStrategy.RETRY for p in history)
    assert len(latest.incident_signatures) == 1
    assert repo.recovery_plans.for_incident(str(second.id)) == latest
    assert repo.recovery_episodes.usage_excluding(str(first.context.experiment_id), "other") == 1
    assert repo.recovery_plans.is_effective_and_fresh(str(latest.id))
    assert not repo.recovery_plans.is_effective_and_fresh(str(initial.id))


@pytest.mark.parametrize("limit,expected", [(0, [True, False]), (2, [True, True, True, False])])
def test_repeat_limit_counts_prior_episodes_with_unchanged_execution(
    connection, seeded, limit, expected
):
    repo = ControlPlaneRepository(connection)
    request = RecoveryRequest(
        allow_retry_without_checkpoint=True,
        limits=RecoveryLimits(
            max_attempts_per_run=20,
            max_recoveries_per_experiment=20,
            max_same_incident_repeats=limit,
        ),
    )
    attempt = seeded["attempt"]
    fingerprints = []
    for index, allowed in enumerate(expected):
        if index:
            attempt = successor(repo, seeded["run"], index + 1)
        observation = incident(repo, attempt, signal=signal("preemption"))
        plan = decide(repo, observation, request)
        # Duplicate observations/revisions in earlier episodes cannot inflate this count.
        duplicate = incident(repo, attempt, sequence=10, signal=signal("preemption"))
        plan = decide(repo, duplicate)
        assert plan.inputs.repeat_counts[0].prior_matching_episodes == index
        assert (plan.strategy is RecoveryStrategy.RETRY) is allowed
        fingerprints.append(plan.execution_state_fingerprint)
    assert len(set(fingerprints)) == 1


@pytest.mark.parametrize("successor_status", [RunAttemptStatus.FAILED, RunAttemptStatus.CANCELLED])
def test_closure_freezes_decisions_and_late_evidence_does_not_change_repeats(
    connection, seeded, successor_status
):
    repo = ControlPlaneRepository(connection)
    request = RecoveryRequest(
        allow_retry_without_checkpoint=True,
        limits=RecoveryLimits(max_attempts_per_run=10, max_same_incident_repeats=0),
    )
    first = incident(repo, seeded["attempt"], signal=signal("process-failure"))
    original = decide(repo, first, request)
    next_attempt = successor(repo, seeded["run"], 2)
    changed = next_attempt.with_status(successor_status)
    with write_transaction(connection):
        repo.aggregates._update_attempt(changed)
    late = incident(repo, seeded["attempt"], sequence=10, signal=signal("preemption"))
    membership = repo.recovery_episodes.membership(str(late.id))
    assert membership.disposition is RecoveryEvidenceDisposition.LATE_AFTER_CLOSURE
    assert not repo.recovery_episodes.is_open(str(original.episode_id))
    assert not repo.recovery_plans.is_effective_and_fresh(str(original.id))
    assert decide(repo, late) == original
    assert repo.recovery_plans.revisions_for_episode(str(original.episode_id)) == (original,)
    current = incident(repo, next_attempt, signal=signal("preemption"))
    current_plan = decide(repo, current, request)
    assert current_plan.inputs.repeat_counts[0].prior_matching_episodes == 0
    assert (
        current_plan.strategy is RecoveryStrategy.RETRY
        or successor_status is RunAttemptStatus.CANCELLED
    )


def test_closure_during_input_read_replays_frozen_decision(connection, seeded, monkeypatch):
    repo = ControlPlaneRepository(connection)
    first = incident(repo, seeded["attempt"])
    original = decide(repo, first, RecoveryRequest(allow_retry_without_checkpoint=True))
    second = incident(repo, seeded["attempt"], sequence=10, signal=signal("cuda-oom"))
    assert not repo.recovery_plans.is_effective_and_fresh(str(original.id))
    snapshot = repo.recovery_snapshot

    def close_before_snapshot(episode):
        successor(repo, seeded["run"], 2)
        return snapshot(episode)

    monkeypatch.setattr(repo, "recovery_snapshot", close_before_snapshot)
    assert decide(repo, second) == original
    assert not repo.recovery_episodes.is_open(str(original.episode_id))
    assert repo.recovery_plans.revisions_for_episode(str(original.episode_id)) == (original,)


def test_resume_to_pause_releases_only_effective_reservation(connection, seeded, tmp_path):
    repo, attempt, _, request, coordinator = prepare(connection, seeded, tmp_path)
    first = incident(repo, attempt)
    resume = asyncio.run(coordinator.plan(str(first.id), request))
    assert resume.strategy is RecoveryStrategy.RESUME
    experiment = str(first.context.experiment_id)
    assert repo.recovery_episodes.usage_excluding(experiment, "other") == 1
    incident(repo, attempt, sequence=10, signal=signal("cuda-oom"))
    pause = asyncio.run(coordinator.plan(str(first.id)))
    assert pause.strategy is RecoveryStrategy.PAUSE_FOR_APPROVAL
    assert repo.recovery_episodes.usage_excluding(experiment, "other") == 0
    assert (
        repo.recovery_episodes.pending_excluding(
            first.context.target, first.context.run_id, "other"
        )
        == 0
    )
    assert repo.recovery_plans.get(str(resume.id)) == resume
    assert pause.checkpoint_eligibility == ()
    assert pause.inputs.checkpoint_reports == ()


def test_restart_repairs_coverage_using_original_request(connection, seeded, db_path):
    repo = ControlPlaneRepository(connection)
    first = incident(repo, seeded["attempt"])
    request = RecoveryRequest(allow_retry_without_checkpoint=True)
    original = decide(repo, first, request)
    episode = repo.recovery_episodes.get(str(original.episode_id))
    incident(repo, seeded["attempt"], sequence=10)
    incident(repo, seeded["attempt"], sequence=11)
    assert not repo.recovery_plans.is_effective_and_fresh(str(original.id))
    reopened = connect(db_path)
    try:
        restarted = ControlPlaneRepository(reopened)

        def forbidden(_):
            pytest.fail("existing episode request must suffice")

        plans = asyncio.run(
            RecoveryCoordinator(restarted).reconcile(
                str(first.context.experiment_id), request_for_incident=forbidden
            )
        )
        assert len(plans) == 1 and plans[0].sequence == 3
        assert plans[0].strategy is RecoveryStrategy.RETRY
        assert restarted.recovery_episodes.get(str(episode.id)) == episode
        assert len(restarted.recovery_episodes.memberships(str(episode.id))) == 3
        assert len(restarted.recovery_plans.revisions_for_episode(str(episode.id))) == 3
        assert restarted.recovery_plans.is_effective_and_fresh(str(plans[0].id))
        # Explicit new caller policy cannot replace original provenance.
        assert (
            asyncio.run(RecoveryCoordinator(restarted).plan(str(first.id), RecoveryRequest()))
            == plans[0]
        )
    finally:
        reopened.close()


def test_unrelated_nonreserving_revision_does_not_stale_inputs(connection, seeded):
    repo = ControlPlaneRepository(connection)
    target = incident(repo, seeded["attempt"])
    target_plan = draft(repo, target, RecoveryRequest(allow_retry_without_checkpoint=True))
    _, other = another_run(repo, seeded)
    unrelated = incident(repo, other, signal=signal("cuda-oom"))
    decide(repo, unrelated, RecoveryRequest())
    incident(repo, other, sequence=10, signal=signal("numerical-nan"))
    decide(repo, unrelated)
    assert repo.recovery_snapshot(_EPISODES[str(target_plan.episode_id)]) == target_plan.inputs
    assert record_plan(repo, target_plan, actor=ACTOR) == target_plan


def test_typed_inputs_are_small_versioned_canonical_and_immutable(connection, seeded):
    repo = ControlPlaneRepository(connection)
    observation = incident(repo, seeded["attempt"])
    plan = decide(repo, observation, RecoveryRequest(allow_retry_without_checkpoint=True))
    inputs = plan.inputs
    assert RecoveryInputsV1.model_validate_json(inputs.model_dump_json()) == inputs
    assert inputs.schema_version == "xaytune.recovery-inputs/v1alpha1"
    assert "resource_usage" not in inputs.model_dump_json()
    assert "runtime_ref" not in inputs.model_dump_json()
    assert "execution_overrides" not in inputs.model_dump_json()
    with pytest.raises(ValidationError):
        inputs.actual_attempt_count = 2
    with pytest.raises(ValidationError):
        inputs.model_copy(update={"schema_version": "unknown"})
    with pytest.raises(ValueError, match="request differs"):
        decide_recovery(inputs, RecoveryRequest(), ())
    episode = repo.recovery_episodes.get(str(plan.episode_id))
    assert RecoveryEpisode.model_validate_json(episode.model_dump_json()) == episode
    assert episode.context == observation.context
    assert episode.attempt_number == 1


@pytest.mark.parametrize("reason", ["cuda-oom", "numerical-nan", "config-error"])
def test_lazy_checkpoint_paths_never_inspect_files(
    connection, seeded, tmp_path, reason, monkeypatch
):
    repo, attempt, _, request, coordinator = prepare(connection, seeded, tmp_path)

    async def forbidden(*args):
        pytest.fail("this authority path does not consult checkpoints")

    monkeypatch.setattr(coordinator.checkpoint_manager, "validate_recorded", forbidden)
    observation = incident(repo, attempt, signal=signal(reason))
    plan = asyncio.run(coordinator.plan(str(observation.id), request))
    assert plan.checkpoint_eligibility == ()
    assert plan.inputs.checkpoint_reports == ()


def test_byte_validation_outside_lock_and_report_binding_inside(
    connection, seeded, tmp_path, monkeypatch
):
    repo, attempt, report, request, coordinator = prepare(connection, seeded, tmp_path)
    observation = incident(repo, attempt)
    validate = coordinator.checkpoint_manager.validate_recorded

    async def inspect(record, context):
        assert not connection.in_transaction
        return await validate(record, context)

    monkeypatch.setattr(coordinator.checkpoint_manager, "validate_recorded", inspect)
    plan = asyncio.run(coordinator.plan(str(observation.id), request))
    assert plan.checkpoint_eligibility[0].report_fingerprint == fingerprint(report)
    incident(repo, attempt, sequence=10)
    episode = repo.recovery_episodes.get(str(plan.episode_id))
    inputs = repo.recovery_snapshot(episode)
    forged = plan.checkpoint_eligibility[0].model_copy(
        update={"report_fingerprint": "sha256:" + "0" * 64}
    )
    from xaytune.core.domain.recovery import RecoveryPlan

    proposed = RecoveryPlan.from_decision(
        inputs, decide_recovery(inputs, request, (forged,)), (forged,)
    )
    with pytest.raises(ProvenanceError, match="eligibility"):
        repo.record_recovery_plan(proposed, actor=ACTOR)


@pytest.mark.parametrize(
    "table,operation",
    itertools.product(
        ["recovery_episodes", "recovery_episode_incidents", "recovery_plans"], ["UPDATE", "DELETE"]
    ),
)
def test_all_recovery_tables_are_append_only(connection, seeded, table, operation):
    repo = ControlPlaneRepository(connection)
    observation = incident(repo, seeded["attempt"])
    decide(repo, observation, RecoveryRequest())
    sql = (
        f"DELETE FROM {table}"
        if operation == "DELETE"
        else f"UPDATE {table} SET payload_json = payload_json"
    )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"), write_transaction(connection):
        connection.execute(sql)


@pytest.mark.parametrize(
    "table,changes",
    [
        ("recovery_episodes", {"id": "new", "attempt_number": 2}),
        ("recovery_episode_incidents", {"membership_sequence": 3}),
        (
            "recovery_episode_incidents",
            {"disposition": "LATE_AFTER_CLOSURE", "membership_sequence": 2},
        ),
        ("recovery_episode_incidents", {"evidence_fingerprint": "bad", "membership_sequence": 2}),
        ("recovery_plans", {"id": "new", "sequence": 3}),
        ("recovery_plans", {"id": "new", "sequence": 2, "supersedes_plan_id": "missing"}),
    ],
)
def test_migration_rejects_invalid_ownership_sequences_and_dispositions(
    connection, seeded, table, changes
):
    repo = ControlPlaneRepository(connection)
    observation = incident(repo, seeded["attempt"])
    decide(repo, observation, RecoveryRequest())
    columns = [r["name"] for r in connection.execute(f"PRAGMA table_info({table})")]
    expressions = ["?" if c in changes else c for c in columns]
    values = [changes[c] for c in columns if c in changes]
    with pytest.raises(sqlite3.IntegrityError), write_transaction(connection):
        connection.execute(
            f"INSERT INTO {table} ({', '.join(columns)}) "
            f"SELECT {', '.join(expressions)} FROM {table}",
            values,
        )


@pytest.mark.parametrize("initial", [True, False])
def test_plan_transaction_rolls_back_without_losing_prior_history(
    connection, seeded, initial, monkeypatch
):
    repo = ControlPlaneRepository(connection)
    observation = incident(repo, seeded["attempt"])
    if not initial:
        decide(repo, observation, RecoveryRequest(allow_retry_without_checkpoint=True))
        observation = incident(repo, seeded["attempt"], sequence=10)
    before = {
        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in [
            "recovery_episodes",
            "recovery_episode_incidents",
            "recovery_plans",
            "events",
            "outbox",
        ]
    }
    emit = repo._emit

    def failure(*args, **kwargs):
        emit(*args, **kwargs)
        if args[1] == "RecoveryPlanned":
            raise RuntimeError("injected failure after event/outbox")

    monkeypatch.setattr(repo, "_emit", failure)
    with pytest.raises(RuntimeError, match="injected"):
        asyncio.run(
            RecoveryCoordinator(repo, destinations=("audit",)).plan(
                str(observation.id), RecoveryRequest(allow_retry_without_checkpoint=True)
            )
        )
    for table, count in before.items():
        assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == count


def test_superseded_target_without_episode_is_historical_only(connection, seeded):
    repo = ControlPlaneRepository(connection)
    historical = incident(repo, seeded["attempt"])
    successor(repo, seeded["run"], 2)

    def forbidden(_):
        pytest.fail("historical observation needs no reconstructed planning request")

    assert (
        asyncio.run(
            RecoveryCoordinator(repo).reconcile(
                str(historical.context.experiment_id), request_for_incident=forbidden
            )
        )
        == ()
    )
    assert repo.incidents.get(str(historical.id)) == historical
    assert repo.recovery_episodes.for_attempt(historical.context.target) is None
    for table in ["recovery_episodes", "recovery_episode_incidents", "recovery_plans"]:
        assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_initial_gap_stops_later_episode_creation(connection, seeded):
    repo = ControlPlaneRepository(connection)
    first = incident(repo, seeded["attempt"])
    _, other = another_run(repo, seeded)
    later = incident(repo, other)
    seen = []

    def unresolved(observation):
        seen.append(observation.id)
        return None

    with pytest.raises(RecoveryRequestUnavailableError):
        asyncio.run(
            RecoveryCoordinator(repo).reconcile(
                str(first.context.experiment_id), request_for_incident=unresolved
            )
        )
    assert seen == [first.id]
    assert repo.recovery_plans.for_incident(str(later.id)) is None
    assert connection.execute("SELECT COUNT(*) FROM recovery_episodes").fetchone()[0] == 0


def test_detector_disagreement_retains_blocking_candidate(connection, seeded):
    repo = ControlPlaneRepository(connection)
    owner = repo.incident_context(
        RuntimeOperationTarget(kind="training-attempt", id=str(seeded["attempt"].id))
    )
    event = envelope(owner, signal("process-failure"))
    proposed = classify(event, owner)
    first = proposed.detectors[0]
    second = proposed.detectors[1]
    proposed = proposed.model_copy(
        update={
            "category": IncidentCategory.UNKNOWN,
            "candidates": (
                IncidentCandidate(
                    category=IncidentCategory.PROCESS_FAILURE, detector=first, reason="generic"
                ),
                IncidentCandidate(
                    category=IncidentCategory.CONFIG_ERROR, detector=second, reason="blocking"
                ),
            ),
        }
    )
    recorded = repo.record_incident(proposed, actor=ACTOR)
    plan = decide(repo, recorded, RecoveryRequest(allow_retry_without_checkpoint=True))
    assert plan.strategy is RecoveryStrategy.FAIL
    assert set(plan.inputs.accepted_evidence[0].categories) == {
        IncidentCategory.CONFIG_ERROR,
        IncidentCategory.PROCESS_FAILURE,
    }
    assert repo.incidents.get(str(recorded.id)).candidates == proposed.candidates


@pytest.mark.parametrize(
    "changed", ["attempt_count", "membership", "status", "execution", "checkpoint"]
)
def test_relevant_database_changes_refuse_prepared_decision(connection, seeded, tmp_path, changed):
    repo, attempt, _, request, coordinator = prepare(connection, seeded, tmp_path)
    observation = incident(repo, attempt)
    episode = repo.prepare_recovery_episode(str(observation.id), request, ACTOR)
    inputs = repo.recovery_snapshot(episode)
    eligibility = asyncio.run(coordinator._checkpoints(inputs, request))
    from xaytune.core.domain.recovery import RecoveryPlan

    plan = RecoveryPlan.from_decision(
        inputs, decide_recovery(inputs, request, eligibility), eligibility
    )
    if changed == "attempt_count":
        successor(repo, seeded["run"], 2)
    elif changed == "membership":
        incident(repo, attempt, sequence=10)
    elif changed == "status":
        updated = attempt.with_status(RunAttemptStatus.CANCELLED)
        with write_transaction(connection):
            repo.aggregates._update_attempt(updated)
    elif changed == "execution":
        updated = type(attempt).model_validate(
            {
                **attempt.model_dump(),
                "execution_fingerprint": "changed",
                "revision": attempt.revision + 1,
            }
        )
        with write_transaction(connection):
            repo.aggregates._update_attempt(updated)
    else:
        from tests.test_checkpoints.helpers import make_bundle
        from tests.test_resilience.test_incidents import envelope as observation_envelope
        from tests.test_storage.test_checkpoints import record
        from xaytune.checkpoints import (
            CheckpointManager,
            LocalCheckpointStore,
            SerializedStateCodec,
        )

        state, context, _ = make_bundle(
            tmp_path / "another-source",
            attempt_id=attempt.id,
            candidate=episode.candidate_fingerprint,
        )
        store = LocalCheckpointStore(tmp_path / "another-store")
        manager = CheckpointManager(SerializedStateCodec(), store)
        ref = asyncio.run(manager.save(state, context))
        manifest = asyncio.run(store.get(ref)).manifest
        event = observation_envelope(episode.context, manifest.committed_payload(ref))
        record(repo, attempt, event.model_copy(update={"sequence": 8}))
    with pytest.raises(StaleRecoveryContextError):
        repo.record_recovery_plan(plan, actor=ACTOR, episode=episode)
    assert repo.recovery_episodes.for_attempt(observation.context.target) is None


def test_concurrent_admission_and_planning_converge_to_one_revision_chain(
    connection, seeded, db_path, tmp_path
):
    repo = ControlPlaneRepository(connection)
    first = incident(repo, seeded["attempt"])
    initial = decide(repo, first, RecoveryRequest(allow_retry_without_checkpoint=True))
    code = """
import asyncio, pathlib, sys, time
from tests.test_storage.test_recovery_episodes import incident
from xaytune.resilience.recovery import RecoveryCoordinator
from xaytune.storage import ControlPlaneRepository, connect
repo = ControlPlaneRepository(connect(sys.argv[1]))
root = pathlib.Path(sys.argv[4])
(root / ('ready-' + sys.argv[3])).touch()
deadline = time.monotonic() + 10
while not (root / 'release').exists():
    if time.monotonic() > deadline: raise RuntimeError('barrier timeout')
    time.sleep(0.01)
attempt = repo.aggregates.load_attempt(sys.argv[2])
observed = incident(repo, attempt, sequence=int(sys.argv[3]))
plan = asyncio.run(RecoveryCoordinator(repo).plan(str(observed.id)))
print(plan.sequence)
"""
    children = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                code,
                str(db_path),
                str(seeded["attempt"].id),
                str(seq),
                str(tmp_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for seq in (10, 11)
    ]
    deadline = time.monotonic() + 10
    while len(list(tmp_path.glob("ready-*"))) != 2:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    (tmp_path / "release").touch()
    for child in children:
        output, error = child.communicate(timeout=20)
        assert child.returncode == 0, error
        assert int(output.strip()) in (2, 3)
    latest = decide(repo, first)
    history = repo.recovery_plans.revisions_for_episode(str(initial.episode_id))
    assert [p.sequence for p in history] == [1, 2, 3]
    assert all(history[i].supersedes_plan_id == history[i - 1].id for i in (1, 2))
    assert len(repo.recovery_episodes.memberships(str(initial.episode_id))) == 3
    assert repo.recovery_plans.is_effective_and_fresh(str(latest.id))
    assert repo.recovery_episodes.usage_excluding(str(first.context.experiment_id), "other") == 1


@pytest.mark.parametrize(
    "first_role,disposition",
    [
        ("evidence", RecoveryEvidenceDisposition.ACCEPTED_FOR_DECISION),
        ("successor", RecoveryEvidenceDisposition.LATE_AFTER_CLOSURE),
    ],
)
def test_successor_evidence_race_is_classified_under_write_lock(
    connection, seeded, db_path, tmp_path, first_role, disposition
):
    repo = ControlPlaneRepository(connection)
    first = incident(repo, seeded["attempt"])
    original = decide(repo, first, RecoveryRequest(allow_retry_without_checkpoint=True))
    next_attempt = make_attempt(seeded["run"], attempt_number=2)
    code = """
import pathlib, sys, time
from tests.test_storage.test_recovery_episodes import incident
from xaytune.core.domain.run import RunAttempt
from xaytune.storage import ControlPlaneRepository, connect, write_transaction
repo = ControlPlaneRepository(connect(sys.argv[1]))
root = pathlib.Path(sys.argv[4])
role, first = sys.argv[3], sys.argv[5] == 'first'
def hold():
    (root / 'holding').touch()
    deadline = time.monotonic() + 10
    while not (root / 'release').exists():
        if time.monotonic() > deadline: raise RuntimeError('lock barrier timeout')
        time.sleep(0.01)
if not first: (root / 'second-ready').touch()
if role == 'successor':
    with write_transaction(repo._connection):
        repo.aggregates._insert_attempt(RunAttempt.model_validate_json(sys.argv[6]))
        if first: hold()
else:
    if first:
        emit = repo._emit
        def holding_emit(*args, **kwargs):
            result = emit(*args, **kwargs)
            if args[1] == 'RecoveryEvidenceAttached': hold()
            return result
        repo._emit = holding_emit
    attempt = repo.aggregates.load_attempt(sys.argv[2])
    observed = incident(repo, attempt, sequence=10)
    print(repo.recovery_episodes.membership(str(observed.id)).disposition.value)
"""

    def launch(role, order):
        return subprocess.Popen(
            [
                sys.executable,
                "-c",
                code,
                str(db_path),
                str(seeded["attempt"].id),
                role,
                str(tmp_path),
                order,
                next_attempt.model_dump_json(),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    left = launch(first_role, "first")
    deadline = time.monotonic() + 10
    while not (tmp_path / "holding").exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    right = launch("successor" if first_role == "evidence" else "evidence", "second")
    while not (tmp_path / "second-ready").exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    (tmp_path / "release").touch()
    outputs = []
    for child in (left, right):
        output, error = child.communicate(timeout=20)
        assert child.returncode == 0, error
        outputs.append(output.strip())
    assert disposition.value in outputs
    assert not repo.recovery_episodes.is_open(str(original.episode_id))
    assert not repo.recovery_plans.is_effective_and_fresh(str(original.id))
    assert decide(repo, first) == original
    assert len(repo.recovery_plans.revisions_for_episode(str(original.episode_id))) == 1


@pytest.mark.parametrize("limit_kind", ["attempt", "experiment"])
def test_closed_history_and_current_reservations_are_counted_once(connection, seeded, limit_kind):
    repo = ControlPlaneRepository(connection)
    request = RecoveryRequest(
        allow_retry_without_checkpoint=True,
        limits=RecoveryLimits(
            max_attempts_per_run=2 if limit_kind == "attempt" else 20,
            max_recoveries_per_experiment=1 if limit_kind == "experiment" else 20,
        ),
    )
    first = incident(repo, seeded["attempt"])
    initial = decide(repo, first, request)
    incident(repo, seeded["attempt"], sequence=10)
    replacement = decide(repo, first)
    assert replacement.strategy is RecoveryStrategy.RETRY
    assert repo.recovery_episodes.usage_excluding(str(first.context.experiment_id), "other") == 1
    second = successor(repo, seeded["run"], 2)
    assert (
        repo.recovery_episodes.pending_excluding(
            first.context.target, first.context.run_id, "other"
        )
        == 0
    )
    second_incident = incident(repo, second)
    refused = decide(repo, second_incident, request)
    assert refused.strategy is RecoveryStrategy.FAIL
    assert refused.inputs.actual_attempt_count == 2
    assert refused.inputs.experiment_recovery_usage_excluding_target == 1
    assert repo.recovery_plans.get(str(initial.id)) == initial


def test_sql_refuses_decision_after_closure_even_for_uncovered_accepted_evidence(
    connection, seeded
):
    repo = ControlPlaneRepository(connection)
    first = incident(repo, seeded["attempt"])
    original = decide(repo, first, RecoveryRequest(allow_retry_without_checkpoint=True))
    incident(repo, seeded["attempt"], sequence=10)
    successor(repo, seeded["run"], 2)
    with (
        pytest.raises(sqlite3.IntegrityError, match="closed episode"),
        write_transaction(connection),
    ):
        connection.execute(
            "INSERT INTO recovery_plans (id, episode_id, sequence, supersedes_plan_id, "
            "accepted_through_sequence, accepted_evidence_fingerprint, strategy, payload_json) "
            "SELECT 'new', episode_id, 2, id, 2, 'new-evidence', strategy, payload_json "
            "FROM recovery_plans WHERE id = ?",
            (str(original.id),),
        )


def test_episode_uniqueness_and_conflicting_replay_are_durable(connection, seeded):
    repo = ControlPlaneRepository(connection)
    first = incident(repo, seeded["attempt"])
    original = decide(repo, first, RecoveryRequest())
    episode = repo.recovery_episodes.get(str(original.episode_id))
    with pytest.raises(sqlite3.IntegrityError), write_transaction(connection):
        repo.recovery_episodes._insert(
            episode.model_copy(update={"id": RecoveryEpisodeId.generate()})
        )
    replay = original.model_copy(update={"id": RecoveryPlanId.generate()})
    assert repo.record_recovery_plan(replay, actor=ACTOR) == original
    with pytest.raises(IdempotencyConflictError):
        repo.record_recovery_plan(original.model_copy(update={"reason": "changed"}), actor=ACTOR)
