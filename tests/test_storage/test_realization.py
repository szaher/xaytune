"""PR-026: a branched node's first run is accepted in one commit -- or nothing is written.

```text
node_B PLANNED (branch_origin: the recorded planner's proposal)
   ↓ realize_planned_node   (one commit)
node_B READY → ACTIVE
Run_B CREATED → ACTIVE, seed inherited from Run_A, max_runs reserved
Attempt_B CREATED, submit INTENDED, parallel-run slot reserved
```

The runtime is called after the commit, by the controller; nothing here issues.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.test_storage.conftest import make_node
from tests.test_storage.test_branching import _world
from tests.test_storage.test_evaluation_lifecycle import _ACTOR
from tests.test_storage.test_evaluation_lifecycle import repo as repo  # noqa: F401 (fixture)
from tests.test_storage.test_planning_context import _rows
from xaytune.core.domain.budget import (
    BudgetDimension,
    BudgetExhaustedError,
    BudgetSubjectKind,
    CapacityUnavailableError,
    LedgerEntryKind,
)
from xaytune.core.domain.experiment import CandidateSpecSnapshot, ExperimentNode
from xaytune.core.domain.objective import BudgetSpec
from xaytune.core.domain.planning import CandidateBranchOrigin
from xaytune.core.domain.run import Run, RunAttempt, RunSeedOrigin, run_history_fingerprint
from xaytune.core.ids import ExperimentNodeId, RunAttemptId, RunId
from xaytune.core.state.status import (
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
)
from xaytune.storage.control_plane import (
    BranchRefusedError,
    ControlPlaneRepository,
    ProvenanceError,
)
from xaytune.storage.errors import StorageError

DIGEST = "sha256:first-run-request"


def _source_run(
    repo: ControlPlaneRepository,
    parent: Any,
    *,
    seed: int | None = 42,
    replicate: int = 1,
    status: RunStatus = RunStatus.SUCCEEDED,
) -> Run:
    """A training run of *parent*, carried to *status*."""
    run = repo.create_run(
        Run(
            id=RunId.generate(),
            node_id=parent.id,
            experiment_id=parent.experiment_id,
            seed=seed,
            replicate=replicate,
            candidate_fingerprint=parent.candidate_fingerprint,
        ),
        actor=_ACTOR,
    )
    path = {
        RunStatus.CREATED: (),
        RunStatus.ACTIVE: (RunStatus.ACTIVE,),
        RunStatus.SUCCEEDED: (RunStatus.ACTIVE, RunStatus.SUCCEEDED),
        RunStatus.FAILED: (RunStatus.ACTIVE, RunStatus.FAILED),
    }[status]
    for step in path:
        run = repo.transition_run(
            run.id, expected_revision=run.revision, new_status=step, actor=_ACTOR
        )
    return run


def _planned(
    repo: ControlPlaneRepository,
    budget: BudgetSpec | None = None,
    *,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """node_A with one seeded, successful training run, decided BRANCH; node_B PLANNED from it."""
    w = _world(repo, budget=budget or BudgetSpec(max_runs=4, max_parallel_runs=1))
    run = _source_run(repo, w["node"], **(source or {}))
    # The run changed the budget, so plan against the record as it is now.
    (proposal,) = asyncio.run(w["planner"].propose(repo.planning_context(w["experiment"].id)))
    child = repo.materialize_candidate_proposal(w["experiment"].id, proposal, actor=_ACTOR)
    return {**w, "source": run, "child": child, "proposal": proposal}


def _bystander(repo: ControlPlaneRepository, w: dict[str, Any]) -> Run:
    """A run of an unrelated node: it takes a run, and a slot if it gets an attempt."""
    other = repo.create_node(make_node(w["experiment"], fingerprint="bystander"), actor=_ACTOR)
    return repo.create_run(
        Run(
            id=RunId.generate(),
            node_id=other.id,
            experiment_id=other.experiment_id,
            seed=1,
            replicate=1,
            candidate_fingerprint=other.candidate_fingerprint,
        ),
        actor=_ACTOR,
    )


def _first_run(child: Any, source: Run, **changes: Any) -> tuple[Run, RunAttempt]:
    fields: dict[str, Any] = {
        "id": RunId.generate(),
        "node_id": child.id,
        "experiment_id": child.experiment_id,
        "seed": source.seed,
        "seed_origin": RunSeedOrigin(source_run_id=source.id),
        "replicate": 1,
        "candidate_fingerprint": child.candidate_fingerprint,
    }
    fields.update(changes)
    run = Run(**fields)
    return run, RunAttempt(id=RunAttemptId.generate(), run_id=run.id, attempt_number=1)


def _realize(repo: ControlPlaneRepository, child: Any, run: Run, attempt: RunAttempt) -> Any:
    return repo.realize_planned_node(
        child.id,
        expected_revision=child.revision,
        run=run,
        attempt=attempt,
        request_digest=DIGEST,
        actor=_ACTOR,
    )


def _refused(repo: ControlPlaneRepository, error: Any, child: Any, run: Run, attempt: RunAttempt):
    before = _rows(repo)
    with pytest.raises(error) as refused:
        _realize(repo, child, run, attempt)
    assert _rows(repo) == before, "a refusal writes nothing"
    return refused.value


def test_the_first_run_is_accepted_whole_in_one_commit(repo) -> None:
    w = _planned(repo)
    run, attempt = _first_run(w["child"], w["source"])
    events_before = len(repo.events.events_for_experiment(str(w["experiment"].id)))

    node, active, created, operation = _realize(repo, w["child"], run, attempt)

    assert node.status is ExperimentNodeStatus.ACTIVE
    assert active.status is RunStatus.ACTIVE
    assert active.seed == 42 and active.seed_origin == RunSeedOrigin(source_run_id=w["source"].id)
    assert active.replicate == 1 and active.id != w["source"].id
    stored = repo.aggregates.load_attempt(str(created.id))
    assert stored.status is RunAttemptStatus.CREATED and stored.attempt_number == 1
    assert operation.type == "submit" and operation.state == "intended"
    assert operation.request_digest == DIGEST
    assert operation.target.id == str(attempt.id)

    events = repo.events.events_for_experiment(str(w["experiment"].id))[events_before:]
    assert [e.event_type for e in events] == [
        "ExperimentNodeStatusChanged",
        "ExperimentNodeStatusChanged",
        "RunCreated",
        "RunStatusChanged",
        "RunAttemptCreated",
        "RuntimeOperationIntended",
    ]
    reserved = {
        (entry.dimension, entry.subject_kind, entry.subject_id)
        for entry in repo.budget.entries(str(w["experiment"].id))
        if entry.kind is LedgerEntryKind.RESERVE
    }
    assert (BudgetDimension.RUNS, BudgetSubjectKind.RUN, str(run.id)) in reserved
    assert (
        BudgetDimension.PARALLEL_RUNS,
        BudgetSubjectKind.TRAINING_ATTEMPT,
        str(attempt.id),
    ) in reserved


def test_a_node_already_realized_is_not_realized_again(repo) -> None:
    w = _planned(repo)
    _realize(repo, w["child"], *_first_run(w["child"], w["source"]))
    before = _rows(repo)

    assert _realize(repo, w["child"], *_first_run(w["child"], w["source"])) is None
    assert _rows(repo) == before
    assert len(repo.aggregates.runs_for_node(str(w["child"].id))) == 1


def test_a_node_no_planner_branched_is_refused(repo) -> None:
    w = _planned(repo)
    manual = repo.create_node(make_node(w["experiment"], fingerprint="manual"), actor=_ACTOR)
    manual = repo.transition_node(
        manual.id,
        expected_revision=manual.revision,
        new_status=ExperimentNodeStatus.PLANNED,
        actor=_ACTOR,
    )
    run, attempt = _first_run(manual, w["source"], seed_origin=None)
    error = _refused(repo, ProvenanceError, manual, run, attempt)
    assert "not branched" in str(error)


def test_no_run_left_is_refused_with_nothing_reserved(repo) -> None:
    # The source run never submitted anything, so its reservation was released.
    w = _planned(repo, BudgetSpec(max_runs=1))
    # Another run takes the last one after branching: the next run is the one refused.
    _bystander(repo, w)
    _refused(repo, BudgetExhaustedError, w["child"], *_first_run(w["child"], w["source"]))


def test_a_full_capacity_is_refused_with_nothing_written(repo) -> None:
    w = _planned(repo)
    holder = _bystander(repo, w)
    holder = repo.transition_run(
        holder.id, expected_revision=holder.revision, new_status=RunStatus.ACTIVE, actor=_ACTOR
    )
    repo.create_attempt_with_submit_intent(
        RunAttempt(id=RunAttemptId.generate(), run_id=holder.id, attempt_number=1),
        request_digest="sha256:holder",
        actor=_ACTOR,
    )
    _refused(repo, CapacityUnavailableError, w["child"], *_first_run(w["child"], w["source"]))


@pytest.mark.parametrize("lie", ["seed", "source"])
def test_an_inherited_seed_must_be_the_parent_runs(repo, lie) -> None:
    w = _planned(repo)
    if lie == "seed":
        run, attempt = _first_run(w["child"], w["source"], seed=43)
        match = "not parent run"
    else:
        run, attempt = _first_run(
            w["child"], w["source"], seed_origin=RunSeedOrigin(source_run_id=RunId.generate())
        )
        match = "names seed source"
    error = _refused(repo, StorageError, w["child"], run, attempt)
    assert match in str(error)


def test_a_first_run_without_a_seed_origin_is_refused(repo) -> None:
    w = _planned(repo)
    run, attempt = _first_run(w["child"], w["source"], seed_origin=None)
    error = _refused(repo, StorageError, w["child"], run, attempt)
    assert "records no seed origin" in str(error)


@pytest.mark.parametrize("status", [RunStatus.CREATED, RunStatus.ACTIVE, RunStatus.FAILED])
def test_a_parent_run_that_did_not_succeed_is_no_seed_source(repo, status) -> None:
    w = _planned(repo, source={"status": status})
    error = _refused(repo, StorageError, w["child"], *_first_run(w["child"], w["source"]))
    assert f"is {status.value}" in str(error)


def test_a_parent_run_with_no_seed_is_no_seed_source(repo) -> None:
    w = _planned(repo, source={"seed": None})
    error = _refused(repo, StorageError, w["child"], *_first_run(w["child"], w["source"]))
    assert "seed None" in str(error)


def test_a_parent_with_two_runs_is_refused_even_when_one_is_named(repo) -> None:
    w = _planned(repo)
    _source_run(repo, w["node"], seed=43, replicate=2)
    error = _refused(repo, StorageError, w["child"], *_first_run(w["child"], w["source"]))
    assert "has 2 runs" in str(error)


def test_a_child_of_two_parents_is_refused(repo) -> None:
    """A planner-branched node with two parents: whose run's seed would it take?"""
    w = _planned(repo)
    other = repo.create_node(make_node(w["experiment"], fingerprint="second-parent"), actor=_ACTOR)
    proposal = w["proposal"]
    child = repo.create_node(
        ExperimentNode(
            id=ExperimentNodeId.generate(),
            experiment_id=w["experiment"].id,
            parent_ids=(w["node"].id, other.id),
            candidate=CandidateSpecSnapshot(candidate=proposal.candidate),
            candidate_fingerprint=proposal.candidate_fingerprint,
            branch_origin=CandidateBranchOrigin.of(proposal),
            created_by=_ACTOR,
        ),
        actor=_ACTOR,
    )
    child = repo.transition_node(
        child.id,
        expected_revision=child.revision,
        new_status=ExperimentNodeStatus.PLANNED,
        actor=_ACTOR,
    )
    error = _refused(repo, StorageError, child, *_first_run(child, w["source"]))
    assert "has 2 parents" in str(error)


def test_a_first_run_is_replicate_one(repo) -> None:
    w = _planned(repo)
    error = _refused(
        repo, StorageError, w["child"], *_first_run(w["child"], w["source"], replicate=2)
    )
    assert "replicate 2" in str(error)


def test_any_run_claiming_an_inherited_seed_must_be_honest(repo) -> None:
    """Not only a first realization: every created run's seed origin is checked."""
    w = _planned(repo)
    lying, _ = _first_run(w["child"], w["source"], seed=43)
    with pytest.raises(StorageError, match="whose seed is 42"):
        repo.create_run(lying, actor=_ACTOR)


def test_a_run_that_is_not_the_nodes_first_is_refused(repo) -> None:
    w = _planned(repo)
    run, _ = _first_run(w["child"], w["source"])
    second = RunAttempt(id=RunAttemptId.generate(), run_id=run.id, attempt_number=2)
    _refused(repo, StorageError, w["child"], run, second)


def test_only_an_active_experiment_starts_a_run(repo) -> None:
    w = _planned(repo)
    experiment = repo.aggregates.load_experiment(str(w["experiment"].id))
    repo.transition_experiment(
        experiment.id,
        expected_revision=experiment.revision,
        new_status=ExperimentStatus.PAUSED,
        actor=_ACTOR,
    )
    _refused(repo, BranchRefusedError, w["child"], *_first_run(w["child"], w["source"]))


def test_seed_origin_is_provenance_not_identity(repo) -> None:
    """Where the seed came from never changes what the run is."""
    w = _planned(repo)
    inherited, _ = _first_run(w["child"], w["source"])
    given = Run(**{**inherited.model_dump(mode="python"), "seed_origin": None})
    assert run_history_fingerprint(inherited, ()) == run_history_fingerprint(given, ())

    payload = given.model_dump(mode="json")
    payload.pop("seed_origin")
    assert Run.model_validate(payload).seed_origin is None, "a run recorded before PR-026 loads"
