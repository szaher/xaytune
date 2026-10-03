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
from xaytune.core.domain.objective import BudgetSpec
from xaytune.core.domain.run import Run, RunAttempt, RunSeedOrigin, run_history_fingerprint
from xaytune.core.ids import RunAttemptId, RunId
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


def _planned(repo: ControlPlaneRepository, budget: BudgetSpec | None = None) -> dict[str, Any]:
    """node_A with one seeded training run, decided BRANCH; node_B branched from it, PLANNED."""
    w = _world(repo, budget=budget or BudgetSpec(max_runs=4, max_parallel_runs=1))
    parent = w["node"]
    source = repo.create_run(
        Run(
            id=RunId.generate(),
            node_id=parent.id,
            experiment_id=parent.experiment_id,
            seed=42,
            replicate=1,
            candidate_fingerprint=parent.candidate_fingerprint,
        ),
        actor=_ACTOR,
    )
    # The run changed the budget, so plan against the record as it is now.
    (proposal,) = asyncio.run(w["planner"].propose(repo.planning_context(w["experiment"].id)))
    child = repo.materialize_candidate_proposal(w["experiment"].id, proposal, actor=_ACTOR)
    return {**w, "source": source, "child": child}


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
    w = _planned(repo, BudgetSpec(max_runs=2))
    # Another run takes the last one after branching: the next run is the one refused.
    repo.create_run(
        Run(
            id=RunId.generate(),
            node_id=w["node"].id,
            experiment_id=w["experiment"].id,
            seed=1,
            replicate=2,
            candidate_fingerprint=w["node"].candidate_fingerprint,
        ),
        actor=_ACTOR,
    )
    _refused(repo, BudgetExhaustedError, w["child"], *_first_run(w["child"], w["source"]))


def test_a_full_capacity_is_refused_with_nothing_written(repo) -> None:
    w = _planned(repo)
    holder = repo.transition_run(
        w["source"].id,
        expected_revision=w["source"].revision,
        new_status=RunStatus.ACTIVE,
        actor=_ACTOR,
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
        match = "whose seed is 42"
    else:
        run, attempt = _first_run(
            w["child"], w["source"], seed_origin=RunSeedOrigin(source_run_id=RunId.generate())
        )
        match = "not a run of node"
    error = _refused(repo, StorageError, w["child"], run, attempt)
    assert match in str(error)


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
