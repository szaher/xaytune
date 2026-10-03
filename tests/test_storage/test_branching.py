"""PR-025: a candidate proposal becomes a PLANNED child node -- or nothing is written.

```text
node_A COMPLETED (BRANCH, task_success 0.79, LoRA 16)
   ↓ planning_context → RuleBasedPlanner → CandidateProposal (LoRA 32)
   ↓ materialize_candidate_proposal   (one commit)
node_B PLANNED, parent node_A, branch_origin = the proposal
```

No run, attempt, operation, action or ledger entry: realizing node_B is PR-026's.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import pytest

from tests.test_storage.conftest import make_experiment, make_node
from tests.test_storage.test_evaluation_lifecycle import _ACTOR
from tests.test_storage.test_evaluation_lifecycle import repo as repo  # noqa: F401 (fixture)
from tests.test_storage.test_planning_context import _evaluated, _rows
from xaytune.core.domain.budget import BudgetDimension, BudgetSubjectKind, LedgerEntryKind
from xaytune.core.domain.objective import BudgetSpec
from xaytune.core.domain.planning import (
    CandidateProposal,
    EvidenceRef,
    candidate_proposal_identity_v1,
)
from xaytune.core.domain.specs import PlannerSpec
from xaytune.core.ids import ExperimentNodeId
from xaytune.core.immutable import FrozenDict
from xaytune.core.state.status import ExperimentNodeStatus, ExperimentStatus
from xaytune.planning import RuleBasedPlanner, _provenance_for, bind_planner
from xaytune.storage import write_transaction
from xaytune.storage.control_plane import (
    BranchRefusedError,
    CandidateConflictError,
    ControlPlaneRepository,
    ProvenanceError,
    StaleProposalError,
)
from xaytune.storage.graph import LineageError

GROW_2 = FrozenDict({"rules": [{"kind": "increase-lora-rank", "factor": 2, "max_rank": 64}]})
GROW_4 = FrozenDict({"rules": [{"kind": "increase-lora-rank", "factor": 4, "max_rank": 64}]})


def _bound(config: FrozenDict = GROW_2) -> RuleBasedPlanner:
    planner = bind_planner(PlannerSpec(kind="rule-based", config=config))
    assert isinstance(planner, RuleBasedPlanner)
    return planner


def _world(repo: ControlPlaneRepository, **options: Any) -> dict[str, Any]:
    """node_A decided BRANCH under a recorded rule-based planner, and its proposal."""
    planner = _bound(options.pop("config", GROW_2))
    experiment, node, result, decision = _evaluated(
        repo, rank=16, value=0.79, planner=planner.spec, **options
    )
    context = repo.planning_context(experiment.id)
    (proposal,) = asyncio.run(planner.propose(context))
    assert isinstance(proposal, CandidateProposal)
    return {
        "experiment": experiment,
        "node": node,
        "result": result,
        "decision": decision,
        "planner": planner,
        "context": context,
        "proposal": proposal,
    }


def _branch(repo: ControlPlaneRepository, w: dict[str, Any], proposal: Any = None) -> Any:
    return repo.materialize_candidate_proposal(
        w["experiment"].id, proposal or w["proposal"], actor=_ACTOR
    )


def _refused(repo: ControlPlaneRepository, w: dict[str, Any], error: Any, proposal: Any = None):
    """*proposal* is refused with *error* and not one row anywhere changes."""
    before = _rows(repo)
    with pytest.raises(error) as refused:
        _branch(repo, w, proposal)
    assert _rows(repo) == before, "a refusal writes nothing"
    return refused.value


def _rehashed(proposal: CandidateProposal, **provenance: Any) -> CandidateProposal:
    return proposal.model_copy(
        update={"provenance": proposal.provenance.model_copy(update=provenance)}
    )


# ---- the exit criterion ------------------------------------------------------------------


def test_a_proposal_becomes_a_planned_child_and_nothing_runs(repo) -> None:
    w = _world(repo)
    before = _rows(repo)

    node_b = _branch(repo, w)

    assert node_b.status is ExperimentNodeStatus.PLANNED
    assert node_b.parent_ids == (w["node"].id,)
    adapter = node_b.candidate.candidate.training.adapter
    assert adapter is not None and adapter.rank == 32
    assert node_b.candidate.candidate == w["proposal"].candidate, "stored exactly as proposed"
    assert node_b.candidate_fingerprint == w["proposal"].candidate_fingerprint
    assert (node_b.hypothesis, node_b.reason) == (w["proposal"].hypothesis, w["proposal"].reason)
    assert node_b.created_by == _ACTOR

    origin = node_b.branch_origin
    assert origin is not None
    assert origin.proposal_fingerprint == w["proposal"].proposal_fingerprint()
    assert origin.provenance == w["proposal"].provenance
    assert origin.provenance.context_fingerprint == w["context"].input_fingerprint()
    assert dict(origin.mutation) == {
        "rule": "increase-lora-rank",
        "field": "training.adapter.rank",
        "from": 16,
        "to": 32,
    }
    assert origin.evidence_refs == (
        EvidenceRef(kind="decision", id=str(w["decision"].id)),
        EvidenceRef(kind="evaluation-result", id=str(w["result"].id)),
    )
    assert repo.aggregates.load_node(str(node_b.id)) == node_b
    assert [p.id for p in repo.graph.parents(str(node_b.id))] == [w["node"].id]

    after = _rows(repo)
    changed = {
        table: after[table] - before[table] for table in after if after[table] != before[table]
    }
    assert set(changed) <= {"experiment_nodes", "experiment_edges", "events", "outbox"}
    assert (changed["experiment_nodes"], changed["experiment_edges"], changed["events"]) == (
        1,
        1,
        2,
    )
    events = repo.events.events_for_experiment(str(w["experiment"].id))
    assert [e.event_type for e in events[-2:]] == ["NodeCreated", "ExperimentNodeStatusChanged"]
    assert events[-2].payload["proposal_fingerprint"] == origin.proposal_fingerprint
    assert repo.aggregates.runs_for_node(str(node_b.id)) == ()
    assert repo.aggregates.load_experiment(str(w["experiment"].id)).status is (
        ExperimentStatus.ACTIVE
    )


def test_the_proposal_identity_is_explicit_and_versioned(repo) -> None:
    w = _world(repo)
    identity = candidate_proposal_identity_v1(w["proposal"])
    assert (identity["kind"], identity["identity_version"]) == ("candidate-proposal", 1)
    assert set(identity) == {
        "kind",
        "identity_version",
        "candidate_fingerprint",
        "candidate",
        "parent_ids",
        "hypothesis",
        "reason",
        "mutation",
        "evidence_refs",
        "provenance",
    }
    assert set(identity["provenance"]) == set(type(w["proposal"].provenance).model_fields)
    reworded = w["proposal"].model_copy(update={"reason": "another reason"})
    assert reworded.proposal_fingerprint() != w["proposal"].proposal_fingerprint()


# ---- idempotency and duplicates ----------------------------------------------------------


def test_the_same_proposal_twice_returns_the_same_node_and_writes_nothing(repo) -> None:
    w = _world(repo)
    first = _branch(repo, w)
    before = _rows(repo)
    assert _branch(repo, w) == first
    assert _rows(repo) == before


def test_another_proposal_for_an_existing_candidate_is_a_conflict(repo) -> None:
    w = _world(repo)
    _branch(repo, w)
    reworded = w["proposal"].model_copy(update={"hypothesis": "Capacity, phrased otherwise."})
    _refused(repo, w, CandidateConflictError, reworded)


def test_a_proposal_for_a_submitted_candidate_is_a_conflict(repo) -> None:
    """node_A came from no proposal; proposing its candidate again is not a new node."""
    w = _world(repo)
    same_as_a = w["node"].candidate.candidate
    proposal = w["proposal"].model_copy(
        update={"candidate": same_as_a, "candidate_fingerprint": same_as_a.candidate_fingerprint()}
    )
    assert "no proposal" in str(_refused(repo, w, CandidateConflictError, proposal))


# ---- staleness ---------------------------------------------------------------------------


def test_a_new_node_since_planning_makes_the_proposal_stale(repo) -> None:
    w = _world(repo)
    repo.create_node(make_node(w["experiment"], fingerprint="other"), actor=_ACTOR)
    _refused(repo, w, StaleProposalError)


def test_a_budget_balance_change_since_planning_makes_the_proposal_stale(repo) -> None:
    w = _world(repo, budget=BudgetSpec(max_runs=4))
    with write_transaction(repo._connection):
        repo._ledger(
            str(w["experiment"].id),
            BudgetDimension.RUNS,
            LedgerEntryKind.RESERVE,
            Decimal(1),
            BudgetSubjectKind.RUN,
            "run_spent_elsewhere",
            _ACTOR,
            (),
        )
    _refused(repo, w, StaleProposalError)


def test_a_proposal_claiming_another_context_is_stale(repo) -> None:
    w = _world(repo)
    _refused(repo, w, StaleProposalError, _rehashed(w["proposal"], context_fingerprint="sha256:0"))


# ---- planner provenance ------------------------------------------------------------------


def test_a_proposal_from_another_configuration_is_refused(repo) -> None:
    """Recorded factor=2; a factor=4 planner's proposal is refused even for the same candidate."""
    w = _world(repo)
    other = _bound(GROW_4)
    forged = w["proposal"].model_copy(
        update={"provenance": _provenance_for(other, w["context"].input_fingerprint())}
    )
    assert "configured differently" in str(_refused(repo, w, ProvenanceError, forged))


@pytest.mark.parametrize(
    "change",
    [
        {"planner_name": "llm", "planner_spec_kind": "llm"},
        {"planner_version": "9.9.9", "planner_spec_version": "9.9.9"},
        {"planner_spec_identity_version": 2},
        {"planner_api_version": "xaytune.plugins/v9"},
    ],
    ids=["kind", "version", "spec-identity-version", "api-version"],
)
def test_a_proposal_naming_another_planner_is_refused(repo, change) -> None:
    w = _world(repo)
    _refused(repo, w, ProvenanceError, _rehashed(w["proposal"], **change))


def test_an_experiment_without_a_recorded_planner_cannot_branch(repo) -> None:
    planner = _bound()
    experiment, *_ = _evaluated(repo, rank=16, value=0.79)
    context = repo.planning_context(experiment.id)
    (proposal,) = asyncio.run(planner.propose(context))
    with pytest.raises(ProvenanceError, match="records no planner"):
        repo.materialize_candidate_proposal(experiment.id, proposal, actor=_ACTOR)


# ---- the candidate, the lineage and the evidence -----------------------------------------


def test_a_candidate_fingerprint_that_does_not_describe_the_candidate_is_refused(repo) -> None:
    w = _world(repo)
    forged = CandidateProposal.model_construct(
        **{**dict(w["proposal"]), "candidate_fingerprint": "sha256:forged"}
    )
    _refused(repo, w, ProvenanceError, forged)


def _without_evidence(w: dict[str, Any], **changes: Any) -> CandidateProposal:
    return w["proposal"].model_copy(update={"evidence_refs": (), **changes})


def test_a_missing_parent_is_refused(repo) -> None:
    w = _world(repo)
    _refused(
        repo,
        w,
        LineageError,
        _without_evidence(w, parent_ids=(ExperimentNodeId.generate(),)),
    )


def test_a_parent_from_another_experiment_is_refused(repo) -> None:
    w = _world(repo)
    elsewhere = repo.create_experiment(make_experiment("elsewhere"), actor=_ACTOR)
    foreign = repo.create_node(make_node(elsewhere), actor=_ACTOR)
    _refused(repo, w, LineageError, _without_evidence(w, parent_ids=(foreign.id,)))


def test_a_parent_named_twice_is_refused(repo) -> None:
    w = _world(repo)
    a = w["node"].id
    _refused(repo, w, LineageError, w["proposal"].model_copy(update={"parent_ids": (a, a)}))


@pytest.mark.parametrize("kind", ["decision", "evaluation-result"])
def test_evidence_that_is_not_the_parents_is_refused(repo, kind) -> None:
    w = _world(repo)
    other = _evaluated(repo, rank=8, value=0.5)  # another experiment's decision and result
    foreign = str(other[3].id) if kind == "decision" else str(other[2].id)
    for ref_id in (foreign, "does-not-exist"):
        proposal = w["proposal"].model_copy(
            update={"evidence_refs": (EvidenceRef(kind=kind, id=ref_id),)}
        )
        _refused(repo, w, ProvenanceError, proposal)


# ---- the experiment's state and budget ---------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [ExperimentStatus.PAUSED, ExperimentStatus.CANCELLED, ExperimentStatus.FAILED],
)
def test_only_an_active_experiment_branches(repo, status) -> None:
    w = _world(repo)
    experiment = repo.aggregates.load_experiment(str(w["experiment"].id))
    repo.transition_experiment(
        experiment.id, expected_revision=experiment.revision, new_status=status, actor=_ACTOR
    )
    _refused(repo, w, BranchRefusedError)


def test_an_exhausted_quota_refuses_the_branch_and_no_quota_is_reserved(repo) -> None:
    w = _world(repo, budget=BudgetSpec(max_runs=1))
    with write_transaction(repo._connection):
        repo._ledger(
            str(w["experiment"].id),
            BudgetDimension.RUNS,
            LedgerEntryKind.RESERVE,
            Decimal(1),
            BudgetSubjectKind.RUN,
            "run_spent",
            _ACTOR,
            (),
        )
    exhausted = repo.planning_context(w["experiment"].id)
    assert exhausted.budget is not None and exhausted.budget.exhausted
    # A proposal made against the exhausted context itself: fresh, yet refused.
    fresh = _rehashed(
        w["proposal"],
        context_fingerprint=exhausted.input_fingerprint(),
    )
    assert "exhausted quota" in str(_refused(repo, w, BranchRefusedError, fresh))


def test_branching_reserves_no_run(repo) -> None:
    w = _world(repo, budget=BudgetSpec(max_runs=4))
    entries = repo.budget.entries(str(w["experiment"].id))
    _branch(repo, w)
    assert repo.budget.entries(str(w["experiment"].id)) == entries


# ---- backward compatibility --------------------------------------------------------------


def test_a_node_recorded_before_branching_loads_without_an_origin(repo) -> None:
    experiment = repo.create_experiment(make_experiment(), actor=_ACTOR)
    node = repo.create_node(make_node(experiment), actor=_ACTOR)
    payload = node.model_dump(mode="json")
    payload.pop("branch_origin")
    from xaytune.core.domain.experiment import ExperimentNode

    assert ExperimentNode.model_validate(payload).branch_origin is None
    assert repo.aggregates.load_node(str(node.id)).branch_origin is None
