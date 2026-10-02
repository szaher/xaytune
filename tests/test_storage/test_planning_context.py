"""The planning context, projected from the durable record -- and nothing written.

The PR-024 exit criterion, against a real repository:

```text
node_A  COMPLETED after BRANCH, task_success = 0.79, LoRA rank 16
   ↓ repository.planning_context()  (read-only)
   ↓ RuleBasedPlanner(increase-lora-rank ×2, max 64)
CandidateProposal: parent node_A, LoRA rank 32, novel, with provenance
```

and no node, run, action, decision, ledger entry or event is written.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tests.test_planning.test_rule_based_planner import candidate
from tests.test_storage.conftest import make_experiment
from tests.test_storage.test_evaluation_lifecycle import (
    _ACTOR,
    _attempt,
    _begin,
    _metric,
    _result,
    _run,
    _running,
)
from tests.test_storage.test_evaluation_lifecycle import repo as repo  # noqa: F401 (fixture)
from xaytune.core.domain.decision import DecisionContext, DecisionOutcome
from xaytune.core.domain.experiment import CandidateSpecSnapshot, ExperimentNode
from xaytune.core.domain.planning import CandidateProposal
from xaytune.core.domain.specs import PlannerSpec
from xaytune.core.ids import ExperimentNodeId
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import Actor
from xaytune.core.state.status import ExperimentNodeStatus, ExperimentStatus
from xaytune.decision import AdaptiveThresholdDecisionEngine
from xaytune.planning import bind_planner
from xaytune.storage.control_plane import ControlPlaneRepository, EvaluationReconciliation


def _evaluated(repo: ControlPlaneRepository, *, rank: int, value: float) -> Any:
    """An ACTIVE experiment with one LoRA node, evaluated at *value* and decided adaptively."""
    experiment = repo.create_experiment(make_experiment(), actor=_ACTOR)
    repo.transition_experiment(
        experiment.id, expected_revision=0, new_status=ExperimentStatus.ACTIVE, actor=_ACTOR
    )
    spec = candidate(rank)
    node = repo.create_node(
        ExperimentNode(
            id=ExperimentNodeId.generate(),
            experiment_id=experiment.id,
            candidate=CandidateSpecSnapshot(candidate=spec),
            candidate_fingerprint=spec.candidate_fingerprint(),
            created_by=Actor(type="system", id="controller"),
        ),
        actor=_ACTOR,
    )
    for status in (
        ExperimentNodeStatus.PLANNED,
        ExperimentNodeStatus.READY,
        ExperimentNodeStatus.ACTIVE,
    ):
        node = repo.transition_node(
            node.id, expected_revision=node.revision, new_status=status, actor=_ACTOR
        )
    run = _run(node)
    _begin(repo, node, run)
    attempt = _running(repo, _attempt(repo, run)[0])
    result = repo.record_evaluation_result(
        attempt.id,
        _result(run, metrics=(_metric(run, name="task_success", value=value),)),
        expected_revision=attempt.revision,
        actor=_ACTOR,
    )
    assert (
        repo.reconcile_evaluating_node(node.id, actor=_ACTOR) is EvaluationReconciliation.DECIDING
    )
    deciding = repo.aggregates.load_node(str(node.id))
    proposal = AdaptiveThresholdDecisionEngine().decide(
        DecisionContext(
            experiment_id=experiment.id,
            node_id=node.id,
            evaluation_cycle=deciding.evaluation_cycle,
            objective=repo.aggregates.load_experiment(str(experiment.id)).objective,
            results=(result,),
        )
    )
    decision = repo.record_decision(
        proposal, expected_node_revision=deciding.revision, actor=_ACTOR
    )
    return experiment, repo.aggregates.load_node(str(node.id)), result, decision


def _rows(repo: ControlPlaneRepository) -> dict[str, int]:
    tables = [
        row[0]
        for row in repo._connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    return {
        table: repo._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in tables
    }


def test_the_context_is_the_record_projected(repo: ControlPlaneRepository) -> None:
    experiment, node, result, decision = _evaluated(repo, rank=16, value=0.79)
    assert decision.outcome is DecisionOutcome.BRANCH

    context = repo.planning_context(experiment.id)

    assert (context.experiment_id, context.experiment_status) == (
        experiment.id,
        ExperimentStatus.ACTIVE,
    )
    assert context.objective == repo.aggregates.load_experiment(str(experiment.id)).objective
    (summary,) = context.nodes
    assert (summary.node_id, summary.status) == (node.id, ExperimentNodeStatus.COMPLETED)
    assert summary.candidate == node.candidate.candidate
    assert summary.candidate_fingerprint == node.candidate.candidate.candidate_fingerprint()
    (decided,) = summary.decisions
    assert (decided.decision_id, decided.outcome) == (decision.id, DecisionOutcome.BRANCH)
    assert decided.evaluation_result_ids == (result.id,)
    (evaluation,) = summary.evaluations
    assert (evaluation.evaluation_result_id, evaluation.evaluation_cycle) == (result.id, 1)
    assert [(m.name, m.value) for m in evaluation.metrics] == [("task_success", 0.79)]
    assert context.budget is None
    assert repo.planning_context(experiment.id) == context, "the same record, the same context"


def test_the_exit_criterion_proposes_lora_32_and_writes_nothing(
    repo: ControlPlaneRepository,
) -> None:
    experiment, node, result, decision = _evaluated(repo, rank=16, value=0.79)
    before = _rows(repo)

    context = repo.planning_context(experiment.id)
    planner = bind_planner(
        PlannerSpec(
            kind="rule-based",
            config=FrozenDict(
                {"rules": [{"kind": "increase-lora-rank", "factor": 2, "max_rank": 64}]}
            ),
        )
    )
    (proposal,) = asyncio.run(planner.propose(context))

    assert isinstance(proposal, CandidateProposal)
    assert proposal.parent_ids == (node.id,)
    adapter = proposal.candidate.training.adapter
    assert adapter is not None and adapter.rank == 32
    assert proposal.candidate_fingerprint not in {
        n.candidate_fingerprint for n in repo.aggregates.nodes_for_experiment(str(experiment.id))
    }
    assert proposal.provenance.context_fingerprint == context.input_fingerprint()
    assert proposal.evidence_refs == (f"decision:{decision.id}", f"evaluation-result:{result.id}")

    assert _rows(repo) == before, "planning wrote nothing"
    (only,) = repo.aggregates.nodes_for_experiment(str(experiment.id))
    assert only.id == node.id
    assert repo.aggregates.load_experiment(str(experiment.id)).status is ExperimentStatus.ACTIVE
