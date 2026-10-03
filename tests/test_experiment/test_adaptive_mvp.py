"""PR-026: submit one adaptive experiment, do nothing else, and it finishes (spec 18).

```text
node_A  LoRA 16   Run A (seed S)  A1 CUDA OOM → A2 restored, resized → 0.79 → BRANCH
   ↓ planner (recorded rule-based) → CandidateProposal LoRA 32 → node_B PLANNED
   ↓ automatic realization
node_B  LoRA 32   Run B (seed S, inherited from Run A)  B1 → 0.83 → STOP_SUCCEEDED
Experiment SUCCEEDED, best_node_id = node_B
```

Nobody calls anything between A's BRANCH and B's training. The loop is driven
from the record: a host that dies at either new resting point -- BRANCH with no
child yet, or a PLANNED child with no run -- is succeeded by one that carries on.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation_fixtures import EVALUATORS
from tests.test_experiment.adaptive_fixtures import (
    GROW_2,
    SEED,
    AdaptiveRuntime,
    LoRACompiler,
    adaptive_spec,
)
from tests.test_storage.test_planning_context import _rows
from xaytune.checkpoints import CheckpointManager, LocalCheckpointStore, SerializedStateCodec
from xaytune.core.domain.budget import BudgetDimension
from xaytune.core.domain.decision import DecisionOutcome
from xaytune.core.domain.incident import IncidentCategory
from xaytune.core.domain.objective import BudgetSpec
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.domain.planning import ActionProposal, CandidateBranchOrigin
from xaytune.core.domain.policy import PolicyVerdict
from xaytune.core.domain.recovery import RecoveryRequest
from xaytune.core.domain.specs import PlannerSpec
from xaytune.core.immutable import FrozenDict
from xaytune.core.state.status import (
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
)
from xaytune.decision import AdaptiveThresholdDecisionEngine
from xaytune.experiment import EmbeddedControllerHost, ReconciliationEscalatedError
from xaytune.planning import PLANNERS, _provenance_for, bind_planner
from xaytune.policy import RulePolicyEngine

_TIMEOUT = 30


def _world(tmp_path: Path, **options: Any) -> dict[str, Any]:
    manager = CheckpointManager(SerializedStateCodec(), LocalCheckpointStore(tmp_path / "bundles"))
    runtime = AdaptiveRuntime(
        manager, tmp_path, **{k: options.pop(k) for k in ("values", "oom_rank") if k in options}
    )
    return {"manager": manager, "runtime": runtime, **options}


def _host(tmp_path: Path, world: dict[str, Any], **overrides: Any) -> EmbeddedControllerHost:
    runtime = world["runtime"]
    options: dict[str, Any] = {
        "compilers": {"native": lambda: LoRACompiler(world.get("max_rank", 64))},
        "runtimes": {"local": lambda config: runtime},
        "evaluators": EVALUATORS,
        "decision_engine": AdaptiveThresholdDecisionEngine(),
        "policy": RulePolicyEngine(default=PolicyVerdict.ALLOW),
        "checkpoint_manager": world["manager"],
        "recovery_request_for_incident": lambda incident: RecoveryRequest(
            restore_context=runtime.restore_context
        ),
    }
    options.update(overrides)
    return EmbeddedControllerHost(tmp_path / "state.db", **options)


def _record(host: EmbeddedControllerHost, experiment_id: Any) -> dict[str, Any]:
    """Every node (submitted first), its runs, attempts and decisions, from the record."""
    aggregates = host.repository.aggregates
    nodes = sorted(
        aggregates.nodes_for_experiment(str(experiment_id)), key=lambda node: node.created_at
    )
    return {
        "experiment": aggregates.load_experiment(str(experiment_id)),
        "nodes": nodes,
        "runs": {node.id: aggregates.runs_for_node(str(node.id)) for node in nodes},
        "attempts": {
            run.id: aggregates.attempts_for_run(str(run.id))
            for node in nodes
            for run in aggregates.runs_for_node(str(node.id))
        },
        "decisions": {node.id: aggregates.decisions_for_node(str(node.id)) for node in nodes},
        "results": {
            node.id: aggregates.evaluation_results_for_node(str(node.id)) for node in nodes
        },
    }


def _rank(node: Any) -> int:
    adapter = node.candidate.candidate.training.adapter
    assert adapter is not None and adapter.rank is not None
    return adapter.rank


async def _submit_and_wait(host: EmbeddedControllerHost, spec: Any) -> Any:
    handle = await host.submit(spec)
    return handle, await asyncio.wait_for(handle.wait(), timeout=_TIMEOUT)


# ---- the exit criterion ------------------------------------------------------------------


def test_one_submission_runs_the_whole_adaptive_loop(tmp_path: Path) -> None:
    world = _world(tmp_path)

    async def scenario() -> None:
        host = _host(tmp_path, world)
        try:
            handle, result = await _submit_and_wait(host, adaptive_spec(tmp_path))
            record = _record(host, handle.experiment_id)
            _assert_spec_18(host, world["runtime"], result, record)
        finally:
            await host.close()

    asyncio.run(scenario())


def _assert_spec_18(host: Any, runtime: Any, result: Any, record: dict[str, Any]) -> None:
    # Experiment
    assert result.status is ExperimentStatus.SUCCEEDED
    assert result.next_stage is None and result.quiescent
    node_a, node_b = record["nodes"]
    assert record["experiment"].best_node_id == node_b.id
    assert {outcome.node_id for outcome in result.nodes} == {node_a.id, node_b.id}

    # Scientific lineage
    assert node_b.parent_ids == (node_a.id,)
    assert (_rank(node_a), _rank(node_b)) == (16, 32)
    assert node_a.candidate_fingerprint != node_b.candidate_fingerprint
    assert node_a.branch_origin is None
    origin = node_b.branch_origin
    assert isinstance(origin, CandidateBranchOrigin)
    assert origin.proposal_fingerprint
    assert origin.mutation["field"] == "training.adapter.rank"
    assert (origin.mutation["from"], origin.mutation["to"]) == (16, 32)

    # Planner
    provenance = origin.provenance
    planner = bind_planner(PlannerSpec(kind="rule-based", config=GROW_2))
    assert (provenance.planner_name, provenance.planner_version) == ("rule-based", "1.0.0")
    assert record["experiment"].planner == planner.spec
    assert provenance == _provenance_for(planner, provenance.context_fingerprint)
    (decision_a,) = record["decisions"][node_a.id]
    (result_a,) = record["results"][node_a.id]
    assert {(ref.kind, ref.id) for ref in origin.evidence_refs} == {
        ("decision", str(decision_a.id)),
        ("evaluation-result", str(result_a.id)),
    }

    # Run comparison: same seed, separate run, provenance of the seed
    (run_a,) = record["runs"][node_a.id]
    (run_b,) = record["runs"][node_b.id]
    assert run_a.id != run_b.id
    assert run_a.seed == run_b.seed == SEED
    assert run_a.seed_origin is None
    assert run_b.seed_origin is not None and run_b.seed_origin.source_run_id == run_a.id
    assert run_b.replicate == 1
    assert run_a.status is run_b.status is RunStatus.SUCCEEDED

    # Recovery: the same Run A, a new attempt -- never a hidden second run
    attempts_a = record["attempts"][run_a.id]
    assert len(attempts_a) == 2
    first, successor = attempts_a
    assert first.status is RunAttemptStatus.FAILED
    assert successor.status is RunAttemptStatus.SUCCEEDED
    incidents = host.repository.incidents.for_attempt(
        RuntimeOperationTarget(kind="training-attempt", id=str(first.id))
    )
    assert any(c.category is IncidentCategory.CUDA_OOM for i in incidents for c in i.candidates)
    assert successor.checkpoint_ref is not None
    assert runtime.restored == [successor.checkpoint_ref]
    assert successor.execution_overrides[-1].kind == "checkpoint_restore"
    a1, a2, b1 = runtime.training_plans
    before, after = a1.spec.config["optimization"], a2.spec.config["optimization"]
    assert (before["micro_batch_size"], before["gradient_accumulation"]) == (4, 8)
    assert (after["micro_batch_size"], after["gradient_accumulation"]) == (2, 16)
    assert (
        before["micro_batch_size"] * before["gradient_accumulation"]
        == after["micro_batch_size"] * after["gradient_accumulation"]
    ), "effective batch preserved"
    (b_attempt,) = record["attempts"][run_b.id]
    assert b_attempt.status is RunAttemptStatus.SUCCEEDED
    assert b1.spec.config["adapter_rank"] == 32

    # Evaluation and decision
    (decision_b,) = record["decisions"][node_b.id]
    (result_b,) = record["results"][node_b.id]
    assert [m.value for m in result_a.metrics] == [0.79]
    assert [m.value for m in result_b.metrics] == [0.83]
    assert decision_a.outcome is DecisionOutcome.BRANCH
    assert decision_b.outcome is DecisionOutcome.STOP_SUCCEEDED
    assert node_a.status is ExperimentNodeStatus.COMPLETED
    assert node_b.status is ExperimentNodeStatus.COMPLETED

    # Budget: two training runs, from the ledger; no GPU-hour dimension exists
    assert result.budget is not None
    runs = result.budget.of(BudgetDimension.RUNS)
    assert runs is not None
    assert (runs.limit, runs.consumed, runs.outstanding) == (4, 2, 0)
    assert [d.dimension for d in result.budget.dimensions] == [BudgetDimension.RUNS]


def test_the_loop_does_not_depend_on_anyone_waiting(tmp_path: Path) -> None:
    """Submitted and never waited on: the host still plans, branches and trains B."""
    world = _world(tmp_path)

    async def scenario() -> None:
        host = _host(tmp_path, world)
        try:
            handle = await host.submit(adaptive_spec(tmp_path))
            for _ in range(400):
                experiment = host.repository.aggregates.load_experiment(str(handle.experiment_id))
                if experiment.is_terminal:
                    break
                await asyncio.sleep(0.05)
            assert experiment.status is ExperimentStatus.SUCCEEDED
            record = _record(host, handle.experiment_id)
            assert experiment.best_node_id == record["nodes"][1].id
        finally:
            await host.close()

    asyncio.run(scenario())


# ---- record-driven: the two new resting points survive a restart -------------------------


async def _never(*args: Any, **kwargs: Any) -> None:
    return None


@pytest.mark.parametrize(
    ("stopped", "stage"),
    [("_continue_adaptive_experiment", "planning"), ("_realize_planned_candidate", "training")],
    ids=["crash-after-branch", "crash-after-planning"],
)
def test_a_new_host_carries_on_from_the_record(tmp_path: Path, stopped: str, stage: str) -> None:
    world = _world(tmp_path)

    async def scenario() -> None:
        first = _host(tmp_path, world)
        try:
            setattr(first, stopped, _never)
            handle, resting = await _submit_and_wait(first, adaptive_spec(tmp_path))
            assert resting.status is ExperimentStatus.ACTIVE
            assert resting.next_stage == stage
            experiment_id = handle.experiment_id
        finally:
            await first.close()

        second = _host(tmp_path, world)
        try:
            handle = await second.attach(experiment_id)
            result = await asyncio.wait_for(handle.wait(), timeout=_TIMEOUT)
            _assert_spec_18(second, world["runtime"], result, _record(second, experiment_id))
        finally:
            await second.close()

    asyncio.run(scenario())


def test_a_realized_run_never_issued_is_reconciled_not_recreated(tmp_path: Path) -> None:
    """Crash point C: Run B and its INTENDED submission are recorded; the runtime never heard."""
    world = _world(tmp_path)

    async def scenario() -> None:
        first = _host(tmp_path, world)
        issue = first._issue

        async def crash_before_issuing_b(experiment_id, run_id, attempt_id, operation, plan, rt):
            if plan.target.kind == "training-attempt" and plan.spec.config["adapter_rank"] == 32:
                return None
            return await issue(experiment_id, run_id, attempt_id, operation, plan, rt)

        try:
            first._issue = crash_before_issuing_b  # type: ignore[method-assign]
            handle = await first.submit(adaptive_spec(tmp_path))
            experiment_id = handle.experiment_id
            for _ in range(200):
                record = _record(first, experiment_id)
                if len(record["nodes"]) == 2 and record["runs"][record["nodes"][1].id]:
                    break
                await asyncio.sleep(0.05)
            node_b = record["nodes"][1]
            (run_b,) = record["runs"][node_b.id]
            (attempt_b,) = record["attempts"][run_b.id]
            (submission,) = first.repository.operations.for_target(
                "training-attempt", str(attempt_b.id)
            )
            assert node_b.status is ExperimentNodeStatus.ACTIVE
            assert run_b.status is RunStatus.ACTIVE
            assert submission.state == "intended"
        finally:
            await first.close()

        second = _host(tmp_path, world)
        try:
            handle = await second.attach(experiment_id)
            result = await asyncio.wait_for(handle.wait(), timeout=_TIMEOUT)
            record = _record(second, experiment_id)
            _assert_spec_18(second, world["runtime"], result, record)
            (rerun,) = record["runs"][node_b.id]
            assert rerun.id == run_b.id, "the recorded run, not another"
            assert [a.id for a in record["attempts"][run_b.id]] == [attempt_b.id]
        finally:
            await second.close()

    asyncio.run(scenario())


def test_two_hosts_continuing_at_once_realize_the_node_once(tmp_path: Path) -> None:
    world = _world(tmp_path)

    async def scenario() -> None:
        first = _host(tmp_path, world)
        try:
            first._realize_planned_candidate = _never  # type: ignore[method-assign]
            handle, resting = await _submit_and_wait(first, adaptive_spec(tmp_path))
            assert resting.next_stage == "training"
            experiment_id = handle.experiment_id
        finally:
            await first.close()

        one, two = _host(tmp_path, world), _host(tmp_path, world)
        try:
            await asyncio.gather(
                one._continue_adaptive_experiment(experiment_id),
                two._continue_adaptive_experiment(experiment_id),
            )
            record = _record(one, experiment_id)
            node_b = record["nodes"][1]
            (run_b,) = record["runs"][node_b.id]
            assert len(record["attempts"][run_b.id]) == 1
            assert len(record["nodes"]) == 2
            await asyncio.wait_for(
                asyncio.gather(
                    (await one.attach(experiment_id)).wait(),
                    (await two.attach(experiment_id)).wait(),
                ),
                timeout=_TIMEOUT,
            )
            assert len(_record(one, experiment_id)["runs"][node_b.id]) == 1
        finally:
            await one.close()
            await two.close()

    asyncio.run(scenario())


# ---- what the loop refuses to run -------------------------------------------------------


def _resting_at_training(tmp_path: Path, world: dict[str, Any]) -> Any:
    """Run to node_B PLANNED (no run), with realization held back; return the experiment id."""

    async def scenario() -> Any:
        host = _host(tmp_path, world)
        try:
            host._realize_planned_candidate = _never  # type: ignore[method-assign]
            handle, resting = await _submit_and_wait(host, adaptive_spec(tmp_path))
            assert resting.next_stage == "training"
            return handle.experiment_id
        finally:
            await host.close()

    return asyncio.run(scenario())


def _continue(tmp_path: Path, world: dict[str, Any], experiment_id: Any, prepare=None) -> Any:
    """One continuation by a fresh host; the rows before and after, and its escalation."""

    async def scenario() -> Any:
        host = _host(tmp_path, world)
        try:
            if prepare is not None:
                prepare(host)
            before = _rows(host.repository)
            await host._continue_adaptive_experiment(experiment_id)
            return (
                before,
                _rows(host.repository),
                host._escalations.get(str(experiment_id)),
                (_record(host, experiment_id)),
            )
        finally:
            await host.close()

    return asyncio.run(scenario())


def _relabel_child(host: Any, experiment_id: Any, origin: Any) -> None:
    """Rewrite node_B's recorded origin in place: a node branched some other way."""
    (node_b,) = [
        n
        for n in host.repository.aggregates.nodes_for_experiment(str(experiment_id))
        if n.status is ExperimentNodeStatus.PLANNED
    ]
    payload = node_b.model_dump(mode="json")
    payload["branch_origin"] = None if origin is None else origin.model_dump(mode="json")
    host.repository._connection.execute(
        "UPDATE experiment_nodes SET payload_json = ? WHERE id = ?",
        (json.dumps(payload), str(node_b.id)),
    )
    host.repository._connection.commit()


def test_a_planned_node_no_planner_branched_never_runs(tmp_path: Path) -> None:
    world = _world(tmp_path)
    experiment_id = _resting_at_training(tmp_path, world)
    before, after, escalation, record = _continue(
        tmp_path, world, experiment_id, lambda host: _relabel_child(host, experiment_id, None)
    )
    assert after == before, "no run, attempt, operation or reservation"
    assert escalation is None, "someone else's planned node is theirs to run, not an error"
    assert record["runs"][record["nodes"][1].id] == ()


def test_a_planned_node_from_another_planner_configuration_never_runs(tmp_path: Path) -> None:
    world = _world(tmp_path)
    experiment_id = _resting_at_training(tmp_path, world)
    other = bind_planner(
        PlannerSpec(
            kind="rule-based",
            config=FrozenDict(
                {"rules": [{"kind": "increase-lora-rank", "factor": 4, "max_rank": 64}]}
            ),
        )
    )

    def forge(host: Any) -> None:
        node_b = [
            n
            for n in host.repository.aggregates.nodes_for_experiment(str(experiment_id))
            if n.status is ExperimentNodeStatus.PLANNED
        ][0]
        assert node_b.branch_origin is not None
        forged = node_b.branch_origin.model_copy(
            update={
                "provenance": _provenance_for(
                    other, node_b.branch_origin.provenance.context_fingerprint
                )
            }
        )
        _relabel_child(host, experiment_id, forged)

    before, after, escalation, record = _continue(tmp_path, world, experiment_id, forge)
    assert after == before
    assert escalation is not None and "planner_spec_fingerprint" in escalation
    assert record["runs"][record["nodes"][1].id] == ()


def test_a_parent_with_two_runs_has_no_seed_to_inherit(tmp_path: Path) -> None:
    world = _world(tmp_path)
    experiment_id = _resting_at_training(tmp_path, world)

    def second_parent_run(host: Any) -> None:
        from xaytune.core.domain.run import Run
        from xaytune.core.ids import RunId
        from xaytune.core.refs import Actor

        node_a = sorted(
            host.repository.aggregates.nodes_for_experiment(str(experiment_id)),
            key=lambda n: n.created_at,
        )[0]
        host.repository.create_run(
            Run(
                id=RunId.generate(),
                node_id=node_a.id,
                experiment_id=node_a.experiment_id,
                seed=SEED + 1,
                replicate=2,
                candidate_fingerprint=node_a.candidate_fingerprint,
            ),
            actor=Actor(type="system", id="test"),
        )

    before, after, escalation, record = _continue(tmp_path, world, experiment_id, second_parent_run)
    assert after == before, "no run, attempt, operation or reservation for node_B"
    assert escalation is not None and "2 training runs" in escalation
    assert record["runs"][record["nodes"][1].id] == ()


def test_a_branched_candidate_the_compiler_cannot_run_is_refused_before_any_effect(
    tmp_path: Path,
) -> None:
    world = _world(tmp_path, max_rank=16)

    async def scenario() -> None:
        host = _host(tmp_path, world)
        try:
            handle = await host.submit(adaptive_spec(tmp_path))
            with pytest.raises(ReconciliationEscalatedError, match="cannot run branched node"):
                await asyncio.wait_for(handle.wait(), timeout=_TIMEOUT)
            record = _record(host, handle.experiment_id)
            node_a, node_b = record["nodes"]
            assert node_b.status is ExperimentNodeStatus.PLANNED, "planning was scientific intent"
            assert record["runs"][node_b.id] == ()
            assert len(world["runtime"].training_plans) == 2, "A1 and A2 only"
            runs = host.repository.budget_status(handle.experiment_id).of(BudgetDimension.RUNS)
            assert (runs.consumed, runs.outstanding) == (1, 0), "no reservation for B"
        finally:
            await host.close()

    asyncio.run(scenario())


def test_no_run_left_ends_the_experiment_budget_exhausted(tmp_path: Path) -> None:
    world = _world(tmp_path, oom_rank=None)

    async def scenario() -> None:
        host = _host(tmp_path, world)
        try:
            handle, result = await _submit_and_wait(
                host, adaptive_spec(tmp_path, budget=BudgetSpec(max_runs=1))
            )
            assert result.status is ExperimentStatus.BUDGET_EXHAUSTED
            assert result.next_stage is None
            record = _record(host, handle.experiment_id)
            (node_a,) = record["nodes"]
            (decision,) = record["decisions"][node_a.id]
            assert decision.outcome is DecisionOutcome.BRANCH, "the science stands; money ran out"
            assert node_a.status is ExperimentNodeStatus.COMPLETED
        finally:
            await host.close()

    asyncio.run(scenario())


def test_a_planner_with_nothing_to_propose_leaves_the_experiment_planning(tmp_path: Path) -> None:
    world = _world(tmp_path, oom_rank=None)
    capped = FrozenDict({"rules": [{"kind": "increase-lora-rank", "factor": 2, "max_rank": 16}]})

    async def scenario() -> None:
        host = _host(tmp_path, world)
        try:
            handle, result = await _submit_and_wait(
                host, adaptive_spec(tmp_path, planner_config=capped)
            )
            assert result.status is ExperimentStatus.ACTIVE
            assert result.next_stage == "planning"
            assert len(_record(host, handle.experiment_id)["nodes"]) == 1
        finally:
            await host.close()

    asyncio.run(scenario())


class _Wrapped:
    """The bound rule-based planner, its proposals rewritten by *rewrite*."""

    def __init__(self, inner: Any, rewrite: Any) -> None:
        self.inner, self.rewrite = inner, rewrite
        self.descriptor, self.spec = inner.descriptor, inner.spec

    def capabilities(self) -> Any:
        return self.inner.capabilities()

    async def propose(self, context: Any) -> Any:
        return self.rewrite(await self.inner.propose(context), context)


def _action(proposals: Any, context: Any) -> Any:
    from xaytune.core.domain.action import ActionTarget
    from xaytune.core.domain.actions import RejectCandidate

    (proposal,) = proposals
    return (
        ActionProposal(
            action=RejectCandidate(
                target=ActionTarget(kind="node", id=str(proposal.parent_ids[0]))
            ),
            reason="r",
            provenance=proposal.provenance,
        ),
    )


@pytest.mark.parametrize(
    ("rewrite", "match"),
    [
        (lambda proposals, context: proposals + proposals, "proposed 2 things"),
        (_action, "does not execute action proposals"),
    ],
    ids=["two-proposals", "action-proposal"],
)
def test_planner_output_without_a_selection_policy_is_escalated(
    tmp_path: Path, rewrite: Any, match: str
) -> None:
    world = _world(tmp_path, oom_rank=None)
    planners = {
        **PLANNERS,
        "rule-based": lambda spec: _Wrapped(PLANNERS["rule-based"](spec), rewrite),
    }

    async def scenario() -> None:
        host = _host(tmp_path, world, planners=planners)
        try:
            handle = await host.submit(adaptive_spec(tmp_path))
            with pytest.raises(ReconciliationEscalatedError, match=match):
                await asyncio.wait_for(handle.wait(), timeout=_TIMEOUT)
            record = _record(host, handle.experiment_id)
            assert len(record["nodes"]) == 1, "nothing chosen, nothing branched"
            assert host.repository.actions.for_target("node", str(record["nodes"][0].id)) == ()
        finally:
            await host.close()

    asyncio.run(scenario())
