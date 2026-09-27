"""Proposing actions through the public API (PR-023).

```text
handle.propose(spec)      validated, judged by the host's policy, recorded -- never applied
host.approve_action(...)  a human approves what policy judged
no policy configured      every proposal denied, with a durable decision
cancel-* specs            refused: cancellation has its own path
```
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation_fixtures import EVALUATORS, evaluation
from tests.test_experiment.test_restart_reconciliation import _spec
from xaytune.core.domain.action import ActionStatus, ActionTarget
from xaytune.core.domain.actions import (
    CancelExperiment,
    ChangeWorkerCount,
    MutationClass,
    RejectCandidate,
)
from xaytune.core.refs import Actor
from xaytune.core.state.status import ExperimentNodeStatus, ExperimentStatus
from xaytune.experiment import EmbeddedControllerHost, PolicyVerdict
from xaytune.policy import PolicyRule, RulePolicyEngine
from xaytune.storage.control_plane import CancellationNotGovernedError

AGENT = Actor(type="llm_agent", id="planner")
ANA = Actor(type="human", id="ana")

REVIEWED = RulePolicyEngine(
    [
        PolicyRule(
            id="judgements-reviewed",
            verdict=PolicyVerdict.REQUIRE_APPROVAL,
            reason="a human signs off on judging a candidate",
            mutation_classes=(MutationClass.EXPERIMENT,),
        )
    ],
    default=PolicyVerdict.ALLOW,
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")


def _deciding(tmp_path: Path, scenario: Any, **host_options: Any) -> Any:
    """Train and evaluate one candidate, left DECIDING, then run *scenario* on the open host."""

    async def run() -> Any:
        host = EmbeddedControllerHost(tmp_path / "state.db", evaluators=EVALUATORS, **host_options)
        try:
            spec = _spec(tmp_path).model_copy(update={"evaluation": evaluation(value=0.8)})
            handle = await host.submit(spec)
            result = await asyncio.wait_for(handle.wait(), timeout=180)
            (node,) = result.nodes
            assert node.status is ExperimentNodeStatus.DECIDING
            return await scenario(host, handle, node)
        finally:
            await host.close()

    return asyncio.run(run())


def test_with_no_policy_configured_a_proposal_is_denied_and_recorded(tmp_path: Path) -> None:
    async def scenario(host: Any, handle: Any, node: Any) -> Any:
        target = ActionTarget(kind="node", id=str(node.node_id))
        governed = await handle.propose(
            RejectCandidate(target=target), reason="off target", proposed_by=AGENT
        )
        return governed, await handle.actions(), await handle.wait()

    governed, actions, result = _deciding(tmp_path, scenario)

    assert governed.action.status is ActionStatus.REJECTED
    assert governed.decision.engine_name == "deny-all"
    assert actions == (governed,)
    assert result.next_stage == "decision", "a rejected action leaves nothing to wait on"


def test_a_human_approves_and_nothing_is_carried_out(tmp_path: Path) -> None:
    async def scenario(host: Any, handle: Any, node: Any) -> Any:
        target = ActionTarget(kind="node", id=str(node.node_id))
        governed = await handle.propose(
            RejectCandidate(target=target), reason="off target", proposed_by=AGENT
        )
        assert governed.action.status is ActionStatus.APPROVAL_PENDING
        awaiting = await handle.wait()
        approved = await host.approve_action(
            governed.action.id, approver=ANA, reason="agreed, it is off target"
        )
        result = await handle.wait()
        return awaiting, approved, result, await handle.status()

    awaiting, approved, result, status = _deciding(tmp_path, scenario, policy=REVIEWED)

    assert awaiting.quiescent, "waiting for a human is not a controller working"
    assert awaiting.next_stage == "action-approval", "the approval, not another decision"
    assert result.quiescent and result.next_stage == "action-execution"
    assert approved.action.status is ActionStatus.APPROVED
    assert approved.decision.verdict is PolicyVerdict.REQUIRE_APPROVAL
    (node,) = result.nodes
    assert node.status is ExperimentNodeStatus.DECIDING, "approved is not applied"
    assert status is ExperimentStatus.ACTIVE


def test_a_worker_count_the_local_runtime_cannot_change_is_rejected_before_policy(
    tmp_path: Path,
) -> None:
    async def scenario(host: Any, handle: Any, node: Any) -> Any:
        (run,) = host.repository.aggregates.runs_for_node(str(node.node_id))
        return await handle.propose(
            ChangeWorkerCount(target=ActionTarget(kind="run", id=str(run.id)), workers=2),
            reason="more throughput",
            proposed_by=AGENT,
        )

    governed = _deciding(tmp_path, scenario, policy=REVIEWED)

    assert governed.action.status is ActionStatus.REJECTED
    assert governed.decision is None
    assert any("has ended" in p for p in governed.problems), "trained, so no next attempt"
    assert "the runtime does not declare whether its worker count can change" in (
        governed.problems
    ), "the local runtime declares no elasticity: fail closed"


def test_cancellation_is_not_proposed(tmp_path: Path) -> None:
    async def scenario(host: Any, handle: Any, node: Any) -> Any:
        target = ActionTarget(kind="experiment", id=str(handle.experiment_id))
        with pytest.raises(CancellationNotGovernedError, match="ExperimentHandle.cancel"):
            await handle.propose(CancelExperiment(target=target), reason="x", proposed_by=ANA)
        return await handle.actions()

    assert _deciding(tmp_path, scenario, policy=REVIEWED) == ()
