"""Cancelling through the handle: intent, effect, observation, and only then CANCELLED.

```text
handle.cancel()
   ↓ one commit
cancel-experiment Action  ──parent──  cancel-attempt Action → cancel operation
   ↓
runtime.cancel(), operation CONFIRMED
   ↓ the controller observes the workload stop
attempt CANCELLED → child APPLIED → run, node, experiment CANCELLED → parent APPLIED
```

ADR-013 §6's invariant is the thing under test: ``CANCELLED`` means Xaytune
believes no owned workload is executing. So the experiment must not get there
before the attempt does, and a cancellation that arrives after training ended
must not rewrite what the run did.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.training_fixtures import sft_candidate, tiny_dataset, tiny_model
from xaytune.core.domain.objective import Objective, ObjectiveMetric
from xaytune.core.state.status import (
    ActionStatus,
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")


def _spec(tmp_path: Path):
    from xaytune.experiment import CompilerSpec, ExperimentSpec, RuntimeSpec

    return ExperimentSpec(
        name="tiny-sft",
        objective=Objective(primary=ObjectiveMetric(name="loss", direction="minimize")),
        candidate=sft_candidate(
            tiny_model(tmp_path / "model"),
            tiny_dataset(tmp_path / "data" / "train.jsonl", "text"),
            "text",
        ),
        seed=7,
        compiler=CompilerSpec(name="native"),
        runtime=RuntimeSpec(kind="local", config={"root": str(tmp_path / "runtime")}),
        artifact_root=str(tmp_path / "artifacts"),
    )


def _actions(host, experiment_id):
    repo = host.repository
    (parent,) = [
        a
        for a in repo.actions.for_target("experiment", str(experiment_id))
        if a.type == "cancel-experiment"
    ]
    return parent, repo.actions.children(str(parent.id))


def test_cancelling_a_running_experiment_stops_its_workload_first(tmp_path: Path) -> None:
    from xaytune.experiment import EmbeddedControllerHost

    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            handle = await host.submit(_spec(tmp_path))
            await handle.cancel()
            result = await asyncio.wait_for(handle.wait(), timeout=120)
            parent, children = _actions(host, handle.experiment_id)
            events = host.repository.events.events_for_experiment(str(handle.experiment_id))
            (attempt,) = [
                a
                for node in host.repository.aggregates.nodes_for_experiment(
                    str(handle.experiment_id)
                )
                for run in host.repository.aggregates.runs_for_node(str(node.id))
                for a in host.repository.aggregates.attempts_for_run(str(run.id))
            ]
            (reference,) = [
                op.runtime_ref
                for op in host.repository.operations.for_target("training-attempt", str(attempt.id))
                if op.type == "submit"
            ]
            runtime = host._runtime(
                host.repository.aggregates.load_experiment(str(handle.experiment_id)).runtime
            )
            workload = await runtime.get_status(reference)
            cancel_ops = [
                op
                for op in host.repository.operations.for_target("training-attempt", str(attempt.id))
                if op.type == "cancel"
            ]
            return result, parent, children, events, workload, cancel_ops
        finally:
            await host.close()

    result, parent, children, events, workload, cancel_ops = asyncio.run(scenario())

    assert workload.state == "cancelled", "the workload itself stopped, not just the record"

    assert result.status is ExperimentStatus.CANCELLED
    assert result.next_stage is None
    (node,) = result.nodes
    assert node.status is ExperimentNodeStatus.CANCELLED
    (run,) = node.runs
    assert (run.status, run.attempt_status) == (RunStatus.CANCELLED, RunAttemptStatus.CANCELLED)

    assert (parent.status, parent.outcome.value) == (ActionStatus.SUCCEEDED, "applied")
    (child,) = children
    assert child.parent_action_id == parent.id
    assert (child.status, child.outcome.value) == (ActionStatus.SUCCEEDED, "applied")
    (cancel,) = cancel_ops
    assert (cancel.state, cancel.caused_by_action_id) == ("confirmed", child.id)

    # The invariant, in the order the record wrote it: the attempt stopped,
    # and only then was the experiment called cancelled.
    def position(aggregate_type: str, status: str) -> int:
        return next(
            i
            for i, e in enumerate(events)
            if e.aggregate_type == aggregate_type and e.payload.get("status") == status
        )

    assert position("RunAttempt", "cancelled") < position("Experiment", "cancelled")
    requested = next(i for i, e in enumerate(events) if e.event_type == "CancellationRequested")
    assert not any(
        e.aggregate_type == "Experiment" and e.payload.get("status") != "active"
        for e in events[requested : position("Experiment", "cancelled")]
    ), "the experiment stays ACTIVE while cancellation propagates"


def test_cancelling_twice_records_one_intent(tmp_path: Path) -> None:
    from xaytune.experiment import EmbeddedControllerHost

    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            handle = await host.submit(_spec(tmp_path))
            await handle.cancel()
            await handle.cancel()
            await asyncio.wait_for(handle.wait(), timeout=120)
            actions = host.repository.actions.for_target("experiment", str(handle.experiment_id))
            return [a for a in actions if a.type == "cancel-experiment"]
        finally:
            await host.close()

    assert len(asyncio.run(scenario())) == 1


def test_cancelling_after_training_does_not_rewrite_the_run(tmp_path: Path) -> None:
    """The run succeeded; cancelling the experiment stops what is left, which is nothing.

    No attempt is live, so no effect is issued and the experiment is cancelled
    at once. The run keeps SUCCEEDED and its artifact: cancellation stops work,
    it does not revise history.
    """
    from xaytune.experiment import EmbeddedControllerHost

    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            handle = await host.submit(_spec(tmp_path))
            await asyncio.wait_for(handle.wait(), timeout=180)
            await handle.cancel()
            result = await handle.wait()
            return result, *_actions(host, handle.experiment_id)
        finally:
            await host.close()

    result, parent, children = asyncio.run(scenario())

    assert result.status is ExperimentStatus.CANCELLED
    (node,) = result.nodes
    assert node.status is ExperimentNodeStatus.CANCELLED
    (run,) = node.runs
    assert (run.status, run.attempt_status) == (RunStatus.SUCCEEDED, RunAttemptStatus.SUCCEEDED)
    assert len(run.artifacts) == 1
    assert children == ()
    assert (parent.status, parent.outcome.value) == (ActionStatus.SUCCEEDED, "applied")
