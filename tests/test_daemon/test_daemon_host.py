"""LocalDaemonControllerHost: the caller's side of the daemon, in one process (PR-029).

```text
submit / attach         a request, until handed off; never the experiment
handle reads            status, actions, events: the record
handle.wait()           the record, until the daemon's controller is at rest
                        on it -- not merely quiescent between two steps
cancel / propose        mailbox requests the daemon carries out
approve / reject        the same, for an action awaiting a human
a refused request       FAILED, nothing changed, raised to the caller
a request repeated      by its id, or after a crash part-way: done once
```

The daemon runs as a task beside the client here; :mod:`.test_daemon_process`
has the separate processes, and the CLI.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation_fixtures import EVALUATORS
from tests.test_daemon.file_runtime import FileRuntime, calls, finish, workloads
from tests.test_daemon.test_daemon_server import (
    _TIMEOUT,
    _daemon,
    _file_spec,
    _serving,
)
from tests.test_experiment.adaptive_fixtures import AdaptiveRuntime, LoRACompiler, adaptive_spec
from xaytune.checkpoints import CheckpointManager, LocalCheckpointStore, SerializedStateCodec
from xaytune.compilation.native import NativeCompiler
from xaytune.core.domain.action import ActionStatus, ActionTarget
from xaytune.core.domain.actions import CancelExperiment, RejectCandidate
from xaytune.core.domain.controller_request import ControllerRequest, ControllerRequestState
from xaytune.core.domain.policy import PolicyVerdict
from xaytune.core.domain.recovery import RecoveryRequest
from xaytune.core.ids import ActionId, ControllerRequestId, ExperimentId
from xaytune.core.refs import Actor
from xaytune.core.state.status import ExperimentNodeStatus, ExperimentStatus, RunStatus
from xaytune.daemon import (
    ControllerRequestFailedError,
    DaemonConfig,
    LocalDaemonControllerHost,
    LocalDaemonControllerServer,
)
from xaytune.decision import AdaptiveThresholdDecisionEngine
from xaytune.experiment import (
    CompilerSpec,
    EmbeddedControllerHost,
    ReconciliationEscalatedError,
    UnknownImplementationError,
)
from xaytune.planning import PLANNERS
from xaytune.policy import RulePolicyEngine
from xaytune.storage import AggregateNotFoundError
from xaytune.storage.control_plane import CancellationNotGovernedError

ANA = Actor(type="human", id="ana")
AGENT = Actor(type="llm_agent", id="planner")
REVIEWED = RulePolicyEngine(default=PolicyVerdict.REQUIRE_APPROVAL)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")


def _host(tmp_path: Path) -> LocalDaemonControllerHost:
    return LocalDaemonControllerHost(
        tmp_path / "state.db", poll_interval=0.02, handoff_timeout=_TIMEOUT
    )


def _only_workload(tmp_path: Path) -> str:
    (operation_id,) = workloads(tmp_path / "runtime")
    return operation_id


def _requests(host: LocalDaemonControllerHost, experiment_id: Any) -> list[ControllerRequest]:
    return list(host.client.requests.for_experiment(str(experiment_id)))


async def _still_waiting(task: asyncio.Future[Any], seconds: float = 0.5) -> None:
    await asyncio.sleep(seconds)
    assert not task.done(), task.result() if task.done() else None


# ---- submit, read, wait ------------------------------------------------------------------


def test_submit_returns_at_the_handoff_and_the_daemon_carries_the_experiment_on(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        host = _host(tmp_path)
        async with _serving(_daemon(tmp_path)):
            handle = await host.submit(_file_spec(tmp_path))
            (request,) = _requests(host, handle.experiment_id)
            assert (request.kind, request.state) == ("submit", ControllerRequestState.COMPLETED)
            assert await handle.status() is ExperimentStatus.ACTIVE, "submitted, not finished"
            assert workloads(tmp_path / "runtime")[_only_workload(tmp_path)]["state"] == "running"

            waiting = asyncio.ensure_future(handle.wait())
            await _still_waiting(waiting)
            finish(tmp_path / "runtime", _only_workload(tmp_path))
            result = await asyncio.wait_for(waiting, _TIMEOUT)

        assert result.quiescent
        ((run,),) = (node.runs for node in result.nodes)
        assert run.status is RunStatus.SUCCEEDED
        assert [event.event_type async for event in _replay(handle)][:2] == [
            "ExperimentCreated",
            "ExperimentStatusChanged",
        ]
        assert len(_requests(host, handle.experiment_id)) == 1, "reading sends nothing"
        await host.close()

    asyncio.run(scenario())


async def _replay(handle: Any) -> Any:
    """The events committed so far, then stop: ``events()`` would follow forever."""
    cursor = 0
    while batch := handle._host._events_after(handle.experiment_id, cursor):
        for event in batch:
            cursor = event.sequence
            yield event


def test_wait_does_not_mistake_the_step_between_training_and_evaluation_for_rest(
    tmp_path: Path,
) -> None:
    """Trained, evaluation not yet begun: the record alone looks settled. wait() waits.

    The daemon is held in exactly that window. ``result()`` shows the trap --
    quiescent, next stage evaluation -- and ``wait()`` still waits, because the
    daemon's controller has not come to rest. Released, the experiment runs
    to its decision, and wait() returns there.
    """
    manager = CheckpointManager(SerializedStateCodec(), LocalCheckpointStore(tmp_path / "bundles"))
    runtime = AdaptiveRuntime(manager, tmp_path)
    config = DaemonConfig(
        compilers={"native": lambda: LoRACompiler(64)},
        runtimes={"local": lambda config: runtime},
        evaluators=EVALUATORS,
        planners=PLANNERS,
        decision_engine=AdaptiveThresholdDecisionEngine(),
        policy=RulePolicyEngine(default=PolicyVerdict.ALLOW),
        checkpoint_manager=manager,
        recovery_request_for_incident=lambda incident: RecoveryRequest(
            restore_context=runtime.restore_context
        ),
    )

    async def scenario() -> None:
        host = _host(tmp_path)
        daemon = LocalDaemonControllerServer(tmp_path / "state.db", config, poll_interval=0.05)
        held, release = asyncio.Event(), asyncio.Event()
        async with _serving(daemon):
            controller = daemon.controller
            original = controller._continue_to_evaluation

            async def hold(*args: Any, **kwargs: Any) -> Any:
                if not release.is_set():
                    held.set()
                    await release.wait()
                return await original(*args, **kwargs)

            controller._continue_to_evaluation = hold  # type: ignore[method-assign]
            handle = await host.submit(adaptive_spec(tmp_path))
            waiting = asyncio.ensure_future(handle.wait())
            await asyncio.wait_for(held.wait(), _TIMEOUT)

            between = host.result(handle.experiment_id)
            assert between.quiescent and between.next_stage == "evaluation", "the trap"
            await _still_waiting(waiting)

            release.set()
            result = await asyncio.wait_for(waiting, _TIMEOUT * 4)

        assert result.status is ExperimentStatus.SUCCEEDED, "the loop ran to its end"
        assert len(result.nodes) == 2
        rest = host.client.requests.rest(str(handle.experiment_id))
        assert rest is not None and rest.controller_id == daemon.instance_id
        assert rest.sequence == host.client._repository.events.latest_sequence_for_experiment(
            str(handle.experiment_id)
        )
        await host.close()

    asyncio.run(scenario())


def test_a_caller_that_stops_waiting_loses_nothing(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _serving(_daemon(tmp_path)):
            first = _host(tmp_path)
            handle = await first.submit(_file_spec(tmp_path))
            waiting = asyncio.ensure_future(handle.wait())
            await asyncio.sleep(0.2)
            waiting.cancel()
            await first.close()  # the caller is gone

            finish(tmp_path / "runtime", _only_workload(tmp_path))
            second = _host(tmp_path)
            again = second.handle(handle.experiment_id)
            result = await asyncio.wait_for(again.wait(), _TIMEOUT)
            assert result.quiescent and result.nodes[0].runs[0].status is RunStatus.SUCCEEDED
            assert len(_requests(second, handle.experiment_id)) == 1, "a handle sends nothing"
            with pytest.raises(AggregateNotFoundError):
                second.handle(ExperimentId.generate())
            await second.close()

    asyncio.run(scenario())


def test_attach_is_a_request_after_which_the_daemon_owns_the_experiment(tmp_path: Path) -> None:
    async def scenario() -> None:
        host = _host(tmp_path)
        async with _serving(_daemon(tmp_path)):
            handle = await host.submit(_file_spec(tmp_path))
        async with _serving(_daemon(tmp_path)):
            attached = await host.attach(handle.experiment_id)
            kinds = [(r.kind, r.state) for r in _requests(host, handle.experiment_id)]
            assert kinds == [
                ("submit", ControllerRequestState.COMPLETED),
                ("attach", ControllerRequestState.COMPLETED),
            ]
            finish(tmp_path / "runtime", _only_workload(tmp_path))
            result = await asyncio.wait_for(attached.wait(), _TIMEOUT)
            assert result.nodes[0].runs[0].status is RunStatus.SUCCEEDED
            with pytest.raises(ControllerRequestFailedError, match="AggregateNotFoundError"):
                await host.attach(ExperimentId.generate())
        assert len(calls(tmp_path / "runtime", "submit")) == 1, "adopted, never resubmitted"
        await host.close()

    asyncio.run(scenario())


# ---- cancellation ------------------------------------------------------------------------


def test_cancel_is_a_request_the_daemon_carries_out(tmp_path: Path) -> None:
    async def scenario() -> None:
        host = _host(tmp_path)
        async with _serving(_daemon(tmp_path)):
            handle = await host.submit(_file_spec(tmp_path))
            await handle.cancel("enough")
            cancel = _requests(host, handle.experiment_id)[-1]
            assert (cancel.kind, cancel.state) == ("cancel", ControllerRequestState.COMPLETED)
            (cancelling,) = [
                governed.action
                for governed in await handle.actions()
                if governed.action.type == "cancel-experiment"
            ]
            assert cancelling.id == cancel.action_id, "the request named the action it recorded"
            assert workloads(tmp_path / "runtime")[_only_workload(tmp_path)]["cancelled"]

            result = await asyncio.wait_for(handle.wait(), _TIMEOUT)
            assert result.status is ExperimentStatus.CANCELLED

            again = cancel.id
            await host.cancel(handle.experiment_id, reason="enough", request_id=again)
        actions = [g.action for g in await handle.actions() if g.action.type == "cancel-experiment"]
        assert len(actions) == 1, "the same request again is not a second cancellation"
        assert len(calls(tmp_path / "runtime", "cancel")) == 1
        await host.close()

    asyncio.run(scenario())


def test_a_cancel_interrupted_after_its_intent_is_finished_by_the_same_identity(
    tmp_path: Path,
) -> None:
    """The intent commits, then the effects fail: PENDING, carried out again, done once."""

    async def scenario() -> None:
        host = _host(tmp_path)
        daemon = _daemon(tmp_path)
        async with _serving(daemon):
            handle = await host.submit(_file_spec(tmp_path))
            controller = daemon.controller
            original = controller._carry_out_cancellation
            failures = []

            async def fail_once(*args: Any) -> None:
                if not failures:
                    failures.append(args)
                    raise RuntimeError("the runtime is unreachable")
                await original(*args)

            controller._carry_out_cancellation = fail_once  # type: ignore[method-assign]
            await handle.cancel("enough")
            assert failures, "the first attempt failed after recording its intent"

            result = await asyncio.wait_for(handle.wait(), _TIMEOUT)
        assert result.status is ExperimentStatus.CANCELLED
        sagas = [g.action for g in await handle.actions() if g.action.type == "cancel-experiment"]
        cancel = _requests(host, handle.experiment_id)[-1]
        assert [saga.id for saga in sagas] == [cancel.action_id]
        assert len(calls(tmp_path / "runtime", "cancel")) == 1
        await host.close()

    asyncio.run(scenario())


# ---- governed actions ------------------------------------------------------------------------


def test_a_proposal_awaiting_approval_is_approved_through_the_mailbox(tmp_path: Path) -> None:
    async def scenario() -> None:
        host = _host(tmp_path)
        async with _serving(_daemon(tmp_path, policy=REVIEWED)):
            handle = await host.submit(_file_spec(tmp_path))
            finish(tmp_path / "runtime", _only_workload(tmp_path))
            trained = await asyncio.wait_for(handle.wait(), _TIMEOUT)
            (node,) = trained.nodes

            target = ActionTarget(kind="node", id=str(node.node_id))
            governed = await handle.propose(
                RejectCandidate(target=target), reason="off target", proposed_by=AGENT
            )
            assert governed.action.status is ActionStatus.APPROVAL_PENDING
            assert governed.decision is not None and governed.decision.engine_name == "rules"
            assert (await handle.actions()) == (governed,)
            awaiting = await asyncio.wait_for(handle.wait(), _TIMEOUT)
            assert awaiting.next_stage == "action-approval"

            approved = await host.approve_action(
                governed.action.id, approver=ANA, reason="agreed, it is off target"
            )
            assert approved.action.status is ActionStatus.APPROVED
            again = await host.approve_action(
                governed.action.id, approver=ANA, reason="agreed, it is off target"
            )
            assert again == approved, "the same approval is recognized, not repeated"
            with pytest.raises(ControllerRequestFailedError, match="ApprovalConflictError"):
                await host.reject_action(governed.action.id, approver=ANA, reason="changed mind")
            result = await asyncio.wait_for(handle.wait(), _TIMEOUT)

        assert result.next_stage == "action-execution"
        assert result.nodes[0].status is ExperimentNodeStatus.ACTIVE, "approved, not applied"
        kinds = [(r.kind, r.state.value) for r in _requests(host, handle.experiment_id)]
        assert kinds == [
            ("submit", "completed"),
            ("propose-action", "completed"),
            ("approve-action", "completed"),
            ("approve-action", "completed"),
            ("reject-action", "failed"),
        ]
        await host.close()

    asyncio.run(scenario())


def test_a_refused_request_fails_and_changes_nothing(tmp_path: Path) -> None:
    async def scenario() -> None:
        host = _host(tmp_path)
        async with _serving(_daemon(tmp_path, policy=REVIEWED)):
            with pytest.raises(ControllerRequestFailedError) as unknown:
                await host.submit(_file_spec(tmp_path, compiler=CompilerSpec(name="nope")))
            assert unknown.value.error_type == "UnknownImplementationError"

            handle = await host.submit(_file_spec(tmp_path))
            (node,) = host.result(handle.experiment_id).nodes
            target = ActionTarget(kind="node", id=str(node.node_id))
            governed = await handle.propose(
                RejectCandidate(target=target), reason="off target", proposed_by=AGENT
            )

            with pytest.raises(ControllerRequestFailedError, match="ApprovalError"):
                await host.approve_action(governed.action.id, approver=AGENT, reason="I agree")
            other = await host.submit(_file_spec(tmp_path, seed=9))
            misdirected = ControllerRequest.resolve(
                "approve-action",
                governed.action.id,
                other.experiment_id,
                approver=ANA,
                reason="agreed",
            )
            with pytest.raises(ControllerRequestFailedError, match="MisdirectedRequestError"):
                await host._hand_off(misdirected)

            before = len(_requests(host, handle.experiment_id))
            with pytest.raises(CancellationNotGovernedError):
                await handle.propose(
                    CancelExperiment(
                        target=ActionTarget(kind="experiment", id=str(handle.experiment_id))
                    ),
                    reason="stop",
                    proposed_by=ANA,
                )
            assert len(_requests(host, handle.experiment_id)) == before, "nothing was sent"
            with pytest.raises(AggregateNotFoundError):
                await host.approve_action("act_nope", approver=ANA, reason="agreed")

        (action,) = [g.action for g in await handle.actions()]
        assert action.status is ActionStatus.APPROVAL_PENDING, "every refusal left it as it was"
        await host.close()

    asyncio.run(scenario())


# ---- retry -------------------------------------------------------------------------------


def test_a_submission_retried_by_its_request_id_is_admitted_once(tmp_path: Path) -> None:
    async def scenario() -> None:
        request_id = ControllerRequestId.generate()
        host = _host(tmp_path)
        async with _serving(_daemon(tmp_path)):
            first = await host.submit(_file_spec(tmp_path), request_id=request_id)
            retried = await host.submit(_file_spec(tmp_path), request_id=request_id)
        assert retried.experiment_id == first.experiment_id
        assert len(workloads(tmp_path / "runtime")) == 1
        assert len(calls(tmp_path / "runtime", "submit")) == 1
        await host.close()

    asyncio.run(scenario())


def test_a_request_sent_with_no_daemon_waits_for_one(tmp_path: Path) -> None:
    async def scenario() -> None:
        host = LocalDaemonControllerHost(tmp_path / "state.db", handoff_timeout=0.3)
        request_id = ControllerRequestId.generate()
        with pytest.raises(TimeoutError):
            await host.submit(_file_spec(tmp_path), request_id=request_id)
        recorded = host.client.request(request_id)
        assert recorded.state is ControllerRequestState.PENDING, "handed off all the same"
        async with _serving(_daemon(tmp_path)):
            handle = await _host(tmp_path).submit(_file_spec(tmp_path), request_id=request_id)
        assert handle.experiment_id == recorded.experiment_id
        await host.close()

    asyncio.run(scenario())


# ---- rest ----------------------------------------------------------------------------------


def test_wait_reads_the_rest_and_nothing_else_moves_it(tmp_path: Path) -> None:
    """The rest must name the latest event, with no request unhandled; an escalation raises."""

    async def scenario() -> None:
        host = _host(tmp_path)
        async with _serving(_daemon(tmp_path)):
            handle = await host.submit(_file_spec(tmp_path))
            finish(tmp_path / "runtime", _only_workload(tmp_path))
            await asyncio.wait_for(handle.wait(), _TIMEOUT)
        experiment_id = handle.experiment_id
        assert host._at_rest(experiment_id) is not None

        host.client.attach(experiment_id)  # sent; no daemon to carry it out
        assert host._at_rest(experiment_id) is None, "a request still unhandled"

        other = _host(tmp_path / "other")
        async with _serving(_daemon(tmp_path / "other")):
            other_handle = await other.submit(_file_spec(tmp_path / "other"))
        other.client._repository.record_controller_rest(
            other_handle.experiment_id,
            controller_id="daemon-x",
            escalation={"type": "ReconciliationEscalatedError", "message": "no outcome"},
        )
        with pytest.raises(ReconciliationEscalatedError, match="no outcome"):
            await asyncio.wait_for(other_handle.wait(), _TIMEOUT)
        await host.close()
        await other.close()

    asyncio.run(scenario())


# ---- review of #64: request identity and recovery ------------------------------------------


class _CountingPolicy(RulePolicyEngine):
    def __init__(self) -> None:
        super().__init__(default=PolicyVerdict.REQUIRE_APPROVAL)
        self.evaluated = 0

    def evaluate(self, spec: Any, context: Any) -> Any:
        self.evaluated += 1
        return super().evaluate(spec, context)


def test_a_refused_mutation_adopts_nothing(tmp_path: Path) -> None:
    """The intent is recorded before the attach: a FAILED request changed nothing, ownership too."""
    runtime_root = tmp_path / "runtime"

    async def scenario() -> None:
        embedded = EmbeddedControllerHost(
            tmp_path / "state.db",
            compilers={"native": NativeCompiler},
            runtimes={"file": lambda config: FileRuntime(config["root"])},
            evaluators={},
        )
        handle = await embedded.submit(_file_spec(tmp_path))
        await embedded.close()  # its workload runs on, unobserved
        watched = len(calls(runtime_root, "watch"))

        host = _host(tmp_path)
        daemon = _daemon(tmp_path)
        async with _serving(daemon):
            refused = ControllerRequest.resolve(
                "approve-action",
                ActionId.generate(),
                handle.experiment_id,
                approver=ANA,
                reason="agreed",
            )
            with pytest.raises(ControllerRequestFailedError, match="AggregateNotFoundError"):
                await host._hand_off(refused)
            await asyncio.sleep(0.3)
            assert str(handle.experiment_id) not in daemon.controller._controllers
            assert len(calls(runtime_root, "watch")) == watched, "not adopted"
            assert handle.experiment_id not in (
                daemon.controller.repository.daemon_responsibilities()
            )
        await host.close()

    asyncio.run(scenario())


def test_a_recorded_mutation_whose_attach_fails_stays_pending_and_is_finished(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        host = _host(tmp_path)
        daemon = _daemon(tmp_path)
        async with _serving(daemon):
            handle = await host.submit(_file_spec(tmp_path))
            controller = daemon.controller
            original = controller.attach
            refused = asyncio.Event()

            async def attach_once_refused(experiment_id: Any) -> Any:
                if not refused.is_set():
                    refused.set()
                    raise UnknownImplementationError("the runtime is not loaded yet")
                return await original(experiment_id)

            controller.attach = attach_once_refused  # type: ignore[method-assign]
            cancelling = asyncio.ensure_future(handle.cancel("enough"))
            await asyncio.wait_for(refused.wait(), _TIMEOUT)
            cancel = _requests(host, handle.experiment_id)[-1]
            assert cancel.kind == "cancel"
            assert host.client.request(cancel.id).state is ControllerRequestState.PENDING, (
                "an attach refused after the intent is recorded does not fail the request"
            )
            assert host.client._repository.actions.get(str(cancel.action_id)) is not None

            await asyncio.wait_for(cancelling, _TIMEOUT)
            result = await asyncio.wait_for(handle.wait(), _TIMEOUT)
        assert host.client.request(cancel.id).state is ControllerRequestState.COMPLETED
        assert result.status is ExperimentStatus.CANCELLED
        sagas = [g.action for g in await handle.actions() if g.action.type == "cancel-experiment"]
        assert [saga.id for saga in sagas] == [cancel.action_id]
        await host.close()

    asyncio.run(scenario())


def test_a_proposal_whose_answer_was_lost_is_retried_as_the_same_request(
    tmp_path: Path,
) -> None:
    policy = _CountingPolicy()

    async def scenario() -> None:
        host = _host(tmp_path)
        async with _serving(_daemon(tmp_path, policy=policy)):
            handle = await host.submit(_file_spec(tmp_path))
        (node,) = host.result(handle.experiment_id).nodes
        spec = RejectCandidate(target=ActionTarget(kind="node", id=str(node.node_id)))
        request_id = ControllerRequestId.generate()

        impatient = LocalDaemonControllerHost(tmp_path / "state.db", handoff_timeout=0.2)
        with pytest.raises(TimeoutError):  # committed; the answer never came
            await impatient.handle(handle.experiment_id).propose(
                spec, reason="off target", proposed_by=AGENT, request_id=request_id
            )
        await impatient.close()

        async with _serving(_daemon(tmp_path, policy=policy)):
            first = await handle.propose(
                spec, reason="off target", proposed_by=AGENT, request_id=request_id
            )
            again = await handle.propose(
                spec, reason="off target", proposed_by=AGENT, request_id=request_id
            )
        assert again == first
        assert first.action.status is ActionStatus.APPROVAL_PENDING
        proposals = [r for r in _requests(host, handle.experiment_id) if r.kind != "submit"]
        assert [r.id for r in proposals] == [request_id]
        assert proposals[0].action_id == first.action.id
        assert [g.action.id for g in await handle.actions()] == [first.action.id]
        connection = host.client._connection
        assert connection.execute("SELECT COUNT(*) FROM policy_decisions").fetchone()[0] == 1
        assert policy.evaluated == 1, "judged once"
        await host.close()

    asyncio.run(scenario())


def test_a_cancel_request_never_acts_through_another_cancellation(tmp_path: Path) -> None:
    """request.action_id is the cancellation the request used -- before, during and after."""

    async def scenario() -> None:
        host = _host(tmp_path)
        daemon = _daemon(tmp_path)
        async with _serving(daemon):
            handle = await host.submit(_file_spec(tmp_path))
            controller = daemon.controller
            # B1: a cancellation already in flight, its effects not yet issued.
            first, _ = controller.repository.request_experiment_cancellation(
                handle.experiment_id, reason="first", actor=Actor(type="system", id="test")
            )
            second = ControllerRequest.cancel(handle.experiment_id, reason="second")
            with pytest.raises(ControllerRequestFailedError, match="CancellationInFlightError"):
                await host._hand_off(second)
            assert host.client._repository.actions.get(str(second.action_id)) is None

            await handle.cancel("again")  # as embedded: already being cancelled
            await controller._cancel(handle.experiment_id, reason="first")  # B1 carried out
            result = await asyncio.wait_for(handle.wait(), _TIMEOUT)
            assert result.status is ExperimentStatus.CANCELLED

            # The same request again -- a retry after a lost answer -- still
            # means its own Action, which it never recorded; never B1, never now.
            with pytest.raises(ControllerRequestFailedError, match="CancellationInFlightError"):
                await host._hand_off(second)
            await daemon.process_requests()
        sagas = [g.action for g in await handle.actions() if g.action.type == "cancel-experiment"]
        assert [saga.id for saga in sagas] == [first.id]
        assert host.client.request(second.id).state is ControllerRequestState.FAILED
        await host.close()

    asyncio.run(scenario())


def test_cancelling_beside_an_unadopted_cancellation_adopts_and_finishes_it(
    tmp_path: Path,
) -> None:
    """An embedded host recorded cancellation B and went away before carrying it out.

    The daemon-backed cancel is refused -- it names its own Action, and B is
    in flight -- so it joins B the only truthful way: an attach request,
    after which the daemon owns the experiment and has carried B out.
    """
    runtime_root = tmp_path / "runtime"

    async def scenario() -> None:
        embedded = EmbeddedControllerHost(
            tmp_path / "state.db",
            compilers={"native": NativeCompiler},
            runtimes={"file": lambda config: FileRuntime(config["root"])},
            evaluators={},
        )
        handle = await embedded.submit(_file_spec(tmp_path))
        first, _ = embedded.repository.request_experiment_cancellation(
            handle.experiment_id, reason="first", actor=Actor(type="system", id="embedded")
        )
        await embedded.close()  # B recorded, its effect never issued
        assert calls(runtime_root, "cancel") == []

        host = _host(tmp_path)
        daemon = _daemon(tmp_path)
        async with _serving(daemon):
            assert handle.experiment_id not in (
                daemon.controller.repository.daemon_responsibilities()
            ), "nothing has adopted it yet"
            await host.handle(handle.experiment_id).cancel("enough")

            kinds = [(r.kind, r.state) for r in _requests(host, handle.experiment_id)]
            assert kinds == [
                ("cancel", ControllerRequestState.FAILED),
                ("attach", ControllerRequestState.COMPLETED),
            ], "the refused cancel changed nothing; the attach is the durable adoption"
            assert len(calls(runtime_root, "cancel")) == 1, "B's effect, issued by the daemon"
            result = await asyncio.wait_for(host.handle(handle.experiment_id).wait(), _TIMEOUT)
        assert result.status is ExperimentStatus.CANCELLED
        sagas = [
            g.action
            for g in await host.handle(handle.experiment_id).actions()
            if g.action.type == "cancel-experiment"
        ]
        assert [saga.id for saga in sagas] == [first.id]
        await host.close()

    asyncio.run(scenario())


def test_a_refused_cancel_retried_after_the_other_cancellation_failed_is_not_success(
    tmp_path: Path,
) -> None:
    """The refusal is permanent; success needs the experiment cancelled or cancelling now."""

    async def scenario() -> None:
        host = _host(tmp_path)
        daemon = _daemon(tmp_path)
        async with _serving(daemon):
            handle = await host.submit(_file_spec(tmp_path))
            repository = daemon.controller.repository
            system = Actor(type="system", id="test")
            first, children = repository.request_experiment_cancellation(
                handle.experiment_id, reason="first", actor=system
            )
            request_id = ControllerRequestId.generate()
            await host.cancel(handle.experiment_id, reason="second", request_id=request_id)
            assert host.client.request(request_id).state is ControllerRequestState.FAILED

            # B's effect fails: the cancellation ends, the experiment runs on.
            ((_child, operation),) = children
            assert operation is not None
            repository.fail_operation(
                operation.id, expected_revision=operation.revision, actor=system
            )
            assert repository.reconcile_experiment_cancellation(first.id, actor=system).status is (
                ActionStatus.FAILED
            )
            assert await handle.status() is ExperimentStatus.ACTIVE

            with pytest.raises(ControllerRequestFailedError, match="CancellationInFlightError"):
                await host.cancel(handle.experiment_id, reason="second", request_id=request_id)
            assert await handle.status() is ExperimentStatus.ACTIVE
        await host.close()

    asyncio.run(scenario())


def test_cancelled_or_cancelling_describes_one_database_state(tmp_path: Path) -> None:
    """A cancellation settling between the status and the action reads changes neither."""
    system = Actor(type="system", id="test")

    async def scenario() -> None:
        embedded = EmbeddedControllerHost(
            tmp_path / "state.db",
            compilers={"native": NativeCompiler},
            runtimes={"file": lambda config: FileRuntime(config["root"])},
            evaluators={},
        )
        handle = await embedded.submit(_file_spec(tmp_path))
        repository = embedded.repository
        first, ((_child, operation),) = repository.request_experiment_cancellation(
            handle.experiment_id, reason="first", actor=system
        )
        assert operation is not None

        host = LocalDaemonControllerHost(tmp_path / "state.db")
        aggregates = host.client._repository.aggregates
        read_status = aggregates.load_experiment

        def then_settle(experiment_id: str) -> Any:
            experiment = read_status(experiment_id)
            # Another connection ends the cancellation between the two reads.
            repository.fail_operation(
                operation.id, expected_revision=operation.revision, actor=system
            )
            repository.reconcile_experiment_cancellation(first.id, actor=system)
            return experiment

        aggregates.load_experiment = then_settle  # type: ignore[method-assign]
        assert host._cancelled_or_cancelling(handle.experiment_id), (
            "ACTIVE with a cancellation in flight: the state the snapshot began in"
        )
        aggregates.load_experiment = read_status  # type: ignore[method-assign]
        assert not host._cancelled_or_cancelling(handle.experiment_id), "and now it has ended"
        await host.close()
        await embedded.close()

    asyncio.run(scenario())
