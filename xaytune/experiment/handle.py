"""ExperimentHandle: a way of asking the durable record about one experiment.

A handle holds an experiment id and the host it asks -- nothing else. Not a
process, not a task, not a compiler or a runtime. Every answer comes from the
record, which is what lets ``attach()`` in another process, or after this one
has gone, return a handle that says the same things.

The host is either the controller itself
(:class:`~xaytune.experiment.EmbeddedControllerHost`) or a client of one
running elsewhere (:class:`~xaytune.daemon.LocalDaemonControllerHost`, PR-029),
whose handle reads the record and sends every mutation to the daemon as a
mailbox request. The handle is the same either way.

**What ``wait()`` means.** Controller quiescence: every piece of work this
controller can currently execute for the experiment is settled, and its
telemetry has been drained. It does *not* mean the experiment is finished.
With evaluation configured, ``wait()`` waits through it and through the
decision that follows: a decided candidate ends ``COMPLETED`` or ``REJECTED``
and the experiment ``SUCCEEDED`` or ``FAILED``. A candidate the decision
engine cannot decide -- no target, a missing metric -- stays ``DECIDING``,
with the experiment ``ACTIVE``. Without evaluation, a trained node stays
``ACTIVE``. Either way :class:`ExperimentResult` says it is quiescent and
names the stage that would run next, so a caller never has to infer success
from ``wait()`` having returned.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Literal, Protocol

from xaytune.core.domain.actions import ActionSpec
from xaytune.core.domain.budget import BudgetStatus
from xaytune.core.domain.evaluation import EvaluationResult
from xaytune.core.domain.event import DomainEvent
from xaytune.core.domain.policy import GovernedAction
from xaytune.core.ids import (
    ControllerRequestId,
    EvaluationRunId,
    ExperimentId,
    ExperimentNodeId,
    RunId,
)
from xaytune.core.immutable import FrozenDomainModel
from xaytune.core.refs import Actor, ArtifactRef
from xaytune.core.state.status import (
    EvaluationAttemptStatus,
    EvaluationRunStatus,
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
)

__all__ = [
    "EvaluationOutcome",
    "ExperimentHandle",
    "ExperimentResult",
    "NodeOutcome",
    "RunOutcome",
]

NextStage = Literal[
    "action-approval",
    "action-execution",
    "evaluation",
    "decision",
    "training",
    "planning",
    "failure-handling",
]
"""Advisory: the work that would move the experiment on.

```text
None                 the experiment is terminal
"action-approval"    a proposed action awaits a human's approval
"action-execution"   an authorized action awaits an executor: VALIDATED with an
                     ALLOW decision, or APPROVED with a REQUIRE_APPROVAL one
"decision"           a candidate is DECIDING: its decision was deferred -- or
                     every candidate is settled, but a STOP decision was not
                     applied (the experiment was paused) or a node has none
"evaluation"         a trained candidate is unevaluated, or evaluating
"training"           an accepted scientific candidate exists (PLANNED) and
                     needs its first Run realization
"planning"           the experiment is ACTIVE and every candidate was decided on
                     its merits -- rejected, or completed short of the target
                     (BRANCH): another candidate is needed to go on
"failure-handling"   training or evaluation failed or was cancelled, and did
                     not establish a result: recovery or policy is needed
```

Deliberately not named after a status. ``"evaluation"`` is not
``ExperimentNodeStatus.EVALUATING``: it is the next work for a trained node
nothing has evaluated, or the evaluation still running. ``"decision"`` is a
node in ``DECIDING`` that the decision engine could not decide -- its
``DecisionDeferred`` event says why -- and that someone must decide.
``"planning"`` comes only from a scientific outcome -- every candidate
``REJECTED`` or, after a ``BRANCH`` decision, ``COMPLETED`` -- never from
candidates merely having ended: a failed or cancelled candidate is
``"failure-handling"``, even beside a decided one. ``REJECT`` and ``BRANCH``
judge the candidate, not the experiment, and what comes next is another
candidate: a planner's work. Once a planner's proposal is branched into a
``PLANNED`` node, the next work is ``"training"`` -- realizing that node's
first run -- not more planning, and not failure handling: the node has no run
because nothing has realized it yet, not because one failed. The embedded host
realizes its own planner's nodes itself (PR-026), so it rests at
``"training"`` only when it cannot: a node planned some other way, or a
realization it escalated.

The two action stages come first: an action someone proposed and is waiting
on is what comes next, before anything the candidates' states suggest -- a
candidate ``DECIDING`` with a rejection awaiting approval needs the approval,
not another decision. Nothing executes actions yet, so ``"action-execution"``
is a resting boundary for now (PR-023)."""

_FOLLOW_INTERVAL_SECONDS = 0.05


class RunOutcome(FrozenDomainModel):
    """One run, as the record has it."""

    run_id: RunId
    status: RunStatus
    attempt_status: RunAttemptStatus | None
    artifacts: tuple[ArtifactRef, ...] = ()


class EvaluationOutcome(FrozenDomainModel):
    """One evaluation run, as the record has it, with its result if it produced one."""

    evaluation_run_id: EvaluationRunId
    evaluation_cycle: int
    status: EvaluationRunStatus
    attempt_status: EvaluationAttemptStatus | None
    result: EvaluationResult | None = None


class NodeOutcome(FrozenDomainModel):
    """One candidate, its runs and its evaluations."""

    node_id: ExperimentNodeId
    status: ExperimentNodeStatus
    runs: tuple[RunOutcome, ...] = ()
    evaluations: tuple[EvaluationOutcome, ...] = ()


class ExperimentResult(FrozenDomainModel):
    """Where an experiment stands once its controller has nothing left to do.

    Attributes:
        status: The experiment's own status. ``SUCCEEDED`` or ``FAILED`` once
            its candidate is decided; ``ACTIVE`` while it is not -- after
            training with no evaluation, or with a decision deferred.
        quiescent: Whether all executable work is settled. Always true for a
            result ``wait()`` returns; carried so the result says so rather
            than leaving it implied.
        next_stage: Advisory controller work that would move the experiment
            on, if it could run -- never a status any aggregate is in:
            ``"decision"`` for a node in ``DECIDING`` the engine could not
            decide, ``"training"`` for an accepted ``PLANNED`` candidate
            that needs its first run, ``"planning"`` for an active experiment whose candidates
            were all decided -- rejected, or completed short of the target --
            and which needs another,
            ``"evaluation"`` for a trained candidate nothing has evaluated (or
            whose evaluation is still running), ``"failure-handling"`` for one
            whose runs or evaluations failed or were cancelled (retry, recover
            or give up: none exists yet). ``None`` once the experiment is
            terminal.
        budget: Each limited dimension's balance, derived from the budget
            ledger -- limit, reserved, committed, consumed and remaining --
            or ``None`` if the experiment limits nothing. ``BUDGET_EXHAUSTED``
            as the status says a used-up quota stopped the next effect.
    """

    experiment_id: ExperimentId
    status: ExperimentStatus
    quiescent: bool
    next_stage: NextStage | None
    nodes: tuple[NodeOutcome, ...] = ()
    budget: BudgetStatus | None = None


class _HandleHost(Protocol):
    """What a handle asks its host. Private: a handle is made by a host, not by callers."""

    def _experiment_status(self, experiment_id: ExperimentId) -> ExperimentStatus: ...

    async def _wait(self, experiment_id: ExperimentId) -> ExperimentResult: ...

    async def _cancel(self, experiment_id: ExperimentId, *, reason: str) -> None: ...

    async def _propose(
        self,
        experiment_id: ExperimentId,
        spec: ActionSpec,
        *,
        reason: str,
        proposed_by: Actor,
        request_id: ControllerRequestId | str | None = None,
    ) -> GovernedAction: ...

    def _actions(self, experiment_id: ExperimentId) -> tuple[GovernedAction, ...]: ...

    def _events_after(
        self, experiment_id: ExperimentId, sequence: int
    ) -> tuple[DomainEvent, ...]: ...


class ExperimentHandle:
    """The public face of one experiment: status, wait, cancel, events."""

    def __init__(self, experiment_id: ExperimentId, host: _HandleHost) -> None:
        self.experiment_id = experiment_id
        self._host = host

    def __repr__(self) -> str:
        return f"ExperimentHandle({self.experiment_id!s})"

    async def status(self) -> ExperimentStatus:
        """The experiment's status, read from the record."""
        return self._host._experiment_status(self.experiment_id)

    async def wait(self) -> ExperimentResult:
        """Wait until the controller is quiescent, and say where things stand.

        Daemon-backed, it polls the record instead: until the daemon's
        controller has come to rest on the experiment and nothing has
        happened to it since, with no request for it still unhandled -- or
        until it is terminal. Nothing in the caller drives anything, so a
        caller that stops waiting, or exits, and waits again later loses
        nothing.

        Raises:
            ReconciliationEscalatedError: If reconciliation reached a question
                it must not answer by guessing -- a workload that ended with no
                recorded outcome, or a submission a runtime cannot account for.
            ControllerNotRunningError: If work is unsettled and this host
                cannot adopt it, because the record names no runtime spec.
        """
        return await self._host._wait(self.experiment_id)

    async def cancel(self, reason: str = "cancelled through ExperimentHandle") -> None:
        """Request cancellation, and issue the effects it needs.

        Cancellation is intent recorded as an Action, then carried out as a
        cancel operation for each live attempt (ADR-013 §6). The experiment
        stays ``ACTIVE`` while that propagates -- there is no ``CANCELLING``
        status -- and reaches ``CANCELLED`` only once no workload it owns is
        executing. ``wait()`` returns once that is settled.

        Calling it again while a cancellation is in flight returns without
        recording a second one.

        Daemon-backed, this sends a ``cancel`` request and returns once the
        daemon has carried it out -- the intent recorded and its effects
        issued -- never calling a runtime itself. Beside another
        cancellation in flight, it joins that one through an ``attach``
        request, so the daemon owns the experiment and carries it on.
        """
        await self._host._cancel(self.experiment_id, reason=reason)

    async def propose(
        self,
        spec: ActionSpec,
        *,
        reason: str,
        proposed_by: Actor,
        request_id: ControllerRequestId | str | None = None,
    ) -> GovernedAction:
        """Propose a typed action: validated, judged by the host's policy, recorded.

        Nothing is carried out. The result says where governance left it:
        ``REJECTED`` with the problems validation found, or with policy's
        decision; ``VALIDATED`` with an ``ALLOW`` decision; or
        ``APPROVAL_PENDING``, for a human to approve or reject through the
        host. With no policy configured, every proposal is denied.

        Cancellation is not proposed: use :meth:`cancel`.

        Daemon-backed, the proposal is a ``propose-action`` request: the
        daemon validates, judges and records it under its own policy. To
        retry one whose answer was lost -- the caller died, or timed out --
        pass the same *request_id*: it is the same request, for the same
        Action, judged once. The embedded host records the proposal within
        this call, so it has nothing to retry by and ignores *request_id*.

        Raises:
            CancellationNotGovernedError: For a ``cancel-*`` spec.
            UnsupportedActionError: If the action's plugin refuses the spec.
            ControllerRequestFailedError: Daemon-backed, if the daemon refused
                the request: it says what the daemon raised.
        """
        return await self._host._propose(
            self.experiment_id, spec, reason=reason, proposed_by=proposed_by, request_id=request_id
        )

    async def actions(self) -> tuple[GovernedAction, ...]:
        """Every action recorded for the experiment, oldest first, as governance left it.

        Includes cancellations, which carry no policy decision.
        """
        return self._host._actions(self.experiment_id)

    async def events(self, *, after: int = 0) -> AsyncIterator[DomainEvent]:
        """The experiment's durable history, then everything committed after.

        Ordered by database sequence, which is also the cursor: replay and
        follow are one query, ``sequence > last delivered``, so nothing
        committed between them can be missed or delivered twice. Follows
        until the caller stops iterating.

        These are control-plane events. Per-step training telemetry stays with
        the runtime that produced it.
        """
        cursor = after
        while True:
            batch = self._host._events_after(self.experiment_id, cursor)
            for event in batch:
                assert event.sequence is not None
                cursor = event.sequence
                yield event
            # Always give the loop a turn, not only when caught up. Yielding an
            # event does not: a consumer replaying a long history would
            # otherwise hold the event loop until the replay ended, starving
            # the controller task that shares it -- including the one writing
            # the events being read.
            await asyncio.sleep(0 if batch else _FOLLOW_INTERVAL_SECONDS)
