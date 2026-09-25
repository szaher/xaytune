"""ExperimentHandle: a way of asking the durable record about one experiment.

A handle holds an experiment id and the host it asks -- nothing else. Not a
process, not a task, not a compiler or a runtime. Every answer comes from the
record, which is what lets ``attach()`` in another process, or after this one
has gone, return a handle that says the same things.

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
from typing import TYPE_CHECKING, Literal

from xaytune.core.domain.evaluation import EvaluationResult
from xaytune.core.domain.event import DomainEvent
from xaytune.core.ids import EvaluationRunId, ExperimentId, ExperimentNodeId, RunId
from xaytune.core.immutable import FrozenDomainModel
from xaytune.core.refs import ArtifactRef
from xaytune.core.state.status import (
    EvaluationAttemptStatus,
    EvaluationRunStatus,
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
)

if TYPE_CHECKING:
    from xaytune.experiment.host import EmbeddedControllerHost

__all__ = [
    "EvaluationOutcome",
    "ExperimentHandle",
    "ExperimentResult",
    "NodeOutcome",
    "RunOutcome",
]

NextStage = Literal["evaluation", "decision", "planning", "failure-handling"]
"""Advisory: the controller work that would move the experiment on.

Deliberately not named after a status. ``"evaluation"`` is not
``ExperimentNodeStatus.EVALUATING``: it is the next work for a trained node
nothing has evaluated, or the evaluation still running. ``"decision"`` is a
node in ``DECIDING`` that the decision engine could not decide -- its
``DecisionDeferred`` event says why -- and that someone must decide.
``"planning"`` is an experiment still ``ACTIVE`` whose candidates are all
settled, one of them rejected: a ``REJECT`` judges the candidate, not the
experiment, and what comes next is another candidate -- a planner's work,
which does not exist yet."""

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
            decide, ``"planning"`` for an active experiment whose candidate
            was rejected and which needs another,
            ``"evaluation"`` for a trained candidate nothing has evaluated (or
            whose evaluation is still running), ``"failure-handling"`` for one
            whose runs or evaluations failed or were cancelled (retry, recover
            or give up: none exists yet). ``None`` once the experiment is
            terminal.
    """

    experiment_id: ExperimentId
    status: ExperimentStatus
    quiescent: bool
    next_stage: NextStage | None
    nodes: tuple[NodeOutcome, ...] = ()


class ExperimentHandle:
    """The public face of one experiment: status, wait, cancel, events."""

    def __init__(self, experiment_id: ExperimentId, host: EmbeddedControllerHost) -> None:
        self.experiment_id = experiment_id
        self._host = host

    def __repr__(self) -> str:
        return f"ExperimentHandle({self.experiment_id!s})"

    async def status(self) -> ExperimentStatus:
        """The experiment's status, read from the record."""
        return self._host._experiment_status(self.experiment_id)

    async def wait(self) -> ExperimentResult:
        """Wait until the controller is quiescent, and say where things stand.

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
        """
        await self._host._cancel(self.experiment_id, reason=reason)

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
