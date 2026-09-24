"""ExperimentHandle: a way of asking the durable record about one experiment.

A handle holds an experiment id and the host it asks -- nothing else. Not a
process, not a task, not a compiler or a runtime. Every answer comes from the
record, which is what lets ``attach()`` in another process, or after this one
has gone, return a handle that says the same things.

**What ``wait()`` means.** Controller quiescence: every piece of work this
controller can currently execute for the experiment is settled, and its
telemetry has been drained. It does *not* mean the experiment is finished. In
this phase a successful training run leaves the node ``ACTIVE`` awaiting
evaluation and the experiment ``ACTIVE`` awaiting a decision, neither of which
exists yet -- so :class:`ExperimentResult` says it is quiescent and names the
stage that would run next, and a caller never has to infer success from
``wait()`` having returned. When evaluation becomes executable, ``wait()``
waits through it, and its meaning does not change.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

from xaytune.core.domain.event import DomainEvent
from xaytune.core.ids import ExperimentId, ExperimentNodeId, RunId
from xaytune.core.immutable import FrozenDomainModel
from xaytune.core.refs import ArtifactRef
from xaytune.core.state.status import (
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
)

if TYPE_CHECKING:
    from xaytune.experiment.host import EmbeddedControllerHost

__all__ = ["ExperimentHandle", "ExperimentResult", "NodeOutcome", "RunOutcome"]

NextStage = Literal["evaluation", "failure-handling"]
"""Advisory: the controller work that would move the experiment on.

Deliberately not named after a status. ``"evaluation"`` is not
``ExperimentNodeStatus.EVALUATING`` -- nothing has entered it -- and there is
no ``"decision"``, because ``DECIDING`` means something specific (a node
reached through evaluation) and a failed training run has not been there."""

_FOLLOW_INTERVAL_SECONDS = 0.05


class RunOutcome(FrozenDomainModel):
    """One run, as the record has it."""

    run_id: RunId
    status: RunStatus
    attempt_status: RunAttemptStatus | None
    artifacts: tuple[ArtifactRef, ...] = ()


class NodeOutcome(FrozenDomainModel):
    """One candidate and its runs."""

    node_id: ExperimentNodeId
    status: ExperimentNodeStatus
    runs: tuple[RunOutcome, ...] = ()


class ExperimentResult(FrozenDomainModel):
    """Where an experiment stands once its controller has nothing left to do.

    Attributes:
        status: The experiment's own status -- ``ACTIVE`` after a successful
            training run, because terminating an experiment is a decision
            nothing in this phase makes.
        quiescent: Whether all executable work is settled. Always true for a
            result ``wait()`` returns; carried so the result says so rather
            than leaving it implied.
        next_stage: Advisory controller work that would move the experiment
            on, if it could run -- never a status any aggregate is in:
            ``"evaluation"`` for a trained candidate, ``"failure-handling"``
            for one whose runs all failed or were cancelled (retry, recover or
            give up: none exists yet). ``None`` once the experiment is
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
