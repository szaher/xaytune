"""LocalDaemonControllerHost: the caller's side of a local daemon controller (PR-029).

The ``ControllerHost`` a caller uses when a daemon owns the state database
(ADR-004 §1). It runs no controller. Every mutation is a durable mailbox
request the daemon carries out under its lease; everything else is a read of
the record:

```text
host.submit(spec)          submit request, until handed off      → ExperimentHandle
host.attach(id)            attach request, until handed off      → ExperimentHandle
host.handle(id)            no request: a handle that only reads
host.approve_action(...)   approve-action request                → GovernedAction
host.reject_action(...)    reject-action request                 → GovernedAction

handle.status(), .actions(), .events()   record reads
handle.wait()                            record polling, until the daemon's
                                         controller is at rest on it
handle.cancel()                          cancel request
handle.propose(...)                      propose-action request
```

"Handed off" is the request's ``COMPLETED``: the daemon has done what the
embedded host's call would have done before returning -- admitted and issued
a submission, adopted an attached experiment, recorded a cancellation and
issued its effects, judged a proposal, resolved an approval. Never that the
experiment has finished. A request the daemon refuses is ``FAILED``, and
raises :class:`ControllerRequestFailedError` here.

**Exiting is safe.** A request is durable once sent. A caller killed while
waiting for its handoff has still handed it off, and the daemon carries it
out; waiting again, from any process, reads the same record. Retrying a
request whose answer was lost means sending the *same* request again --
pass its ``request_id`` -- which the mailbox recognizes, and which carries
the same pre-minted experiment or action id, so nothing is done twice.

No daemon running is not an error: the request waits in the mailbox, and is
carried out when one starts. ``handoff_timeout`` bounds how long a call waits
for that.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import TracebackType
from typing import Literal

from xaytune.core.domain.action import CANCELLATION_TYPES
from xaytune.core.domain.actions import ActionSpec, action_from_spec
from xaytune.core.domain.controller_request import ControllerRequest, ControllerRequestState
from xaytune.core.domain.event import DomainEvent
from xaytune.core.domain.policy import GovernedAction
from xaytune.core.errors import XaytuneError
from xaytune.core.ids import ActionId, ControllerRequestId, ExperimentId
from xaytune.core.refs import Actor
from xaytune.core.sqlite import read_snapshot
from xaytune.core.state.status import ExperimentStatus
from xaytune.daemon.client import DaemonClient
from xaytune.experiment.handle import ExperimentHandle, ExperimentResult
from xaytune.experiment.host import ControllerNotRunningError, ReconciliationEscalatedError
from xaytune.experiment.result import experiment_result
from xaytune.experiment.spec import ExperimentSpec
from xaytune.storage.control_plane import CancellationNotGovernedError
from xaytune.storage.errors import AggregateNotFoundError

__all__ = ["ControllerRequestFailedError", "LocalDaemonControllerHost"]

_ESCALATIONS: dict[str, type[XaytuneError]] = {
    "ReconciliationEscalatedError": ReconciliationEscalatedError,
    "ControllerNotRunningError": ControllerNotRunningError,
}


class ControllerRequestFailedError(XaytuneError):
    """The daemon refused a request: ``FAILED``, having changed nothing for it.

    Attributes:
        request: The request as the mailbox records it.
        error_type: The name of the error the daemon raised -- for example
            ``UnsupportedCandidateError`` or ``ApprovalConflictError``.
        detail: Its message.
    """

    def __init__(self, request: ControllerRequest) -> None:
        self.request = request
        error = dict(request.error or {})
        self.error_type = str(error.get("type", "Error"))
        self.detail = str(error.get("message", ""))
        super().__init__(
            f"the daemon refused {request.kind} request {request.id}: "
            f"{self.error_type}: {self.detail}"
        )


class LocalDaemonControllerHost:
    """Experiments controlled by the local daemon serving *state_path*.

    Args:
        state_path: The daemon's state database. Created and migrated if new.
        poll_interval: Seconds between looks at the record while waiting.
        handoff_timeout: Seconds a request may wait for the daemon to carry
            it out before :class:`TimeoutError`; ``None`` waits as long as it
            takes. A request that timed out is still in the mailbox.
    """

    def __init__(
        self,
        state_path: Path | str,
        *,
        poll_interval: float = 0.2,
        handoff_timeout: float | None = None,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self._client = DaemonClient(state_path)
        self._repository = self._client._repository
        self._poll_interval = poll_interval
        self._handoff_timeout = handoff_timeout

    @property
    def client(self) -> DaemonClient:
        """The mailbox client every request goes through."""
        return self._client

    # ---- the public surface ---------------------------------------------

    async def submit(
        self,
        spec: ExperimentSpec,
        *,
        request_id: ControllerRequestId | str | None = None,
        experiment_id: ExperimentId | str | None = None,
    ) -> ExperimentHandle:
        """Hand *spec* to the daemon; return its handle once it is admitted and issued.

        Does not wait for the experiment: training continues in the daemon,
        whether or not this process does.

        Args:
            request_id: To retry a submission whose answer was lost. Its
                recorded experiment id is reused, so the retry is the same
                request.
            experiment_id: The experiment's id, if the caller mints it.

        Raises:
            ControllerRequestFailedError: If the daemon refused the spec --
                validation, an implementation it lacks, a candidate, budget or
                evaluation nothing can run. Nothing was admitted.
            IdempotencyConflictError: If *request_id* names another request.
            TimeoutError: After ``handoff_timeout``.
        """
        recorded = self._recorded(request_id)
        if experiment_id is None and recorded is not None:
            experiment_id = recorded.experiment_id
        request = ControllerRequest.submit(
            spec.submission_payload(),
            experiment_id=None if experiment_id is None else ExperimentId(str(experiment_id)),
            request_id=None if request_id is None else ControllerRequestId(str(request_id)),
        )
        await self._hand_off(request)
        return ExperimentHandle(request.experiment_id, self)

    async def attach(
        self,
        experiment_id: ExperimentId | str,
        *,
        request_id: ControllerRequestId | str | None = None,
    ) -> ExperimentHandle:
        """Ask the daemon to adopt an experiment already in the record, then return its handle.

        Once adopted, the daemon owns it -- reconciling it after every restart
        -- so a later handle needs no request: :meth:`handle`.

        Raises:
            ControllerRequestFailedError: If there is no such experiment, or
                the daemon lacks its recorded implementations.
            TimeoutError: After ``handoff_timeout``.
        """
        request = ControllerRequest.attach(
            ExperimentId(str(experiment_id)),
            request_id=None if request_id is None else ControllerRequestId(str(request_id)),
        )
        await self._hand_off(request)
        return ExperimentHandle(request.experiment_id, self)

    def handle(self, experiment_id: ExperimentId | str) -> ExperimentHandle:
        """A handle to an experiment in the record, sending no request.

        For an experiment the daemon already owns -- submitted through it, or
        attached. Reading and waiting need nothing more; its mutations are
        requests either way.

        Raises:
            AggregateNotFoundError: If no such experiment exists.
        """
        experiment = self._repository.aggregates.load_experiment(str(experiment_id))
        return ExperimentHandle(experiment.id, self)

    async def cancel(
        self,
        experiment_id: ExperimentId | str,
        *,
        reason: str = "cancelled through ExperimentHandle",
        request_id: ControllerRequestId | str | None = None,
    ) -> None:
        """What :meth:`ExperimentHandle.cancel` sends, with a request id to retry by.

        As embedded: while another cancellation of the experiment is in
        flight, this joins it rather than recording a second one. The daemon
        records the request's own Action or none -- never answering it with
        another -- so that request is ``FAILED`` with
        ``CancellationInFlightError``, having changed nothing. Joining is then
        a separate ``attach`` request: the daemon adopts the experiment, and
        its reconciliation carries the cancellation in flight on. This
        returns only if, after that, the experiment is cancelled or a
        cancellation is still in flight; otherwise -- the other cancellation
        ended without cancelling it -- the refusal is raised.

        Raises:
            ControllerRequestFailedError: If the daemon refused it, and it
                could not join another cancellation.
            TimeoutError: After ``handoff_timeout``.
        """
        recorded = self._recorded(request_id)
        experiment = ExperimentId(str(experiment_id))
        request = ControllerRequest.cancel(
            experiment,
            reason=reason,
            action_id=None if recorded is None else recorded.action_id,
            request_id=None if request_id is None else ControllerRequestId(str(request_id)),
        )
        try:
            await self._hand_off(request)
        except ControllerRequestFailedError as refused:
            if refused.error_type != "CancellationInFlightError":
                raise
            await self._hand_off(ControllerRequest.attach(experiment))
            if not self._cancelled_or_cancelling(experiment):
                raise

    def _cancelled_or_cancelling(self, experiment_id: ExperimentId) -> bool:
        """Whether the experiment is cancelled, or a cancellation of it is in flight.

        One read snapshot: the status and the actions describe one state.
        Read apart, a cancellation settling between them -- ACTIVE, then no
        cancellation in flight -- would read as neither.
        """
        with read_snapshot(self._client._connection):
            experiment = self._repository.aggregates.load_experiment(str(experiment_id))
            if experiment.status is ExperimentStatus.CANCELLED:
                return True
            return any(
                action.type == "cancel-experiment" and not action.is_terminal
                for action in self._repository.actions.for_target("experiment", str(experiment_id))
            )

    async def propose(
        self,
        experiment_id: ExperimentId | str,
        spec: ActionSpec,
        *,
        reason: str,
        proposed_by: Actor,
        request_id: ControllerRequestId | str | None = None,
    ) -> GovernedAction:
        """What :meth:`ExperimentHandle.propose` sends: the daemon validates, judges, records.

        The Action is built here -- the spec's own validation -- with an id
        minted here, and judged by the daemon's policy. Retried with a
        *request_id* already sent, the recorded Action id is reused: the same
        request, the same Action, one policy decision.

        Raises:
            CancellationNotGovernedError: For a ``cancel-*`` spec; nothing is sent.
            UnsupportedActionError: If the action's plugin refuses the spec here.
            ControllerRequestFailedError: If the daemon refused it.
            IdempotencyConflictError: If *request_id* names another request.
            TimeoutError: After ``handoff_timeout``.
        """
        if spec.type in CANCELLATION_TYPES:  # type: ignore[attr-defined]
            raise CancellationNotGovernedError(spec.type)  # type: ignore[attr-defined]
        recorded = self._recorded(request_id)
        action = action_from_spec(
            spec,
            experiment_id=ExperimentId(str(experiment_id)),
            proposed_by=proposed_by,
            reason=reason,
            action_id=None if recorded is None else recorded.action_id,
        )
        request = ControllerRequest.propose(
            action,
            request_id=None if request_id is None else ControllerRequestId(str(request_id)),
        )
        await self._hand_off(request)
        return self._repository.governed_action(action.id)

    async def approve_action(
        self,
        action_id: ActionId | str,
        *,
        approver: Actor,
        reason: str,
        request_id: ControllerRequestId | str | None = None,
    ) -> GovernedAction:
        """A human approves an action awaiting approval; the daemon carries on from it.

        Raises:
            AggregateNotFoundError: If no such action is recorded.
            ControllerRequestFailedError: If the daemon refused the approval --
                the approver is not a human, the action is not awaiting
                approval, or a human already resolved it otherwise.
            TimeoutError: After ``handoff_timeout``.
        """
        return await self._resolve("approve-action", action_id, approver, reason, request_id)

    async def reject_action(
        self,
        action_id: ActionId | str,
        *,
        approver: Actor,
        reason: str,
        request_id: ControllerRequestId | str | None = None,
    ) -> GovernedAction:
        """A human refuses an action awaiting approval. Raises as :meth:`approve_action`."""
        return await self._resolve("reject-action", action_id, approver, reason, request_id)

    def result(self, experiment_id: ExperimentId | str) -> ExperimentResult:
        """Where the experiment stands now, read from the record, without waiting.

        ``quiescent`` says whether work is unsettled; unlike :meth:`ExperimentHandle.wait`,
        this does not say whether the daemon is about to start more.

        Raises:
            AggregateNotFoundError: If no such experiment exists.
        """
        return experiment_result(self._repository, ExperimentId(str(experiment_id)))

    async def close(self) -> None:
        """Release the database. Nothing is stopped: there is nothing here to stop."""
        self._client.close()

    async def __aenter__(self) -> LocalDaemonControllerHost:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    # ---- what the handle asks -------------------------------------------

    def _experiment_status(self, experiment_id: ExperimentId) -> ExperimentStatus:
        return self._repository.aggregates.load_experiment(str(experiment_id)).status

    def _actions(self, experiment_id: ExperimentId) -> tuple[GovernedAction, ...]:
        return tuple(
            self._repository.governed_action(action.id)
            for action in self._repository.actions.for_experiment(str(experiment_id))
        )

    def _events_after(self, experiment_id: ExperimentId, sequence: int) -> tuple[DomainEvent, ...]:
        return self._repository.events.events_for_experiment_after(str(experiment_id), sequence)

    async def _cancel(self, experiment_id: ExperimentId, *, reason: str) -> None:
        await self.cancel(experiment_id, reason=reason)

    async def _propose(
        self,
        experiment_id: ExperimentId,
        spec: ActionSpec,
        *,
        reason: str,
        proposed_by: Actor,
        request_id: ControllerRequestId | str | None = None,
    ) -> GovernedAction:
        return await self.propose(
            experiment_id, spec, reason=reason, proposed_by=proposed_by, request_id=request_id
        )

    async def _wait(self, experiment_id: ExperimentId) -> ExperimentResult:
        while True:
            result = self._at_rest(experiment_id)
            if result is not None:
                return result
            await asyncio.sleep(self._poll_interval)

    def _at_rest(self, experiment_id: ExperimentId) -> ExperimentResult | None:
        """The result, if the experiment is at rest; ``None`` while the daemon may still move it.

        At rest: terminal with nothing unsettled; or the daemon's controller
        came to rest on it at its latest event, and no request for it is
        waiting to be carried out. Read from one snapshot, so the rest, the
        latest event and the result all describe the same state.
        """
        repository = self._repository
        with read_snapshot(self._client._connection):
            result = experiment_result(repository, experiment_id)
            if result.next_stage is None and result.quiescent:
                return result
            if any(
                request.is_unfinished
                for request in repository.controller_requests.for_experiment(str(experiment_id))
            ):
                return None
            rest = repository.controller_requests.rest(str(experiment_id))
            latest = repository.events.latest_sequence_for_experiment(str(experiment_id))
        if rest is None or rest.sequence != latest:
            return None
        if rest.escalation is not None:
            error = _ESCALATIONS.get(str(rest.escalation.get("type")), ReconciliationEscalatedError)
            raise error(str(rest.escalation.get("message", "")))
        return result if result.quiescent else None

    # ---- requests -----------------------------------------------------------

    async def _resolve(
        self,
        kind: Literal["approve-action", "reject-action"],
        action_id: ActionId | str,
        approver: Actor,
        reason: str,
        request_id: ControllerRequestId | str | None,
    ) -> GovernedAction:
        action = self._repository.actions.get(str(action_id))
        if action is None:
            raise AggregateNotFoundError("Action", str(action_id))
        request = ControllerRequest.resolve(
            kind,
            action.id,
            action.experiment_id,
            approver=approver,
            reason=reason,
            request_id=None if request_id is None else ControllerRequestId(str(request_id)),
        )
        await self._hand_off(request)
        return self._repository.governed_action(action.id)

    def _recorded(self, request_id: ControllerRequestId | str | None) -> ControllerRequest | None:
        return None if request_id is None else self._client.requests.get(str(request_id))

    async def _hand_off(self, request: ControllerRequest) -> ControllerRequest:
        """Send *request*, or find it already sent; wait until the daemon has carried it out.

        Raises:
            ControllerRequestFailedError: If it is ``FAILED``.
            IdempotencyConflictError: If its id was sent with another request.
            TimeoutError: After ``handoff_timeout``.
        """
        sent = self._client.send(request)
        try:
            done = await self._client.wait_for_handoff(
                sent.id, timeout=self._handoff_timeout, poll_interval=self._poll_interval
            )
        except asyncio.TimeoutError:
            # The builtin, on every Python: before 3.11 asyncio's is another class.
            raise TimeoutError(
                f"{sent.kind} request {sent.id} was not carried out within "
                f"{self._handoff_timeout}s; it stays in the mailbox for the daemon"
            ) from None
        if done.state is ControllerRequestState.FAILED:
            raise ControllerRequestFailedError(done)
        return done
