"""LocalDaemonControllerServer: the persistent local controller process (PR-027-PR-029).

The server side of the local daemon. It is not a ``ControllerHost``: callers
hand it work through the request mailbox -- with
:class:`~xaytune.daemon.LocalDaemonControllerHost`, whose ``submit()`` and
``attach()`` return an ``ExperimentHandle`` over it, or the lower-level
:class:`~xaytune.daemon.DaemonClient` (ADR-004 §1).

```text
acquire <db>.lock (flock)          another daemon holds it → refuse to start
acquire the controller lease       a live one is held → wait for it to expire;
                                   each new owner is the next epoch
open the controller                an EmbeddedControllerHost recording
                                   ControllerHostRef(kind="local_daemon"),
                                   every write fenced by this lease and epoch
recover unfinished requests        as below, once
sweep owned experiments            attach() every nonterminal experiment a
                                   daemon admitted or adopted
poll controller_requests           renewing the lease every TTL/3
  submit  PENDING   validate, bind, compile; admit in one transaction
                    (request → ACCEPTED), then issue; → COMPLETED
                    a definitive refusal before admission → FAILED
          ACCEPTED  admitted by an earlier session: attach(), which
                    reconciles its INTENDED submit (ADR-013); → COMPLETED
  attach  PENDING   attach(); → COMPLETED, or FAILED if it cannot be
  cancel, propose-action, approve-action, reject-action
          PENDING   record the intent the request names -- FAILED if that
                    is refused, having written nothing -- then attach() and
                    carry it on; → COMPLETED. Once the intent is recorded
                    nothing fails the request: it stays PENDING to retry
after each request and each swept experiment
                    wait for the controller to come to rest on it, and
                    record that rest for clients' wait()
SIGTERM / SIGINT   stop dequeuing, cancel observers, close the runtimes,
                   stop renewing, expire the lease, release the flock last
lease lost         the same, without touching the lease; LeaseLostError
```

The daemon owns its controller: it is the only component calling runtime
effects for the experiments it owns, and a client never runs a controller
against its database. Workloads are the runtime's; shutting down the daemon
never cancels one, and writes nothing synthetic.

**Ownership.** The flock keeps a second daemon on this machine out; the lease
is the durable ownership every controller write proves (ADR-004 §8). A write
from an epoch that is no longer current, or whose lease expired, raises
:class:`~xaytune.storage.leases.LeaseLostError` having written nothing -- and
wherever it is raised, in a mailbox request, an observer, recovery or the
sweep, it stops the daemon: it no longer has authority over the database.
While the lease is live an embedded host can read the database, not write it.

**Every mutation is a request** (PR-029). A client cancels, proposes,
approves and rejects by mailbox request, never by writing the record or
calling a runtime itself. Each carries the identity of what it records -- the
``cancel-experiment`` or proposed action's id, minted by the client, or the
action it resolves -- so carrying it out again, after a crash or a retry,
finds what the first attempt recorded. The intent is recorded first, and is
the only step that can fail the request; the experiment is then attached, so
the daemon observes the effects it causes and owns the experiment from then
on, and only then are the effects carried out. A ``FAILED`` request has
changed nothing, not even ownership.

**Rest.** A client cannot run the controller's ``wait()``, and the record
alone cannot tell an experiment at rest from one between two steps. So once
the controller has nothing left it can do for an experiment, the daemon
records that rest at the experiment's latest event
(:meth:`~xaytune.storage.ControlPlaneRepository.record_controller_rest`); a
client's ``wait()`` returns while that is still the latest event.

**Restart.** Unfinished requests are carried over first -- ``PENDING`` ones
processed, ``ACCEPTED`` ones resumed. Then every nonterminal experiment the
daemon is responsible for -- admitted by a daemon, or adopted by a
``COMPLETED`` attach request -- is attached, which reconciles it from the
record: a running workload is adopted, not recreated. Its budget ledger is
settled first; whether a used-up quota stops anything is left to the
controller's own effect boundaries, exactly as without a restart. No
time-based limit exists, so the downtime itself costs nothing. A ``PAUSED``
experiment is swept, and starts nothing new.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import uuid
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from xaytune.compilation import UnsupportedCandidateError
from xaytune.core.clock import utc_now
from xaytune.core.domain.action import Action, ActionTarget, UnknownActionTypeError
from xaytune.core.domain.actions import ActionPayloadError, UnsupportedActionError, spec_of
from xaytune.core.domain.budget import UnsupportedBudgetError
from xaytune.core.domain.controller_request import (
    ControllerRequest,
    ControllerRequestState,
    MisdirectedRequestError,
)
from xaytune.core.domain.numerical_recovery import UnsupportedNumericalRecoveryError
from xaytune.core.errors import IdempotencyConflictError, XaytuneError
from xaytune.core.ids import ExperimentId
from xaytune.core.immutable import FrozenDict, thaw
from xaytune.core.refs import Actor, ControllerHostRef
from xaytune.core.sqlite import connect
from xaytune.daemon.config import DaemonConfig
from xaytune.daemon.lock import StateDatabaseLock
from xaytune.evaluation import UnsupportedEvaluationError
from xaytune.experiment.host import (
    ControllerNotRunningError,
    EmbeddedControllerHost,
    ImplementationMismatchError,
    ReconciliationEscalatedError,
    UnknownImplementationError,
)
from xaytune.experiment.spec import ExperimentSpec
from xaytune.planning import PlannerConfigurationError
from xaytune.resilience.provider import (
    ResilienceProviderConfigurationError,
    UnsupportedResilienceError,
)
from xaytune.storage.control_plane import (
    AdmissionRefusedError,
    ApprovalConflictError,
    ApprovalError,
    CancellationInFlightError,
    CancellationNotGovernedError,
)
from xaytune.storage.errors import AggregateNotFoundError
from xaytune.storage.leases import (
    ControllerLease,
    ControllerLeaseFence,
    ControllerLeaseStore,
    LeaseLostError,
)
from xaytune.storage.migrations import migrate

__all__ = ["LocalDaemonControllerServer"]

_LOG = logging.getLogger("xaytune.daemon")

_SUBMIT_REFUSALS: tuple[type[Exception], ...] = (
    ValidationError,
    UnknownImplementationError,
    ImplementationMismatchError,
    UnsupportedCandidateError,
    UnsupportedEvaluationError,
    UnsupportedBudgetError,
    UnsupportedNumericalRecoveryError,
    PlannerConfigurationError,
    ResilienceProviderConfigurationError,
    UnsupportedResilienceError,
    AdmissionRefusedError,
)
"""The refusals that make a submission ``FAILED``, and only before admission.

Each is a definitive answer about the request itself: its payload is not a
valid spec, or not the canonical form of the spec it parses to; it names an
implementation this daemon does not have, or has at another version; the
candidate, evaluation, budget, numerical-recovery policy or planner
configuration is refused by what would run it; its resilience provider refuses
its configuration or its installed engine, or the runtime, compiler and
provider cannot carry the request it delegates (PR-035); its experiment id is
taken. Anything else -- a plugin raising a plain ``ValueError``, a resilience
provider breaking its contract (``ResilienceContractError``, a defect in the
plugin, not in the request), the database busy, an environment problem -- is
not a judgement on the request: it stays
``PENDING`` and is tried again, because ``FAILED`` cannot be undone."""

_ATTACH_REFUSALS: tuple[type[Exception], ...] = (
    AggregateNotFoundError,
    UnknownImplementationError,
    ImplementationMismatchError,
)
"""An attach that cannot succeed unchanged: no such experiment, or its
recorded implementations are absent from this daemon or at other versions."""

_ACTION_REFUSALS: tuple[type[Exception], ...] = (
    AggregateNotFoundError,
    UnknownImplementationError,
    ImplementationMismatchError,
    ValidationError,
    MisdirectedRequestError,
    IdempotencyConflictError,
    CancellationInFlightError,
    CancellationNotGovernedError,
    UnknownActionTypeError,
    ActionPayloadError,
    UnsupportedActionError,
    ApprovalError,
    ApprovalConflictError,
)
"""The refusals that make a cancel, propose or approval request ``FAILED``.

Raised only by the step that records the request's intent, which writes
nothing when it raises: no such experiment or action, a runtime this daemon
cannot judge a proposal against, a payload that is not an actor or a
well-formed action, an action of another experiment, an id already recorded
for something else, another cancellation in flight, a cancellation proposed
as a governed action, a type or spec nothing registered accepts, an approval
by a non-human, of an action not awaiting one, or that a human already
resolved otherwise. Nothing after the intent is recorded -- the attach, the
effects -- is caught here."""

_ESCALATIONS: tuple[type[Exception], ...] = (
    ReconciliationEscalatedError,
    ControllerNotRunningError,
)
"""Why the controller's ``wait()`` stops short; recorded with the rest."""


class LocalDaemonControllerServer:
    """A foreground controller process's work, over one state database.

    Args:
        state_path: The control-plane database. Created and migrated if new.
            It must be on a local filesystem: SQLite on NFS, SMB or another
            shared filesystem is unsupported, for the lock and the lease alike.
        config: Every implementation the controller uses (ADR-004 §6), and
            the lease TTL.
        poll_interval: Seconds between looks at the mailbox. Latency, not
            correctness: a request committed while the daemon sleeps is
            durable, and found on the next look.
        instance_id: This process's id: the ``controller_id`` of the lease it
            holds, and the ``local_daemon`` :class:`ControllerHostRef` of
            every experiment it admits. The second is provenance -- which
            process admitted an experiment -- and is never rewritten when a
            later daemon takes over.
    """

    def __init__(
        self,
        state_path: Path | str,
        config: DaemonConfig,
        *,
        poll_interval: float = 0.5,
        instance_id: str | None = None,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self._lock = StateDatabaseLock(state_path)
        self._state_path = Path(state_path)
        self._config = config
        self._poll_interval = poll_interval
        self._ttl = timedelta(seconds=config.lease_ttl_seconds)
        self.instance_id = instance_id or f"daemon-{uuid.uuid4().hex}"
        self.reference = ControllerHostRef(kind="local_daemon", id=self.instance_id)
        self._controller: EmbeddedControllerHost | None = None
        self._lease: ControllerLease | None = None
        self._lost: LeaseLostError | None = None
        # Experiments a request is being carried out for, whose rest must
        # wait until it is; and the task waiting for each experiment's rest.
        self._busy: set[str] = set()
        self._resting: dict[str, asyncio.Task[None]] = {}

    @property
    def controller(self) -> EmbeddedControllerHost:
        """The controller this daemon delegates to, while it serves."""
        if self._controller is None:
            raise RuntimeError("the daemon is not serving")
        return self._controller

    @property
    def lease(self) -> ControllerLease | None:
        """The lease this daemon holds, as last acquired or renewed; ``None`` before."""
        return self._lease

    async def serve(self, stop: asyncio.Event) -> None:
        """Own the database and carry out its requests until *stop* is set.

        Raises:
            DaemonAlreadyRunningError: If another daemon holds the database.
            UnsupportedPlatformError: If the platform cannot lock it.
            LeaseLostError: If the daemon lost its lease while serving: it
                stopped, having written nothing after the loss.
        """
        self._lock.acquire(
            {
                "pid": os.getpid(),
                "instance_id": self.instance_id,
                "started_at": utc_now().isoformat(),
                "state_db": str(self._state_path.resolve()),
            }
        )
        try:
            connection = connect(self._state_path)
            try:
                migrate(connection)
                await self._serve_owned(ControllerLeaseStore(connection), stop)
            finally:
                connection.close()
        finally:
            # Last: until here another daemon must not start on this database.
            self._lock.release()
            _LOG.info("xaytune daemon %s stopped", self.instance_id)
        if self._lost is not None:
            raise self._lost

    async def _serve_owned(self, leases: ControllerLeaseStore, stop: asyncio.Event) -> None:
        """Take the lease, then control the database under it until stopped or it is lost."""
        lease = await self._acquire_lease(leases, stop)
        if lease is None:
            return
        self._lease = lease
        self._lost = None
        halt = asyncio.Event()
        relay = asyncio.ensure_future(_relay(stop, halt))
        loop = asyncio.get_running_loop()

        def lost(error: LeaseLostError) -> None:
            if self._lost is None:
                self._lost = error
                _LOG.error("xaytune daemon %s stopping: %s", self.instance_id, error)
            loop.call_soon_threadsafe(halt.set)

        config = self._config
        heartbeat = asyncio.ensure_future(self._heartbeat(leases, lost))
        try:
            self._controller = EmbeddedControllerHost(
                self._state_path,
                compilers=config.compilers,
                runtimes=config.runtimes,
                evaluators=config.evaluators,
                planners=config.planners,
                resilience_providers=config.resilience_providers,
                decision_engine=config.decision_engine,
                policy=config.policy,
                checkpoint_manager=config.checkpoint_manager,
                recovery_request_for_incident=config.recovery_request_for_incident,
                controller_host=self.reference,
                lease_fence=ControllerLeaseFence(self.instance_id, lease.epoch, on_lost=lost),
            )
            _LOG.info(
                "xaytune daemon %s serving %s at lease epoch %d",
                self.instance_id,
                self._state_path,
                lease.epoch,
            )
            # The incomplete handoffs first, then everything else it owns.
            await _until(self.process_requests(halt), halt)
            await _until(self.sweep(halt), halt)
            while not halt.is_set():
                await _until(asyncio.sleep(self._poll_interval), halt)
                await _until(self.process_requests(halt), halt)
        except LeaseLostError as error:
            lost(error)
        finally:
            # Observers are cancelled and runtimes closed; workloads keep
            # running, and every uncertain effect stays in the journal. Only
            # then does the lease go, so nothing of this controller writes
            # once another may own the database.
            resting = list(self._resting.values())
            for task in resting:
                task.cancel()
            await asyncio.gather(*resting, return_exceptions=True)
            self._resting.clear()
            if self._controller is not None:
                await self._controller.close()
                self._controller = None
            for task in (heartbeat, relay):
                task.cancel()
            await asyncio.gather(heartbeat, relay, return_exceptions=True)
            if self._lost is None:
                leases.release(self._lease)

    async def _acquire_lease(
        self, leases: ControllerLeaseStore, stop: asyncio.Event
    ) -> ControllerLease | None:
        """The lease as a new epoch, waiting out a live one; ``None`` if stopped first.

        A live lease is waited out, never taken: its owner may still be
        running, and only its expiry -- not a host, PID or process check --
        says it no longer owns the database.
        """
        waiting_on: tuple[str, int] | None = None
        while not stop.is_set():
            lease = leases.acquire(self.instance_id, self._ttl)
            if lease is not None:
                return lease
            held = leases.current()
            if held is None:
                continue
            if waiting_on != (held.controller_id, held.epoch):
                waiting_on = (held.controller_id, held.epoch)
                _LOG.warning(
                    "xaytune daemon %s waiting for the lease of %s (epoch %d) to expire at %s",
                    self.instance_id,
                    held.controller_id,
                    held.epoch,
                    held.lease_expires_at.isoformat(),
                )
            remaining = (held.lease_expires_at - utc_now()).total_seconds()
            await _until(asyncio.sleep(min(max(remaining, 0.01), self._renew_every)), stop)
        return None

    @property
    def _renew_every(self) -> float:
        return self._ttl.total_seconds() / 3

    async def _heartbeat(
        self, leases: ControllerLeaseStore, lost: Callable[[LeaseLostError], None]
    ) -> None:
        """Renew the lease every TTL/3 until cancelled, or until it cannot be renewed.

        A renewal the database is too busy to take is tried again next
        time; one refused because the lease moved on, or already expired,
        is the end of this daemon's ownership.
        """
        while True:
            await asyncio.sleep(self._renew_every)
            assert self._lease is not None
            try:
                self._lease = leases.renew(self._lease, self._ttl)
            except LeaseLostError as error:
                lost(error)
                return
            except sqlite3.OperationalError as busy:
                _LOG.warning("daemon %s could not renew its lease: %s", self.instance_id, busy)

    async def sweep(self, stop: asyncio.Event | None = None) -> tuple[ExperimentId, ...]:
        """Reconcile every nonterminal experiment the daemon owns; return those swept.

        Owned means admitted by a daemon or adopted by a ``COMPLETED`` attach
        request (:meth:`~xaytune.storage.ControlPlaneRepository.daemon_responsibilities`).
        Each has its budget ledger settled, then is attached:
        ``attach()`` reconciles it from the record (ADR-013), and the
        controller paths it resumes carry it on. Running it again finds the
        same record and repeats nothing -- no second run, attempt or
        operation. One experiment that cannot be reconciled is logged and
        left for the next sweep; the rest are still swept.
        """
        controller = self.controller
        swept: list[ExperimentId] = []
        for experiment_id in controller.repository.daemon_responsibilities():
            if stop is not None and stop.is_set():
                break
            self._busy.add(str(experiment_id))
            try:
                controller._settle_ledger(experiment_id)
                await controller.attach(experiment_id)
            except (asyncio.CancelledError, LeaseLostError):
                raise
            except Exception:
                _LOG.exception("experiment %s could not be reconciled at startup", experiment_id)
                continue
            finally:
                self._busy.discard(str(experiment_id))
            swept.append(experiment_id)
            self._rest_when_settled(experiment_id)
        if swept:
            _LOG.info("xaytune daemon %s reconciled %d experiment(s)", self.instance_id, len(swept))
        return tuple(swept)

    async def process_requests(self, stop: asyncio.Event | None = None) -> int:
        """Carry out every unfinished request once, oldest first; return how many were looked at.

        Stops dequeuing as soon as *stop* is set.
        """
        requests = self.controller.repository.controller_requests.unfinished()
        for count, request in enumerate(requests):
            if stop is not None and stop.is_set():
                return count
            await self._process(request)
        return len(requests)

    async def _process(self, request: ControllerRequest) -> None:
        experiment_id = str(request.experiment_id)
        self._busy.add(experiment_id)
        try:
            if request.kind == "submit":
                await self._submit(request)
            elif request.kind == "attach":
                await self._attach(request)
            else:
                await self._act(request)
        except (asyncio.CancelledError, LeaseLostError):
            raise
        except Exception:
            # Left as it is: PENDING is tried again, and ACCEPTED has an
            # admitted experiment whose outcome the journal settles -- never
            # this request.
            _LOG.exception("controller request %s left %s", request.id, self._state_of(request))
        finally:
            self._busy.discard(experiment_id)
        self._rest_when_settled(request.experiment_id)

    async def _submit(self, request: ControllerRequest) -> None:
        controller = self.controller
        if request.state is ControllerRequestState.PENDING:
            try:
                spec = ExperimentSpec.model_validate(thaw(request.payload))
                handle = await controller._submit(
                    spec, experiment_id=request.experiment_id, request_id=request.id
                )
            except _SUBMIT_REFUSALS as refusal:
                if self._fail_if_pending(request, refusal):
                    return
                raise
            if handle is None:
                # It was no longer PENDING: admitted already. Resume it as below.
                await controller.attach(request.experiment_id)
        else:
            # Admitted by an earlier session, which stopped before the handoff
            # completed: the attempt and its INTENDED submit exist, so this is
            # adoption -- reconciliation looks the operation up -- never a
            # second experiment, run or attempt.
            await controller.attach(request.experiment_id)
        current = self._current(request)
        if current.state is ControllerRequestState.ACCEPTED:
            controller.repository.complete_controller_request(
                current.id, expected_revision=current.revision
            )

    async def _attach(self, request: ControllerRequest) -> None:
        try:
            await self.controller.attach(request.experiment_id)
        except _ATTACH_REFUSALS as refusal:
            if self._fail_if_pending(request, refusal):
                return
            raise
        current = self._current(request)
        if current.state is ControllerRequestState.PENDING:
            self.controller.repository.complete_controller_request(
                current.id, expected_revision=current.revision
            )

    async def _act(self, request: ControllerRequest) -> None:
        """Carry out a cancel, propose-action, approve-action or reject-action request.

        Record the intent the request names -- the only step whose refusal
        fails it, and one that calls no runtime. Then attach the experiment,
        so the daemon observes what follows and owns it, and carry the intent
        on: issue a cancellation's effects, drive a recovery an approval
        released. Anything that goes wrong after the intent is recorded
        leaves the request ``PENDING``; repeating it finds the intent
        already recorded, by the identity the request carries.
        """
        controller = self.controller
        try:
            carry_on = await self._record(request)
        except _ACTION_REFUSALS as refusal:
            if self._fail_if_pending(request, refusal):
                return
            raise
        await controller.attach(request.experiment_id)
        await carry_on()
        current = self._current(request)
        if current.state is ControllerRequestState.PENDING:
            controller.repository.complete_controller_request(
                current.id, expected_revision=current.revision
            )

    async def _record(self, request: ControllerRequest) -> Callable[[], Awaitable[Any]]:
        """Record the intent *request* names; return what carries it on."""
        controller = self.controller
        payload = thaw(request.payload)
        action_id = request.action_id
        assert action_id is not None
        if request.kind == "cancel":
            controller._request_cancellation(
                request.experiment_id, reason=payload["reason"], action_id=action_id
            )
            # Carried on from the saga as the record has it then -- the attach
            # in between may have moved its operations -- replayed by its id.
            return lambda: controller._cancel(
                request.experiment_id, reason=payload["reason"], action_id=action_id
            )
        if request.kind == "propose-action":
            proposed_by = Actor.model_validate(payload["proposed_by"])
            proposed = Action(
                id=action_id,
                experiment_id=request.experiment_id,
                type=payload["type"],
                target=ActionTarget.model_validate(payload["target"]),
                proposed_by=proposed_by,
                reason=payload["reason"],
                payload=payload["payload"],
            )
            await controller._propose(
                request.experiment_id,
                spec_of(proposed),
                reason=payload["reason"],
                proposed_by=proposed_by,
                action_id=action_id,
            )
            return _nothing
        approver = Actor.model_validate(payload["approver"])
        action = controller.repository.actions.get(str(action_id))
        if action is None:
            raise AggregateNotFoundError("Action", str(action_id))
        if action.experiment_id != request.experiment_id:
            raise MisdirectedRequestError(
                f"action {action_id} belongs to experiment {action.experiment_id}, "
                f"not {request.experiment_id}"
            )
        if request.kind == "approve-action":
            controller.repository.approve_action(
                action_id, approver=approver, reason=payload["reason"]
            )
            return lambda: controller._carry_on_approval(action_id)
        controller.repository.reject_action(action_id, approver=approver, reason=payload["reason"])
        return lambda: controller._carry_on_rejection(
            action_id, approver=approver, reason=payload["reason"]
        )

    def _rest_when_settled(self, experiment_id: ExperimentId) -> None:
        """Make sure something will record the experiment's next rest.

        A task already waiting for it will: its last look comes after
        whatever was just started. Otherwise start one.
        """
        task = self._resting.get(str(experiment_id))
        if task is not None and not task.done():
            return
        if task is not None and not task.cancelled() and task.exception() is not None:
            # A lost lease, reported to the server already; retrieved here so
            # it is not reported again as never retrieved.
            _LOG.debug("rest of %s ended: %s", experiment_id, task.exception())
        if self.controller.repository.aggregates.get_experiment(str(experiment_id)) is None:
            return
        self._resting[str(experiment_id)] = asyncio.ensure_future(self._rest(experiment_id))

    async def _rest(self, experiment_id: ExperimentId) -> None:
        """Wait until the controller has nothing left it can do for the experiment; record it.

        The controller's own ``wait()``, then the record, with no await
        between them: nothing this daemon does can start in between, so the
        rest names exactly the state it waited for. Skipped while a request
        for the experiment is being carried out, or the sweep is attaching it
        -- it is halfway, not at rest -- and that work's end starts the next
        wait.
        """
        controller = self.controller
        escalation: dict[str, Any] | None = None
        try:
            await controller._wait(experiment_id)
        except _ESCALATIONS as stopped:
            escalation = {"type": type(stopped).__name__, "message": str(stopped)}
        except (asyncio.CancelledError, LeaseLostError):
            raise
        except Exception:
            _LOG.exception("could not wait for experiment %s to come to rest", experiment_id)
            return
        if str(experiment_id) in self._busy:
            return
        controller.repository.record_controller_rest(
            experiment_id, controller_id=self.instance_id, escalation=escalation
        )

    def _fail_if_pending(self, request: ControllerRequest, refusal: Exception) -> bool:
        """FAILED, if nothing was admitted for it -- it is still PENDING. Whether it was."""
        current = self._current(request)
        if current.state is not ControllerRequestState.PENDING:
            return False
        self.controller.repository.fail_controller_request(
            current.id,
            expected_revision=current.revision,
            error=FrozenDict({"type": type(refusal).__name__, "message": str(refusal)}),
        )
        _LOG.warning("controller request %s failed: %s", request.id, refusal)
        return True

    def _current(self, request: ControllerRequest) -> ControllerRequest:
        current = self.controller.repository.controller_requests.get(str(request.id))
        assert current is not None, "controller requests are permanent"
        return current

    def _state_of(self, request: ControllerRequest) -> str:
        try:
            return self._current(request).state.value
        except (XaytuneError, AssertionError, RuntimeError):
            return "unknown"


async def _nothing() -> None:
    """Nothing to carry on: the intent was all the request asked for."""


async def _until(work: Awaitable[Any], stop: asyncio.Event) -> None:
    """Run *work* until it finishes or *stop* is set, cancelling it in the second case.

    A submission cancelled mid-issue stays ``INTENDED`` under an ``ACCEPTED``
    request: the next daemon resumes it through reconciliation.
    """
    task = asyncio.ensure_future(work)
    stopped = asyncio.ensure_future(stop.wait())
    try:
        await asyncio.wait({task, stopped}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for pending in (task, stopped):
            if not pending.done():
                pending.cancel()
        await asyncio.gather(task, stopped, return_exceptions=True)
    if task.done() and not task.cancelled():
        task.result()


async def _relay(stop: asyncio.Event, halt: asyncio.Event) -> None:
    """Set *halt* once *stop* is: a stop request is one of the reasons to halt."""
    await stop.wait()
    halt.set()
