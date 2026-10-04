"""LocalDaemonControllerServer: the persistent local controller process (PR-027, PR-028).

The server side of the local daemon. It is not a ``ControllerHost``: callers
hand it work through :class:`~xaytune.daemon.DaemonClient` and the request
mailbox, and the caller-side ``LocalDaemonControllerHost`` -- ``submit()`` and
``attach()`` returning an ``ExperimentHandle`` over that mailbox -- is PR-029's
(ADR-004 §1).

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
from xaytune.core.domain.budget import UnsupportedBudgetError
from xaytune.core.domain.controller_request import ControllerRequest, ControllerRequestState
from xaytune.core.domain.numerical_recovery import UnsupportedNumericalRecoveryError
from xaytune.core.errors import XaytuneError
from xaytune.core.ids import ExperimentId
from xaytune.core.immutable import FrozenDict, thaw
from xaytune.core.refs import ControllerHostRef
from xaytune.core.sqlite import connect
from xaytune.daemon.config import DaemonConfig
from xaytune.daemon.lock import StateDatabaseLock
from xaytune.evaluation import UnsupportedEvaluationError
from xaytune.experiment.host import (
    EmbeddedControllerHost,
    ImplementationMismatchError,
    UnknownImplementationError,
)
from xaytune.experiment.spec import ExperimentSpec
from xaytune.planning import PlannerConfigurationError
from xaytune.storage.control_plane import AdmissionRefusedError
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
    AdmissionRefusedError,
)
"""The refusals that make a submission ``FAILED``, and only before admission.

Each is a definitive answer about the request itself: its payload is not a
valid spec, or not the canonical form of the spec it parses to; it names an
implementation this daemon does not have, or has at another version; the
candidate, evaluation, budget, numerical-recovery policy or planner
configuration is refused by what would run it; its experiment id is taken.
Anything else -- a plugin raising a plain ``ValueError``, the database busy,
an environment problem -- is not a judgement on the request: it stays
``PENDING`` and is tried again, because ``FAILED`` cannot be undone."""

_ATTACH_REFUSALS: tuple[type[Exception], ...] = (
    AggregateNotFoundError,
    UnknownImplementationError,
    ImplementationMismatchError,
)
"""An attach that cannot succeed unchanged: no such experiment, or its
recorded implementations are absent from this daemon or at other versions."""


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
            try:
                controller._settle_ledger(experiment_id)
                await controller.attach(experiment_id)
            except (asyncio.CancelledError, LeaseLostError):
                raise
            except Exception:
                _LOG.exception("experiment %s could not be reconciled at startup", experiment_id)
                continue
            swept.append(experiment_id)
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
        try:
            if request.kind == "submit":
                await self._submit(request)
            else:
                await self._attach(request)
        except (asyncio.CancelledError, LeaseLostError):
            raise
        except Exception:
            # Left as it is: PENDING is tried again, and ACCEPTED has an
            # admitted experiment whose outcome the journal settles -- never
            # this request.
            _LOG.exception("controller request %s left %s", request.id, self._state_of(request))

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
