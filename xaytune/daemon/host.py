"""LocalDaemonControllerHost: a persistent local controller over one state database (PR-027).

```text
acquire <db>.lock (flock)          another daemon holds it → refuse to start
open the controller                an EmbeddedControllerHost recording
                                   ControllerHostRef(kind="local_daemon")
poll controller_requests
  submit  PENDING   validate, bind, compile; admit in one transaction
                    (request → ACCEPTED), then issue; → COMPLETED
                    a definitive refusal before admission → FAILED
          ACCEPTED  admitted by an earlier session: attach(), which
                    reconciles its INTENDED submit (ADR-013); → COMPLETED
  attach  PENDING   attach(); → COMPLETED, or FAILED if it cannot be
SIGTERM / SIGINT   stop dequeuing, cancel observers, close the runtimes
                   and the database, release the lock last
```

The daemon owns its controller: it is the only component calling runtime
effects for the requests it admitted, and a client never runs a controller
against its database. Workloads are the runtime's; shutting down the daemon
never cancels one, and writes nothing synthetic.

**Restart.** Only *unfinished* requests are carried over -- ``PENDING`` ones
processed, ``ACCEPTED`` ones resumed. An experiment whose request is
``COMPLETED`` is not adopted again by restarting the daemon; an explicit
``attach`` request adopts it. Recovering every active experiment at startup,
leases, and deadline or budget re-evaluation after downtime are PR-028.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

from xaytune.core.clock import utc_now
from xaytune.core.domain.controller_request import ControllerRequest, ControllerRequestState
from xaytune.core.errors import XaytuneError
from xaytune.core.immutable import FrozenDict, thaw
from xaytune.core.refs import ControllerHostRef
from xaytune.daemon.config import DaemonConfig
from xaytune.daemon.lock import StateDatabaseLock
from xaytune.experiment.host import (
    EmbeddedControllerHost,
    ImplementationMismatchError,
    UnknownImplementationError,
)
from xaytune.experiment.spec import ExperimentSpec
from xaytune.storage.errors import StorageError

__all__ = ["LocalDaemonControllerHost"]

_LOG = logging.getLogger("xaytune.daemon")

_DEFINITIVE = (ValueError, UnknownImplementationError, ImplementationMismatchError, StorageError)
"""Refusals that are final for a request nothing was admitted for: an invalid
payload or spec (pydantic's ``ValidationError`` is a ``ValueError``), an
implementation this daemon does not have or has at another version, a
candidate, evaluation, budget or numerical-recovery policy it cannot run, an
experiment id already taken. Anything else -- the database busy, an
unexpected exception -- leaves the request ``PENDING``, to be tried again."""


class LocalDaemonControllerHost:
    """A foreground controller process's work, over one state database.

    Args:
        state_path: The control-plane database. Created and migrated if new.
        config: Every implementation the controller uses (ADR-004 §6).
        poll_interval: Seconds between looks at the mailbox. Latency, not
            correctness: a request committed while the daemon sleeps is
            durable, and found on the next look.
        instance_id: This process's id, recorded as the ``local_daemon``
            :class:`ControllerHostRef` of every experiment it admits.
            Provenance only -- not a lease or a durable controller identity.
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
        self.instance_id = instance_id or f"daemon-{uuid.uuid4().hex}"
        self.reference = ControllerHostRef(kind="local_daemon", id=self.instance_id)
        self._controller: EmbeddedControllerHost | None = None

    @property
    def controller(self) -> EmbeddedControllerHost:
        """The controller this daemon delegates to, while it serves."""
        if self._controller is None:
            raise RuntimeError("the daemon is not serving")
        return self._controller

    async def serve(self, stop: asyncio.Event) -> None:
        """Own the database and carry out its requests until *stop* is set.

        Raises:
            DaemonAlreadyRunningError: If another daemon holds the database.
            UnsupportedPlatformError: If the platform cannot lock it.
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
            config = self._config
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
            )
            _LOG.info("xaytune daemon %s serving %s", self.instance_id, self._state_path)
            try:
                while not stop.is_set():
                    await _until(self.process_requests(stop), stop)
                    if not stop.is_set():
                        await _until(asyncio.sleep(self._poll_interval), stop)
            finally:
                # Observers are cancelled and runtimes closed; workloads keep
                # running, and every uncertain effect stays in the journal.
                await self._controller.close()
                self._controller = None
        finally:
            self._lock.release()
            _LOG.info("xaytune daemon %s stopped", self.instance_id)

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
        except asyncio.CancelledError:
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
            except _DEFINITIVE as refusal:
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
        except _DEFINITIVE as refusal:
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
