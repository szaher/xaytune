"""The client side of the daemon's mailbox: write requests, read durable state (ADR-004 §6).

```python
with DaemonClient("state.db") as client:
    request = client.submit(spec)            # committed: handed off
    request = await client.wait_for_handoff(request.id)
    experiment = client.aggregates.load_experiment(str(request.experiment_id))
```

A client never runs a controller. It records what it wants and reads what the
daemon did; the daemon, which owns the database, is the only process calling
runtime effects for it. Once :meth:`DaemonClient.submit` returns, the request
is durable: the client may exit, and the daemon still finds it.

**Retrying.** A request is idempotent by its id. To retry one whose answer was
lost, send the same :class:`ControllerRequest` again with :meth:`send` -- not
a new :meth:`submit`, which mints a new request and a new experiment id.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import TracebackType

from xaytune.core.domain.controller_request import ControllerRequest, ControllerRequestState
from xaytune.core.ids import ControllerRequestId, ExperimentId
from xaytune.core.sqlite import connect
from xaytune.experiment.spec import ExperimentSpec
from xaytune.storage.control_plane import ControlPlaneRepository
from xaytune.storage.errors import AggregateNotFoundError
from xaytune.storage.migrations import migrate
from xaytune.storage.repository import AggregateStore
from xaytune.storage.requests import ControllerRequestStore

__all__ = ["DaemonClient"]


class DaemonClient:
    """Requests to the daemon serving *state_path*, and reads of its record."""

    def __init__(self, state_path: Path | str) -> None:
        if str(state_path) == ":memory:":
            raise ValueError("an in-memory database is not shared with a daemon")
        Path(state_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = connect(state_path)
        migrate(self._connection)
        self._repository = ControlPlaneRepository(self._connection)

    @property
    def aggregates(self) -> AggregateStore:
        """The durable experiments, nodes, runs and attempts, to read."""
        return self._repository.aggregates

    @property
    def requests(self) -> ControllerRequestStore:
        """The mailbox, to read."""
        return self._repository.controller_requests

    def submit(
        self,
        spec: ExperimentSpec,
        *,
        request_id: ControllerRequestId | None = None,
        experiment_id: ExperimentId | None = None,
    ) -> ControllerRequest:
        """Hand *spec* to the daemon; the experiment it becomes is named now.

        Returns:
            The committed request; its ``experiment_id`` is the experiment the
            daemon will admit, or has admitted.
        """
        return self.send(
            ControllerRequest.submit(
                spec.submission_payload(), experiment_id=experiment_id, request_id=request_id
            )
        )

    def attach(
        self, experiment_id: ExperimentId | str, *, request_id: ControllerRequestId | None = None
    ) -> ControllerRequest:
        """Ask the daemon to adopt an experiment already in the record."""
        return self.send(
            ControllerRequest.attach(ExperimentId(experiment_id), request_id=request_id)
        )

    def send(self, request: ControllerRequest) -> ControllerRequest:
        """Commit *request*, or return it as already recorded.

        Raises:
            IdempotencyConflictError: If its id was recorded with another request.
        """
        return self._repository.record_controller_request(request)

    def request(self, request_id: ControllerRequestId | str) -> ControllerRequest:
        """The request as it stands now.

        Raises:
            AggregateNotFoundError: If no such request was recorded.
        """
        request = self.requests.get(str(request_id))
        if request is None:
            raise AggregateNotFoundError("ControllerRequest", str(request_id))
        return request

    async def wait_for_handoff(
        self,
        request_id: ControllerRequestId | str,
        *,
        timeout: float | None = None,
        poll_interval: float = 0.2,
    ) -> ControllerRequest:
        """Poll until the request is ``COMPLETED`` or ``FAILED``.

        ``COMPLETED`` means the daemon reached the point at which ``submit()``
        or ``attach()`` returns -- not that the experiment has finished.

        Raises:
            TimeoutError: If *timeout* seconds pass first.
        """

        async def poll() -> ControllerRequest:
            while True:
                request = self.request(request_id)
                if request.state in (
                    ControllerRequestState.COMPLETED,
                    ControllerRequestState.FAILED,
                ):
                    return request
                await asyncio.sleep(poll_interval)

        return await asyncio.wait_for(poll(), timeout)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> DaemonClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
