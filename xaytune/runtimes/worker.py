"""What passes between a runtime and the worker it starts.

A worker is told where things are and reports what happened. This module is
the part of that exchange every runtime and every worker agrees on, kept out of
any one backend so that a Ray or Training Hub runtime can supervise the same
worker without either side changing.

Torch-free: both the controller-side launcher and the worker import it.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Generic, TypeVar

from xaytune.core.clock import utc_now
from xaytune.core.immutable import FrozenDomainModel
from xaytune.core.observability import CorrelationContext, TraceContext

__all__ = [
    "OBSERVATIONS_PATH_ENV",
    "TOPOLOGY_VARIABLES",
    "WORKER_CONFIG_PATH_ENV",
    "ObservationWriter",
    "WorkerObservationRecord",
]

ObservationT = TypeVar("ObservationT")

WORKER_CONFIG_PATH_ENV = "XAYTUNE_WORKER_CONFIG_PATH"
"""Where the runtime put the worker's compiled config.

A path rather than the config itself, and set by the runtime rather than the
plan: *which file* is a delivery detail of one backend, while the config's
contents are the plan, and are already part of its request digest.
"""

OBSERVATIONS_PATH_ENV = "XAYTUNE_OBSERVATIONS_PATH"
"""Where the worker appends what it observes.

Internal transport between a worker and its supervisor, not protocol: the
supervisor reads it and writes canonical envelopes elsewhere. Another runtime
can point this at a pipe or replace it outright without the worker changing.
"""

TOPOLOGY_VARIABLES: frozenset[str] = frozenset(
    {
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "ROLE_NAME",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_USE_AGENT_STORE",
        "TORCHELASTIC_ERROR_FILE",
    }
)
"""The variables through which torch.distributed learns where a process sits.

**Placement topology belongs to the runtime, never to whatever shell started
the controller.** A controller launched under ``torchrun`` has these set; a
worker that inherited them would believe it was rank 3 of 8, try to join a
process group at the controller's ``MASTER_ADDR``, and either fail or hang
waiting for seven ranks that were never started. Nothing in the plan would say
so, because nothing in the plan caused it.

So a runtime removes them from what a worker inherits, and sets them only when
it has deliberately placed the worker in a group of its own.

Behavioural knobs that happen to share the prefix -- ``NCCL_*`` tuning,
``TORCH_NCCL_ASYNC_ERROR_HANDLING`` -- are not placement and are left alone.
So is ``CUDA_VISIBLE_DEVICES``: an operator restricting which devices a
controller may use is a constraint a worker should keep, and removing it would
hand the worker every device on the machine.
"""


class WorkerObservationRecord(FrozenDomainModel, Generic[ObservationT]):
    """One observation, as a worker reports it to its supervisor.

    Carries what only the worker knows -- *when* it happened, and what it was
    correlated with -- and deliberately **nothing a supervisor assigns**:

    ```text
    target, event_id, stream_generation, sequence
    ```

    Those belong to exactly one telemetry supervisor per target (ADR-014 §1a).
    A worker that could choose a sequence would be a second writer to one
    stream, which is the condition that makes a gap indistinguishable from a
    reorder.

    ``observed_at`` exists so the envelope's ``emitted_at`` records when the
    observation happened rather than when the supervisor happened to poll.

    Generic over the observation family so the supervisor validates against
    the one its target may carry: training and evaluation vocabularies share
    members, and a single union would have duplicate discriminators.
    """

    observed_at: datetime
    observation: ObservationT
    context: CorrelationContext | None = None
    trace_context: TraceContext | None = None


class ObservationWriter:
    """Appends a worker's observations for its supervisor to read.

    **Flushed, not fsynced.** This file is transport, read from the page cache
    by a supervisor on the same machine, which sees a flushed line at once. The
    durable record is the envelope the supervisor writes -- and fsyncs -- into
    the event stream. Syncing here as well would put a disk round-trip on the
    training step for every metric, to protect a copy nobody replays.
    """

    __slots__ = ("_path",)

    def __init__(self, path: Path) -> None:
        self._path = path

    @classmethod
    def from_environment(cls) -> ObservationWriter | None:
        """The writer the runtime asked for, or ``None`` if none was given.

        ``None`` rather than a default path: a worker run by hand outside any
        runtime has no supervisor, and inventing a file nobody reads would
        make "not observed" look like "observed and lost".
        """
        location = os.environ.get(OBSERVATIONS_PATH_ENV)
        return None if location is None else cls(Path(location))

    def write(
        self,
        observation: object,
        *,
        context: CorrelationContext | None = None,
        trace_context: TraceContext | None = None,
    ) -> None:
        """Record *observation* as having happened now."""
        record = WorkerObservationRecord(
            observed_at=utc_now(),
            observation=observation,
            context=context,
            trace_context=trace_context,
        )
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(record.model_dump_json() + "\n")
            stream.flush()
