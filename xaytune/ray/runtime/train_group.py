"""A training plan as a worker group: what each rank does, and the one driver over them.

```text
Ray job (submission_id = the operation id)
  └─ python -m xaytune.ray.runtime.train_driver <workload directory>   the driver
       │   claims the workload; the ONE sequencer of its event stream
       └─ TorchTrainer(ScalingConfig(num_workers=N))                    the group
            ├─ rank 0 ─ the plan's worker ─ observations ──▶ the stream
            ├─ rank 1 ─ the plan's worker ─ diagnostics only
            └─ ...
```

Ray-free, so it is testable without Ray: :mod:`~xaytune.ray.runtime.train_driver`
supplies the real :class:`WorkerGroup` (``TorchTrainer``), a test supplies
another.

**Each rank runs the plan's worker as LocalRuntime would** -- the same argv,
the same environment, its own session -- with the placement Ray Train decided
added: ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``, ``LOCAL_WORLD_SIZE``,
``GROUP_RANK`` and a ``MASTER_ADDR``/``MASTER_PORT`` rank 0 chose for the
workers' own process group. The workers form their group; Ray Train's actors
place them, hold their devices, and supervise them. A worker does not know it
runs under Ray.

**One sequencer (ADR-014 §1a).** Only the driver appends to the workload's
event stream, and only rank 0's observations reach it: the controller-
significant facts -- progress, checkpoints, completion -- are rank 0's to
report. Every other rank's observations are kept beside it as diagnostics,
never interleaved: N writers to one stream is the condition that makes a gap
indistinguishable from a reorder.

**The group ends together.** A rank whose worker fails marks the group
aborted, and every other rank stops its worker -- once, with ``SIGTERM`` to its
process group, as a cancellation is delivered -- so a failed group leaves
nothing running. A cancellation reaches every rank the same way. The outcome
is the group's, and what was observed outranks what was intended: failed with
the first rank that failed on its own -- even if its siblings were stopped for
a cancellation -- or failed because Ray Train failed the group; otherwise
cancelled if any worker was stopped for one; otherwise succeeded.

``WorkerReady`` means the group started: it is emitted, once, only after every
rank's worker has started -- never for a group that failed or was cancelled
before all of them did.

The coordination is files in the workload directory, which the v1 shared
state root makes visible to the driver and every rank.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from xaytune.core.clock import utc_now
from xaytune.core.execution import ResolvedExecutionPlan
from xaytune.core.immutable import thaw
from xaytune.ray.runtime.supervisor import CLAIM
from xaytune.runtimes.local.jsonl import AppendOnlyJsonlReader
from xaytune.runtimes.local.launcher import (
    EventWriter,
    observation_adapter,
    relay_observations,
    worker_argv,
    worker_environment,
)
from xaytune.runtimes.local.paths import WorkloadPaths, read_json, write_atomic
from xaytune.runtimes.worker import OBSERVATIONS_PATH_ENV

__all__ = [
    "ABORT",
    "RANKS",
    "RankContext",
    "RankPaths",
    "WorkerGroup",
    "drive",
    "free_port",
    "run_rank",
]

RANKS = "ranks"
ABORT = "abort.json"

_POLL_SECONDS = 0.05


@dataclass(frozen=True)
class RankContext:
    """Where Ray Train placed one rank, and where its workers' group meets."""

    rank: int
    world_size: int
    local_rank: int
    local_world_size: int
    node_rank: int
    master_addr: str
    master_port: int


class WorkerGroup(Protocol):
    """Runs *rank_fn* once on every rank of a group, returning when all have.

    Raises whatever the group raised when it failed as a group -- a rank
    whose process died, a placement that could not be made.
    """

    world_size: int

    def run(self, rank_fn: Callable[[RankContext], None]) -> None: ...


class RankPaths:
    """One rank's files in the workload directory."""

    def __init__(self, workload: WorkloadPaths, rank: int) -> None:
        self.workload = workload
        self.rank = rank
        self.directory = workload.directory / RANKS

    @property
    def spawning(self) -> Path:
        return self.directory / f"{self.rank}.spawning"

    @property
    def started(self) -> Path:
        return self.directory / f"{self.rank}.started.json"

    @property
    def finished(self) -> Path:
        return self.directory / f"{self.rank}.finished.json"

    @property
    def observations(self) -> Path:
        """Rank 0 reports to the workload's channel; the others keep diagnostics."""
        if self.rank == 0:
            return self.workload.observations
        return self.directory / f"{self.rank}.observations.jsonl"

    @property
    def stdout(self) -> Path:
        return self.workload.stdout if self.rank == 0 else self.directory / f"{self.rank}.stdout"

    @property
    def stderr(self) -> Path:
        return self.workload.stderr if self.rank == 0 else self.directory / f"{self.rank}.stderr"


def free_port() -> int:
    """A port free on this node now, for the workers' process group to meet on."""
    with socket.socket() as probe:
        probe.bind(("", 0))
        return int(probe.getsockname()[1])


# ---- one rank ---------------------------------------------------------------------------


def run_rank(directory: str, context: RankContext) -> None:
    """Run the plan's worker as rank *context.rank*, and record how it ended.

    Returns normally however the worker ended: the outcome is the group's to
    decide from every rank's record, and a rank that raised would have Ray
    tear its siblings down before they could stop their workers.

    **Spawning is announced before cancellation is checked.** The driver
    reads the announcements after a cancellation is durable, so a rank either
    sees the cancellation and never spawns, or is seen spawning and waited
    for -- never a worker started behind a driver that already left.
    """
    paths = WorkloadPaths(Path(directory))
    plan = ResolvedExecutionPlan.model_validate_json(paths.plan.read_text(encoding="utf-8"))
    rank = RankPaths(paths, context.rank)
    rank.directory.mkdir(parents=True, exist_ok=True)
    rank.spawning.touch()

    if paths.cancel.exists() or _aborted(paths):
        write_atomic(
            rank.finished,
            {
                "exit_code": None,
                "cancelled": paths.cancel.exists(),
                "never_started": True,
                "finished_at": utc_now().isoformat(),
            },
        )
        return

    environment = worker_environment(plan, paths)
    environment[OBSERVATIONS_PATH_ENV] = str(rank.observations)
    environment.update(
        {
            "RANK": str(context.rank),
            "WORLD_SIZE": str(context.world_size),
            "LOCAL_RANK": str(context.local_rank),
            "LOCAL_WORLD_SIZE": str(context.local_world_size),
            "GROUP_RANK": str(context.node_rank),
            "MASTER_ADDR": context.master_addr,
            "MASTER_PORT": str(context.master_port),
        }
    )
    with rank.stdout.open("ab") as out, rank.stderr.open("ab") as err:
        try:
            child = subprocess.Popen(
                worker_argv(plan),
                stdout=out,
                stderr=err,
                env=environment,
                cwd=str(plan.runtime_options.get("working_directory", paths.directory)),
                # Its own session, so stopping it reaches everything it started.
                start_new_session=True,
            )
        except OSError as exc:
            _abort(paths, context.rank, f"spawn failed: {exc}")
            write_atomic(
                rank.finished,
                {
                    "exit_code": None,
                    "spawn_error": str(exc),
                    "finished_at": utc_now().isoformat(),
                },
            )
            return

    write_atomic(
        rank.started,
        {
            "pid": child.pid,
            "host": socket.gethostname(),
            "started_at": utc_now().isoformat(),
        },
    )
    stopped_by = _wait(child, paths)
    code = child.returncode
    if code != 0 and stopped_by is None:
        _abort(paths, context.rank, f"rank {context.rank} exited {code}")
    write_atomic(
        rank.finished,
        {
            "exit_code": code,
            "signal": -code if code < 0 else None,
            "cancelled": stopped_by == "cancel",
            "stopped_by": stopped_by,
            "finished_at": utc_now().isoformat(),
        },
    )


def _wait(child: subprocess.Popen[bytes], paths: WorkloadPaths) -> str | None:
    """Wait for the worker, stopping it once if the workload is cancelled or the group aborts."""
    stopped_by: str | None = None
    while child.poll() is None:
        if stopped_by is None:
            reason = "cancel" if paths.cancel.exists() else "group" if _aborted(paths) else None
            if reason is not None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    stopped_by = reason
                except (ProcessLookupError, PermissionError):  # pragma: no cover - raced its exit
                    pass
        time.sleep(_POLL_SECONDS)
    return stopped_by


def _aborted(paths: WorkloadPaths) -> bool:
    return (paths.directory / RANKS / ABORT).exists()


def _abort(paths: WorkloadPaths, rank: int | None, reason: str) -> None:
    marker = paths.directory / RANKS / ABORT
    if not marker.exists():
        write_atomic(marker, {"rank": rank, "reason": reason, "at": utc_now().isoformat()})


# ---- the driver -------------------------------------------------------------------------


def drive(
    directory: Path,
    group_for: Callable[[ResolvedExecutionPlan, WorkloadPaths], WorkerGroup],
    *,
    on_termination: Callable[[], None] | None = None,
) -> int:
    """Claim the workload, run its worker group, and be its one telemetry sequencer.

    Returns the exit code the Ray job ends with -- 0 for a group that
    succeeded or was cancelled, the failing rank's code otherwise -- after
    ``finished.json`` records the group's ending. Returns 0 at once if
    another driver already owns the workload.
    """
    paths = WorkloadPaths(directory)
    try:
        descriptor = os.open(directory / CLAIM, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return 0  # Ray ran this entrypoint again; the first driver owns the group.
    with os.fdopen(descriptor, "w", encoding="utf-8") as claim:
        claim.write(f"{os.getpid()} {utc_now().isoformat()}\n")

    plan = ResolvedExecutionPlan.model_validate_json(paths.plan.read_text(encoding="utf-8"))
    events = EventWriter(paths, plan)
    if on_termination is not None:
        _on_termination(on_termination)

    if paths.cancel.exists():
        return _finish(paths, events, {"cancelled": True, "never_started": True, "exit_code": None})

    paths.worker_config.write_text(
        json.dumps(thaw(plan.spec.config), sort_keys=True), encoding="utf-8"
    )
    group = group_for(plan, paths)
    world_size = group.world_size
    ranks = [RankPaths(paths, rank) for rank in range(world_size)]
    failure: list[BaseException] = []

    def run_group() -> None:
        try:
            group.run(_RankFunction(str(directory)))
        except BaseException as error:  # noqa: BLE001 -- recorded, never swallowed silently
            failure.append(error)

    runner = threading.Thread(target=run_group, name="ray-train-group", daemon=True)
    runner.start()

    observations = AppendOnlyJsonlReader(paths.observations, observation_adapter(plan))
    ready = False
    while runner.is_alive():
        if not ready and all(rank.started.exists() for rank in ranks):
            ready = _ready(paths, events, ranks)
        if ready:
            relay_observations(observations, events)
        if paths.cancel.exists() and not any(rank.spawning.exists() for rank in ranks):
            # No rank has begun to spawn, and none will now: each checks for
            # the cancellation after announcing itself. Ray ends the group
            # still waiting for resources when this job exits.
            return _finish(
                paths, events, {"cancelled": True, "never_started": True, "exit_code": None}
            )
        runner.join(_POLL_SECONDS)

    if failure:
        # Whatever ranks survive the group's failure stop their workers too.
        _abort(paths, None, f"the worker group failed: {type(failure[0]).__name__}")
        _await_ranks(ranks)
    if not ready and all(rank.started.exists() for rank in ranks):
        # Every rank started and the group ended between two polls. A group
        # some rank of which never started was never ready: no WorkerReady,
        # no started.json, and rank 0's observations stay diagnostics.
        ready = _ready(paths, events, ranks)
    if ready:
        relay_observations(observations, events)
        if observations.pending_bytes:
            events.emit(
                "IncidentObserved",
                reason="truncated-observation",
                detail=(
                    f"rank 0's worker exited mid-write, leaving {observations.pending_bytes} "
                    f"bytes that were never a complete record"
                ),
            )
    return _finish(paths, events, _outcome(ranks, failure, events))


class _RankFunction:
    """``run_rank`` bound to a workload: a plain, picklable callable for Ray to ship."""

    def __init__(self, directory: str) -> None:
        self.directory = directory

    def __call__(self, context: RankContext) -> None:
        run_rank(self.directory, context)


def _ready(paths: WorkloadPaths, events: EventWriter, ranks: list[RankPaths]) -> bool:
    started = [read_json(rank.started) or {} for rank in ranks]
    write_atomic(
        paths.started,
        {
            "pid": started[0].get("pid"),
            "launcher_pid": os.getpid(),
            "ranks": [
                {"rank": rank.rank, "pid": record.get("pid"), "host": record.get("host")}
                for rank, record in zip(ranks, started, strict=True)
            ],
            "generation": events.generation,
            "started_at": utc_now().isoformat(),
        },
    )
    # First, before rank 0's observations are relayed: sequence 0.
    events.emit("WorkerReady", pid=started[0].get("pid"))
    return True


def _await_ranks(ranks: list[RankPaths], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(rank.finished.exists() or not rank.started.exists() for rank in ranks):
            return
        time.sleep(_POLL_SECONDS)


def _outcome(
    ranks: list[RankPaths], failure: list[BaseException], events: EventWriter
) -> dict[str, Any]:
    """The group's ending, from every rank's record and the group's own fate."""
    records = [read_json(rank.finished) for rank in ranks]
    cancelled = any(record is not None and record.get("cancelled") for record in records)
    never_started = all(record is None or record.get("never_started") for record in records)
    failing = next(
        (
            (rank.rank, record)
            for rank, record in zip(ranks, records, strict=True)
            if record is not None
            and not record.get("stopped_by")
            and (record.get("exit_code") not in (0, None) or record.get("spawn_error"))
        ),
        None,
    )
    # Observed failure outranks cancellation intent: a rank that failed on
    # its own -- not stopped by anyone -- fails the group even if its
    # siblings were then stopped for a cancellation.
    if failing is not None:
        rank, record = failing
        code = record.get("exit_code")
        events.emit(
            "IncidentObserved",
            reason="spawn-failed"
            if record.get("spawn_error")
            else "nonzero-exit"
            if isinstance(code, int) and code > 0
            else "signalled",
            exit_code=code if isinstance(code, int) else None,
            metadata={"rank": rank, "world_size": len(ranks)},
        )
        return {
            "exit_code": code if isinstance(code, int) else None,
            "signal": record.get("signal"),
            "spawn_error": record.get("spawn_error"),
            "cancelled": False,
            "failed_rank": rank,
        }
    if failure:
        events.emit(
            "IncidentObserved",
            reason="worker-group-failed",
            detail=f"Ray Train's worker group failed: {type(failure[0]).__name__}",
            metadata={"world_size": len(ranks)},
        )
        return {"exit_code": 1, "signal": None, "cancelled": False, "group_failed": True}
    if cancelled:
        return {"exit_code": None, "cancelled": True, "never_started": never_started}
    return {"exit_code": 0, "signal": None, "cancelled": False}


def _finish(paths: WorkloadPaths, events: EventWriter, outcome: dict[str, Any]) -> int:
    write_atomic(
        paths.finished,
        {
            "signal": None,
            **outcome,
            "finished_at": utc_now().isoformat(),
            "generation": events.generation,
        },
    )
    code = outcome.get("exit_code")
    if outcome.get("cancelled") or code == 0:
        return 0
    # A process exit code is never negative; finished.json keeps the signal.
    return code if isinstance(code, int) and code > 0 else 1


def _on_termination(action: Callable[[], None]) -> None:
    """A termination request -- Ray stopping the job -- becomes the workload's cancellation."""

    def handle(_signum: int, _frame: object) -> None:
        action()

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)
