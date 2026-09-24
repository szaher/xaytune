"""The process that supervises one local workload.

``LocalRuntime`` does not run the worker directly. It runs this module, which
runs the worker, and that indirection buys the one property a direct
``Popen`` cannot provide: **a workload's outcome outlives the controller that
started it.** Only a process's own parent can reap it, so a recreated
``LocalRuntime`` could never learn the exit code of a worker it did not spawn.
The launcher is that parent, and it writes what it reaps to disk.

It is also the telemetry supervisor of ADR-014 §1a: the single writer that
assigns ``sequence`` for this target. Nothing else appends to ``events.jsonl``,
so the sequence is gapless by construction rather than by agreement.

And it owns two decisions the controller cannot safely make from outside the
process tree. It **claims the workload** before spawning anything, so that two
launchers started either side of a crash cannot both produce a worker. And it
**delivers cancellations**, because it is the only participant that knows the
pid it is signalling is still its own child rather than a number the operating
system has since handed to something else.

**It reports what it observes and nothing more.** The events it emits are the
ones in both workload vocabularies -- a worker started, something went wrong --
because a process exiting zero is not evidence that training converged, and a
runtime that claimed ``TrainingCompleted`` from an exit code would be inventing
a scientific fact out of an operating-system one. Whether the workload did its
job is the worker's to report and the controller's to conclude.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from xaytune.core.clock import utc_now
from xaytune.core.execution import CommandEntrypoint, ResolvedExecutionPlan
from xaytune.core.immutable import thaw
from xaytune.core.telemetry import EvaluationObservation, TrainingObservation
from xaytune.runtimes import (
    EvaluationEventPayload,
    RuntimeEventEnvelope,
    RuntimeEventPayload,
    TrainingEventPayload,
)
from xaytune.runtimes.local.jsonl import AppendOnlyJsonlReader, CorruptRecordError
from xaytune.runtimes.local.paths import WorkloadPaths, write_atomic
from xaytune.runtimes.local.registry import LocalWorkloadRegistry
from xaytune.runtimes.worker import (
    OBSERVATIONS_PATH_ENV,
    TOPOLOGY_VARIABLES,
    WORKER_CONFIG_PATH_ENV,
    WorkerObservationRecord,
)

__all__ = ["main", "run"]

_CANCEL_POLL_SECONDS = 0.05


def _argv(plan: ResolvedExecutionPlan) -> list[str]:
    """Turn the plan's entrypoint into an argument vector.

    The two entrypoint kinds are handled mechanically. Note what is *not* here:
    nothing reads ``spec.config``, the algorithm, or anything else describing
    what the workload is for. A runtime that understood the workload would be a
    second place scientific intent lived.
    """
    entrypoint = plan.spec.entrypoint
    arguments = list(plan.spec.arguments)

    if isinstance(entrypoint, CommandEntrypoint):
        return [*entrypoint.argv, *arguments]

    if entrypoint.function is None:
        return [sys.executable, "-m", entrypoint.module, *arguments]

    # The module and function arrive as ``argv`` rather than interpolated into
    # the source: a name spliced into code would execute as code.
    return [
        sys.executable,
        "-c",
        "import importlib, sys\n"
        "module = importlib.import_module(sys.argv[1])\n"
        "sys.exit(getattr(module, sys.argv[2])(*sys.argv[3:]))\n",
        entrypoint.module,
        entrypoint.function,
        *arguments,
    ]


def _environment(plan: ResolvedExecutionPlan, paths: WorkloadPaths) -> dict[str, str]:
    """The worker's environment: this process's, plus what the plan declares.

    Inherited rather than replaced, because a bare environment has no ``PATH``
    and most workers would fail for a reason that has nothing to do with the
    plan.

    **Except the distributed topology**, which is removed. It describes where
    the *controller* happens to be running, not where this worker is, and a
    worker that inherited it from a controller started under ``torchrun``
    would try to join a process group nobody created. This runtime runs one
    worker and places it in no group, so it sets none of them.
    """
    environment = {key: value for key, value in os.environ.items() if key not in TOPOLOGY_VARIABLES}
    environment.update({key: str(value) for key, value in plan.spec.environment.items()})
    environment["XAYTUNE_TARGET_KIND"] = plan.target.kind
    environment["XAYTUNE_TARGET_ID"] = plan.target.id
    # Set after the plan's own environment, so a plan cannot redirect them:
    # where the config and observations live is this runtime's decision.
    environment[WORKER_CONFIG_PATH_ENV] = str(paths.worker_config)
    environment[OBSERVATIONS_PATH_ENV] = str(paths.observations)
    return environment


class _EventWriter:
    """The single assigner of ``sequence`` for one target (ADR-014 §1a)."""

    def __init__(self, paths: WorkloadPaths, plan: ResolvedExecutionPlan) -> None:
        self._paths = paths
        self._plan = plan
        self._generation = _next_generation(paths.events)
        self._sequence = 0

    @property
    def generation(self) -> int:
        return self._generation

    def emit(self, event_type: str, **data: Any) -> None:
        """Record something the launcher itself observed."""
        self._append(_payload(self._plan, event_type, data), emitted_at=utc_now())

    def emit_observation(self, record: WorkerObservationRecord[Any]) -> None:
        """Wrap something the worker observed, and record it.

        The same sequencer as :meth:`emit`, on purpose: one counter, one file,
        one writer. Two paths each assigning sequence numbers would be the two
        writers ADR-014 §1a forbids, just inside one process.

        ``emitted_at`` is when the worker observed it, not now -- otherwise it
        would record when this loop happened to poll.

        Raises:
            ValueError: If the observation cannot be enveloped for this target,
                for instance a correlation context naming a different attempt.
        """
        if self._plan.target.kind == "training-attempt":
            payload: RuntimeEventPayload = TrainingEventPayload(data=record.observation)
        else:
            payload = EvaluationEventPayload(data=record.observation)
        self._append(
            payload,
            emitted_at=record.observed_at,
            context=record.context,
            trace_context=record.trace_context,
        )

    def _append(
        self,
        payload: RuntimeEventPayload,
        *,
        emitted_at: Any,
        context: Any = None,
        trace_context: Any = None,
    ) -> None:
        envelope = RuntimeEventEnvelope(
            event_id=f"{self._plan.target.id}-{self._generation}-{self._sequence}",
            target=self._plan.target,
            stream_generation=self._generation,
            sequence=self._sequence,
            emitted_at=emitted_at,
            payload=payload,
            context=context,
            trace_context=trace_context,
        )
        # Only after the envelope is known valid: a refused observation must
        # not consume a sequence number, or the stream would have a gap in it.
        self._sequence += 1

        with self._paths.events.open("a", encoding="utf-8") as stream:
            stream.write(envelope.model_dump_json() + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _payload(
    plan: ResolvedExecutionPlan, event_type: str, data: dict[str, Any]
) -> RuntimeEventPayload:
    """Build the payload family this target is allowed to carry.

    Selected from the typed ``target.kind``, not from anything about the
    workload itself. Only event types present in both vocabularies are used
    here, so neither branch can construct a payload the envelope would refuse.
    """
    if plan.target.kind == "training-attempt":
        return TrainingEventPayload.model_validate({"data": {"type": event_type, **data}})
    return EvaluationEventPayload.model_validate({"data": {"type": event_type, **data}})


def _next_generation(events: Path) -> int:
    """The generation this launcher writes under.

    A relaunch of the same workload starts a new stream rather than continuing
    the old one, because its sequence restarts at zero and a controller holding
    a cursor would otherwise read the new events as duplicates of events it had
    already recorded (ADR-014 §1a).

    **Provisional, and local to this backend.** ADR-014 §2 allocates the
    generation from durable controller-side attempt state; deriving it from the
    stream a replacement supervisor happens to find is a weaker rule, and two
    supervisors that both read the file before either wrote to it would pick the
    same number. It is enough because nothing relaunches a local supervisor:
    one launcher per workload, so this is always 0.

    Since PR-012a the authoritative generation is the attempt's
    ``telemetry_generation``, which the controller advances when a stream dies
    over a live workload (ADR-014 §1a). Passing that into a *relaunched*
    supervisor arrives with the first backend that relaunches one; LocalRuntime
    does not, so there is nothing to pass it to yet.
    """
    highest = -1
    try:
        text = events.read_text(encoding="utf-8")
    except FileNotFoundError:
        return 0

    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            highest = max(highest, int(json.loads(line)["stream_generation"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return highest + 1


def run(directory: Path, registry_path: Path, external_id: str) -> int:
    """Supervise the workload in *directory* and return its exit code.

    The first thing it does is try to own the workload, and the second is check
    whether the workload has already been cancelled. Neither can be deferred
    until after the worker is spawned: by then the duplicate exists, or the
    cancellation has been ignored.
    """
    paths = WorkloadPaths(directory)
    plan = ResolvedExecutionPlan.model_validate_json(paths.plan.read_text(encoding="utf-8"))

    registry = LocalWorkloadRegistry(registry_path)
    try:
        owned = registry.claim_launcher_ownership(external_id, os.getpid())
    finally:
        registry.close()

    if not owned:
        # Another launcher owns this workload. Exit before spawning anything:
        # the whole point of the claim is that only one worker exists.
        return 0

    events = _EventWriter(paths, plan)

    if paths.cancel.exists():
        # Cancelled before there was anything to cancel. Recorded as an ending
        # rather than left running, because a controller waiting on a workload
        # that will never start has no way to find that out.
        write_atomic(
            paths.finished,
            {
                "exit_code": None,
                "signal": None,
                "cancelled": True,
                "never_started": True,
                "finished_at": utc_now().isoformat(),
                "generation": events.generation,
            },
        )
        return 0

    # Written before the worker exists, so it can never start without it.
    # thaw(), not dict(): a real config is nested, and dict() unfreezes only
    # the top level, leaving FrozenDicts json cannot serialise.
    paths.worker_config.write_text(
        json.dumps(thaw(plan.spec.config), sort_keys=True), encoding="utf-8"
    )

    with paths.stdout.open("ab") as out, paths.stderr.open("ab") as err:
        try:
            child = subprocess.Popen(
                _argv(plan),
                stdout=out,
                stderr=err,
                env=_environment(plan, paths),
                cwd=str(plan.runtime_options.get("working_directory", directory)),
                # Its own session, so the worker leads a process group that
                # contains it and everything it starts. Cancelling a training
                # job has to reach the dataloader workers and helpers it
                # spawned, and signalling the worker alone would leave them
                # running while the workload reported itself cancelled.
                start_new_session=True,
            )
        except OSError as exc:
            # The worker never existed, so there is nothing to wait for and no
            # exit code to report. Recorded as a finish anyway: a workload that
            # never started and one whose outcome is unknown lead a controller
            # to different places.
            events.emit("IncidentObserved", reason="spawn-failed", detail=str(exc))
            write_atomic(
                paths.finished,
                {
                    "exit_code": None,
                    "signal": None,
                    "spawn_error": str(exc),
                    "finished_at": utc_now().isoformat(),
                    "generation": events.generation,
                },
            )
            return 127

    _survive_signals()
    write_atomic(
        paths.started,
        {
            "pid": child.pid,
            "launcher_pid": os.getpid(),
            "generation": events.generation,
            "started_at": utc_now().isoformat(),
        },
    )
    # First, and before any drain: whatever the worker writes waits in its
    # file until the loop below reads it, so this is sequence 0 however the
    # operating system schedules the two processes.
    events.emit("WorkerReady", pid=child.pid)

    observations = AppendOnlyJsonlReader(paths.observations, _observation_adapter(plan))
    cancelled = _supervise(child, paths, events, observations)
    code = child.returncode

    if code != 0:
        events.emit(
            "IncidentObserved",
            reason="nonzero-exit" if code > 0 else "signalled",
            exit_code=code,
        )

    write_atomic(
        paths.finished,
        {
            "exit_code": code,
            "signal": -code if code < 0 else None,
            "cancelled": cancelled,
            "finished_at": utc_now().isoformat(),
            "generation": events.generation,
        },
    )
    return code


def _observation_adapter(plan: ResolvedExecutionPlan) -> TypeAdapter[Any]:
    """Validate worker records against the family this target may carry.

    The two vocabularies share members, so the family is chosen by the typed
    target and not inferred from the record: an evaluation worker reporting
    ``CheckpointCommitted`` is refused here, before it becomes an envelope.
    """
    if plan.target.kind == "training-attempt":
        return TypeAdapter(WorkerObservationRecord[TrainingObservation])
    return TypeAdapter(WorkerObservationRecord[EvaluationObservation])


def _drain(observations: AppendOnlyJsonlReader[Any], events: _EventWriter) -> None:
    """Move everything the worker has reported so far into the event stream.

    **The supervisor does not die of bad worker output.** The worker is the
    user's code, and if a corrupt line or an unwrappable observation killed the
    launcher, nothing would ever write ``finished.json`` -- a workload that
    finished would report ``unknown`` forever. So each is recorded as an
    incident, the valid records around it still go through, and supervision
    continues.
    """
    try:
        records = observations.read_new()
        corrupt: CorruptRecordError | None = None
    except CorruptRecordError as exc:
        records, corrupt = exc.records, exc

    for record in records:
        try:
            events.emit_observation(record)
        except (ValidationError, ValueError) as exc:
            events.emit(
                "IncidentObserved",
                reason="invalid-observation",
                detail=f"the worker reported an observation this target cannot carry: {exc}",
            )

    if corrupt is not None:
        events.emit(
            "IncidentObserved",
            reason="corrupt-observation",
            detail=(
                f"lines {', '.join(str(n) for n in corrupt.line_numbers)} of the worker's "
                f"observations were not valid records: {corrupt}"
            ),
        )


def _supervise(
    child: subprocess.Popen[bytes],
    paths: WorkloadPaths,
    events: _EventWriter,
    observations: AppendOnlyJsonlReader[Any],
) -> bool:
    """Relay observations and honour cancellation until the worker exits.

    **Drained once more after exit, before anything else.** A worker commonly
    reports ``TrainingCompleted`` and exits immediately. If the loop saw the
    exit first and returned, that last observation would still be sitting in
    the file, unread, when ``finished.json`` was written -- a completed run
    recorded without its completion, intermittently, and mostly not on the
    machine where anyone was looking. Nothing can be written after the worker
    has exited, so one more read is enough.

    A record still cut off after that read means the worker died mid-write;
    that is reported rather than silently dropped.
    """
    cancelled = _wait_honouring_cancellation(child, paths, lambda: _drain(observations, events))
    _drain(observations, events)

    if observations.pending_bytes:
        events.emit(
            "IncidentObserved",
            reason="truncated-observation",
            detail=(
                f"the worker exited mid-write, leaving {observations.pending_bytes} "
                f"bytes that were never a complete record"
            ),
        )
    return cancelled


def _wait_honouring_cancellation(
    child: subprocess.Popen[bytes],
    paths: WorkloadPaths,
    on_poll: Any = None,
) -> bool:
    """Wait for the worker, stopping it if a cancellation appears.

    The launcher signals, and the controller does not, because only the
    launcher knows that the pid it is about to signal is still its own child.
    A controller signalling a pid it remembered from before a restart is
    signalling a number, and the operating system may have given that number to
    something else.

    Signalled **once**, and to the worker's whole process group. A training
    worker starts dataloader workers and helpers, and terminating only the
    process the launcher can see would leave those running while the workload
    reported itself cancelled -- which is not what cancellation promises. One
    signal, because a worker that treats a second SIGTERM as "stop being
    graceful" would be denied the shutdown the first one asked for.

    Returns whether a signal was actually delivered to a running worker, which
    is a different question from whether one was ever requested, and it is what
    the terminal state is later classified from.
    """
    signalled = False
    while child.poll() is None:
        if on_poll is not None:
            on_poll()
        if not signalled and paths.cancel.exists():
            signalled = True
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):  # pragma: no cover
                # It finished in the moment between the poll above and this
                # line. Nothing was delivered, so nothing was cancelled.
                signalled = False
        time.sleep(_CANCEL_POLL_SECONDS)
    return signalled


def _survive_signals() -> None:
    """Keep waiting through a termination request instead of dying of it.

    A cancellation is delivered to the whole process group, so the worker has
    already received it by the time this handler runs. **It must not be passed
    on again**: a worker that treats a second SIGTERM as "stop being graceful"
    -- which is the common convention -- would be denied the shutdown the first
    one asked for, and one cancellation would have become two.

    The launcher stays alive because it is the only process that can write
    ``finished.json``. If it died of the same signal, a cancelled workload
    would be indistinguishable from one whose outcome nobody observed.
    """

    def _keep_waiting(_signum: int, _frame: object) -> None:
        return

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, _keep_waiting)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 3:
        print(
            "usage: python -m xaytune.runtimes.local.launcher "
            "<workload-directory> <registry-path> <external-id>"
        )
        return 2
    return run(Path(arguments[0]), Path(arguments[1]), arguments[2])


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    raise SystemExit(main())
