"""``LocalRuntime``: the first real :class:`~xaytune.runtimes.RuntimeBackend`.

Runs a plan as a local subprocess. It exists to prove the compile/execute
boundary holds against something real before Ray or Training Hub are involved,
and to be the backend a laptop actually uses.

**It sees only a** :class:`~xaytune.core.execution.ResolvedExecutionPlan`. It
does not import ``CandidateSpec`` or the candidate's ``TrainingSpec``, and it
never asks what algorithm is being run: an entrypoint, arguments, an
environment and an output location are everything it needs, and everything it
could not misinterpret. That is enforced by a test, not just asserted here.

What it deliberately does not do is reconcile. It provides the primitives a
controller needs -- an operation that can be looked up after a restart, a
status that distinguishes "finished" from "I cannot tell", telemetry that can
be replayed from a cursor -- and stops there. The loop that decides what to do
about them is ADR-013 §3 work and belongs to the controller.
"""

from __future__ import annotations

import asyncio
import errno
import os
import signal
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from xaytune.core.capabilities import (
    CapabilityDocument,
    CheckpointCapabilities,
    DistributedCapabilities,
    PluginDescriptor,
    ResilienceCapabilities,
)
from xaytune.core.clock import utc_now
from xaytune.core.execution import ResolvedExecutionPlan
from xaytune.core.ids import OperationId
from xaytune.core.refs import RuntimeRef
from xaytune.runtimes import (
    OperationOutcome,
    RuntimeEventEnvelope,
    RuntimeLog,
    RuntimeState,
    RuntimeStatus,
    StreamCursor,
)
from xaytune.runtimes.local.paths import WorkloadPaths, read_json
from xaytune.runtimes.local.registry import LocalWorkloadRecord, LocalWorkloadRegistry

__all__ = ["BACKEND", "LocalRuntime", "UnsupportedPlanError"]

BACKEND = "local"

_TERMINAL: frozenset[RuntimeState] = frozenset({"succeeded", "failed", "cancelled", "preempted"})
_POLL_SECONDS = 0.02


class UnsupportedPlanError(Exception):
    """This runtime cannot execute the plan it was handed.

    Raised to the caller *and* recorded as a rejected operation, because those
    answer different questions. The caller needs to stop; a controller that
    restarts and looks the operation up needs to learn that nothing was
    started, which is the one answer that makes re-issuing safe.
    """


class LocalRuntime:
    """Executes plans as local subprocesses.

    Args:
        root: Where the registry and workload directories live. Durable on
            purpose: a runtime rebuilt against the same root answers for the
            workloads the previous instance started.
    """

    descriptor = PluginDescriptor(
        api_version="xaytune.plugins/v1alpha1",
        name="local",
        plugin_version="0.1.0",
        provider="xaytune",
        xaytune_version="0.6.0",
    )

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._workloads = self._root / "workloads"
        self._workloads.mkdir(parents=True, exist_ok=True)
        self._registry = LocalWorkloadRegistry(self._root / "registry.db")

    def close(self) -> None:
        self._registry.close()

    # -- what this backend can do -----------------------------------------

    def capabilities(self) -> CapabilityDocument:
        """What a local subprocess can execute, and what it can report.

        ``max_workers=1`` rather than a launcher that shells out to
        ``torchrun``: choosing a world size, a rendezvous endpoint and a
        restart policy means reading the training configuration, and a runtime
        that read the training configuration would be interpreting the
        workload. A distributed local backend is a separate plugin with a
        separate capability document, not a flag on this one.
        """
        return CapabilityDocument(
            distributed=DistributedCapabilities(strategies=(), min_workers=1, max_workers=1),
            checkpoint=CheckpointCapabilities(atomic_commit=False),
            resilience=ResilienceCapabilities(
                per_step=False,
                provider="local-subprocess",
                provider_version=self.descriptor.plugin_version,
                # Both true because both are backed by files the launcher
                # writes: telemetry is appended to events.jsonl, and an exit
                # code outlives the controller in finished.json.
                supports_event_replay=True,
                reports_completed_operations=True,
            ),
        )

    # -- submitting --------------------------------------------------------

    async def submit_or_get(
        self, operation_id: OperationId, plan: ResolvedExecutionPlan
    ) -> RuntimeRef:
        """Start *plan*, or return the workload this operation already started.

        The durable claim is written before anything is spawned, so a crash
        anywhere in this method leaves either "intent, no process" or "intent
        and process", never a process with nothing pointing at it.
        """
        digest = plan.request_digest("submit")
        external_id = str(operation_id)

        refusal = _refuse(plan)
        if refusal is not None:
            self._registry.record_rejection(
                operation_id=operation_id, request_digest=digest, detail=refusal
            )
            raise UnsupportedPlanError(refusal)

        directory = self._workloads / external_id
        claimed = self._registry.claim_submission(
            operation_id=operation_id,
            request_digest=digest,
            external_id=external_id,
            target_kind=plan.target.kind,
            target_id=plan.target.id,
            directory=directory,
        )
        if claimed.rejected_detail is not None:
            raise UnsupportedPlanError(claimed.rejected_detail)

        paths = WorkloadPaths(directory)
        record = self._registry.workload(external_id)
        assert record is not None

        if record.launcher_pid is None and read_json(paths.started) is None:
            self._spawn(paths, plan, external_id)

        return RuntimeRef(backend=BACKEND, external_id=external_id)

    def _spawn(self, paths: WorkloadPaths, plan: ResolvedExecutionPlan, external_id: str) -> None:
        """Start the launcher for a claim that has no process yet.

        ``start_new_session`` puts the launcher in its own process group, which
        is what lets a cancellation reach the worker as well as its supervisor,
        and what stops a Ctrl-C in the controller's terminal reaching either.
        """
        paths.directory.mkdir(parents=True, exist_ok=True)
        paths.plan.write_text(plan.model_dump_json(), encoding="utf-8")

        process = subprocess.Popen(
            [sys.executable, "-m", "xaytune.runtimes.local.launcher", str(paths.directory)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._registry.record_launcher_pid(external_id, process.pid)

    # -- asking what happened ---------------------------------------------

    async def lookup_operation(self, operation_id: OperationId) -> OperationOutcome | None:
        """What became of *operation_id*, or ``None`` if it never arrived.

        Answerable after a restart because every branch reads durable state:
        the registry for whether the request was received, and the launcher's
        own files for what became of the process.
        """
        record = self._registry.operation(operation_id)
        if record is None:
            return None

        if record.rejected_detail is not None:
            return OperationOutcome(
                operation_id=operation_id, disposition="rejected", detail=record.rejected_detail
            )

        assert record.external_id is not None
        workload = self._registry.workload(record.external_id)
        assert workload is not None

        status = self._status(workload)
        return OperationOutcome(
            operation_id=operation_id,
            disposition="completed" if status.state in _TERMINAL else "accepted",
            runtime_ref=RuntimeRef(backend=BACKEND, external_id=record.external_id),
            status=status,
        )

    async def get_status(self, runtime_ref: RuntimeRef) -> RuntimeStatus:
        """Observe a workload, saying ``unknown`` when that is the true answer."""
        workload = self._require(runtime_ref)
        return self._status(workload)

    def _status(self, workload: LocalWorkloadRecord) -> RuntimeStatus:
        """Derive a state from what is on disk, never from a live handle.

        A recreated runtime has no ``Popen`` for a workload it did not start,
        so the launcher's markers are the only evidence there is -- and reading
        them is the same code path in both cases, which means the restart path
        is exercised by every test rather than by a special one.
        """
        paths = WorkloadPaths(workload.directory)
        cancelled = workload.cancel_requested_at is not None

        finished = read_json(paths.finished)
        if finished is not None:
            return self._finished_status(finished, cancelled=cancelled)

        started = read_json(paths.started)
        pid = workload.launcher_pid
        alive = pid is not None and _alive(pid)

        if not alive:
            # Nothing is running and nothing recorded an ending. Either the
            # launcher died before it could write one, or it was never spawned
            # -- and this runtime cannot tell those apart, so it says so
            # rather than picking the convenient one.
            return RuntimeStatus(
                state="unknown",
                detail=(
                    "no exit was recorded and the launcher is gone: the workload "
                    "ended in a way nothing observed, or never started"
                ),
                observed_at=utc_now(),
            )

        return RuntimeStatus(
            state="running" if started is not None else "starting",
            detail="cancellation requested" if cancelled else None,
            observed_at=utc_now(),
        )

    def _finished_status(self, finished: dict[str, object], *, cancelled: bool) -> RuntimeStatus:
        """Read an exit code as a runtime state, and nothing more.

        A workload that exits cleanly after a cancellation request finished
        before the signal reached it. That is a success, not a cancellation:
        the work is done and its artifacts are real, and recording it as
        cancelled would discard them (ADR-013 §5).
        """
        raw = finished.get("exit_code")
        code = raw if isinstance(raw, int) else None

        if finished.get("spawn_error") is not None:
            return RuntimeStatus(
                state="failed",
                detail=f"the worker could not be started: {finished['spawn_error']}",
                observed_at=utc_now(),
            )

        if code == 0:
            state: RuntimeState = "succeeded"
        elif cancelled:
            state = "cancelled"
        else:
            state = "failed"

        return RuntimeStatus(
            state=state,
            exit_code=code,
            detail=f"terminated by signal {-code}" if code is not None and code < 0 else None,
            observed_at=utc_now(),
        )

    # -- cancelling --------------------------------------------------------

    async def cancel(self, runtime_ref: RuntimeRef, operation_id: OperationId) -> None:
        """Ask the workload to stop, once per operation.

        Keyed by ``operation_id`` like every other effect: a retried
        cancellation must not become a second signal, because the process that
        replaced the first one on the same pid would receive it.
        """
        workload = self._require(runtime_ref)
        digest = _cancel_digest(workload)

        first = self._registry.claim_cancellation(
            operation_id=operation_id, request_digest=digest, external_id=workload.external_id
        )
        if not first:
            return

        pid = workload.launcher_pid
        if pid is None:
            return

        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            # Already gone, or no longer ours. Both mean this cancellation has
            # nothing left to do; neither means it failed.
            return

    # -- watching ----------------------------------------------------------

    async def watch(
        self, runtime_ref: RuntimeRef, cursor: StreamCursor | None = None
    ) -> AsyncIterator[RuntimeEventEnvelope]:
        """Stream telemetry after *cursor*, in ``(generation, sequence)`` order.

        Replayed from the file the launcher wrote, so a controller that
        reconnects after a restart receives what it missed rather than being
        told the stream is a gap. That is the whole reason
        ``supports_event_replay`` is true here.

        The cursor is the last sequence the controller **durably recorded**, so
        delivery starts strictly after it; an event received and lost in a
        crash is redelivered.
        """
        workload = self._require(runtime_ref)
        paths = WorkloadPaths(workload.directory)
        position = (cursor.generation, cursor.sequence) if cursor else (0, -1)
        delivered: set[tuple[int, int]] = set()

        while True:
            drained = True
            for envelope in _read_events(paths.events):
                key = (envelope.stream_generation, envelope.sequence)
                if key <= position or key in delivered:
                    continue
                delivered.add(key)
                drained = False
                yield envelope

            if drained and self._status(workload).state in _TERMINAL:
                return
            await asyncio.sleep(_POLL_SECONDS)

    async def get_logs(self, runtime_ref: RuntimeRef) -> AsyncIterator[RuntimeLog]:
        """Stream worker output until the workload ends.

        Read straight from the files the worker wrote to, and never recorded in
        the registry: logs are for humans, and a controller that had to durably
        record every line before acting on it would be rate-limited by its own
        observability.

        Order *within* a stream is the order the worker wrote it. Order
        *between* stdout and stderr is not meaningful and is not claimed --
        they are separate files with separate buffers.
        """
        workload = self._require(runtime_ref)
        paths = WorkloadPaths(workload.directory)
        offsets = {paths.stdout: 0, paths.stderr: 0}

        while True:
            drained = True
            for path, stream in ((paths.stdout, "stdout"), (paths.stderr, "stderr")):
                lines, offsets[path] = _read_lines(path, offsets[path])
                for line in lines:
                    drained = False
                    yield RuntimeLog(stream=stream, line=line)  # type: ignore[arg-type]

            if drained and self._status(workload).state in _TERMINAL:
                return
            await asyncio.sleep(_POLL_SECONDS)

    # -- helpers -----------------------------------------------------------

    def _require(self, runtime_ref: RuntimeRef) -> LocalWorkloadRecord:
        if runtime_ref.backend != BACKEND:
            raise KeyError(
                f"{runtime_ref.backend!r} is not a local workload: this runtime "
                f"can only answer for references it issued"
            )
        workload = self._registry.workload(runtime_ref.external_id)
        if workload is None:
            raise KeyError(f"unknown local workload {runtime_ref.external_id!r}")
        return workload


def _refuse(plan: ResolvedExecutionPlan) -> str | None:
    """Why this plan cannot run locally, or ``None`` if it can.

    Refusing beats ignoring. A plan that declares secrets or an image has said
    the workload needs them; running it anyway would start a process that fails
    somewhere inside the worker, for a reason the logs would attribute to the
    training code rather than to a runtime that quietly dropped part of the
    request.
    """
    if plan.spec.secrets:
        names = ", ".join(secret.name for secret in plan.spec.secrets)
        return (
            f"this runtime cannot resolve secret references ({names}); running "
            f"the workload without them would fail inside the worker instead of here"
        )
    if plan.spec.container is not None:
        return (
            f"this runtime runs subprocesses, not containers; "
            f"{plan.spec.container.image!r} needs a container runtime"
        )
    workers = plan.spec.resources.workers
    if workers is not None and workers > 1:
        return (
            f"this runtime runs one worker, not {workers}; a distributed local "
            f"launcher is a separate backend with its own capability document"
        )
    return None


def _cancel_digest(workload: LocalWorkloadRecord) -> str:
    """The idempotency key for cancelling this workload.

    Derived from the workload rather than the plan, because a caller cancelling
    after a restart holds a :class:`RuntimeRef` and need not still have the
    plan that produced it.
    """
    return f"cancel:{workload.external_id}"


def _alive(pid: int) -> bool:
    """Whether *pid* is still running.

    A launcher this process started stays in the process table as a zombie
    until someone waits on it, and a zombie answers ``kill(pid, 0)`` exactly
    like a live process. Left at that, a workload whose launcher had been dead
    for hours would still be reported as running. So a child of ours is reaped
    here, non-blockingly, which both answers the question and stops a
    long-running controller accumulating one zombie per workload.

    ``ChildProcessError`` means the pid is not ours -- the usual case after a
    restart -- and then the signal probe is the only evidence available.
    ``EPERM`` from it counts as alive: the process is there, it simply is not
    ours to signal, and reading that as "gone" would turn a running workload
    into an outcome nobody observed.
    """
    try:
        reaped, _status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    except OSError:  # pragma: no cover - platform specific
        pass
    else:
        return reaped == 0

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:  # pragma: no cover - platform specific
        return exc.errno == errno.EPERM
    return True


def _read_events(path: Path) -> list[RuntimeEventEnvelope]:
    """Every complete envelope in the file, in stream order.

    A trailing partial line is skipped rather than raised on: the launcher
    appends and flushes, so a reader can arrive mid-write, and the next poll
    will see the whole line.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []

    envelopes = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            envelopes.append(RuntimeEventEnvelope.model_validate_json(line))
        except ValueError:
            continue
    envelopes.sort(key=lambda envelope: (envelope.stream_generation, envelope.sequence))
    return envelopes


def _read_lines(path: Path, offset: int) -> tuple[list[str], int]:
    """Complete lines written after *offset*, and the new offset.

    An incomplete final line is left for the next read, so a log line is never
    delivered in two pieces.
    """
    try:
        with path.open("rb") as stream:
            stream.seek(offset)
            data = stream.read()
    except FileNotFoundError:
        return [], offset

    if not data:
        return [], offset

    text = data.decode("utf-8", errors="replace")
    complete, _, remainder = text.rpartition("\n")
    if not complete and remainder:
        return [], offset

    consumed = len(text) - len(remainder)
    return complete.splitlines(), offset + len(text[:consumed].encode("utf-8"))
