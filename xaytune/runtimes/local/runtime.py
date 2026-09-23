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
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from pydantic import TypeAdapter

from xaytune.core.capabilities import (
    CapabilityDocument,
    CheckpointCapabilities,
    DistributedCapabilities,
    PluginDescriptor,
    ResilienceCapabilities,
    require_supported_plugin,
)
from xaytune.core.clock import utc_now
from xaytune.core.errors import IncompatiblePluginError
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
from xaytune.runtimes.local.jsonl import AppendOnlyJsonlReader
from xaytune.runtimes.local.paths import WorkloadPaths, read_json, write_atomic
from xaytune.runtimes.local.registry import LocalWorkloadRecord, LocalWorkloadRegistry
from xaytune.runtimes.worker import TOPOLOGY_VARIABLES

__all__ = ["BACKEND", "LocalRuntime", "UnsupportedPlanError"]

BACKEND = "local"

_TERMINAL: frozenset[RuntimeState] = frozenset({"succeeded", "failed", "cancelled", "preempted"})
_LIVE: frozenset[RuntimeState] = frozenset({"pending", "queued", "starting", "running"})
"""States a workload can still leave. ``unknown`` is in neither set: it is
not an ending, and it is not somewhere a caller should keep waiting."""
_POLL_SECONDS = 0.02

_TELEMETRY_PROTOCOL = "xaytune.telemetry/v1alpha2"
"""The telemetry contract this backend's launcher emits.

Named here rather than read from the envelope so a plan compiled against
an older protocol is refused at submission, which is what
:class:`~xaytune.core.execution.TelemetryContract` exists to make
possible."""

_ENVELOPE: TypeAdapter[RuntimeEventEnvelope] = TypeAdapter(RuntimeEventEnvelope)

_RUNTIME_OPTIONS = frozenset({"working_directory"})
"""Every runtime option this backend implements.

Checked as a closed set rather than read opportunistically. An unknown
option is a caller asking for behaviour -- ``{'launcher': 'torchrun'}`` is
the obvious one -- and silently running a plain subprocess instead would be
exactly the "ignore part of the request" failure this runtime refuses
elsewhere."""


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
        """Start a launcher for a claim that appears to have no process yet.

        Only *appears*: this check is an optimisation, not the guarantee. A
        controller that died between ``Popen`` returning and any write it could
        make would restart, see no owner, and spawn a second launcher -- so the
        launcher claims the workload itself, and the loser of that claim exits
        without spawning a worker. Correctness lives in the claim; this check
        only keeps the common path from starting a process that would
        immediately exit.

        ``start_new_session`` puts the launcher in its own process group, so a
        Ctrl-C in the controller's terminal does not reach the worker.
        """
        paths.directory.mkdir(parents=True, exist_ok=True)
        paths.plan.write_text(plan.model_dump_json(), encoding="utf-8")

        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "xaytune.runtimes.local.launcher",
                str(paths.directory),
                str(self._root / "registry.db"),
                external_id,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._registry.record_spawned_pid(external_id, process.pid)

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

        status = self._status(record.external_id)
        return OperationOutcome(
            operation_id=operation_id,
            disposition="completed" if status.state in _TERMINAL else "accepted",
            runtime_ref=RuntimeRef(backend=BACKEND, external_id=record.external_id),
            status=status,
        )

    async def get_status(self, runtime_ref: RuntimeRef) -> RuntimeStatus:
        """Observe a workload, saying ``unknown`` when that is the true answer."""
        self._require(runtime_ref)
        return self._status(runtime_ref.external_id)

    def _status(self, external_id: str) -> RuntimeStatus:
        """Derive a state from what is on disk, never from a live handle.

        A recreated runtime has no ``Popen`` for a workload it did not start,
        so the launcher's markers are the only evidence there is -- and reading
        them is the same code path in both cases, which means the restart path
        is exercised by every test rather than by a special one.

        **The launcher is not the workload.** It is the worker's parent and the
        telemetry supervisor, and it can die while the worker, reparented, runs
        on. Reading the supervisor's absence as the workload's end would report
        a live training job as an unobserved ending, and ADR-014 §1a separates
        the two precisely so a controller can lose telemetry without losing the
        run. So the worker's own pid is what decides whether work is happening,
        and the launcher's only decides whether anything is still watching it.
        """
        workload = self._registry.workload(external_id)
        assert workload is not None
        paths = WorkloadPaths(workload.directory)

        finished = read_json(paths.finished)
        if finished is not None:
            return self._finished_status(finished)

        started = read_json(paths.started)
        supervisor = workload.launcher_pid or workload.spawned_pid
        supervised = supervisor is not None and _alive(supervisor)

        if started is not None:
            worker_pid = started.get("pid")
            if isinstance(worker_pid, int) and _alive(worker_pid):
                return RuntimeStatus(
                    state="running",
                    detail=None if supervised else "the telemetry supervisor is gone",
                    observed_at=utc_now(),
                )
            if supervised:
                # The worker has gone but its supervisor has not, so the
                # outcome is moments away rather than lost. Reporting
                # ``unknown`` here would send a controller to reconcile a
                # workload that is about to report a clean exit -- the same
                # confusion between "no answer yet" and "no answer possible"
                # that separating these two pids exists to remove.
                return RuntimeStatus(
                    state="running",
                    detail="the worker has exited; its outcome is being recorded",
                    observed_at=utc_now(),
                )

            # Nobody is left to answer. A pid can also be reused, so "gone" is
            # the safer reading of a pid that no longer answers than "still
            # running" would be.
            return RuntimeStatus(
                state="unknown",
                detail=(
                    "the worker is no longer running and no exit was recorded: "
                    "it ended in a way nothing observed"
                ),
                observed_at=utc_now(),
            )

        if supervised:
            return RuntimeStatus(
                state="starting",
                detail=(
                    "cancellation requested" if workload.cancel_requested_at is not None else None
                ),
                observed_at=utc_now(),
            )

        return RuntimeStatus(
            state="unknown",
            detail=(
                "no worker was recorded and the launcher is gone: the workload "
                "never started, or started in a way nothing observed"
            ),
            observed_at=utc_now(),
        )

    def _finished_status(self, finished: dict[str, object]) -> RuntimeStatus:
        """Classify an ending from what the launcher observed.

        Entirely from the finished record, and deliberately **not** from the
        registry's ``cancel_requested_at``. Those answer different questions --
        whether a cancellation was ever wanted, and whether one reached this
        execution before it ended -- and only the second describes what
        happened. ADR-013 §5 puts the observed terminal state above pending
        intent, and two races make the difference visible:

        ```text
        cancel requested, worker exits 3 first    -> FAILED, not cancelled
        SIGTERM delivered, worker exits 0 cleanly -> CANCELLED, not succeeded
        ```

        The second is the one an exit code alone gets wrong. A worker that
        handles SIGTERM and shuts down tidily did not finish its work, and
        recording it as a success would put a partial run into the record as a
        complete one.
        """
        raw = finished.get("exit_code")
        code = raw if isinstance(raw, int) else None

        if finished.get("spawn_error") is not None:
            return RuntimeStatus(
                state="failed",
                detail=f"the worker could not be started: {finished['spawn_error']}",
                observed_at=utc_now(),
            )

        if finished.get("cancelled") is True:
            state: RuntimeState = "cancelled"
        elif code == 0:
            state = "succeeded"
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

        This runtime **does not signal anything**. It makes the cancellation
        durable and the launcher delivers it, which fixes two problems that a
        direct ``killpg`` from here cannot.

        A signal is not re-assertable. A controller that crashed between
        recording the claim and sending it would, on retry, find the claim
        already recorded and conclude the work was done -- so the cancellation
        would be lost permanently, which is the same "durable intent, no
        effect" ambiguity one layer down that ADR-013 exists to remove. A
        durable request is simply written again.

        And a remembered pid is only a number. After a restart the operating
        system may have given it to something else, and signalling it would
        stop an unrelated process group. The launcher signals instead, because
        it is the only participant that knows the pid is still its own child.
        """
        workload = self._require(runtime_ref)
        digest = _cancel_digest(workload)

        self._registry.claim_cancellation(
            operation_id=operation_id, request_digest=digest, external_id=workload.external_id
        )

        # Unconditional, and after the claim. Every retry of this operation has
        # to reach the effect, or a crash between the two lines above and this
        # one would silence the cancellation for good.
        self._registry.record_cancellation_request(workload.external_id)
        paths = WorkloadPaths(workload.directory)
        paths.directory.mkdir(parents=True, exist_ok=True)
        write_atomic(paths.cancel, {"requested_at": utc_now().isoformat()})

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

        # A fresh reader starts at byte zero, so a controller that restarted
        # still sees the whole file and filters it against its own durable
        # cursor. The byte offset never leaves this generator, and no set of
        # delivered keys grows with the run.
        reader: AppendOnlyJsonlReader[RuntimeEventEnvelope] = AppendOnlyJsonlReader(
            paths.events, _ENVELOPE
        )

        def _pending() -> list[RuntimeEventEnvelope]:
            return [
                envelope
                for envelope in reader.read_new()
                if (envelope.stream_generation, envelope.sequence) > position
            ]

        while True:
            for envelope in _pending():
                yield envelope

            if self._stream_ended(workload.external_id):
                # Nothing can append to this file any more, so one last pass
                # cannot miss an event that a slower check would have caught.
                for envelope in _pending():
                    yield envelope
                return
            await asyncio.sleep(_POLL_SECONDS)

    def _stream_ended(self, external_id: str) -> bool:
        """Whether more telemetry can still arrive for this workload.

        A stream ends when the workload does -- or when the supervisor writing
        it does, which is **not** the same thing and is why this is not simply
        a terminal-state check. A worker outliving its launcher keeps running
        and stops being observed, and ADR-014 calls that a gap to reconcile.
        Waiting for more events that nobody is left to write would hang the
        controller instead of sending it down that path.
        """
        workload = self._registry.workload(external_id)
        assert workload is not None
        if self._status(external_id).state not in _LIVE:
            return True
        supervisor = workload.launcher_pid or workload.spawned_pid
        return supervisor is None or not _alive(supervisor)

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

            if drained and self._status(workload.external_id).state not in _LIVE:
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
    if plan.runtime != BACKEND:
        return (
            f"this plan was resolved for {plan.runtime!r}, not {BACKEND!r}; a "
            f"resolver's decisions are made against one runtime's capabilities "
            f"and running them on another discards the resolution"
        )

    unsupported = sorted(set(plan.runtime_options) - _RUNTIME_OPTIONS)
    if unsupported:
        return (
            f"this runtime does not implement the runtime options "
            f"{', '.join(repr(option) for option in unsupported)}; it understands "
            f"{', '.join(repr(option) for option in sorted(_RUNTIME_OPTIONS))}. "
            f"They are part of the request, so honouring some and ignoring the "
            f"rest would run something other than what was asked for"
        )

    descriptor = plan.spec.compiler.descriptor
    if descriptor is None:
        return (
            f"the plan names compiler {plan.spec.compiler.name!r} but carries no "
            f"PluginDescriptor; ADR-008 requires every plugin to declare one, and "
            f"a plan whose producer cannot be identified cannot be version-checked "
            f"or traced back to what built it"
        )

    try:
        require_supported_plugin(descriptor)
    except IncompatiblePluginError as exc:
        # Refused here rather than raised, so the operation is recorded as
        # rejected and the controller learns nothing was started (ADR-008).
        return str(exc)

    if plan.spec.telemetry.protocol_version != _TELEMETRY_PROTOCOL:
        return (
            f"this runtime speaks {_TELEMETRY_PROTOCOL}, and the plan asks for "
            f"{plan.spec.telemetry.protocol_version!r}; a worker and a controller "
            f"that disagree about the telemetry contract should fail at submission "
            f"rather than halfway through a run"
        )

    placed = sorted(set(plan.spec.environment) & TOPOLOGY_VARIABLES)
    if placed:
        return (
            f"the plan sets {', '.join(placed)}; those place a worker in a "
            f"distributed process group, and this runtime runs one worker in "
            f"none -- honouring them would start a process waiting for peers "
            f"that were never launched"
        )

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
