"""How a Ray runtime tracks the jobs it submits -- shared, private, composed.

Every Ray-backed :class:`~xaytune.runtimes.RuntimeBackend` keeps the same
promises about a submitted job, whatever runs inside it: get-or-create under
the operation id, adoption after any restart, "unknown" rather than a guess,
cancellation idempotent by its own operation id, telemetry replayed from the
one event stream its single sequencer writes. :class:`RayWorkloads` keeps them
once. A runtime composes it with what is its own -- the name it answers to,
the plans it refuses, and how each plan is launched (:class:`Launch`) --
rather than inheriting another runtime's.

Not public: ``RayJobsRuntime`` and ``RayTrainRuntime`` are the concepts.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import tempfile
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from xaytune.core.clock import utc_now
from xaytune.core.errors import IdempotencyConflictError
from xaytune.core.execution import ResolvedExecutionPlan
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import OperationId
from xaytune.core.immutable import FrozenDict, FrozenDomainModel
from xaytune.core.refs import RuntimeRef
from xaytune.ray.runtime.supervisor import CLAIM
from xaytune.ray.submission import RayJob, RaySubmissionBackend, RayUnavailableError
from xaytune.runtimes import (
    OperationOutcome,
    RuntimeEventEnvelope,
    RuntimeLog,
    RuntimeState,
    RuntimeStatus,
    StreamCursor,
    UnsupportedPlanError,
)
from xaytune.runtimes.local.jsonl import AppendOnlyJsonlReader
from xaytune.runtimes.local.paths import WorkloadPaths, read_json, write_atomic
from xaytune.runtimes.local.runtime import _read_lines, finished_status

__all__ = [
    "ACCEPTED",
    "CANCELLATIONS",
    "LAUNCH",
    "REJECTED",
    "Launch",
    "RayClusterConfig",
    "RayWorkloads",
]

REJECTED = "rejected.json"
ACCEPTED = "accepted.json"
LAUNCH = "launch.json"
CANCELLATIONS = "cancellations"

_TERMINAL: frozenset[RuntimeState] = frozenset({"succeeded", "failed", "cancelled", "preempted"})
_LIVE: frozenset[RuntimeState] = frozenset({"pending", "queued", "starting", "running"})
_POLL_SECONDS = 0.05
_ENVELOPE: TypeAdapter[RuntimeEventEnvelope] = TypeAdapter(RuntimeEventEnvelope)

_DIGEST = "xaytune.request_digest"
_ENVIRONMENT = "xaytune.environment_digest"
_TARGET_KIND = "xaytune.target_kind"
_TARGET_ID = "xaytune.target_id"


class RayClusterConfig(FrozenDomainModel):
    """Which cluster, its environment, and where state lives: what every Ray runtime needs.

    All three are required; none is read from the environment or defaulted,
    so the record says which cluster and which environment ran a workload.
    It is persisted with the experiment: credentials are never configuration
    -- the address carries none, and neither may ``runtime_env``.
    """

    address: str
    """The job-submission address of an existing cluster, such as ``http://127.0.0.1:8265``."""
    runtime_env: FrozenDict
    """Ray's ``runtime_env`` for every job, as Ray defines it; ``{}`` runs in the cluster's own.

    The job's ``python`` must import xaytune there: install it in the
    cluster's image, or ask Ray for it here (``pip``, ``working_dir``).
    """
    shared_state_root: str
    """v1: an absolute directory mounted at this same path on the controller and every node."""

    def model_post_init(self, __context: Any) -> None:
        reasons = []
        if not self.address.startswith(("http://", "https://")):
            reasons.append(
                f"address must be an http(s) job-submission address, not {self.address!r}"
            )
        if not Path(self.shared_state_root).is_absolute():
            reasons.append(
                f"shared_state_root must be an absolute path, not {self.shared_state_root!r}"
            )
        if reasons:
            raise ValueError("; ".join(reasons))


@dataclass(frozen=True)
class Launch:
    """How one plan becomes a Ray job: the entrypoint module, and what it asks Ray for.

    ``record`` is written to the workload directory (``launch.json``) before
    Ray is asked, so how a job was placed is on record next to its plan.
    """

    module: str
    resources: Mapping[str, Any] = field(default_factory=dict)
    record: Mapping[str, Any] = field(default_factory=dict)


class RayWorkloads:
    """The jobs one Ray runtime submitted, tracked through Ray and the shared state root.

    Args:
        backend: The runtime's name, which its references carry.
        config: The cluster, its environment and the shared state root.
        submission: How jobs reach the cluster.
        refuse: Why a plan cannot run on this runtime, or ``None``.
        launch: How a plan this runtime accepts is launched.
    """

    def __init__(
        self,
        *,
        backend: str,
        config: RayClusterConfig,
        submission: RaySubmissionBackend,
        refuse: Callable[[ResolvedExecutionPlan], str | None],
        launch: Callable[[ResolvedExecutionPlan], Launch],
    ) -> None:
        self._backend = backend
        self._jobs = submission
        self._runtime_env = config.runtime_env
        self._refuse = refuse
        self._launch = launch
        self._root = Path(config.shared_state_root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def _environment(self) -> str:
        """The digest of the ``runtime_env`` every job runs in -- part of its identity."""
        return fingerprint(self._runtime_env)

    async def submit_or_get(
        self, operation_id: OperationId, plan: ResolvedExecutionPlan
    ) -> RuntimeRef:
        """Submit *plan* as Ray job *operation_id*, or return the job it already is.

        The plan is written to the workload directory before Ray is asked, so
        a job never exists without the plan its supervisor reads. A job Ray
        already holds under this id is returned if it was submitted for the
        same plan in the same ``runtime_env``, and refused as a conflict
        otherwise -- including when the submission itself failed but a retry
        finds Ray accepted it. Once Ray has the job, ``accepted.json`` records
        it, so a cluster that later forgets the job cannot make it look new.

        Raises:
            UnsupportedPlanError: The plan cannot run here; recorded, so a
                lookup later learns nothing was started.
            IdempotencyConflictError: This id names a different request.
            RayUnavailableError: Ray could not be asked; nothing is concluded.
        """
        digest = plan.request_digest("submit")
        environment = self._environment
        external_id = str(operation_id)
        paths = WorkloadPaths(self._root / external_id)
        _require_unclaimed_by_cancel(self._root, external_id)

        rejected = read_json(paths.directory / REJECTED)
        if rejected is not None:
            # Refused for the plan alone; the environment never reached Ray.
            if rejected.get("request_digest") != digest:
                raise IdempotencyConflictError(external_id, ("request_digest",))
            raise UnsupportedPlanError(str(rejected.get("detail")))

        refusal = self._refuse(plan)
        if refusal is not None:
            write_atomic(
                paths.directory / REJECTED,
                {"request_digest": digest, "detail": refusal, "at": utc_now().isoformat()},
            )
            raise UnsupportedPlanError(refusal)

        existing = await self._info(external_id)
        if existing is not None:
            _require_same(external_id, existing.metadata, digest, environment)
            _record_accepted(paths, digest, environment)
            return self._ref(external_id)

        recorded_acceptance = read_json(paths.directory / ACCEPTED)
        if recorded_acceptance is not None:
            _require_same(
                external_id,
                {
                    _DIGEST: str(recorded_acceptance.get("request_digest")),
                    _ENVIRONMENT: str(recorded_acceptance.get("environment_digest")),
                },
                digest,
                environment,
            )
        if self._forgotten(operation_id) is not None:
            # Ray forgot a job it had: returned, never submitted a second time.
            return self._ref(external_id)

        paths.directory.mkdir(parents=True, exist_ok=True)
        recorded = read_json(paths.plan)
        if recorded is None:
            write_atomic(paths.plan, plan.model_dump(mode="json"))
        elif ResolvedExecutionPlan.model_validate(recorded).request_digest("submit") != digest:
            raise IdempotencyConflictError(external_id, ("request_digest",))
        launch = self._launch(plan)
        if launch.record:
            write_atomic(paths.directory / LAUNCH, dict(launch.record))

        try:
            await asyncio.to_thread(
                self._jobs.submit,
                external_id,
                # ``python`` as the job's environment resolves it: the cluster
                # image's, or the one runtime_env builds -- never a path the
                # controller assumes exists on the node.
                shlex.join(["python", "-m", launch.module, str(paths.directory)]),
                metadata={
                    _DIGEST: digest,
                    _ENVIRONMENT: environment,
                    _TARGET_KIND: plan.target.kind,
                    _TARGET_ID: plan.target.id,
                },
                resources=dict(launch.resources),
                runtime_env=self._runtime_env,
            )
        except RayUnavailableError:
            # Ray may have accepted it and the answer been lost -- or refused
            # a duplicate id raced in by another submitter. Its job store is
            # what decides, not this exception.
            accepted = await self._info(external_id)
            if accepted is None:
                raise
            _require_same(external_id, accepted.metadata, digest, environment)
        _record_accepted(paths, digest, environment)
        return self._ref(external_id)

    # -- asking what happened ---------------------------------------------

    async def lookup_operation(self, operation_id: OperationId) -> OperationOutcome | None:
        """What became of *operation_id*; ``None`` only when Ray says it never had it.

        Raises:
            RayUnavailableError: Ray could not be asked. Not ``None``: that
                would tell the controller re-submitting is safe.
        """
        external_id = str(operation_id)
        rejected = read_json(self._root / external_id / REJECTED)
        if rejected is not None:
            return OperationOutcome(
                operation_id=operation_id,
                disposition="rejected",
                detail=str(rejected.get("detail")),
            )
        job = await self._info(external_id)
        if job is None:
            return self._forgotten(operation_id)
        status = self._status(external_id, job)
        return OperationOutcome(
            operation_id=operation_id,
            disposition="completed" if status.state in _TERMINAL else "accepted",
            runtime_ref=self._ref(external_id),
            status=status,
        )

    def _forgotten(self, operation_id: OperationId) -> OperationOutcome | None:
        """What Ray's "no such job" means, given what this runtime recorded.

        ``None`` -- never received, safe to re-submit -- only if nothing
        durable says Ray ever had the job. A 404 from a cluster that restarted
        without its job store proves nothing about a job it once accepted, and
        re-submitting one that already ran would run it twice.
        """
        external_id = str(operation_id)
        paths = WorkloadPaths(self._root / external_id)
        finished = read_json(paths.finished)
        if finished is not None:
            return OperationOutcome(
                operation_id=operation_id,
                disposition="completed",
                runtime_ref=self._ref(external_id),
                status=finished_status(finished),
            )
        evidence = [
            path.name
            for path in (
                paths.directory / ACCEPTED,
                paths.directory / CLAIM,
                paths.started,
                paths.events,
            )
            if path.exists()
        ]
        if not evidence:
            return None
        return OperationOutcome(
            operation_id=operation_id,
            disposition="accepted",
            runtime_ref=self._ref(external_id),
            status=RuntimeStatus(
                state="unknown",
                detail=(
                    f"Ray has no record of this job, but {', '.join(evidence)} shows it "
                    f"accepted the job; it is not re-submitted, and how it ended is unknown"
                ),
                observed_at=utc_now(),
            ),
        )

    async def get_status(self, runtime_ref: RuntimeRef) -> RuntimeStatus:
        """Observe a workload, saying ``unknown`` when that is the true answer."""
        external_id = self._require(runtime_ref)
        finished = read_json(self._root / external_id / "finished.json")
        if finished is not None:
            return finished_status(finished)
        try:
            job = await self._info(external_id)
        except RayUnavailableError as unavailable:
            return RuntimeStatus(state="unknown", detail=str(unavailable), observed_at=utc_now())
        return self._status(external_id, job)

    def _status(self, external_id: str, job: RayJob | None) -> RuntimeStatus:
        """The workload's state: its supervisor's record first, then Ray's.

        The supervisor's ``finished.json`` says how the worker ended, which
        Ray cannot -- a worker cancelled gracefully exits 0, and is still
        cancelled. Ray answers what the files cannot: a job not yet started,
        or one that ended without its supervisor recording an ending.
        """
        paths = WorkloadPaths(self._root / external_id)
        finished = read_json(paths.finished)
        if finished is not None:
            return finished_status(finished)
        now = utc_now()
        if job is None:
            return RuntimeStatus(
                state="unknown",
                detail="Ray has no record of this job: the cluster forgot it, or never had it",
                observed_at=now,
            )
        cancel_requested = read_json(paths.cancel) is not None
        never_started = read_json(paths.started) is None
        if job.status == "PENDING":
            return RuntimeStatus(
                state="pending",
                # Stopping is asynchronous, and Ray may keep a job it cannot
                # schedule pending after being asked. Its status decides.
                detail="cancellation requested; Ray has not started the job"
                if cancel_requested
                else None,
                observed_at=now,
            )
        if job.status == "RUNNING":
            state: RuntimeState = "running" if read_json(paths.started) else "starting"
            return RuntimeStatus(state=state, observed_at=now)
        if job.status == "STOPPED":
            return RuntimeStatus(state="cancelled", detail="Ray stopped the job", observed_at=now)
        if job.status == "FAILED" and cancel_requested and never_started:
            return RuntimeStatus(
                state="cancelled",
                # Ray ended a job whose worker never ran, after it was asked to
                # stop it (a start timeout, for one it could not schedule).
                detail="cancelled before it started; Ray then ended the job",
                observed_at=now,
            )
        if job.status == "FAILED":
            return RuntimeStatus(
                state="failed",
                exit_code=job.exit_code,
                detail="the job failed before its supervisor recorded an ending",
                observed_at=now,
            )
        return RuntimeStatus(
            state="unknown",
            detail=(
                f"Ray reports the job {job.status} but its supervisor recorded no ending "
                f"under {paths.directory}: is shared_state_root mounted on the Ray node?"
            ),
            observed_at=now,
        )

    # -- cancelling --------------------------------------------------------

    async def cancel(self, runtime_ref: RuntimeRef, operation_id: OperationId) -> None:
        """Ask the workload to stop, once per cancellation operation.

        The operation is claimed for this workload first, durably, as
        LocalRuntime's registry claims it: the same *operation_id* again for
        the same workload is a retry, and the same id for another workload is
        an :class:`~xaytune.core.errors.IdempotencyConflictError`. A retry
        reasserts the effect rather than trusting the claim, so a crash
        between the two never silences a cancellation.

        The effect is durable too: ``cancel.request`` is written and the
        supervisor delivers it to the worker's process group, once. A job Ray
        has not started yet has no supervisor to deliver it, so Ray is asked
        to stop that one too -- and the supervisor, should it start anyway,
        finds the request and records a cancellation that never ran.
        """
        external_id = self._require(runtime_ref)
        _claim_cancellation(self._root, operation_id, external_id)
        paths = WorkloadPaths(self._root / external_id)
        if read_json(paths.cancel) is None:
            write_atomic(
                paths.cancel,
                {"requested_at": utc_now().isoformat(), "operation_id": str(operation_id)},
            )
        if read_json(paths.started) is None:
            job = await self._info(external_id)
            if job is not None and job.status == "PENDING":
                await asyncio.to_thread(self._jobs.stop, external_id)

    # -- watching ----------------------------------------------------------

    async def watch(
        self, runtime_ref: RuntimeRef, cursor: StreamCursor | None = None
    ) -> AsyncIterator[RuntimeEventEnvelope]:
        """Stream telemetry after *cursor*, replayed from the supervisor's event file."""
        external_id = self._require(runtime_ref)
        paths = WorkloadPaths(self._root / external_id)
        position = (cursor.generation, cursor.sequence) if cursor else (0, -1)
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
            if (await self.get_status(runtime_ref)).state not in _LIVE:
                # Nothing appends after the workload ends, so one more pass
                # cannot miss an event.
                for envelope in _pending():
                    yield envelope
                return
            await asyncio.sleep(_POLL_SECONDS)

    async def get_logs(self, runtime_ref: RuntimeRef) -> AsyncIterator[RuntimeLog]:
        """Stream the worker's output from the files its supervisor wrote."""
        external_id = self._require(runtime_ref)
        paths = WorkloadPaths(self._root / external_id)
        offsets = {paths.stdout: 0, paths.stderr: 0}
        while True:
            drained = True
            for path, stream in ((paths.stdout, "stdout"), (paths.stderr, "stderr")):
                lines, offsets[path] = _read_lines(path, offsets[path])
                for line in lines:
                    drained = False
                    yield RuntimeLog(stream=stream, line=line)  # type: ignore[arg-type]
            if drained and (await self.get_status(runtime_ref)).state not in _LIVE:
                return
            await asyncio.sleep(_POLL_SECONDS)

    # -- helpers -----------------------------------------------------------

    async def _info(self, external_id: str) -> RayJob | None:
        return await asyncio.to_thread(self._jobs.info, external_id)

    def _ref(self, external_id: str) -> RuntimeRef:
        return RuntimeRef(backend=self._backend, external_id=external_id)

    def _require(self, runtime_ref: RuntimeRef) -> str:
        if runtime_ref.backend != self._backend:
            raise KeyError(
                f"{runtime_ref.backend!r} is not a {self._backend} workload: this runtime can "
                f"only answer for references it issued"
            )
        return runtime_ref.external_id


def _require_same(
    external_id: str, metadata: Mapping[str, str], digest: str, environment: str
) -> None:
    """The job under this id was submitted for this plan, in this ``runtime_env``."""
    differing = tuple(
        field
        for field, recorded, expected in (
            ("request_digest", metadata.get(_DIGEST), digest),
            ("runtime_env", metadata.get(_ENVIRONMENT), environment),
        )
        if recorded != expected
    )
    if differing:
        raise IdempotencyConflictError(external_id, differing)


def _record_accepted(paths: WorkloadPaths, digest: str, environment: str) -> None:
    """Durable evidence that Ray accepted the job, outliving Ray's own memory of it."""
    if read_json(paths.directory / ACCEPTED) is None:
        write_atomic(
            paths.directory / ACCEPTED,
            {
                "request_digest": digest,
                "environment_digest": environment,
                "at": utc_now().isoformat(),
            },
        )


def _claim_cancellation(root: Path, operation_id: OperationId, external_id: str) -> None:
    """Bind cancellation *operation_id* to *external_id*, first come first served.

    Created exclusively (``os.link`` refuses an existing name), so two
    controllers racing with one id cannot both believe they bound it.

    Raises:
        IdempotencyConflictError: The id already names a submission, or a
            cancellation of another workload.
    """
    cancel_id = str(operation_id)
    if (root / cancel_id).is_dir():
        raise IdempotencyConflictError(cancel_id, ("operation_type",))
    directory = root / CANCELLATIONS
    directory.mkdir(parents=True, exist_ok=True)
    claim = directory / f"{cancel_id}.json"
    if not claim.exists():
        descriptor, staged = tempfile.mkstemp(dir=directory, prefix=".claim-")
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"target": external_id, "requested_at": utc_now().isoformat()}, handle)
        try:
            os.link(staged, claim)
        except FileExistsError:
            pass
        finally:
            os.unlink(staged)
    recorded = read_json(claim)
    if recorded is None or recorded.get("target") != external_id:
        raise IdempotencyConflictError(cancel_id, ("request_digest",))


def _require_unclaimed_by_cancel(root: Path, external_id: str) -> None:
    """A submission id must not already name a cancellation."""
    if (root / CANCELLATIONS / f"{external_id}.json").exists():
        raise IdempotencyConflictError(external_id, ("operation_type",))
