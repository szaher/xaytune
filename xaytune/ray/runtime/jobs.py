"""``RayJobsRuntime``: a :class:`~xaytune.runtimes.RuntimeBackend` that runs plans as Ray jobs.

```text
ResolvedExecutionPlan
      ↓ submit_or_get(operation_id, plan)
RaySubmissionBackend.submit(submission_id = operation_id)  ── an existing Ray cluster
      ↓ entrypoint
python -m xaytune.ray.runtime.supervisor <shared_state_root>/<operation_id>
      ↓
one worker, writing the same telemetry as under LocalRuntime
```

**Generic, one process.** It runs whatever plan it is handed -- a training
run or an evaluation -- as one supervised worker, exactly as LocalRuntime
would, only on a Ray cluster. It is not Ray Train: a plan asking for several
workers is refused here, and placing a worker group with ``ScalingConfig`` is
``RayTrainRuntime``'s job (PR-033b), composed with the same submission
backends, not derived from this class.

**Mechanical.** It reads the plan's entrypoint, arguments, environment and
resources, never the candidate: it does not know what SFT, LoRA or a learning
rate is, plans nothing, recovers nothing, and searches nothing. The
controller needs no Ray-specific code.

**Get-or-create through Ray's own idempotency key.** The operation id is the
job's ``submission_id``, which Ray refuses to reuse, and the job records in
its metadata the digests of everything it was submitted with: the plan's
request and the ``runtime_env`` it runs in. The same id with either changed
is a different request, and refused. A controller that died after Ray
accepted the job but before it recorded the reference finds the same job by
the same id after a restart -- :meth:`lookup_operation` -- and adopts it;
submitting again returns it too. Restarting never submits a second job.

**v1 state transport: a shared filesystem.** The supervisor writes the
workload directory under ``shared_state_root`` -- the plan, the event stream,
how the worker ended -- exactly as the local launcher does, so ``watch``
replays from a cursor and an ending outlives the controller. In this version
that directory must be mounted at the same absolute path on the controller
and on every Ray node (one machine, or NFS / a PVC). That is a limitation of
how state travels today, not part of the runtime contract: object storage or
a remote telemetry transport can replace it without changing
``RuntimeBackend``. The *code* does not travel this way: the job runs in the
cluster's environment as ``runtime_env`` shapes it -- a ``working_dir`` Ray
uploads, pip packages, an image -- never a path assumed to exist on both
sides.

**It says "unknown" rather than guess.** A cluster that cannot be reached, or
that has forgotten a job, yields ``unknown`` from :meth:`get_status`, and an
error -- never ``None`` -- from :meth:`lookup_operation`, whose ``None`` would
tell the controller re-submitting is safe. Ray's "no such job" is believed
only when nothing durable says otherwise: a job this runtime recorded as
accepted, or whose supervisor ever ran, is not "never received" because a
restarted cluster forgot it -- a finished one is reported completed, any
other accepted with an unknown status.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

from xaytune._version import __version__
from xaytune.core.capabilities import (
    CapabilityDocument,
    CheckpointCapabilities,
    DistributedCapabilities,
    PluginDescriptor,
    ResilienceCapabilities,
)
from xaytune.core.execution import ResolvedExecutionPlan
from xaytune.core.ids import OperationId
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import RuntimeRef
from xaytune.core.telemetry import TELEMETRY_V1ALPHA2, TELEMETRY_V1ALPHA3
from xaytune.ray.runtime._workloads import (
    ACCEPTED,
    REJECTED,
    Launch,
    RayClusterConfig,
    RayWorkloads,
)
from xaytune.ray.submission import RaySubmissionBackend, submission_backend
from xaytune.runtimes import (
    OperationOutcome,
    RuntimeEventEnvelope,
    RuntimeLog,
    RuntimeStatus,
    StreamCursor,
    refusals,
)

__all__ = [
    "ACCEPTED",
    "BACKEND",
    "REJECTED",
    "SUPERVISOR",
    "RayJobsConfig",
    "RayJobsRuntime",
    "ray_jobs_runtime",
]

BACKEND = "ray-jobs"
SUPERVISOR = "xaytune.ray.runtime.supervisor"

# What this backend honours. The same as LocalRuntime's today because the
# same supervisor runs the same worker -- declared here, not borrowed.
_WORKER_REQUESTS = frozenset({"managed_numerical_recovery", "training_interventions"})
_RUNTIME_OPTIONS = frozenset({"working_directory", "checkpoint_restore"}) | _WORKER_REQUESTS
_TELEMETRY_PROTOCOLS = frozenset({TELEMETRY_V1ALPHA2, TELEMETRY_V1ALPHA3})


class RayJobsConfig(RayClusterConfig):
    """The ``ray-jobs`` runtime's configuration (see :class:`RayClusterConfig`)."""


class RayJobsRuntime:
    """Runs plans as Ray jobs, one worker each, supervised as LocalRuntime supervises them.

    Args:
        config: The cluster, its environment and the shared state root.
        submission: How jobs reach the cluster; the backend
            ``config.submission`` describes unless given.
    """

    descriptor = PluginDescriptor(
        api_version="xaytune.plugins/v1alpha1",
        name=BACKEND,
        plugin_version="0.1.0",
        provider="xaytune",
        xaytune_version=__version__,
    )

    def __init__(
        self, config: RayJobsConfig, submission: RaySubmissionBackend | None = None
    ) -> None:
        self.config = config
        self._workloads = RayWorkloads(
            backend=BACKEND,
            config=config,
            submission=submission
            if submission is not None
            else submission_backend(config.submission),
            refuse=_refuse,
            launch=supervised_launch,
        )

    def close(self) -> None:
        """Nothing to release: every answer is read from Ray or the shared state root."""

    def capabilities(self) -> CapabilityDocument:
        return CapabilityDocument(
            distributed=DistributedCapabilities(strategies=(), min_workers=1, max_workers=1),
            checkpoint=CheckpointCapabilities(
                formats=("native-torch/v1",), atomic_commit=False, full_exact_restore=True
            ),
            extensions=FrozenDict(
                {
                    "checkpoint_restore_entrypoints": ("xaytune.workers.native",),
                    "worker_requests": {
                        "xaytune.workers.native": tuple(sorted(_WORKER_REQUESTS)),
                    },
                }
            ),
            resilience=ResilienceCapabilities(
                per_step=False,
                provider="ray-jobs",
                provider_version=self.descriptor.plugin_version,
                # The event stream is a file the supervisor appends to, and an
                # ending is recorded by it and by Ray's job store.
                supports_event_replay=True,
                reports_completed_operations=True,
            ),
        )

    async def submit_or_get(
        self, operation_id: OperationId, plan: ResolvedExecutionPlan
    ) -> RuntimeRef:
        """Submit *plan* as Ray job *operation_id*, or return the job it already is.

        Raises:
            UnsupportedPlanError: The plan cannot run here; recorded, so a
                lookup later learns nothing was started.
            IdempotencyConflictError: This id names a different request.
            RayUnavailableError: Ray could not be asked; nothing is concluded.
        """
        return await self._workloads.submit_or_get(operation_id, plan)

    async def lookup_operation(self, operation_id: OperationId) -> OperationOutcome | None:
        """What became of *operation_id*; ``None`` only when nothing says Ray ever had it."""
        return await self._workloads.lookup_operation(operation_id)

    async def get_status(self, runtime_ref: RuntimeRef) -> RuntimeStatus:
        """Observe a workload, saying ``unknown`` when that is the true answer."""
        return await self._workloads.get_status(runtime_ref)

    async def cancel(self, runtime_ref: RuntimeRef, operation_id: OperationId) -> None:
        """Ask the workload to stop, once per cancellation operation."""
        await self._workloads.cancel(runtime_ref, operation_id)

    def watch(
        self, runtime_ref: RuntimeRef, cursor: StreamCursor | None = None
    ) -> AsyncIterator[RuntimeEventEnvelope]:
        """Stream telemetry after *cursor*, replayed from the supervisor's event file."""
        return self._workloads.watch(runtime_ref, cursor)

    def get_logs(self, runtime_ref: RuntimeRef) -> AsyncIterator[RuntimeLog]:
        """Stream the worker's output from the files its supervisor wrote."""
        return self._workloads.get_logs(runtime_ref)


def ray_jobs_runtime(config: Mapping[str, Any]) -> RayJobsRuntime:
    """The host's factory for ``RuntimeSpec(kind="ray-jobs", config=...)``.

    Register it explicitly::

        EmbeddedControllerHost(..., runtimes={"local": ..., "ray-jobs": ray_jobs_runtime})

    Raises:
        ValueError: The configuration is incomplete or names anything else.
    """
    return RayJobsRuntime(RayJobsConfig.model_validate(dict(config)))


def supervised_launch(plan: ResolvedExecutionPlan) -> Launch:
    """One supervised worker: the plan's resources as Ray's entrypoint options, nothing added."""
    return Launch(module=SUPERVISOR, resources=_entrypoint_resources(plan))


def _refuse(plan: ResolvedExecutionPlan) -> str | None:
    """Why this plan cannot run as a Ray job here, or ``None`` if it can.

    This backend's own list from :mod:`xaytune.runtimes.refusals`. Several
    rules coincide with LocalRuntime's because the same supervisor runs the
    same worker; each stays here only while that holds for Ray (secrets and
    images, say, are Ray-reachable later through ``runtime_env``).
    """
    reasons = (
        refusals.foreign_plan(plan, BACKEND),
        refusals.unimplemented_options(plan, _RUNTIME_OPTIONS),
        refusals.managed_worker_only(plan, _WORKER_REQUESTS),
        refusals.unidentified_producer(plan),
        refusals.unspoken_telemetry(plan, _TELEMETRY_PROTOCOLS),
        refusals.topology_environment(plan),
        refusals.secrets(plan),
        refusals.container_image(plan),
        _workers(plan),
        _resources(plan),
    )
    return next((reason for reason in reasons if reason is not None), None)


def _workers(plan: ResolvedExecutionPlan) -> str | None:
    workers = plan.spec.resources.workers
    if workers is None or workers <= 1:
        return None
    return (
        f"the ray-jobs runtime runs one worker, not {workers}; a Ray Train worker "
        f"group is a separate runtime (ray-train), not this one stretched"
    )


def _resources(plan: ResolvedExecutionPlan) -> str | None:
    """Resources Ray cannot be asked for as stated are refused, not dropped."""
    resources = plan.spec.resources
    unsupported = [
        name
        for name in ("gpu_type", "accelerator_memory_bytes", "max_runtime_seconds")
        if getattr(resources, name) is not None
    ]
    if resources.extensions:
        unsupported.append("extensions")
    if unsupported:
        return (
            f"this runtime cannot honour the resource requirements {', '.join(unsupported)}; "
            f"Ray would schedule the job without them"
        )
    return None


def _entrypoint_resources(plan: ResolvedExecutionPlan) -> dict[str, Any]:
    """The plan's requirements, as Ray's entrypoint scheduling options -- nothing added."""
    resources = plan.spec.resources
    options: dict[str, Any] = {}
    if resources.cpus is not None:
        options["entrypoint_num_cpus"] = resources.cpus
    if resources.gpus is not None:
        options["entrypoint_num_gpus"] = resources.gpus
    if resources.memory_bytes is not None:
        options["entrypoint_memory"] = resources.memory_bytes
    return options
