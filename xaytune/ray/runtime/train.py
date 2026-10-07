"""``RayTrainRuntime``: training as a Ray Train worker group, evaluation as one supervised job.

```text
RuntimeSpec(kind="ray-train")
  TrainingExecutionSpec    → Ray job → train_driver → TorchTrainer(ScalingConfig(num_workers=N))
                                                          → N ranks, each running the plan's worker
  EvaluationExecutionSpec  → Ray job → supervisor → one worker      (as RayJobsRuntime runs it)
```

**Ray Train for training, at every size.** A training plan always runs as a
``TorchTrainer`` worker group -- one worker included -- so ``workers=1`` here
is Ray Train placing one worker, not the single-process runtime under another
name.

**Evaluation still runs.** An experiment records one runtime, and its
evaluations resolve against that runtime too, so an experiment that trains
on ``ray-train`` must be able to evaluate there. An evaluation plan runs the
way :class:`~xaytune.ray.runtime.jobs.RayJobsRuntime` runs one: a single
supervised worker in one Ray job.

**Composed, not derived.** Job tracking -- get-or-create under the operation
id, adoption after any restart, forgotten jobs, cancellation claims, replayed
telemetry -- is the same private machinery ``RayJobsRuntime`` composes, over
any :class:`~xaytune.ray.submission.RaySubmissionBackend`. Neither runtime
inherits from the other.

**Resources are interpreted only where unambiguous.** The plan states
workload-level requirements; Ray Train asks per worker. Where the plan does
not say how the workload divides, nothing is guessed and the plan is refused:

```text
workers                 required: the group's size
gpus  None or 0         no GPUs
gpus == workers         one GPU per worker
gpus anything else      refused -- how they divide is not stated
cpus, memory_bytes      with one worker, that worker's; with several, refused
```

With no ``cpus`` stated, Ray Train's own per-worker default applies; the
scaling actually used is recorded in ``launch.json`` before Ray is asked.

**One telemetry sequencer**: the driver, relaying rank 0's observations only
(see :mod:`~xaytune.ray.runtime.train_group`).
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
from xaytune.core.execution import (
    CommandEntrypoint,
    PythonModuleEntrypoint,
    ResolvedExecutionPlan,
    TrainingExecutionSpec,
)
from xaytune.core.ids import OperationId
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import RuntimeRef
from xaytune.core.telemetry import TELEMETRY_V1ALPHA2, TELEMETRY_V1ALPHA3
from xaytune.ray.runtime._workloads import Launch, RayClusterConfig, RayWorkloads
from xaytune.ray.runtime.jobs import supervised_launch
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
    "BACKEND",
    "TRAIN_DRIVER",
    "RayTrainConfig",
    "RayTrainRuntime",
    "ray_train_runtime",
    "scaling_for",
]

BACKEND = "ray-train"
TRAIN_DRIVER = "xaytune.ray.runtime.train_driver"

_WORKER_REQUESTS = frozenset({"managed_numerical_recovery", "training_interventions"})
_RUNTIME_OPTIONS = frozenset({"working_directory", "checkpoint_restore"}) | _WORKER_REQUESTS
_TELEMETRY_PROTOCOLS = frozenset({TELEMETRY_V1ALPHA2, TELEMETRY_V1ALPHA3})
_BUILT_IN_WORKERS = "xaytune.workers."


class RayTrainConfig(RayClusterConfig):
    """The ``ray-train`` runtime's configuration (see :class:`RayClusterConfig`)."""


class RayTrainRuntime:
    """Trains as a Ray Train worker group; evaluates as one supervised Ray job.

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
        self, config: RayTrainConfig, submission: RaySubmissionBackend | None = None
    ) -> None:
        self.config = config
        self._workloads = RayWorkloads(
            backend=BACKEND,
            config=config,
            submission=submission
            if submission is not None
            else submission_backend(config.submission),
            refuse=_refuse,
            launch=_launch,
        )

    def close(self) -> None:
        """Nothing to release: every answer is read from Ray or the shared state root."""

    def capabilities(self) -> CapabilityDocument:
        return CapabilityDocument(
            # The runtime places the group; which strategy the workers use
            # over it is the worker's, so none is claimed here.
            distributed=DistributedCapabilities(strategies=(), min_workers=1, max_workers=None),
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
                provider="ray-train",
                provider_version=self.descriptor.plugin_version,
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
        """Ask the workload -- every rank of it -- to stop, once per cancellation operation."""
        await self._workloads.cancel(runtime_ref, operation_id)

    def watch(
        self, runtime_ref: RuntimeRef, cursor: StreamCursor | None = None
    ) -> AsyncIterator[RuntimeEventEnvelope]:
        """Stream telemetry after *cursor*, replayed from the driver's event file."""
        return self._workloads.watch(runtime_ref, cursor)

    def get_logs(self, runtime_ref: RuntimeRef) -> AsyncIterator[RuntimeLog]:
        """Stream rank 0's output (each other rank's is kept beside it)."""
        return self._workloads.get_logs(runtime_ref)


def ray_train_runtime(config: Mapping[str, Any]) -> RayTrainRuntime:
    """The host's factory for ``RuntimeSpec(kind="ray-train", config=...)``.

    Raises:
        ValueError: The configuration is incomplete or names anything else.
    """
    return RayTrainRuntime(RayTrainConfig.model_validate(dict(config)))


def scaling_for(plan: ResolvedExecutionPlan) -> dict[str, Any]:
    """The ``ScalingConfig`` keywords for a training plan this runtime accepts.

    Raises:
        ValueError: The plan's resources do not say how they divide (see the
            module docstring) -- the reason it is refused.
    """
    resources = plan.spec.resources
    workers = resources.workers
    if workers is None or workers < 1:
        raise ValueError(
            "a Ray Train worker group needs its size: the plan must state resources.workers"
        )
    gpus = resources.gpus or 0
    if gpus not in (0, workers):
        raise ValueError(
            f"{gpus} GPUs over {workers} workers does not say how they divide; this runtime "
            f"places either none or one per worker (gpus == workers)"
        )
    per_worker: dict[str, Any] = {}
    for name, ray_name, value in (
        ("cpus", "CPU", resources.cpus),
        ("memory_bytes", "memory", resources.memory_bytes),
    ):
        if value is None:
            continue
        if workers > 1:
            raise ValueError(
                f"resources.{name} is the workload's total, and how it divides over "
                f"{workers} workers is not stated; state it for one worker, or leave it out"
            )
        per_worker[ray_name] = value
    if gpus:
        per_worker["GPU"] = 1
    return {
        "num_workers": workers,
        "use_gpu": bool(gpus),
        "resources_per_worker": per_worker or None,
    }


def _launch(plan: ResolvedExecutionPlan) -> Launch:
    if not isinstance(plan.spec, TrainingExecutionSpec):
        return supervised_launch(plan)
    # The driver itself asks Ray for nothing: the group's workers hold the
    # resources, and they are placed by Ray Train.
    return Launch(module=TRAIN_DRIVER, record={"scaling": scaling_for(plan)})


def _refuse(plan: ResolvedExecutionPlan) -> str | None:
    """Why this plan cannot run here, or ``None`` -- this backend's own list."""
    training = isinstance(plan.spec, TrainingExecutionSpec)
    reasons = (
        refusals.foreign_plan(plan, BACKEND),
        refusals.unimplemented_options(plan, _RUNTIME_OPTIONS),
        refusals.managed_worker_only(plan, _WORKER_REQUESTS),
        refusals.unidentified_producer(plan),
        refusals.unspoken_telemetry(plan, _TELEMETRY_PROTOCOLS),
        # The runtime sets placement; a plan that sets it would contradict it.
        refusals.topology_environment(plan),
        refusals.secrets(plan),
        refusals.container_image(plan),
        _unplaced_resources(plan),
        _scaling(plan) if training else _single_worker(plan),
        _managed_requests_need_one_worker(plan) if training else None,
        _built_in_worker_in_a_group(plan) if training else None,
    )
    return next((reason for reason in reasons if reason is not None), None)


def _scaling(plan: ResolvedExecutionPlan) -> str | None:
    try:
        scaling_for(plan)
    except ValueError as ambiguous:
        return str(ambiguous)
    return None


def _single_worker(plan: ResolvedExecutionPlan) -> str | None:
    workers = plan.spec.resources.workers
    if workers is None or workers <= 1:
        return None
    return (
        f"an evaluation runs as one supervised worker on this runtime, not {workers}; only "
        f"training runs as a Ray Train worker group"
    )


def _managed_requests_need_one_worker(plan: ResolvedExecutionPlan) -> str | None:
    """Restore and managed-worker requests address one worker; a group has several."""
    requested = sorted(set(plan.runtime_options) & (_WORKER_REQUESTS | {"checkpoint_restore"}))
    workers = plan.spec.resources.workers or 0
    if not requested or workers <= 1:
        return None
    return (
        f"{', '.join(repr(item) for item in requested)} address a single managed worker, and "
        f"this plan asks for {workers}"
    )


def _built_in_worker_in_a_group(plan: ResolvedExecutionPlan) -> str | None:
    """Xaytune's own workers run as one process; a group of them is refused.

    The Native and TRL workers publish the trained model from whichever rank
    runs them, to one output path -- and Native, at ``world_size > 1``,
    wraps the model in FSDP, whose ``save_pretrained`` on every rank is no
    full-model publication. Distributed publication (gather, barrier, one
    publishing rank) is a worker's to implement deliberately; until a
    built-in worker does, a group of them would race to publish a model
    none of them holds whole.
    """
    workers = plan.spec.resources.workers or 0
    if workers <= 1:
        return None
    module = _worker_module(plan)
    if module is None or not module.startswith(_BUILT_IN_WORKERS):
        return None
    return (
        f"{module} runs as one process and publishes its model from every rank, so it "
        f"cannot run as a group of {workers}: distributed publication is not implemented "
        f"by Xaytune's built-in workers yet"
    )


def _worker_module(plan: ResolvedExecutionPlan) -> str | None:
    """The Python module the plan's worker runs, however its entrypoint names it."""
    entrypoint = plan.spec.entrypoint
    if isinstance(entrypoint, PythonModuleEntrypoint):
        return entrypoint.module
    if isinstance(entrypoint, CommandEntrypoint):
        argv = list(entrypoint.argv)
        for flag, value in zip(argv, argv[1:], strict=False):
            if flag == "-m":
                return value
        # A module named as an argument -- the launcher's own form for a
        # module function -- is still that worker.
        return next((arg for arg in argv if arg.startswith(_BUILT_IN_WORKERS)), None)
    return None


def _unplaced_resources(plan: ResolvedExecutionPlan) -> str | None:
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
