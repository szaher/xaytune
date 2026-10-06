"""Reasons a runtime refuses a plan, each one a rule a backend chooses to apply.

Refusing beats ignoring. A plan that declares secrets or an image has said
the workload needs them; running it anyway would start a process that fails
somewhere inside the worker, for a reason the logs would attribute to the
training code rather than to a runtime that quietly dropped part of the
request.

Every rule here is runtime-neutral: it takes what a backend implements as an
argument and names no backend. Which rules apply is each backend's decision,
listed in its own ``_refuse`` -- sharing the wording never shares a
limitation, so one backend learning to honour secrets or containers drops
that rule from its list without touching another's.

Each returns why the plan cannot run, or ``None``.
"""

from __future__ import annotations

from xaytune.core.capabilities import require_supported_plugin
from xaytune.core.errors import IncompatiblePluginError
from xaytune.core.execution import (
    PythonModuleEntrypoint,
    ResolvedExecutionPlan,
    TrainingExecutionSpec,
)
from xaytune.runtimes.worker import TOPOLOGY_VARIABLES

__all__ = [
    "MANAGED_NATIVE_WORKER",
    "container_image",
    "foreign_plan",
    "managed_worker_only",
    "secrets",
    "topology_environment",
    "unidentified_producer",
    "unimplemented_options",
    "unspoken_telemetry",
]

MANAGED_NATIVE_WORKER = "xaytune.workers.native"


def foreign_plan(plan: ResolvedExecutionPlan, backend: str) -> str | None:
    """A plan resolved for another runtime."""
    if plan.runtime == backend:
        return None
    return (
        f"this plan was resolved for {plan.runtime!r}, not {backend!r}; a "
        f"resolver's decisions are made against one runtime's capabilities "
        f"and running them on another discards the resolution"
    )


def unimplemented_options(plan: ResolvedExecutionPlan, implemented: frozenset[str]) -> str | None:
    """Runtime options outside the closed set the backend *implemented*.

    Checked as a closed set rather than read opportunistically: an unknown
    option is a caller asking for behaviour, and running without it would run
    something other than what was asked for.
    """
    unsupported = sorted(set(plan.runtime_options) - implemented)
    if not unsupported:
        return None
    return (
        f"this runtime does not implement the runtime options "
        f"{', '.join(repr(option) for option in unsupported)}; it understands "
        f"{', '.join(repr(option) for option in sorted(implemented))}. "
        f"They are part of the request, so honouring some and ignoring the "
        f"rest would run something other than what was asked for"
    )


def managed_worker_only(plan: ResolvedExecutionPlan, worker_requests: frozenset[str]) -> str | None:
    """Checkpoint restore, or a request in *worker_requests*, for any worker but the managed one.

    The managed Native worker is the only one that reads them; any other
    would silently ignore them.
    """
    managed = (
        isinstance(plan.spec, TrainingExecutionSpec)
        and isinstance(plan.spec.entrypoint, PythonModuleEntrypoint)
        and plan.spec.entrypoint.module == MANAGED_NATIVE_WORKER
        and plan.spec.checkpoint.format == "native-torch/v1"
    )
    if "checkpoint_restore" in plan.runtime_options and (
        not managed
        or not isinstance(plan.spec, TrainingExecutionSpec)
        or plan.spec.checkpoint.store_uri is None
    ):
        return "FULL+EXACT restore is supported only by the managed Native worker"
    requested = sorted(set(plan.runtime_options) & worker_requests)
    if requested and not managed:
        return (
            f"worker requests {', '.join(repr(item) for item in requested)} are "
            f"honoured only by the managed Native worker"
        )
    return None


def unidentified_producer(plan: ResolvedExecutionPlan) -> str | None:
    """A producer -- compiler or evaluator alike -- without a supported descriptor (ADR-008)."""
    producer = plan.spec.producer
    descriptor = producer.descriptor
    if descriptor is None:
        return (
            f"the plan names {producer.name!r} as its producer but carries no "
            f"PluginDescriptor; ADR-008 requires every plugin to declare one, and "
            f"a plan whose producer cannot be identified cannot be version-checked "
            f"or traced back to what built it"
        )
    try:
        require_supported_plugin(descriptor)
    except IncompatiblePluginError as exc:
        # Refused rather than raised, so the operation is recorded as
        # rejected and the controller learns nothing was started.
        return str(exc)
    return None


def unspoken_telemetry(plan: ResolvedExecutionPlan, protocols: frozenset[str]) -> str | None:
    """A telemetry protocol outside the ones the backend reads."""
    if plan.spec.telemetry.protocol_version in protocols:
        return None
    return (
        f"this runtime speaks {', '.join(sorted(protocols))}, and the plan asks for "
        f"{plan.spec.telemetry.protocol_version!r}; a worker and a controller "
        f"that disagree about the telemetry contract should fail at submission "
        f"rather than halfway through a run"
    )


def topology_environment(plan: ResolvedExecutionPlan) -> str | None:
    """Variables that place a worker in a process group, for a backend that forms none."""
    placed = sorted(set(plan.spec.environment) & TOPOLOGY_VARIABLES)
    if not placed:
        return None
    return (
        f"the plan sets {', '.join(placed)}; those place a worker in a "
        f"distributed process group, and this runtime runs one worker in "
        f"none -- honouring them would start a process waiting for peers "
        f"that were never launched"
    )


def secrets(plan: ResolvedExecutionPlan) -> str | None:
    """Secret references, for a backend that cannot resolve them."""
    if not plan.spec.secrets:
        return None
    names = ", ".join(secret.name for secret in plan.spec.secrets)
    return (
        f"this runtime cannot resolve secret references ({names}); running "
        f"the workload without them would fail inside the worker instead of here"
    )


def container_image(plan: ResolvedExecutionPlan) -> str | None:
    """A container image, for a backend that runs processes."""
    if plan.spec.container is None:
        return None
    return (
        f"this runtime runs subprocesses, not containers; "
        f"{plan.spec.container.image!r} needs a container runtime"
    )
