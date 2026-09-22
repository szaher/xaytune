"""What crosses the compile/execute boundary.

```text
CandidateSpec            what we are testing
      ↓ TrainerCompiler
TrainingExecutionSpec    how to run it, runtime-neutral
      ↓ CapabilityResolver
ResolvedExecutionPlan    what a specific runtime will actually execute
      ↓ RuntimeBackend.submit_or_get
RuntimeRef
```

Everything here **leaves the controller's process**, which is the constraint
that shapes it: JSON-serializable, versioned, no live objects, no closures, no
implicit filesystem. A spec holding a tokenizer instance or a `Path` that only
resolves on the submitting host is not a spec — it is a local variable that
will fail somewhere else, later, for reasons the record will not explain
(ADR-016).

Deterministic too: compiling the same candidate in the same context twice must
produce the same spec, or the `request_digest` built from it cannot mean
anything.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from xaytune.core.capabilities import CapabilityRequirements, PluginDescriptor
from xaytune.core.fingerprint import fingerprint
from xaytune.core.immutable import FrozenDict, FrozenDomainModel

__all__ = [
    "ArtifactInput",
    "ArtifactOutput",
    "CheckpointExecutionContract",
    "CompilerIdentity",
    "ContainerSpec",
    "DependencySpec",
    "EntrypointSpec",
    "ResolvedExecutionPlan",
    "ResourceRequirements",
    "SecretRef",
    "TelemetryContract",
    "TrainingExecutionSpec",
]


class CompilerIdentity(FrozenDomainModel):
    """Which compiler produced a spec, and at what version.

    Part of `ExecutionFingerprint`: the same candidate compiled by two
    compilers, or by two versions of one, can execute differently, and
    provenance has to be able to say which one ran.
    """

    name: str
    version: str
    descriptor: PluginDescriptor | None = None


class EntrypointSpec(FrozenDomainModel):
    """How the worker process is started.

    A module and function rather than a callable: a callable cannot cross a
    process boundary, and pickling one would bind the plan to the exact
    interpreter that built it.
    """

    module: str
    function: str | None = None
    command: tuple[str, ...] = Field(default_factory=tuple)


class DependencySpec(FrozenDomainModel):
    """What the worker environment must contain.

    ``lock_digest`` is what makes this reproducible rather than approximate --
    a requirement list resolves differently on different days.
    """

    requirements: tuple[str, ...] = Field(default_factory=tuple)
    lock_digest: str | None = None
    python_version: str | None = None


class ContainerSpec(FrozenDomainModel):
    """The image a workload runs in, when it runs in one.

    ``digest`` rather than a tag alone: a tag is mutable, so recording only
    ``:latest`` means the provenance record cannot say what actually ran.
    """

    image: str
    digest: str | None = None
    pull_policy: str | None = None


class SecretRef(FrozenDomainModel):
    """A reference to a secret, never its value (ADR-016).

    A plan is persisted, fingerprinted and shipped between processes. A secret
    inlined here would be written to durable storage and hashed into an
    identity; a rotation would also change that identity and turn a routine
    credential change into an `IdempotencyConflict`.
    """

    name: str
    source: str
    version: str | None = None


class ArtifactInput(FrozenDomainModel):
    """Something the workload reads, named so a remote worker can find it."""

    name: str
    uri: str
    digest: str | None = None
    optional: bool = False


class ArtifactOutput(FrozenDomainModel):
    """Something the workload is expected to produce."""

    name: str
    uri: str
    kind: str | None = None


class ResourceRequirements(FrozenDomainModel):
    """What the workload needs to be scheduled.

    Requirements, not placement. Which node, queue or cluster satisfies them is
    the runtime's decision -- the control plane says what is needed and never
    how to find it, which is what lets one plan run on Local, Ray or Training
    Hub unchanged.
    """

    gpus: int | None = None
    gpu_type: str | None = None
    workers: int | None = None
    cpus: float | None = None
    memory_bytes: int | None = None
    accelerator_memory_bytes: int | None = None
    max_runtime_seconds: int | None = None

    extensions: FrozenDict = Field(default_factory=FrozenDict)


class CheckpointExecutionContract(FrozenDomainModel):
    """What the worker must do about checkpoints, and what it must record.

    The controller learns a resumable position exists only when a checkpoint is
    reported (ADR-014), so this is what makes recovery possible at all rather
    than a storage preference.

    ``boundary`` is ``optimizer-step`` because a checkpoint taken mid-window
    holds partial gradients the optimizer state does not reflect, and is
    therefore ineligible for batch-size-changing resume (ADR-012 §3).
    """

    store_uri: str | None = None
    format: str | None = None
    every_optimizer_steps: int | None = None
    keep_last: int | None = None
    boundary: Literal["optimizer-step", "mid-accumulation"] = "optimizer-step"
    require_atomic_commit: bool = True


class TelemetryContract(FrozenDomainModel):
    """Which telemetry protocol the worker speaks (ADR-014).

    Named in the spec rather than assumed, so a worker and a controller that
    disagree fail at submission instead of halfway through a run.
    """

    protocol_version: str = "xaytune.telemetry/v1alpha1"
    heartbeat_seconds: int | None = None
    endpoint: str | None = None


class TrainingExecutionSpec(FrozenDomainModel):
    """How to run a candidate — runtime-neutral, and ready to cross a boundary.

    Produced by a :class:`~xaytune.compilation.TrainerCompiler`, consumed by a
    resolver. It says everything about *what to execute* and nothing about
    *where*: no cluster, no queue, no node. That separation is what lets the
    same spec be resolved onto Local, Ray or Training Hub without recompiling
    the candidate.

    ``candidate_fingerprint`` travels with it so an artifact can be traced back
    to the hypothesis it tested, across a boundary the candidate itself does
    not cross.
    """

    api_version: str = "xaytune.execution/v1alpha1"

    compiler: CompilerIdentity
    candidate_fingerprint: str

    entrypoint: EntrypointSpec
    arguments: tuple[str, ...] = Field(default_factory=tuple)
    config: FrozenDict = Field(default_factory=FrozenDict)

    environment: FrozenDict = Field(default_factory=FrozenDict)
    secrets: tuple[SecretRef, ...] = Field(default_factory=tuple)

    dependencies: DependencySpec = Field(default_factory=DependencySpec)
    container: ContainerSpec | None = None

    inputs: tuple[ArtifactInput, ...] = Field(default_factory=tuple)
    outputs: tuple[ArtifactOutput, ...] = Field(default_factory=tuple)

    resources: ResourceRequirements = Field(default_factory=ResourceRequirements)

    checkpoint: CheckpointExecutionContract = Field(default_factory=CheckpointExecutionContract)
    telemetry: TelemetryContract = Field(default_factory=TelemetryContract)

    required_capabilities: CapabilityRequirements = Field(default_factory=CapabilityRequirements)

    metadata: FrozenDict = Field(default_factory=FrozenDict)


class ResolvedExecutionPlan(FrozenDomainModel):
    """What a specific runtime will actually execute.

    A :class:`TrainingExecutionSpec` plus the decisions a resolver made against
    one runtime's capabilities. The spec is what to run; this is what will run.

    Separate objects because the same spec resolves differently per runtime,
    and a controller comparing two runs needs to see which differences were
    scientific and which were resolution.
    """

    api_version: str = "xaytune.plan/v1alpha1"

    spec: TrainingExecutionSpec
    runtime: str
    resolved_capabilities: FrozenDict = Field(default_factory=FrozenDict)
    resolution_notes: tuple[str, ...] = Field(default_factory=tuple)

    def request_digest(self, operation_type: str) -> str:
        """The idempotency key for submitting or cancelling this plan (ADR-013).

        Hashes the **whole external request** — the operation type and the
        entire plan — and deliberately **not** ``ExecutionFingerprint``. Those
        answer different questions::

            ExecutionFingerprint   are these executions equivalent?
            request_digest         is this literally the same request?

        Two submissions can agree on compiler, runtime, GPU type and topology
        while differing in entrypoint, arguments, dataset or output location.
        Deriving the key from the fingerprint would return the *original*
        workload for a request that was not the same one — the failure
        get-or-create exists to prevent, arriving through the key rather than
        through the call.

        Secrets are referenced, never inlined (:class:`SecretRef`), so a
        credential rotation does not change the digest and turn a routine
        rotation into an ``IdempotencyConflict``.
        """
        return fingerprint({"type": "runtime-request", "operation": operation_type, "plan": self})
