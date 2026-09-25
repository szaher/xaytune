"""What crosses the compile/execute boundary.

```text
CandidateSpec            what we are testing        EvaluationSpec + subject
      ↓ TrainerCompiler                                   ↓ Evaluator
TrainingExecutionSpec    how to run it,             EvaluationExecutionSpec
      └──────────── runtime-neutral ──────────────────────┘
                              ↓ CapabilityResolver
ResolvedExecutionPlan    what a specific runtime will actually execute
      ↓ RuntimeBackend.submit_or_get
RuntimeRef
```

The two specs are siblings, not one type. Training and evaluation stay
separate domain objects (ADR-015 §2); what they share is **transport** -- an
entrypoint, a config, an environment, resources -- because a runtime executes
those mechanically whatever the workload is. :data:`ExecutionSpec` is that
union at the wire, and only there.

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

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from xaytune.core.capabilities import CapabilityRequirements, PluginDescriptor
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.fingerprint import fingerprint
from xaytune.core.immutable import FrozenDict, FrozenDomainModel
from xaytune.core.observability import ObservabilitySpec
from xaytune.core.refs import ArtifactRef
from xaytune.core.telemetry import TELEMETRY_V1ALPHA2, TELEMETRY_V1ALPHA3

__all__ = [
    "ArtifactInput",
    "ArtifactOutput",
    "CheckpointExecutionContract",
    "CompilerIdentity",
    "ContainerSpec",
    "DependencySpec",
    "CommandEntrypoint",
    "EntrypointSpec",
    "EvaluationExecutionSpec",
    "EvaluatorIdentity",
    "ExecutionSpec",
    "PythonModuleEntrypoint",
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


class EvaluatorIdentity(FrozenDomainModel):
    """Which evaluator prepared a spec, and at what version.

    The evaluation counterpart of :class:`CompilerIdentity`: the runtime checks
    the descriptor the same way (ADR-008), and provenance can say which
    implementation measured.
    """

    name: str
    version: str
    descriptor: PluginDescriptor | None = None


class PythonModuleEntrypoint(FrozenDomainModel):
    """Start the worker by importing a module.

    A module and function name rather than a callable: a callable cannot cross
    a process boundary, and pickling one would bind the plan to the exact
    interpreter that built it.
    """

    kind: Literal["python-module"] = "python-module"
    module: str
    function: str | None = None


class CommandEntrypoint(FrozenDomainModel):
    """Start the worker by running an argument vector.

    ``argv``, never a shell string. A string would be re-parsed by whatever
    shell the runtime happens to use, so quoting and word-splitting would vary
    between backends -- and an argument containing a space would mean different
    things in different places.
    """

    kind: Literal["command"] = "command"
    argv: tuple[str, ...] = Field(min_length=1)


EntrypointSpec = Annotated[PythonModuleEntrypoint | CommandEntrypoint, Field(discriminator="kind")]
"""How the worker process is started: exactly one way.

A single model carrying both a module and a command would let a spec declare
both, and nothing would say which the runtime should prefer -- so every backend
would invent its own precedence, and two backends would run the same spec
differently. A tagged union makes the ambiguity unrepresentable."""


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

    protocol_version: str = TELEMETRY_V1ALPHA2
    """A string rather than the known versions, so a plan asking for one this
    runtime does not speak reaches the runtime and is refused at submission,
    naming it -- rather than failing to construct somewhere upstream."""

    heartbeat_seconds: int | None = None
    endpoint: str | None = None


class _WorkloadExecutionSpec(FrozenDomainModel):
    """What a runtime executes, whatever the workload is for.

    Transport, not domain: an entrypoint, its configuration, its environment
    and what it needs to be scheduled. A runtime executes these mechanically
    -- LocalRuntime starts a process from them and never asks whether it
    trains or evaluates -- so both workload specs carry them, declared once.
    What a workload *means* is on the subclasses.
    """

    entrypoint: EntrypointSpec
    arguments: tuple[str, ...] = Field(default_factory=tuple)
    config: FrozenDict = Field(default_factory=FrozenDict)

    environment: FrozenDict = Field(default_factory=FrozenDict)
    """Process environment. Values are strings, and validated as such.

    An OS environment holds strings, so a spec carrying ``{"WORKERS": 4}``
    would force every runtime to invent its own coercion -- and they would
    differ on booleans and floats. Rejecting it here means one rule instead of
    one per backend.
    """

    secrets: tuple[SecretRef, ...] = Field(default_factory=tuple)

    dependencies: DependencySpec = Field(default_factory=DependencySpec)
    container: ContainerSpec | None = None

    inputs: tuple[ArtifactInput, ...] = Field(default_factory=tuple)
    outputs: tuple[ArtifactOutput, ...] = Field(default_factory=tuple)

    resources: ResourceRequirements = Field(default_factory=ResourceRequirements)

    telemetry: TelemetryContract = Field(default_factory=TelemetryContract)
    observability: ObservabilitySpec = Field(default_factory=ObservabilitySpec)

    required_capabilities: CapabilityRequirements = Field(default_factory=CapabilityRequirements)

    metadata: FrozenDict = Field(default_factory=FrozenDict)

    @field_validator("environment")
    @classmethod
    def _environment_values_are_strings(cls, value: FrozenDict) -> FrozenDict:
        offending = {k: type(v).__name__ for k, v in value.items() if not isinstance(v, str)}
        if offending:
            raise ValueError(
                f"environment values must be strings; got {offending}. An OS "
                f"environment holds strings, and coercing here would mean every "
                f"runtime inventing its own rules for booleans and numbers"
            )
        return value

    @property
    def producer(self) -> CompilerIdentity | EvaluatorIdentity:
        """The plugin that produced this spec, which a runtime version-checks."""
        raise NotImplementedError


class TrainingExecutionSpec(_WorkloadExecutionSpec):
    """How to run a candidate — runtime-neutral, and ready to cross a boundary.

    Produced by a :class:`~xaytune.compilation.TrainerCompiler`, consumed by a
    resolver. It says everything about *what to execute* and nothing about
    *where*: no cluster, no queue, no node. That separation is what lets the
    same spec be resolved onto Local, Ray or Training Hub without recompiling
    the candidate.

    ``candidate_fingerprint`` travels with it so an artifact can be traced back
    to the hypothesis it tested, across a boundary the candidate itself does
    not cross.

    Its fields are exactly those it had before evaluation existed: a plan's
    ``request_digest`` hashes all of them, and a field added here would change
    the digest of every training submission already recorded -- so a restart
    after an upgrade could not re-issue one it proved was never received.
    """

    api_version: Literal["xaytune.execution/v1alpha1"] = "xaytune.execution/v1alpha1"

    compiler: CompilerIdentity
    candidate_fingerprint: str

    checkpoint: CheckpointExecutionContract = Field(default_factory=CheckpointExecutionContract)

    @property
    def producer(self) -> CompilerIdentity:
        return self.compiler


class EvaluationExecutionSpec(_WorkloadExecutionSpec):
    """How to run one evaluation of one subject — the sibling of training's spec.

    Produced by an evaluator, consumed by the same resolver and runtimes. It
    carries no checkpoint contract, because evaluation writes none, and no
    candidate: what it names is the **subject** being measured and the
    ``evaluation_fingerprint`` of what measures it.

    Always telemetry ``v1alpha3``: its completion must carry the metrics,
    which is what makes the result durable without the controller reading
    the worker's files.
    """

    api_version: Literal["xaytune.evaluation-execution/v1alpha1"] = (
        "xaytune.evaluation-execution/v1alpha1"
    )

    evaluator: EvaluatorIdentity
    evaluation_fingerprint: str
    subject: ArtifactRef

    telemetry: TelemetryContract = Field(
        default_factory=lambda: TelemetryContract(protocol_version=TELEMETRY_V1ALPHA3)
    )

    @field_validator("telemetry")
    @classmethod
    def _results_travel_inline(cls, value: TelemetryContract) -> TelemetryContract:
        if value.protocol_version != TELEMETRY_V1ALPHA3:
            raise ValueError(
                f"an evaluation reports its result under {TELEMETRY_V1ALPHA3}, not "
                f"{value.protocol_version!r}: earlier versions have no way to carry it"
            )
        return value

    @property
    def producer(self) -> EvaluatorIdentity:
        return self.evaluator


ExecutionSpec = Annotated[
    TrainingExecutionSpec | EvaluationExecutionSpec, Field(discriminator="api_version")
]
"""Either workload's spec, told apart by ``api_version`` -- at the wire only.

Not a domain ``Execution`` (ADR-015 §2 declines one): nothing above the
resolver handles this union, and training and evaluation remain separate
types everywhere they mean something."""


class ResolvedExecutionPlan(FrozenDomainModel):
    """What a specific runtime will actually execute.

    An :data:`ExecutionSpec` plus the decisions a resolver made against
    one runtime's capabilities, and the attempt it is being run for. The spec
    is what to run; this is what will run, and for whom.

    Separate objects because the same spec resolves differently per runtime,
    and a controller comparing two runs needs to see which differences were
    scientific and which were resolution.
    """

    api_version: str = "xaytune.plan/v1alpha1"

    spec: ExecutionSpec
    runtime: str

    target: RuntimeOperationTarget
    """Which attempt this plan is being executed for.

    Required, and it travels with the plan rather than being passed beside it,
    because the runtime needs it to do two things it cannot otherwise do.

    ``watch()`` returns :class:`~xaytune.runtimes.RuntimeEventEnvelope`, whose
    ``target`` is mandatory and whose payload family is pinned to the target
    kind -- so a backend that did not know its target could not emit a single
    valid telemetry event. And a workload the runtime finds still running after
    a restart has to be reported against *something*; a plan that did not say
    which attempt it belonged to would leave the controller holding a live
    process it could not attribute.

    Part of ``request_digest`` as a result, which is correct rather than
    incidental: the same spec submitted for a different attempt is a different
    request, and get-or-create must not return the first attempt's workload to
    the second.
    """

    resolved_capabilities: FrozenDict = Field(default_factory=FrozenDict)
    """What the resolver decided the runtime will provide.

    A record of resolution, not a place to configure the runtime -- those are
    different things, and conflating them would make the record unreadable.
    """

    runtime_options: FrozenDict = Field(default_factory=FrozenDict)
    """Backend-specific execution settings: launcher, working directory, and
    later Ray or Training Hub configuration.

    Part of the external request, so it is part of ``request_digest``
    (ADR-013): changing a launcher changes what the runtime is being asked to
    do, even when the spec is identical.
    """

    resolution_notes: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _spec_matches_target(self) -> ResolvedExecutionPlan:
        # The runtime picks the telemetry family from the target, so a
        # training spec aimed at an evaluation attempt would run a trainer
        # whose every observation is refused -- or worse, one that looks
        # like evaluation to everything downstream.
        expected = _SPEC_FOR_TARGET[self.target.kind]
        if not isinstance(self.spec, expected):
            raise ValueError(
                f"a {self.target.kind} is executed from a {expected.__name__}, not a "
                f"{type(self.spec).__name__}: the spec and the target are two statements "
                f"about the same workload"
            )
        return self

    def request_digest(self, operation_type: Literal["submit", "cancel"]) -> str:
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


_SPEC_FOR_TARGET: dict[str, type[_WorkloadExecutionSpec]] = {
    "training-attempt": TrainingExecutionSpec,
    "evaluation-attempt": EvaluationExecutionSpec,
}
"""Which spec executes each target kind -- the plan's counterpart of the
envelope's payload-family pairing (ADR-014 §1)."""
