"""The compile/execute boundary: what crosses it, and what must not.

Contract tests only. There is no real training here and no real runtime — a
fake compiler and a fake backend exist to prove the seam holds, which is the
point of having a seam.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from xaytune.compilation import CompilationContext, SupportResult, TrainerCompiler
from xaytune.core.capabilities import (
    AlgorithmCapabilities,
    CapabilityDocument,
    CapabilityRequirements,
    PluginDescriptor,
)
from xaytune.core.domain.candidate import (
    CandidateSpec,
    DataSpec,
    ModelSpec,
    OptimizationSpec,
    TrainingKind,
    TrainingSpec,
)
from xaytune.core.domain.operation import OperationTargetKind, RuntimeOperationTarget
from xaytune.core.execution import (
    CommandEntrypoint,
    CompilerIdentity,
    PythonModuleEntrypoint,
    ResolvedExecutionPlan,
    ResourceRequirements,
    SecretRef,
    TrainingExecutionSpec,
)
from xaytune.core.ids import CheckpointId, OperationId
from xaytune.core.refs import CheckpointRef, DatasetRef, ModelRef, RuntimeRef
from xaytune.core.resume import ResumeGuarantee
from xaytune.core.telemetry import (
    CheckpointCommittedPayload,
    EvaluationStartedPayload,
    HeartbeatPayload,
)
from xaytune.runtimes import (
    EvaluationEventPayload,
    OperationOutcome,
    RuntimeBackend,
    RuntimeEventEnvelope,
    StreamCursor,
    TrainingEventPayload,
)

_REF = RuntimeRef(backend="fake", external_id="pid-1")
_TARGET = RuntimeOperationTarget(kind="training-attempt", id="ra_1")

DESCRIPTOR = PluginDescriptor(
    api_version="xaytune.plugins/v1alpha1",
    name="fake",
    plugin_version="0.1.0",
    provider="tests",
    xaytune_version="0.6.0",
)


class FakeCompiler:
    """A compiler that compiles and does nothing else.

    Deliberately total: it reads the candidate, emits a spec, and has no way
    to reach a runtime. If this class could submit, the protocol would not be
    enforcing anything.
    """

    descriptor = DESCRIPTOR

    def capabilities(self) -> CapabilityDocument:
        return CapabilityDocument(algorithms=AlgorithmCapabilities(supported=("sft",)))

    def supports(self, candidate: CandidateSpec) -> SupportResult:
        if candidate.training.kind is not TrainingKind.SFT:
            return SupportResult(
                supported=False,
                reasons=(f"{candidate.training.kind.value} is not supported",),
            )
        if candidate.environment is not None:
            return SupportResult(supported=False, reasons=("no agent environments",))
        return SupportResult(supported=True)

    def compile(
        self, candidate: CandidateSpec, context: CompilationContext
    ) -> TrainingExecutionSpec:
        return TrainingExecutionSpec(
            compiler=CompilerIdentity(name="fake", version="0.1.0"),
            candidate_fingerprint=candidate.candidate_fingerprint(),
            entrypoint=PythonModuleEntrypoint(module="xaytune.trainer.worker", function="main"),
            arguments=("--seed", str(context.seed)) if context.seed is not None else (),
            config={
                "kind": candidate.training.kind.value,
                "learning_rate": candidate.training.optimization.learning_rate,
            },
            resources=ResourceRequirements(gpus=1, workers=1),
            required_capabilities=CapabilityRequirements(precision="bf16"),
        )


def _candidate(**overrides: Any) -> CandidateSpec:
    defaults: dict[str, Any] = dict(
        model=ModelSpec(model=ModelRef(uri="Qwen/Qwen3-8B")),
        data=DataSpec(dataset=DatasetRef(uri="./support-v4.jsonl")),
        training=TrainingSpec(
            kind=TrainingKind.SFT,
            optimization=OptimizationSpec(learning_rate=2e-5),
        ),
    )
    defaults.update(overrides)
    return CandidateSpec(**defaults)


def _context() -> CompilationContext:
    return CompilationContext(run_id="run_1", seed=7, output_uri="./out")


# ---- a compiler compiles -------------------------------------------------


def test_a_compiler_turns_a_candidate_into_a_spec() -> None:
    compiler = FakeCompiler()
    candidate = _candidate()

    assert compiler.supports(candidate)
    spec = compiler.compile(candidate, _context())

    assert spec.candidate_fingerprint == candidate.candidate_fingerprint()
    assert spec.entrypoint.module == "xaytune.trainer.worker"


def test_refusal_carries_reasons() -> None:
    """A bare False is not actionable: a planner cannot propose anything better."""
    result = FakeCompiler().supports(_candidate(training=TrainingSpec(kind=TrainingKind.GRPO)))

    assert not result
    assert result.reasons and "grpo" in result.reasons[0]


def test_compilation_is_deterministic() -> None:
    """Otherwise the request_digest built from the plan means nothing.

    A compiler that read the clock, the environment or a global would break
    this silently, and a retry would look like a different request.
    """
    compiler, candidate, context = FakeCompiler(), _candidate(), _context()

    assert compiler.compile(candidate, context) == compiler.compile(candidate, context)


def test_a_compiler_cannot_execute() -> None:
    """The seam's whole purpose, asserted against the protocol surface.

    Filters dunders rather than reading ``__protocol_attrs__``: that is a
    CPython 3.12 implementation detail, so the assertion would pass on the
    interpreter it was written against and fail on 3.10 and 3.11.
    """
    declared = {
        name
        for name, _ in inspect.getmembers(TrainerCompiler, inspect.isfunction)
        if not name.startswith("_")
    }

    assert declared == {"capabilities", "supports", "compile"}
    for forbidden in ("submit", "submit_or_get", "run", "execute", "train"):
        assert not hasattr(FakeCompiler(), forbidden)


# ---- the spec is a wire contract ----------------------------------------


def test_the_spec_round_trips_through_json() -> None:
    """It leaves the process, so this is the property that matters most."""
    spec = FakeCompiler().compile(_candidate(), _context())

    restored = TrainingExecutionSpec.model_validate_json(spec.model_dump_json())

    assert restored == spec


def test_the_plan_round_trips_through_json() -> None:
    plan = ResolvedExecutionPlan(
        target=_TARGET,
        spec=FakeCompiler().compile(_candidate(), _context()),
        runtime="local",
        resolved_capabilities={"precision": "bf16"},
    )

    assert ResolvedExecutionPlan.model_validate_json(plan.model_dump_json()) == plan


def test_the_spec_cannot_hold_a_live_object() -> None:
    """A tokenizer or a closure in a spec is a local variable that will fail
    somewhere else, later, for reasons the record cannot explain."""
    spec = FakeCompiler().compile(_candidate(), _context())

    with pytest.raises(Exception):
        spec.model_copy(update={"config": {"tokenizer": object()}})

    with pytest.raises(Exception):
        spec.model_copy(update={"config": {"callback": lambda: None}})


def test_the_spec_carries_candidate_identity_only() -> None:
    """Not run history or artifact lineage.

    Those describe what a run *did*; a spec is compiled before anything has
    happened, so carrying them would be a claim about the future.
    """
    spec = FakeCompiler().compile(_candidate(), _context())
    fields = set(type(spec).model_fields)

    assert "candidate_fingerprint" in fields
    assert not {"run_history_fingerprint", "artifact_lineage_fingerprint"} & fields


def test_the_spec_says_what_to_run_not_where() -> None:
    """The separation that lets one spec resolve onto any runtime."""
    fields = set(TrainingExecutionSpec.model_fields)

    for placement in ("runtime", "cluster", "queue", "node", "namespace"):
        assert placement not in fields

    assert "runtime" in ResolvedExecutionPlan.model_fields


# ---- the runtime executes ------------------------------------------------


class FakeRuntime:
    """A backend that runs plans and understands nothing about candidates."""

    descriptor = DESCRIPTOR

    def __init__(self) -> None:
        self.submitted: dict[str, ResolvedExecutionPlan] = {}

    def capabilities(self) -> CapabilityDocument:
        return CapabilityDocument()

    async def submit_or_get(self, operation_id: OperationId, plan: ResolvedExecutionPlan):
        self.submitted.setdefault(str(operation_id), plan)
        return None

    async def lookup_operation(self, operation_id: OperationId) -> OperationOutcome | None:
        if str(operation_id) not in self.submitted:
            return None
        return OperationOutcome(
            operation_id=operation_id,
            disposition="accepted",
            runtime_ref=RuntimeRef(backend="fake", external_id=str(operation_id)),
        )


def test_the_runtime_api_is_operation_keyed_from_the_start() -> None:
    """Retrofitting idempotency onto a runtime API is not a refactor."""
    signature = inspect.signature(RuntimeBackend.submit_or_get)
    parameters = list(signature.parameters)

    assert parameters[1] == "operation_id", "identity, not a tag on the call"
    assert not hasattr(RuntimeBackend, "submit"), "ADR-013 rejects create semantics"
    assert "operation_id" in inspect.signature(RuntimeBackend.cancel).parameters


def test_the_runtime_never_receives_a_candidate() -> None:
    """A runtime that understood SFT would be a second home for scientific
    intent, and the two would drift."""
    for method in ("submit_or_get", "lookup_operation", "get_status", "cancel"):
        annotations = inspect.signature(getattr(RuntimeBackend, method)).parameters
        rendered = " ".join(str(p.annotation) for p in annotations.values())
        assert "CandidateSpec" not in rendered
        assert "TrainingSpec" not in rendered


def test_resubmitting_one_operation_does_not_start_a_second_workload() -> None:
    """Get-or-create, never create. Driven with asyncio.run rather than a
    plugin, so the contract test needs no extra dependency."""
    runtime = FakeRuntime()
    plan = ResolvedExecutionPlan(
        target=_TARGET, spec=FakeCompiler().compile(_candidate(), _context()), runtime="local"
    )
    operation_id = OperationId.generate()

    async def submit_twice() -> None:
        await runtime.submit_or_get(operation_id, plan)
        await runtime.submit_or_get(operation_id, plan)

    asyncio.run(submit_twice())

    assert len(runtime.submitted) == 1


# ---- the idempotency key -------------------------------------------------


def _digest(spec: TrainingExecutionSpec) -> str:
    return ResolvedExecutionPlan(target=_TARGET, spec=spec, runtime="local").request_digest(
        "submit"
    )


def test_the_request_digest_hashes_the_whole_request() -> None:
    """Not ExecutionFingerprint. Two submissions can agree on compiler,
    runtime and topology while differing in entrypoint or dataset."""
    spec = FakeCompiler().compile(_candidate(), _context())
    plan = ResolvedExecutionPlan(target=_TARGET, spec=spec, runtime="local")

    other_data = FakeCompiler().compile(
        _candidate(data=DataSpec(dataset=DatasetRef(uri="./support-v5.jsonl"))), _context()
    )
    other_plan = ResolvedExecutionPlan(target=_TARGET, spec=other_data, runtime="local")

    assert plan.request_digest("submit") != other_plan.request_digest("submit")


def test_submitting_and_cancelling_are_different_requests() -> None:
    plan = ResolvedExecutionPlan(
        target=_TARGET, spec=FakeCompiler().compile(_candidate(), _context()), runtime="local"
    )

    assert plan.request_digest("submit") != plan.request_digest("cancel")


def test_the_same_request_digests_the_same() -> None:
    """Which is what makes a retry recognisable as a retry."""
    spec = FakeCompiler().compile(_candidate(), _context())

    first = ResolvedExecutionPlan(target=_TARGET, spec=spec, runtime="local").request_digest(
        "submit"
    )
    second = ResolvedExecutionPlan(target=_TARGET, spec=spec, runtime="local").request_digest(
        "submit"
    )

    assert first == second


def test_a_stable_secret_reference_keeps_the_digest_stable() -> None:
    """Rotating the value behind a reference must not read as a new request.

    An earlier version of this test compared a plan with itself, which proved
    nothing. The observable contract is about the *reference*: the model holds
    no secret value at all, so a rotation is invisible here -- which is the
    point.
    """
    spec = FakeCompiler().compile(_candidate(), _context())
    reference = SecretRef(name="hf-token", source="env", version="v1")

    first = spec.model_copy(update={"secrets": (reference,)})
    second = spec.model_copy(
        update={"secrets": (SecretRef(name="hf-token", source="env", version="v1"),)}
    )

    assert _digest(first) == _digest(second)


def test_a_changed_secret_reference_changes_the_digest() -> None:
    """A different version *is* a different external request (ADR-013)."""
    spec = FakeCompiler().compile(_candidate(), _context())

    v1 = spec.model_copy(update={"secrets": (SecretRef(name="t", source="env", version="v1"),)})
    v2 = spec.model_copy(update={"secrets": (SecretRef(name="t", source="env", version="v2"),)})
    elsewhere = spec.model_copy(
        update={"secrets": (SecretRef(name="t", source="vault", version="v1"),)}
    )

    assert len({_digest(v1), _digest(v2), _digest(elsewhere)}) == 3


def test_a_secret_value_cannot_be_represented() -> None:
    """The model has nowhere to put one, which is stronger than a rule saying
    not to: a plan is persisted and hashed, so an inlined secret would be
    written to durable storage and baked into an identity."""
    assert set(SecretRef.model_fields) == {"name", "source", "version"}
    assert "value" not in SecretRef.model_fields


def test_a_changed_runtime_option_changes_the_digest() -> None:
    """Runtime options are part of the external request (ADR-013).

    Changing a launcher changes what the runtime is being asked to do, even
    when the spec is byte-identical.
    """
    spec = FakeCompiler().compile(_candidate(), _context())

    torchrun = ResolvedExecutionPlan(
        target=_TARGET, spec=spec, runtime="local", runtime_options={"launcher": "torchrun"}
    )
    subprocess_ = ResolvedExecutionPlan(
        target=_TARGET, spec=spec, runtime="local", runtime_options={"launcher": "subprocess"}
    )

    assert torchrun.request_digest("submit") != subprocess_.request_digest("submit")


# ---- the resolved plan is a separate object ------------------------------


def test_the_same_plan_for_a_different_attempt_is_a_different_request() -> None:
    """The target is part of the request, not decoration beside it.

    Two attempts of the same candidate compile to the same spec. If the digest
    ignored who the plan was for, get-or-create would hand the second attempt
    the first one's running workload.
    """
    spec = FakeCompiler().compile(_candidate(), _context())

    first = ResolvedExecutionPlan(spec=spec, runtime="local", target=_TARGET)
    second = ResolvedExecutionPlan(
        spec=spec,
        runtime="local",
        target=RuntimeOperationTarget(kind="training-attempt", id="ra_2"),
    )

    assert first.spec == second.spec
    assert first.request_digest("submit") != second.request_digest("submit")


def test_one_spec_resolves_onto_different_runtimes() -> None:
    """The reason spec and plan are not one object."""
    spec = FakeCompiler().compile(_candidate(), _context())

    local = ResolvedExecutionPlan(target=_TARGET, spec=spec, runtime="local")
    ray = ResolvedExecutionPlan(target=_TARGET, spec=spec, runtime="ray")

    assert local.spec == ray.spec
    assert local.request_digest("submit") != ray.request_digest("submit")


# ---- the entrypoint is exactly one thing --------------------------------


def test_an_entrypoint_cannot_be_both_a_module_and_a_command() -> None:
    """Nothing would say which the runtime should prefer.

    Every backend would invent its own precedence, and two backends would run
    the same spec differently. The tagged union makes it unrepresentable.
    """
    with pytest.raises(ValidationError):
        PythonModuleEntrypoint(module="x.worker", argv=("python", "other.py"))

    with pytest.raises(ValidationError):
        CommandEntrypoint(argv=("python",), module="x.worker")


def test_a_command_entrypoint_is_an_argv_not_a_shell_string() -> None:
    """A string would be re-parsed by whichever shell the runtime uses, so
    quoting would vary between backends."""
    entrypoint = CommandEntrypoint(argv=("python", "-m", "x", "--flag", "a b"))

    assert entrypoint.argv[-1] == "a b", "one argument, not two"
    with pytest.raises(ValidationError):
        CommandEntrypoint(argv=())


def test_both_entrypoint_kinds_round_trip_through_the_spec() -> None:
    """The discriminator has to survive the wire, or the union is decoration."""
    for entrypoint in (
        PythonModuleEntrypoint(module="xaytune.trainer.worker", function="main"),
        CommandEntrypoint(argv=("python", "-m", "xaytune.trainer.worker")),
    ):
        spec = (
            FakeCompiler()
            .compile(_candidate(), _context())
            .model_copy(update={"entrypoint": entrypoint})
        )
        restored = TrainingExecutionSpec.model_validate_json(spec.model_dump_json())

        assert restored.entrypoint == entrypoint
        assert type(restored.entrypoint) is type(entrypoint)


# ---- the telemetry envelope implements ADR-014 --------------------------


def test_an_evaluation_cannot_emit_a_checkpoint_event() -> None:
    """A type error, not a convention someone has to remember.

    Evaluation produces no checkpoints, which is why its state machine has
    neither CHECKPOINTING nor RECOVERING.
    """
    assert (
        TrainingEventPayload(
            data=CheckpointCommittedPayload(
                checkpoint_ref=CheckpointRef(id=CheckpointId.generate(), uri="file:///checkpoint"),
                optimizer_step=0,
                data_cursor=None,
                resume_guarantee=ResumeGuarantee(
                    state="model-only", data="none", boundary="optimizer-step"
                ),
            )
        ).type
        == "CheckpointCommitted"
    )

    with pytest.raises(ValidationError):
        EvaluationEventPayload.model_validate({"data": {"type": "CheckpointCommitted"}})


def test_a_target_cannot_carry_the_other_workloads_telemetry() -> None:
    """Splitting the payload is not enough on its own.

    The discriminated union constrains a payload in isolation, so it only
    helps a caller who already chose the right family. Pairing the families
    with the target is what stops an evaluation attempt reporting a
    ``CheckpointCommitted`` (ADR-014 §1).
    """
    with pytest.raises(ValidationError):
        RuntimeEventEnvelope(
            event_id="e",
            target=RuntimeOperationTarget(kind="evaluation-attempt", id="eval_1"),
            sequence=0,
            payload=TrainingEventPayload(
                data=CheckpointCommittedPayload(
                    checkpoint_ref=CheckpointRef(
                        id=CheckpointId.generate(), uri="file:///checkpoint"
                    ),
                    optimizer_step=0,
                    data_cursor=None,
                    resume_guarantee=ResumeGuarantee(
                        state="model-only", data="none", boundary="optimizer-step"
                    ),
                )
            ),
        )

    with pytest.raises(ValidationError):
        RuntimeEventEnvelope(
            event_id="e",
            target=RuntimeOperationTarget(kind="training-attempt", id="run_1"),
            sequence=0,
            payload=EvaluationEventPayload(data=EvaluationStartedPayload()),
        )


def test_the_shared_vocabulary_stays_shared() -> None:
    """The pairing is by family, not by event name.

    Both workloads emit ``Heartbeat``; each must emit it as its own family, and
    neither may borrow the other's.
    """
    for kind, payload in (
        (
            "training-attempt",
            TrainingEventPayload(data=HeartbeatPayload(expected_interval_seconds=5)),
        ),
        (
            "evaluation-attempt",
            EvaluationEventPayload(data=HeartbeatPayload(expected_interval_seconds=5)),
        ),
    ):
        envelope = RuntimeEventEnvelope(
            event_id="e",
            target=RuntimeOperationTarget(kind=kind, id="x"),
            sequence=0,
            payload=payload,
        )
        assert envelope.payload.type == "Heartbeat"


def test_every_target_kind_admits_exactly_one_telemetry_family() -> None:
    """No kind is left unpaired, and none accepts both.

    Driven from ``OperationTargetKind`` itself rather than a list written here,
    so a target kind added without a telemetry family fails this test instead
    of quietly accepting anything.
    """
    families = (
        TrainingEventPayload(data=HeartbeatPayload(expected_interval_seconds=5)),
        EvaluationEventPayload(data=HeartbeatPayload(expected_interval_seconds=5)),
    )

    for kind in get_args(OperationTargetKind):
        accepted = []
        for payload in families:
            try:
                RuntimeEventEnvelope(
                    event_id="e",
                    target=RuntimeOperationTarget(kind=kind, id="x"),
                    sequence=0,
                    payload=payload,
                )
            except ValidationError:
                continue
            accepted.append(payload.workload)

        assert len(accepted) == 1, f"{kind} accepts {accepted}, expected exactly one family"


def test_the_envelope_names_its_target_the_way_the_journal_does() -> None:
    """So an event stream and the operation that started it agree."""
    envelope = RuntimeEventEnvelope(
        event_id="evt_1",
        target=RuntimeOperationTarget(kind="evaluation-attempt", id="ea_1"),
        sequence=0,
        payload=EvaluationEventPayload(data=EvaluationStartedPayload()),
    )

    restored = RuntimeEventEnvelope.model_validate_json(envelope.model_dump_json())
    assert restored == envelope
    assert restored.target.kind == "evaluation-attempt"


def test_sequences_and_generations_are_counters() -> None:
    """Both start at zero; -1 belongs only to a cursor that recorded nothing."""
    target = RuntimeOperationTarget(kind="training-attempt", id="a_1")

    with pytest.raises(ValidationError):
        RuntimeEventEnvelope(
            event_id="e",
            target=target,
            sequence=-1,
            payload=TrainingEventPayload(data=HeartbeatPayload(expected_interval_seconds=5)),
        )

    with pytest.raises(ValidationError):
        StreamCursor(generation=-1)
    with pytest.raises(ValidationError):
        StreamCursor(sequence=-2)

    assert StreamCursor().sequence == -1, "nothing recorded yet"


# ---- an outcome cannot carry contradictory evidence ---------------------


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        ("rejected with a workload", {"disposition": "rejected", "runtime_ref": _REF}),
        ("accepted without one", {"disposition": "accepted"}),
        ("completed without one", {"disposition": "completed"}),
    ],
)
def test_an_outcome_cannot_be_contradictory(label: str, kwargs: dict) -> None:
    """This type carries evidence during recovery.

    Contradictory evidence is worse than none: it tells the controller a
    workload exists and not where, or that one was refused and also started.
    """
    with pytest.raises(ValidationError):
        OperationOutcome(operation_id=OperationId.generate(), **kwargs)
    assert label


def test_only_a_rejection_makes_reissuing_safe() -> None:
    """An accepted or completed operation must be adopted, not repeated."""

    def outcome(disposition: str, **kw) -> OperationOutcome:
        return OperationOutcome(operation_id=OperationId.generate(), disposition=disposition, **kw)

    assert outcome("rejected").may_reissue
    assert not outcome("accepted", runtime_ref=_REF).may_reissue
    assert not outcome("completed", runtime_ref=_REF).may_reissue


# ---- the environment is what a process can actually hold ----------------


def test_environment_values_must_be_strings() -> None:
    """An OS environment holds strings.

    Coercing here would mean every runtime inventing its own rules, and they
    would differ on booleans and floats.
    """
    spec = FakeCompiler().compile(_candidate(), _context())

    assert spec.model_copy(update={"environment": {"WORKERS": "4"}})

    for bad in ({"WORKERS": 4}, {"DEBUG": True}, {"NESTED": {"a": "b"}}):
        with pytest.raises(ValidationError):
            spec.model_copy(update={"environment": bad})
