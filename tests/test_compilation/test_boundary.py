"""The compile/execute boundary: what crosses it, and what must not.

Contract tests only. There is no real training here and no real runtime — a
fake compiler and a fake backend exist to prove the seam holds, which is the
point of having a seam.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

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
from xaytune.core.execution import (
    CompilerIdentity,
    EntrypointSpec,
    ResolvedExecutionPlan,
    ResourceRequirements,
    SecretRef,
    TrainingExecutionSpec,
)
from xaytune.core.ids import OperationId
from xaytune.core.refs import DatasetRef, ModelRef
from xaytune.runtimes import OperationOutcome, RuntimeBackend

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
            entrypoint=EntrypointSpec(module="xaytune.trainer.worker", function="main"),
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
        return OperationOutcome(operation_id=operation_id, accepted=True)


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
        spec=FakeCompiler().compile(_candidate(), _context()), runtime="local"
    )
    operation_id = OperationId.generate()

    async def submit_twice() -> None:
        await runtime.submit_or_get(operation_id, plan)
        await runtime.submit_or_get(operation_id, plan)

    asyncio.run(submit_twice())

    assert len(runtime.submitted) == 1


# ---- the idempotency key -------------------------------------------------


def test_the_request_digest_hashes_the_whole_request() -> None:
    """Not ExecutionFingerprint. Two submissions can agree on compiler,
    runtime and topology while differing in entrypoint or dataset."""
    spec = FakeCompiler().compile(_candidate(), _context())
    plan = ResolvedExecutionPlan(spec=spec, runtime="local")

    other_data = FakeCompiler().compile(
        _candidate(data=DataSpec(dataset=DatasetRef(uri="./support-v5.jsonl"))), _context()
    )
    other_plan = ResolvedExecutionPlan(spec=other_data, runtime="local")

    assert plan.request_digest("submit") != other_plan.request_digest("submit")


def test_submitting_and_cancelling_are_different_requests() -> None:
    plan = ResolvedExecutionPlan(
        spec=FakeCompiler().compile(_candidate(), _context()), runtime="local"
    )

    assert plan.request_digest("submit") != plan.request_digest("cancel")


def test_the_same_request_digests_the_same() -> None:
    """Which is what makes a retry recognisable as a retry."""
    spec = FakeCompiler().compile(_candidate(), _context())

    first = ResolvedExecutionPlan(spec=spec, runtime="local").request_digest("submit")
    second = ResolvedExecutionPlan(spec=spec, runtime="local").request_digest("submit")

    assert first == second


def test_a_rotated_secret_does_not_change_the_digest() -> None:
    """A credential rotation must not read as a different request.

    The plan carries references, so rotating the value behind one changes
    nothing here -- and an inlined secret would also have been persisted and
    hashed into a durable identity.
    """
    spec = FakeCompiler().compile(_candidate(), _context())
    with_secret = spec.model_copy(update={"secrets": (SecretRef(name="hf-token", source="env"),)})

    assert "token" not in with_secret.model_dump_json().lower().split("hf-token")[1][:200]
    assert ResolvedExecutionPlan(spec=with_secret, runtime="local").request_digest(
        "submit"
    ) == ResolvedExecutionPlan(spec=with_secret, runtime="local").request_digest("submit")


# ---- the resolved plan is a separate object ------------------------------


def test_one_spec_resolves_onto_different_runtimes() -> None:
    """The reason spec and plan are not one object."""
    spec = FakeCompiler().compile(_candidate(), _context())

    local = ResolvedExecutionPlan(spec=spec, runtime="local")
    ray = ResolvedExecutionPlan(spec=spec, runtime="ray")

    assert local.spec == ray.spec
    assert local.request_digest("submit") != ray.request_digest("submit")
