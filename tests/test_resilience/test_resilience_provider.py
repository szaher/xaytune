"""The generic resilience-provider boundary (PR-035, commit 1).

A provider adds one canonical, versioned request to a training plan and changes
nothing else; ``augment_execution_plan`` holds it to that, and refuses --
before anything is submitted -- a combination that would carry the request
untruthfully.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from tests.test_resilience.resilience_support import (
    ENGINE,
    KIND,
    POLICY,
    SCHEMA,
    ReplicaRecoveryProvider,
    hosting,
    per_step_runtime,
    providers,
    resilience_spec,
)
from xaytune import __version__
from xaytune.compilation.attempt_resolution import training_execution_fingerprint
from xaytune.core.capabilities import CapabilityDocument, PluginDescriptor
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.domain.resilience import (
    ResilienceImplementation,
    ResiliencePolicy,
    ResilienceSpec,
    resilience_spec_fingerprint,
)
from xaytune.core.execution import (
    CompilerIdentity,
    EvaluationExecutionSpec,
    EvaluatorIdentity,
    PythonModuleEntrypoint,
    ResolvedExecutionPlan,
    ResourceRequirements,
    TrainingExecutionSpec,
)
from xaytune.core.execution_controls import RESILIENCE, ResilienceRequest
from xaytune.core.ids import ArtifactId
from xaytune.core.immutable import FrozenDict, thaw
from xaytune.core.refs import ArtifactRef
from xaytune.resilience.provider import (
    ExecutionCapabilities,
    ResilienceContractError,
    ResilienceProvider,
    ResilienceProviderConfigurationError,
    UnsupportedResilienceError,
    augment_execution_plan,
    bind_resilience_provider,
)

_DESCRIPTOR = PluginDescriptor(
    api_version="xaytune.plugins/v1alpha1",
    name="fake",
    plugin_version="0.1.0",
    provider="tests",
    xaytune_version=__version__,
)
HOSTED = ExecutionCapabilities(
    runtime=hosting(CapabilityDocument(), SCHEMA), compiler=hosting(CapabilityDocument(), SCHEMA)
)


def _plan(workers: int = 2, **options: Any) -> ResolvedExecutionPlan:
    return ResolvedExecutionPlan(
        spec=TrainingExecutionSpec(
            compiler=CompilerIdentity(name="fake", version="0.1.0", descriptor=_DESCRIPTOR),
            candidate_fingerprint="sha256:" + "0" * 64,
            entrypoint=PythonModuleEntrypoint(module="my.worker"),
            config=FrozenDict({"optimization": {"learning_rate": 1e-4, "micro_batch_size": 4}}),
            resources=ResourceRequirements(workers=workers),
        ),
        runtime="ray-train",
        target=RuntimeOperationTarget(kind="training-attempt", id="ra_1"),
        runtime_options=FrozenDict(options),
    )


def _provider(replicas: int = 2, installed: Any = ENGINE) -> ResilienceProvider:
    return bind_resilience_provider(resilience_spec(replicas), providers(installed))


# ---- the request ---------------------------------------------------------------


def test_the_same_plan_policy_and_engine_make_the_identical_serializable_plan() -> None:
    first = augment_execution_plan(_provider(), _plan(), capabilities=HOSTED)
    again = augment_execution_plan(_provider(), _plan(), capabilities=HOSTED)

    assert first == again
    assert first.model_dump_json() == again.model_dump_json()
    assert ResolvedExecutionPlan.model_validate_json(first.model_dump_json()) == first
    assert first.request_digest("submit") == again.request_digest("submit")


def test_the_request_records_the_provider_the_exact_engine_and_the_bound_spec() -> None:
    provider = _provider()
    request = ResilienceRequest.model_validate(
        thaw(
            augment_execution_plan(provider, _plan(), capabilities=HOSTED).runtime_options[
                RESILIENCE
            ]
        )
    )

    assert request.provider == "test-replica-recovery"
    assert request.provider_version == "1.0.0"
    assert request.engine == FrozenDict(ENGINE)
    assert request.spec_fingerprint == resilience_spec_fingerprint(provider.spec)
    assert request.delegate == ("per-step-worker-recovery",)
    assert request.request_schema == SCHEMA
    assert request.parameters == FrozenDict({"replicas": 2})


def test_augmenting_changes_nothing_but_its_one_request() -> None:
    plan = _plan(checkpoint_restore={"id": "ck"})
    augmented = augment_execution_plan(_provider(), plan, capabilities=HOSTED)

    assert augmented.spec == plan.spec
    assert augmented.target == plan.target
    assert augmented.resolved_capabilities == plan.resolved_capabilities
    assert set(augmented.runtime_options) == {"checkpoint_restore", RESILIENCE}
    assert (
        augmented.runtime_options["checkpoint_restore"]
        == plan.runtime_options["checkpoint_restore"]
    )


def test_the_request_is_part_of_the_request_digest_and_the_execution_identity() -> None:
    plan = _plan()
    augmented = augment_execution_plan(_provider(), plan, capabilities=HOSTED)
    other_engine = augment_execution_plan(
        _provider(installed={"replica-engine": "2.3.5"}), plan, capabilities=HOSTED
    )

    assert augmented.request_digest("submit") != plan.request_digest("submit")
    assert training_execution_fingerprint(augmented) != training_execution_fingerprint(plan)
    assert other_engine.request_digest("submit") != augmented.request_digest("submit")
    assert training_execution_fingerprint(other_engine) != training_execution_fingerprint(augmented)


# ---- refusals, before anything is submitted ----------------------------------------


def _refusal(plan: ResolvedExecutionPlan, capabilities: ExecutionCapabilities, **kw: Any) -> str:
    with pytest.raises(UnsupportedResilienceError) as refused:
        augment_execution_plan(_provider(**kw), plan, capabilities=capabilities)
    return str(refused.value)


def test_a_runtime_that_does_not_host_the_request_is_refused() -> None:
    capabilities = HOSTED.model_copy(update={"runtime": CapabilityDocument()})
    assert "runtime does not declare hosting" in _refusal(_plan(), capabilities)


def test_a_compiler_whose_worker_does_not_honour_the_request_is_refused() -> None:
    capabilities = HOSTED.model_copy(update={"compiler": hosting(CapabilityDocument(), "other/v1")})
    assert "compiler does not declare" in _refusal(_plan(), capabilities)


def test_a_runtime_that_recovers_workers_itself_already_owns_the_responsibility() -> None:
    capabilities = HOSTED.model_copy(update={"runtime": per_step_runtime(HOSTED.runtime)})
    assert "another mechanism already owns" in _refusal(_plan(), capabilities)


def test_a_plan_that_already_carries_a_request_is_refused() -> None:
    once = augment_execution_plan(_provider(), _plan(), capabilities=HOSTED)
    assert "already carries a resilience request" in _refusal(once, HOSTED)


def test_an_evaluation_plan_is_never_augmented() -> None:
    plan = ResolvedExecutionPlan(
        spec=EvaluationExecutionSpec(
            evaluator=EvaluatorIdentity(name="fake", version="0.1.0", descriptor=_DESCRIPTOR),
            evaluation_fingerprint="sha256:" + "1" * 64,
            subject=ArtifactRef(id=ArtifactId.generate(), kind="model", uri="/models/1"),
            entrypoint=PythonModuleEntrypoint(module="my.evaluator"),
        ),
        runtime="ray-train",
        target=RuntimeOperationTarget(kind="evaluation-attempt", id="ea_1"),
    )
    assert "only a training plan" in _refusal(plan, HOSTED)


def test_the_provider_refuses_a_topology_it_cannot_recover() -> None:
    assert "3 replicas need as many workers, not 2" in _refusal(
        _plan(workers=2), HOSTED, replicas=3
    )


# ---- a provider that breaks the contract ---------------------------------------------


class _Misbehaving(ReplicaRecoveryProvider):
    def __init__(self, spec: ResilienceSpec, change: Any) -> None:
        super().__init__(spec, ENGINE)
        self.change = change

    def augment(self, plan: Any, **kw: Any) -> Any:
        return self.change(self, super().augment(plan, **kw))


def _request_with(provider: Any, plan: Any, **fields: Any) -> Any:
    request = {**provider.request(), **fields}
    return plan.model_copy(
        update={"runtime_options": FrozenDict({**thaw(plan.runtime_options), RESILIENCE: request})}
    )


_COUNTER = iter(range(10_000))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda self, plan: plan.model_copy(
                update={"spec": plan.spec.model_copy(update={"config": FrozenDict({"lr": 1.0})})}
            ),
            "change nothing else",
        ),
        (
            lambda self, plan: plan.model_copy(
                update={"target": RuntimeOperationTarget(kind="training-attempt", id="ra_2")}
            ),
            "change nothing else",
        ),
        (
            lambda self, plan: plan.model_copy(
                update={
                    "runtime_options": FrozenDict({**thaw(plan.runtime_options), "extra": True})
                }
            ),
            "change nothing else",
        ),
        (
            lambda self, plan: plan.model_copy(update={"runtime_options": FrozenDict()}),
            "change nothing else",
        ),
        (
            lambda self, plan: _request_with(self, plan, engine={"replica-engine": "9.9.9"}),
            "engine",
        ),
        (
            lambda self, plan: _request_with(self, plan, provider_version="0.9.0"),
            "provider_version",
        ),
        (
            lambda self, plan: _request_with(self, plan, spec_fingerprint="sha256:x"),
            "spec_fingerprint",
        ),
        (lambda self, plan: _request_with(self, plan, request_schema="other/v1"), "request_schema"),
        (
            lambda self, plan: _request_with(
                self, plan, parameters={"replicas": 2, "at": next(_COUNTER)}
            ),
            "differently twice",
        ),
    ],
)
def test_a_provider_that_changes_anything_else_breaks_the_contract(
    change: Any, message: str
) -> None:
    with pytest.raises(ResilienceContractError, match=message):
        provider = bind_resilience_provider(
            resilience_spec(2), {KIND: lambda spec: _Misbehaving(spec, change)}
        )
        augment_execution_plan(provider, _plan(), capabilities=HOSTED)


def test_a_provider_that_was_never_bound_is_refused() -> None:
    unbound = ReplicaRecoveryProvider(resilience_spec(2), ENGINE)
    assert unbound.spec.implementation is None
    with pytest.raises(ResilienceContractError, match="another implementation"):
        augment_execution_plan(unbound, _plan(), capabilities=HOSTED)


# ---- binding ---------------------------------------------------------------------


def test_binding_records_the_provider_version_and_the_installed_engine() -> None:
    bound = _provider().spec
    assert (bound.kind, bound.version, bound.engine) == (KIND, "1.0.0", FrozenDict(ENGINE))
    assert bound.implementation == ResilienceImplementation(
        provider="tests",
        name="test-replica-recovery",
        api_version="xaytune.plugins/v1alpha1",
        plugin_version="1.0.0",
        request_schema=SCHEMA,
    )
    assert bound.policy == POLICY
    # Bound from a recorded spec, it binds to itself.
    assert bind_resilience_provider(bound, providers()).spec == bound


@pytest.mark.parametrize(
    ("spec", "registry", "message"),
    [
        (ResilienceSpec(kind="nope", policy=POLICY), providers(), "no resilience provider"),
        (resilience_spec(), providers(installed=None), "not installed"),
        (
            ResilienceSpec(kind=KIND, config=FrozenDict({"replicas": "x"}), policy=POLICY),
            providers(),
            "replicas",
        ),
    ],
)
def test_a_spec_that_cannot_be_bound_is_refused(
    spec: ResilienceSpec, registry: Any, message: str
) -> None:
    with pytest.raises(ResilienceProviderConfigurationError, match=message):
        bind_resilience_provider(spec, registry)


def test_a_provider_that_binds_another_spec_is_refused() -> None:
    class Rewrites(ReplicaRecoveryProvider):
        def __init__(self, spec: ResilienceSpec) -> None:
            super().__init__(spec, ENGINE)
            self.spec = self.spec.model_copy(update={"config": FrozenDict({"replicas": 9})})

    with pytest.raises(ResilienceProviderConfigurationError, match="another kind, configuration"):
        bind_resilience_provider(resilience_spec(), {KIND: Rewrites})


def test_a_policy_is_canonical_and_delegates_something() -> None:
    with pytest.raises(ValidationError):
        ResiliencePolicy(delegate=())
    with pytest.raises(ValidationError):
        ResiliencePolicy(delegate=("per-step-worker-recovery", "per-step-worker-recovery"))
    with pytest.raises(ValidationError):
        ResiliencePolicy(delegate=("restart-the-attempt",))  # type: ignore[arg-type]


def test_an_unbound_spec_has_no_identity() -> None:
    with pytest.raises(ValueError, match="only a bound"):
        resilience_spec_fingerprint(resilience_spec())


class _Impostor(ReplicaRecoveryProvider):
    """Same kind, version and engine as the real provider -- another implementation."""

    descriptor = ReplicaRecoveryProvider.descriptor.model_copy(update={"provider": "elsewhere"})


class _OtherSchema(ReplicaRecoveryProvider):
    request_schema = "xaytune.test-replica-recovery/v2alpha1"


@pytest.mark.parametrize("substitute", [_Impostor, _OtherSchema])
def test_another_implementation_under_the_same_kind_version_and_engine_binds_differently(
    substitute: type[ReplicaRecoveryProvider],
) -> None:
    recorded = _provider().spec
    rebound = bind_resilience_provider(recorded, {KIND: lambda spec: substitute(spec, ENGINE)}).spec
    assert (rebound.kind, rebound.version, rebound.engine) == (
        recorded.kind,
        recorded.version,
        recorded.engine,
    )
    assert rebound != recorded
    assert resilience_spec_fingerprint(rebound) != resilience_spec_fingerprint(recorded)


def test_a_factory_cannot_misstate_its_implementation() -> None:
    class Misstates(ReplicaRecoveryProvider):
        def __init__(self, spec: ResilienceSpec) -> None:
            super().__init__(spec, ENGINE)
            self.spec = self.spec.model_copy(
                update={"implementation": _provider().spec.implementation}
            )

        descriptor = _Impostor.descriptor

    with pytest.raises(ResilienceProviderConfigurationError, match="another implementation"):
        bind_resilience_provider(resilience_spec(), {KIND: Misstates})
