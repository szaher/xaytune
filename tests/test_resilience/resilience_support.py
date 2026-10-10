"""A resilience provider with no third-party engine, for the generic contract (PR-035).

``ReplicaRecoveryProvider`` stands in for an engine like TorchFT: it binds the
"installed" engine release it is given, refuses a plan with fewer workers than
its configured replicas, and adds one canonical request. The capability
helpers declare -- or do not -- that a runtime hosts its request schema and a
compiler's worker honours it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import ValidationError

from tests.test_experiment.adaptive_fixtures import AdaptiveRuntime, LoRACompiler
from xaytune import __version__
from xaytune.core.capabilities import (
    PLUGIN_API_VERSIONS,
    CapabilityDocument,
    PluginDescriptor,
    ResilienceCapabilities,
)
from xaytune.core.domain.resilience import (
    ResiliencePolicy,
    ResilienceSpec,
    resilience_spec_fingerprint,
)
from xaytune.core.execution import ResolvedExecutionPlan
from xaytune.core.execution_controls import RESILIENCE, ResilienceRequest
from xaytune.core.immutable import FrozenDict, FrozenDomainModel, thaw
from xaytune.resilience.provider import (
    RESILIENCE_REQUESTS,
    ExecutionCapabilities,
    ResilienceProviderConfigurationError,
    UnsupportedResilienceError,
)

KIND = "replica-recovery"
SCHEMA = "xaytune.test-replica-recovery/v1alpha1"
ENGINE = {"replica-engine": "2.3.4"}
POLICY = ResiliencePolicy(delegate=("per-step-worker-recovery",))


class ReplicaRecoveryConfig(FrozenDomainModel):
    replicas: int


class ReplicaRecoveryProvider:
    """Recovers a failed replica in-run -- in name only: it adds a request, nothing else."""

    descriptor = PluginDescriptor(
        api_version=PLUGIN_API_VERSIONS[0],
        name="test-replica-recovery",
        plugin_version="1.0.0",
        provider="tests",
        xaytune_version=__version__,
    )
    request_schema = SCHEMA

    def __init__(self, spec: ResilienceSpec, installed: Mapping[str, str] | None) -> None:
        if installed is None:
            raise ResilienceProviderConfigurationError(
                spec.kind, ("replica-engine is not installed",)
            )
        try:
            self.config = ReplicaRecoveryConfig.model_validate(thaw(spec.config))
        except ValidationError as error:
            raise ResilienceProviderConfigurationError(spec.kind, (str(error),)) from error
        self.spec = spec.model_copy(
            update={
                "version": self.descriptor.plugin_version,
                "engine": FrozenDict(dict(installed)),
            }
        )

    def augment(
        self,
        plan: ResolvedExecutionPlan,
        *,
        capabilities: ExecutionCapabilities,
        policy: ResiliencePolicy,
    ) -> ResolvedExecutionPlan:
        workers = plan.spec.resources.workers or 1
        if workers < self.config.replicas:
            raise UnsupportedResilienceError(
                self.descriptor.name,
                (f"{self.config.replicas} replicas need as many workers, not {workers}",),
            )
        return plan.model_copy(
            update={
                "runtime_options": FrozenDict(
                    {**thaw(plan.runtime_options), RESILIENCE: self.request()}
                )
            }
        )

    def request(self) -> dict[str, Any]:
        assert self.spec.engine is not None
        return ResilienceRequest(
            provider=self.descriptor.name,
            provider_version=self.descriptor.plugin_version,
            engine=self.spec.engine,
            spec_fingerprint=resilience_spec_fingerprint(self.spec),
            delegate=self.spec.policy.delegate,
            request_schema=SCHEMA,
            parameters=FrozenDict({"replicas": self.config.replicas}),
        ).model_dump(mode="json")


def providers(
    installed: Mapping[str, str] | None = ENGINE,
) -> dict[str, Callable[[ResilienceSpec], ReplicaRecoveryProvider]]:
    return {KIND: lambda spec: ReplicaRecoveryProvider(spec, installed)}


def resilience_spec(replicas: int = 1) -> ResilienceSpec:
    return ResilienceSpec(kind=KIND, config=FrozenDict({"replicas": replicas}), policy=POLICY)


def hosting(document: CapabilityDocument, *schemas: str) -> CapabilityDocument:
    """*document*, declaring it handles the resilience request *schemas*."""
    return document.model_copy(
        update={
            "extensions": FrozenDict(
                {**thaw(document.extensions), RESILIENCE_REQUESTS: list(schemas)}
            )
        }
    )


def per_step_runtime(document: CapabilityDocument) -> CapabilityDocument:
    """*document*, recovering workers per step on its own."""
    return document.model_copy(
        update={"resilience": ResilienceCapabilities(per_step=True, provider="built-in")}
    )


class HostingRuntime(AdaptiveRuntime):
    """The scripted adaptive runtime, declaring it hosts the test request schema."""

    def capabilities(self) -> Any:
        return hosting(super().capabilities(), SCHEMA)


class HonouringCompiler(LoRACompiler):
    """The LoRA compiler, declaring its worker honours the test request schema."""

    def capabilities(self) -> Any:
        return hosting(super().capabilities(), SCHEMA)
