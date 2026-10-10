"""The resilience-provider boundary: one versioned request, added to a plan, checked (PR-035).

```text
TrainingExecutionSpec
        ↓  resolve_training_attempt                     Xaytune: attempt lineage
ResolvedExecutionPlan
        ↓  augment_execution_plan(provider, plan, ...)  the provider adds ONE request
ResolvedExecutionPlan + runtime_options["resilience"]
        ↓  RuntimeBackend.submit_or_get                 carried, not interpreted
```

A :class:`ResilienceProvider` translates the experiment's recorded
:class:`~xaytune.core.domain.resilience.ResiliencePolicy` into a request its
engine understands. It launches nothing, records nothing, and decides nothing
about attempts: it returns the plan it was given plus a
:class:`~xaytune.core.execution_controls.ResilienceRequest` under
``runtime_options["resilience"]``. :func:`augment_execution_plan` is the only
caller, and it holds the provider to exactly that -- whatever else a provider
changed, the candidate, the compiled training settings, the target, is refused
as a broken contract rather than run.

Fail closed, before submission. The request is added only when

- the plan is a training plan with no resilience request already in it, and
  the runtime does not itself own the delegated responsibility;
- the runtime declares it hosts the provider's ``request_schema``
  (``CapabilityDocument.extensions["resilience_requests"]``), so it carries
  the request to the workers it starts rather than dropping it;
- the compiler declares the same, so the worker it emits honours it rather
  than running as if nothing had been asked;
- the provider itself accepts the plan's topology, checkpoints and engine.

The request is part of the plan, so of ``request_digest`` and of the training
execution fingerprint: a rebuilt plan after a restart carries the same request
or is refused as a different one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

from xaytune.core.capabilities import CapabilityDocument, PluginDescriptor, require_supported_plugin
from xaytune.core.domain.resilience import (
    ResilienceImplementation,
    ResiliencePolicy,
    ResilienceSpec,
    resilience_spec_fingerprint,
)
from xaytune.core.errors import XaytuneError
from xaytune.core.execution import ResolvedExecutionPlan, TrainingExecutionSpec
from xaytune.core.execution_controls import RESILIENCE, ResilienceRequest
from xaytune.core.immutable import FrozenDict, FrozenDomainModel, thaw

__all__ = [
    "RESILIENCE_REQUESTS",
    "ExecutionCapabilities",
    "ResilienceContractError",
    "ResilienceProvider",
    "ResilienceProviderConfigurationError",
    "UnsupportedResilienceError",
    "augment_execution_plan",
    "bind_resilience_provider",
    "implementation_of",
]

RESILIENCE_REQUESTS = "resilience_requests"
"""``CapabilityDocument.extensions`` key: the resilience request schemas a plugin handles.

On a runtime: the schemas it carries to the workers it starts, with the
semantics they need from it. On a compiler: the schemas the worker it emits
honours. Undeclared means neither -- the default for every built-in plugin."""


class ResilienceProviderConfigurationError(ValueError):
    """A resilience spec cannot be bound: unknown kind, bad configuration, engine missing."""

    def __init__(self, kind: str, reasons: tuple[str, ...]) -> None:
        self.kind = kind
        self.reasons = reasons
        super().__init__(f"resilience provider {kind!r} cannot be bound: " + "; ".join(reasons))


class UnsupportedResilienceError(ValueError):
    """The plan, runtime and compiler cannot carry what the policy delegates. Nothing ran."""

    def __init__(self, provider: str, reasons: tuple[str, ...]) -> None:
        self.provider = provider
        self.reasons = reasons
        super().__init__(
            f"resilience provider {provider!r} cannot be applied: " + "; ".join(reasons)
        )


class ResilienceContractError(XaytuneError):
    """A provider returned something other than the plan plus its one request."""


class ExecutionCapabilities(FrozenDomainModel):
    """What the plan's runtime and compiler declared, as the provider sees them."""

    runtime: CapabilityDocument
    compiler: CapabilityDocument


@runtime_checkable
class ResilienceProvider(Protocol):
    """Adds a versioned resilience request to a training plan, and does nothing else.

    ``spec`` is the bound spec -- ``version`` and ``engine`` set by the
    provider, ``implementation`` by :func:`bind_resilience_provider` -- which
    the host records. ``request_schema`` names the request it adds, which runtime
    and compiler must both declare.
    """

    descriptor: PluginDescriptor
    spec: ResilienceSpec
    request_schema: str

    def augment(
        self,
        plan: ResolvedExecutionPlan,
        *,
        capabilities: ExecutionCapabilities,
        policy: ResiliencePolicy,
    ) -> ResolvedExecutionPlan:
        """*plan* plus this provider's request; deterministic, side-effect free.

        Raises:
            UnsupportedResilienceError: If the plan cannot carry the request
                truthfully -- topology, checkpoints, engine.
        """
        ...


def implementation_of(provider: ResilienceProvider) -> ResilienceImplementation:
    """*provider*'s full identity: its descriptor and the request schema it emits."""
    descriptor = provider.descriptor
    return ResilienceImplementation(
        provider=descriptor.provider,
        name=descriptor.name,
        api_version=descriptor.api_version,
        plugin_version=descriptor.plugin_version,
        request_schema=provider.request_schema,
    )


def bind_resilience_provider(
    spec: ResilienceSpec,
    providers: Mapping[str, Callable[[ResilienceSpec], ResilienceProvider]],
) -> ResilienceProvider:
    """Resolve *spec*, as a request, to a provider bound under it.

    The provider is given the spec with ``version``, ``engine`` and
    ``implementation`` cleared, and binds the first two from what is
    installed; the implementation -- descriptor and request schema -- is
    bound here, from the provider itself, so no factory can misstate it. The
    caller compares the result with a recorded spec, whole. No provider is
    built in: each is registered explicitly by whoever has its dependencies.

    Raises:
        ResilienceProviderConfigurationError: If no provider has the kind, it
            refuses the spec, or it bound something other than the spec asked.
    """
    factory = providers.get(spec.kind)
    if factory is None:
        raise ResilienceProviderConfigurationError(
            spec.kind,
            (f"no resilience provider of kind {spec.kind!r}; known: {sorted(providers)}",),
        )
    unbound = {"version": None, "engine": None, "implementation": None}
    request = spec.model_copy(update=unbound)
    provider = factory(request)
    bound = provider.spec
    implementation = implementation_of(provider)
    reasons = []
    if bound.model_copy(update=unbound) != request:
        reasons.append("the provider bound another kind, configuration or policy than asked")
    if bound.version != provider.descriptor.plugin_version:
        reasons.append(
            f"the bound version {bound.version!r} is not the provider's "
            f"{provider.descriptor.plugin_version!r}"
        )
    if not bound.engine:
        reasons.append("the provider bound no engine release")
    if bound.implementation not in (None, implementation):
        reasons.append("the provider bound another implementation than its own descriptor")
    if reasons:
        raise ResilienceProviderConfigurationError(spec.kind, tuple(reasons))
    require_supported_plugin(provider.descriptor)
    provider.spec = bound.model_copy(update={"implementation": implementation})
    return provider


def augment_execution_plan(
    provider: ResilienceProvider,
    plan: ResolvedExecutionPlan,
    *,
    capabilities: ExecutionCapabilities,
) -> ResolvedExecutionPlan:
    """*plan* with *provider*'s request for the recorded policy, or a refusal.

    Raises:
        UnsupportedResilienceError: If the plan, runtime or compiler cannot
            carry the request, or the provider refuses it.
        ResilienceContractError: If the provider changed anything but its own
            request, made an inconsistent one, or answered differently twice.
    """
    spec = provider.spec
    name = provider.descriptor.name
    require_supported_plugin(provider.descriptor)
    if spec.implementation != implementation_of(provider):
        raise ResilienceContractError(
            f"resilience provider {name!r} runs under a spec bound to another implementation "
            f"({spec.implementation}); only bind_resilience_provider binds one"
        )
    refusals = _refusals(provider, plan, capabilities)
    if refusals:
        raise UnsupportedResilienceError(name, refusals)

    augmented = provider.augment(plan, capabilities=capabilities, policy=spec.policy)
    again = provider.augment(plan, capabilities=capabilities, policy=spec.policy)
    if augmented != again:
        raise ResilienceContractError(
            f"resilience provider {name!r} augmented the same plan differently twice; a "
            f"request rebuilt after a restart would not be the one recorded"
        )
    if not isinstance(augmented, ResolvedExecutionPlan):
        raise ResilienceContractError(f"resilience provider {name!r} returned no plan")
    options = thaw(augmented.runtime_options)
    raw = options.pop(RESILIENCE, None)
    if raw is None or augmented.model_copy(update={"runtime_options": FrozenDict(options)}) != plan:
        raise ResilienceContractError(
            f"resilience provider {name!r} must add its request under "
            f"runtime_options[{RESILIENCE!r}] and change nothing else in the plan"
        )
    request = ResilienceRequest.model_validate(raw)
    expected = {
        "provider": name,
        "provider_version": provider.descriptor.plugin_version,
        "engine": spec.engine,
        "spec_fingerprint": resilience_spec_fingerprint(spec),
        "delegate": spec.policy.delegate,
        "request_schema": provider.request_schema,
    }
    wrong = sorted(key for key, value in expected.items() if getattr(request, key) != value)
    if wrong or augmented.runtime_options[RESILIENCE] != FrozenDict(
        request.model_dump(mode="json")
    ):
        raise ResilienceContractError(
            f"resilience provider {name!r} made a request that is not canonical for its "
            f"bound spec" + (f" ({', '.join(wrong)} differ)" if wrong else "")
        )
    if ResolvedExecutionPlan.model_validate_json(augmented.model_dump_json()) != augmented:
        raise ResilienceContractError(
            f"resilience provider {name!r} made a request that does not survive serialization"
        )
    return augmented


def _refusals(
    provider: ResilienceProvider,
    plan: ResolvedExecutionPlan,
    capabilities: ExecutionCapabilities,
) -> tuple[str, ...]:
    """Why no provider could add a request to *plan* here -- the generic half."""
    schema = provider.request_schema
    reasons = []
    if not isinstance(plan.spec, TrainingExecutionSpec):
        reasons.append("only a training plan delegates recovery; an evaluation is never augmented")
    if RESILIENCE in plan.runtime_options:
        reasons.append(
            "the plan already carries a resilience request; one responsibility has one owner"
        )
    runtime_resilience = capabilities.runtime.resilience
    if (
        "per-step-worker-recovery" in provider.spec.policy.delegate
        and runtime_resilience is not None
        and runtime_resilience.per_step is True
    ):
        reasons.append(
            f"the runtime recovers workers per step itself ({runtime_resilience.provider!r}); "
            f"another mechanism already owns per-step-worker-recovery"
        )
    if schema not in _declared(capabilities.runtime):
        reasons.append(
            f"the runtime does not declare hosting {schema!r}; it would drop the request or "
            f"run without what it needs"
        )
    if schema not in _declared(capabilities.compiler):
        reasons.append(
            f"the compiler does not declare that its worker honours {schema!r}; the worker "
            f"would run as if nothing had been delegated"
        )
    return tuple(reasons)


def _declared(document: CapabilityDocument) -> tuple[str, ...]:
    declared = document.extensions.get(RESILIENCE_REQUESTS, ())
    return tuple(declared) if isinstance(declared, (tuple, list)) else ()
