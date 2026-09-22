"""What a plugin can do, and what a workload needs (ADR-008).

**Booleans are not enough.** ``distributed=True`` says nothing a resolver can
act on: it does not say which strategies, between how many workers, or whether
membership can change without a restart. A capability is therefore a
parameterized, versioned document, and a requirement names a specific value
rather than a flag.

The asymmetry matters. A compiler declares what it *can* emit; a runtime
declares what it *can* execute; a spec declares what it *needs*. The resolver
compares them and either produces a plan or says precisely what is missing —
which is only possible if all three speak in the same parameters.
"""

from __future__ import annotations

from pydantic import Field

from xaytune.core.errors import IncompatiblePluginError
from xaytune.core.immutable import FrozenDict, FrozenDomainModel

__all__ = [
    "AgentRolloutCapabilities",
    "AlgorithmCapabilities",
    "CapabilityDocument",
    "CapabilityRequirements",
    "CheckpointCapabilities",
    "DistributedCapabilities",
    "ElasticityCapabilities",
    "PLUGIN_API_VERSIONS",
    "PluginDescriptor",
    "PrecisionCapabilities",
    "require_supported_plugin",
    "ResilienceCapabilities",
]


class PrecisionCapabilities(FrozenDomainModel):
    """Numeric precisions a plugin can run."""

    supported: tuple[str, ...] = Field(default_factory=tuple)


class DistributedCapabilities(FrozenDomainModel):
    """Distribution strategies, and the worker range they run over."""

    strategies: tuple[str, ...] = Field(default_factory=tuple)
    min_workers: int | None = None
    max_workers: int | None = None


class CheckpointCapabilities(FrozenDomainModel):
    """What a plugin can do with checkpoints.

    ``atomic_commit`` is the load-bearing one. A checkpoint that can be
    observed half-written is not a resume point, so a runtime that cannot
    commit atomically constrains what recovery may claim (ADR-012).
    """

    formats: tuple[str, ...] = Field(default_factory=tuple)
    asynchronous: bool | None = None
    reshardable: bool | None = None
    atomic_commit: bool | None = None


class ElasticityCapabilities(FrozenDomainModel):
    """Whether the worker count can change, and at what cost."""

    supported: bool | None = None
    min_workers: int | None = None
    max_workers: int | None = None
    membership_change: str | None = None


class ResilienceCapabilities(FrozenDomainModel):
    """Recovery a plugin provides on its own, below the controller."""

    per_step: bool | None = None
    provider: str | None = None
    provider_version: str | None = None

    supports_event_replay: bool | None = None
    """Whether ``watch()`` can resume from a cursor (ADR-014 §4).

    A runtime that cannot replay is supported, but the controller treats every
    reconnect as a gap and records that observability was degraded -- rather
    than assuming nothing happened while it was away.
    """

    reports_completed_operations: bool | None = None
    """Whether ``lookup_operation()`` can answer "it already finished".

    Without it, an operation in ``SENT`` after a restart cannot be told apart
    from one never received, and the controller must escalate instead of
    resubmitting (ADR-013 §3). Declared, because "I cannot tell" is a valid
    answer and guessing is not.
    """


class AgentRolloutCapabilities(FrozenDomainModel):
    """Agent-environment rollout behaviour."""

    stateful: bool | None = None
    asynchronous: bool | None = None


class AlgorithmCapabilities(FrozenDomainModel):
    """Which training programs a compiler can emit."""

    supported: tuple[str, ...] = Field(default_factory=tuple)


class CapabilityDocument(FrozenDomainModel):
    """What one plugin can do.

    Every section is optional, and ``None`` means *not declared* rather than
    *not supported*. A resolver treats an undeclared capability as unknown and
    refuses to depend on it, which is safer than reading silence as a promise.
    """

    schema_version: str = "xaytune.capabilities/v1alpha1"

    precision: PrecisionCapabilities | None = None
    distributed: DistributedCapabilities | None = None
    checkpoint: CheckpointCapabilities | None = None
    elasticity: ElasticityCapabilities | None = None
    resilience: ResilienceCapabilities | None = None
    agent_rollout: AgentRolloutCapabilities | None = None
    algorithms: AlgorithmCapabilities | None = None

    extensions: FrozenDict = Field(default_factory=FrozenDict)


class CapabilityRequirements(FrozenDomainModel):
    """What a workload needs in order to run.

    ``None`` means *no requirement*, so an unset field never constrains a
    resolver. Requirements name values rather than set flags, because "needs
    fsdp" and "needs distribution" are different questions.
    """

    precision: str | None = None
    distributed_strategy: str | None = None
    min_workers: int | None = None
    checkpoint_resharding: bool | None = None
    per_step_recovery: bool | None = None
    stateful_rollouts: bool | None = None

    extensions: FrozenDict = Field(default_factory=FrozenDict)


class PluginDescriptor(FrozenDomainModel):
    """Who provides a plugin, and against which contracts.

    ``capabilities_schema`` is separate from ``plugin_version`` on purpose: a
    plugin can be upgraded without changing the vocabulary it speaks, and the
    vocabulary can change without the plugin. Conflating them would make every
    plugin release look like a contract change (ADR-008).
    """

    api_version: str
    name: str
    plugin_version: str
    provider: str

    xaytune_version: str
    capabilities_schema: str = "xaytune.capabilities/v1alpha1"

    metadata: FrozenDict = Field(default_factory=FrozenDict)


PLUGIN_API_VERSIONS: tuple[str, ...] = ("xaytune.plugins/v1alpha1",)
"""Every plugin API version this build implements (ADR-008).

An explicit list rather than a major-version comparison, because during alpha
the revisions are not compatible with each other and pretending otherwise is
the failure this is meant to prevent -- ``v1alpha1`` and ``v1alpha2`` differ in
exactly the way that makes a plugin misread its host. Adding a version here is
a deliberate act by someone who checked. Once the API is stable this becomes a
major-version rule, and the list is the thing that changes.
"""


def require_supported_plugin(descriptor: PluginDescriptor) -> None:
    """Refuse a plugin speaking an API this build does not implement.

    Fails closed, per ADR-008. Called at the boundary where a plugin's output
    is first trusted, not at import: a descriptor that is merely constructed
    has not yet been believed, and refusing it there would make a version check
    impossible to test without a plugin.

    Raises:
        IncompatiblePluginError: Naming the plugin, what it declared, and what
            this build supports -- because "incompatible plugin" alone leaves
            the reader to work out which of the two to change.
    """
    if descriptor.api_version not in PLUGIN_API_VERSIONS:
        raise IncompatiblePluginError(descriptor.name, descriptor.api_version, PLUGIN_API_VERSIONS)
