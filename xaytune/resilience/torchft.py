"""TorchFT as a resilience provider: a versioned request, never a runtime (PR-035).

```text
ResilienceSpec(kind="torchft", config={...}, policy=delegate per-step-worker-recovery)
        ↓ bound: torchft's exact release read from its distribution metadata
TorchFTResilienceProvider.augment(plan)
        ↓ runtime_options["resilience"] = ResilienceRequest(request_schema=
        ↓     "xaytune.torchft/v1alpha1", parameters={lighthouse, replica groups, ...})
a runtime that hosts the schema, a worker that honours it
```

**What TorchFT owns**: in-process, worker-group recovery mechanics -- per-step
quorum, reconfiguring the replicas' process group, live recovery of a
rejoining replica from a healthy peer -- and its own synchronization and
configuration. **What Xaytune owns**: everything else -- incidents, recovery
policy, episodes and plans, checkpoint eligibility, whether to retry or
resume, interventions, attempt and run state, budgets, lineage, evaluation
after recovery. A TorchFT recovery is telemetry inside one attempt. It never
becomes a new attempt, and never makes a failed one succeed.

This module never imports ``torchft``. The engine release is read from the
installed distribution's metadata, so binding a recorded experiment where
another release is installed is refused before anything is built. Exactly
one stable release is supported; ``torchft-nightly`` is not. The ``torchft``
extra installs it on Linux x86_64, the only platform it publishes wheels for;
elsewhere the provider refuses to bind, as not installed.

**Narrow on purpose.** v1 translates one responsibility,
``per-step-worker-recovery``, for a plan that is a replicated worker group:
at least two replica groups of ``replica_group_size`` workers each, all
starting fresh. It is refused for

- a plan of one worker, a worker count the replica groups do not divide, or
  fewer replica groups than ``min_replica_size`` (or than two);
- checkpoints taken mid-accumulation or committed non-atomically, or a runtime
  that does not commit atomically -- a step TorchFT discards must never be a
  resume point;
- a checkpoint restore, intervention directives or managed numerical recovery:
  each addresses one managed worker, and restoring a replicated group is
  PR-036's to prove;
- a runtime or compiler that does not declare the request schema -- which, in
  PR-035, is every built-in one. Hosting TorchFT in Ray Train (launch,
  replica replacement and rejoin, destructive tests) is PR-036.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import metadata
from urllib.parse import urlsplit

from pydantic import Field, ValidationError, field_validator

from xaytune import __version__
from xaytune.core.capabilities import PLUGIN_API_VERSIONS, PluginDescriptor
from xaytune.core.domain.resilience import (
    ResiliencePolicy,
    ResilienceSpec,
    resilience_spec_fingerprint,
)
from xaytune.core.execution import ResolvedExecutionPlan, TrainingExecutionSpec
from xaytune.core.execution_controls import (
    MANAGED_NUMERICAL_RECOVERY,
    RESILIENCE,
    TRAINING_INTERVENTIONS,
    ResilienceRequest,
)
from xaytune.core.immutable import FrozenDict, FrozenDomainModel, thaw
from xaytune.resilience.provider import (
    ExecutionCapabilities,
    ResilienceProviderConfigurationError,
    UnsupportedResilienceError,
)

__all__ = [
    "MANAGER_PARAMETERS",
    "SUPPORTED_TORCHFT",
    "TORCHFT_REQUEST_SCHEMA",
    "TorchFTResilienceConfig",
    "TorchFTResilienceProvider",
    "torchft_resilience_provider",
]

TORCHFT_REQUEST_SCHEMA = "xaytune.torchft/v1alpha1"
"""The request a runtime hosts and a worker honours to run under TorchFT."""

SUPPORTED_TORCHFT = "0.2.0"
"""The one TorchFT release this provider translates for. Kept equal to the extra's pin."""

MANAGER_PARAMETERS: Mapping[str, str] = {
    "lighthouse_address": "lighthouse_addr",
    "min_replica_size": "min_replica_size",
    "quorum_timeout_seconds": "quorum_timeout",
    "timeout_seconds": "timeout",
    "use_async_quorum": "use_async_quorum",
}
"""Which ``torchft.Manager`` argument each request parameter is, in 0.2.0.

The request carries plain JSON -- seconds, not ``timedelta`` -- and a worker
that honours the schema passes each to the Manager under this name. Checked
against the installed release's signature in CI (the ``torchft`` job).
``replica_group_size`` and ``replica_groups`` are placement: they say how a
host forms the replica groups, and are no Manager argument."""

_DISTRIBUTION = "torchft"
_NIGHTLY = "torchft-nightly"


class TorchFTResilienceConfig(FrozenDomainModel):
    """TorchFT's side of the delegation, validated when the spec is bound.

    The lighthouse is operated outside Xaytune -- the provider launches
    nothing -- and is named by an address that carries no credentials. Every
    TorchFT setting is stated, never defaulted: the record says what was
    asked for rather than what one release happened to default to.
    """

    lighthouse_address: str = Field(min_length=1)
    min_replica_size: int = Field(ge=1, strict=True)
    replica_group_size: int = Field(default=1, ge=1, strict=True)
    """Workers per replica group: 1 replicates every worker (DDP-like)."""
    quorum_timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    use_async_quorum: bool = Field(strict=True)

    @field_validator("lighthouse_address")
    @classmethod
    def _an_address_without_credentials(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"lighthouse_address must be an http(s) URL, not {value!r}")
        if parts.username is not None or parts.password is not None:
            raise ValueError(
                "lighthouse_address must not carry credentials: the spec is persisted and "
                "fingerprinted"
            )
        if parts.path not in ("", "/") or parts.query or parts.fragment:
            raise ValueError("lighthouse_address names a host and port, nothing more")
        return value


class TorchFTResilienceProvider:
    """Translates a ``per-step-worker-recovery`` delegation into a TorchFT request.

    Register it explicitly -- no resilience provider is built in::

        EmbeddedControllerHost(..., resilience_providers={"torchft": torchft_resilience_provider})
    """

    descriptor = PluginDescriptor(
        api_version=PLUGIN_API_VERSIONS[0],
        name="torchft",
        plugin_version="1.0.0",
        provider="xaytune",
        xaytune_version=__version__,
    )
    request_schema = TORCHFT_REQUEST_SCHEMA

    def __init__(self, spec: ResilienceSpec) -> None:
        reasons = []
        try:
            self.config = TorchFTResilienceConfig.model_validate(thaw(spec.config))
        except ValidationError as error:
            raise ResilienceProviderConfigurationError(
                spec.kind,
                tuple(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()),
            ) from error
        if spec.policy.delegate != ("per-step-worker-recovery",):
            reasons.append(
                f"TorchFT takes per-step-worker-recovery and nothing else, not "
                f"{spec.policy.delegate}"
            )
        engine = self.engine_versions()
        if spec.engine is not None and dict(spec.engine) != dict(engine):
            reasons.append(f"the spec names engine {dict(spec.engine)}, but {engine} is installed")
        if reasons:
            raise ResilienceProviderConfigurationError(spec.kind, tuple(reasons))
        self.spec = spec.model_copy(
            update={"version": self.descriptor.plugin_version, "engine": FrozenDict(engine)}
        )

    @classmethod
    def engine_versions(cls) -> Mapping[str, str]:
        """The installed TorchFT release, from its distribution's metadata -- never imported.

        Raises:
            ResilienceProviderConfigurationError: If TorchFT is not installed,
                is a nightly, or is another release than the one supported.
        """
        kind = cls.descriptor.name
        nightly = _installed(_NIGHTLY)
        if nightly is not None:
            raise ResilienceProviderConfigurationError(
                kind,
                (
                    f"{_NIGHTLY} {nightly} is installed; only the stable torchft "
                    f"{SUPPORTED_TORCHFT} is supported",
                ),
            )
        installed = _installed(_DISTRIBUTION)
        if installed is None:
            raise ResilienceProviderConfigurationError(
                kind,
                (
                    "torchft is not installed: pip install 'xaytune[torchft]' (Linux x86_64 "
                    "only; TorchFT publishes no other wheels)",
                ),
            )
        if installed != SUPPORTED_TORCHFT:
            raise ResilienceProviderConfigurationError(
                kind,
                (
                    f"torchft {installed} is installed; this provider translates for exactly "
                    f"{SUPPORTED_TORCHFT}",
                ),
            )
        return {_DISTRIBUTION: installed}

    def augment(
        self,
        plan: ResolvedExecutionPlan,
        *,
        capabilities: ExecutionCapabilities,
        policy: ResiliencePolicy,
    ) -> ResolvedExecutionPlan:
        """*plan* plus the TorchFT request for *policy*, or why it cannot carry one.

        Raises:
            UnsupportedResilienceError: If the plan's topology, checkpoints or
                worker requests are outside what v1 translates.
        """
        reasons = self._refusals(plan, capabilities, policy)
        if reasons:
            raise UnsupportedResilienceError(self.descriptor.name, reasons)
        workers = plan.spec.resources.workers
        assert workers is not None
        config = self.config
        assert self.spec.engine is not None
        request = ResilienceRequest(
            provider=self.descriptor.name,
            provider_version=self.descriptor.plugin_version,
            engine=self.spec.engine,
            spec_fingerprint=resilience_spec_fingerprint(self.spec),
            delegate=policy.delegate,
            request_schema=TORCHFT_REQUEST_SCHEMA,
            parameters=FrozenDict(
                {
                    "lighthouse_address": config.lighthouse_address,
                    "min_replica_size": config.min_replica_size,
                    "replica_group_size": config.replica_group_size,
                    "replica_groups": workers // config.replica_group_size,
                    "quorum_timeout_seconds": config.quorum_timeout_seconds,
                    "timeout_seconds": config.timeout_seconds,
                    "use_async_quorum": config.use_async_quorum,
                }
            ),
        )
        options = {**thaw(plan.runtime_options), RESILIENCE: request.model_dump(mode="json")}
        return plan.model_copy(update={"runtime_options": FrozenDict(options)})

    def _refusals(
        self,
        plan: ResolvedExecutionPlan,
        capabilities: ExecutionCapabilities,
        policy: ResiliencePolicy,
    ) -> tuple[str, ...]:
        spec = plan.spec
        if not isinstance(spec, TrainingExecutionSpec):
            return ("TorchFT recovers training worker groups only",)
        reasons = []
        if policy != self.spec.policy:
            reasons.append("the policy is not the one this provider was bound under")
        config = self.config
        workers = spec.resources.workers or 1
        groups, remainder = divmod(workers, config.replica_group_size)
        needed = max(2, config.min_replica_size)
        if remainder:
            reasons.append(
                f"{workers} workers do not divide into replica groups of "
                f"{config.replica_group_size}"
            )
        elif groups < needed:
            reasons.append(
                f"{workers} workers make {groups} replica group(s) of "
                f"{config.replica_group_size}; per-step worker recovery needs at least {needed} "
                f"(min_replica_size {config.min_replica_size}, and a healthy peer to recover from)"
            )
        distributed = capabilities.runtime.distributed
        if distributed is not None and distributed.max_workers is not None:
            if workers > distributed.max_workers:
                reasons.append(
                    f"the runtime places at most {distributed.max_workers} workers, not {workers}"
                )
        checkpoint = spec.checkpoint
        if checkpoint.boundary != "optimizer-step" or not checkpoint.require_atomic_commit:
            reasons.append(
                "checkpoints must be taken at optimizer-step boundaries and committed atomically: "
                "a step TorchFT discards must never become a resume point"
            )
        if checkpoint.store_uri is not None:
            runtime_checkpoints = capabilities.runtime.checkpoint
            if runtime_checkpoints is None or runtime_checkpoints.atomic_commit is not True:
                reasons.append(
                    "the plan writes checkpoints, and the runtime does not declare atomic commit"
                )
        single_worker = sorted(
            set(plan.runtime_options)
            & {"checkpoint_restore", TRAINING_INTERVENTIONS, MANAGED_NUMERICAL_RECOVERY}
        )
        if single_worker:
            reasons.append(
                f"{', '.join(repr(option) for option in single_worker)} address one managed "
                f"worker; restoring or steering a replicated TorchFT group is not supported yet"
            )
        return tuple(reasons)


def _installed(distribution: str) -> str | None:
    """*distribution*'s installed release, read from its metadata, or ``None``."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def torchft_resilience_provider(spec: ResilienceSpec) -> TorchFTResilienceProvider:
    """The host's factory for ``ResilienceSpec(kind="torchft", ...)``.

    Raises:
        ResilienceProviderConfigurationError: If the configuration is invalid,
            the policy delegates anything else, or the supported TorchFT
            release is not what is installed.
    """
    return TorchFTResilienceProvider(spec)
