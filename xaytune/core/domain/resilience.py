"""Which recovery a resilience provider is trusted with, and which provider (PR-035).

```text
TrainingExecutionSpec
        ↓  resolve_training_attempt        (Xaytune: attempt lineage)
ResolvedExecutionPlan
        ↓  ResilienceProvider.augment      (adds one versioned request, nothing else)
ResolvedExecutionPlan + runtime_options["resilience"]
        ↓  RuntimeBackend.submit_or_get    (carries it; the worker honours it)
```

A provider recovers *inside* a running attempt -- a worker of a replicated
group fails and the group carries on. Everything above that stays Xaytune's:
incidents, recovery policy and episodes, checkpoint eligibility, whether to
retry or resume, interventions, attempt and run state, budgets, lineage and
evaluation after recovery. A provider's recovery is telemetry, never a new
attempt, and never a reason to call a failed attempt successful.

So the policy names **responsibilities delegated**, not provider mechanics.
v1 has one: ``per-step-worker-recovery``. A responsibility is owned by exactly
one mechanism; a provider asked to take one that the runtime already performs
itself is refused rather than layered.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from xaytune.core.fingerprint import fingerprint
from xaytune.core.immutable import FrozenDict, FrozenDomainModel

__all__ = [
    "RESILIENCE_RESPONSIBILITIES",
    "ResilienceImplementation",
    "ResiliencePolicy",
    "ResilienceResponsibility",
    "ResilienceSpec",
    "resilience_spec_fingerprint",
]

ResilienceResponsibility = Literal["per-step-worker-recovery"]
"""A recovery responsibility Xaytune can delegate below the controller.

``per-step-worker-recovery``: a worker of the attempt's group fails, and the
surviving workers continue training at step granularity, without the attempt
ending. Restarting the attempt, resuming from a checkpoint, and every other
recovery decision are not delegable: they are the controller's."""

RESILIENCE_RESPONSIBILITIES: tuple[ResilienceResponsibility, ...] = ("per-step-worker-recovery",)


class ResiliencePolicy(FrozenDomainModel):
    """What Xaytune delegates to the experiment's resilience provider.

    Explicit and canonical: the responsibilities are listed once each, in
    order, so the same policy has one fingerprint. Nothing is delegated by
    default -- an experiment without a policy keeps every recovery decision.
    """

    schema_version: Literal["xaytune.resilience-policy/v1alpha1"] = (
        "xaytune.resilience-policy/v1alpha1"
    )
    delegate: tuple[ResilienceResponsibility, ...] = Field(min_length=1)

    @field_validator("delegate")
    @classmethod
    def _canonical(
        cls, value: tuple[ResilienceResponsibility, ...]
    ) -> tuple[ResilienceResponsibility, ...]:
        if list(value) != sorted(set(value)):
            raise ValueError(
                "delegated responsibilities are listed once each, in sorted order, so one "
                "policy has one identity"
            )
        return value


class ResilienceImplementation(FrozenDomainModel):
    """Which implementation a resilience spec was bound to: its descriptor and request schema.

    ``kind`` is only a registry key, and a version only a number: two
    implementations registered under the same kind can share both. The
    descriptor's publisher, name, plugin API and version, and the request
    schema it emits, say which one it was -- so a host that registers another
    one under the same kind is refused before it builds a single plan.
    """

    provider: str = Field(min_length=1)
    name: str = Field(min_length=1)
    api_version: str = Field(min_length=1)
    plugin_version: str = Field(min_length=1)
    request_schema: str = Field(min_length=1)


class ResilienceSpec(FrozenDomainModel):
    """Which ``ResilienceProvider`` the experiment delegates to, and what (ADR-016).

    Like a ``PlannerSpec``: ``kind`` names the provider in the host's
    registry; ``version``, ``engine`` and ``implementation`` are ``None`` as a
    request and set when bound -- the provider's ``plugin_version``, the exact
    releases of the third-party engine it drives, and the implementation's
    full identity -- so the record says which implementation recovers, and a
    host with another one installed refuses to continue rather than silently
    changing what a rebuilt or a new attempt's request asks for.
    ``config`` is the provider's own, canonical JSON, validated when bound;
    credentials never belong in it.
    """

    kind: str = Field(min_length=1)
    version: str | None = None
    engine: FrozenDict | None = None
    implementation: ResilienceImplementation | None = None
    config: FrozenDict = Field(default_factory=FrozenDict)
    policy: ResiliencePolicy


def resilience_spec_fingerprint(spec: ResilienceSpec) -> str:
    """Identity of a *bound* resilience spec: implementation, version, engine, config, policy."""
    if spec.version is None or spec.engine is None or spec.implementation is None:
        raise ValueError("only a bound resilience spec has an identity")
    return fingerprint({"kind": "resilience-spec/v1", "spec": spec})
