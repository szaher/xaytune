"""Versioned worker requests carried in ``ResolvedExecutionPlan.runtime_options``.

The runtime treats both as opaque: it neither reads nor interprets them, and a
runtime that submits the plan unchanged is all they need. The control plane
writes them from durable records, so a plan rebuilt after a restart carries the
same request and digest; the managed worker knows how to honour them.

``managed_numerical_recovery``  an execution control, not candidate intent: a
    managed worker fails the attempt when its loss is nonfinite, after reporting
    the observation, so governed numerical recovery can begin. Part of the
    execution identity.

``training_interventions``  directives to apply on restore, each under its
    pre-assigned application id. Attempt-specific scientific history, so they
    are excluded from the execution fingerprint, like the restore target.

``resilience``  a resilience provider's request (PR-035): which responsibility
    is delegated, to which provider and engine release, with the provider's own
    parameters. Added by ``augment_execution_plan`` and by nothing else, from
    the experiment's recorded spec, so it is part of the execution identity and
    the request digest. A runtime carries it only if it declares hosting its
    ``request_schema``; the worker honours it only if its compiler declares so.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from xaytune.core.domain.intervention import InterventionDirective
from xaytune.core.domain.resilience import ResilienceResponsibility
from xaytune.core.immutable import FrozenDict, FrozenDomainModel

__all__ = [
    "MANAGED_NUMERICAL_RECOVERY",
    "RESILIENCE",
    "TRAINING_INTERVENTIONS",
    "ManagedNumericalRecoveryControl",
    "ResilienceRequest",
    "TrainingInterventionDirectives",
]

MANAGED_NUMERICAL_RECOVERY = "managed_numerical_recovery"
TRAINING_INTERVENTIONS = "training_interventions"
RESILIENCE = "resilience"


class ManagedNumericalRecoveryControl(FrozenDomainModel):
    schema_version: Literal["xaytune.managed-numerical-recovery/v1alpha1"] = (
        "xaytune.managed-numerical-recovery/v1alpha1"
    )
    fail_on_nonfinite_loss: Literal[True] = True


class TrainingInterventionDirectives(FrozenDomainModel):
    schema_version: Literal["xaytune.training-interventions/v1alpha1"] = (
        "xaytune.training-interventions/v1alpha1"
    )
    directives: tuple[InterventionDirective, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _ordered(self) -> TrainingInterventionDirectives:
        attempts = {directive.attempt_id for directive in self.directives}
        if len(attempts) != 1 or [d.ordinal for d in self.directives] != list(
            range(len(self.directives))
        ):
            raise ValueError("directives belong to one attempt, in contiguous ordinal order")
        return self


class ResilienceRequest(FrozenDomainModel):
    """What a resilience provider asks the worker group to do, versioned (PR-035).

    The envelope is Xaytune's; ``parameters`` are the provider's, under its
    ``request_schema``. A runtime never interprets either: it carries the
    request to the workers it starts, or refuses a schema it does not host.
    """

    schema_version: Literal["xaytune.resilience-request/v1alpha1"] = (
        "xaytune.resilience-request/v1alpha1"
    )
    provider: str = Field(min_length=1)
    provider_version: str = Field(min_length=1)
    engine: FrozenDict
    """The exact releases of the engine the request was made for."""
    spec_fingerprint: str = Field(min_length=1)
    delegate: tuple[ResilienceResponsibility, ...] = Field(min_length=1)
    request_schema: str = Field(min_length=1)
    parameters: FrozenDict = Field(default_factory=FrozenDict)
