"""Governing a proposed action: validation, authorization, approval (PR-023).

```text
ActionSpec + durable state + runtime capabilities
      ↓ PolicyContext                     a snapshot, from the record
      ↓ applicability_problems()          validation: always, built in
      ↓ PolicyEngine.evaluate()           authorization: pure, like DecisionEngine
PolicyProposal                            verdict, reasons, the rules that matched
      ↓ recorded with the Action, in one commit
PolicyDecision                            the proposal and the snapshot it judged
```

Three questions, kept apart (04-state-machines.md §5)::

    validation      is this action applicable, here and now?     always
    authorization   does policy permit it?                        every governed action
    approval        does a human have to say yes?                 when policy says so

**A decision authorizes a snapshot.** ``input_fingerprint`` hashes
:func:`policy_input_identity_v1` of the context the engine saw, and the whole
context is recorded with the decision, because runtime capabilities exist
nowhere else durable. The repository recomputes the context in the
transaction that records it, and refuses to record a decision about a state
that has since changed.

**Approval approves that recorded proposal**; it never re-runs the policy.
**Execution is not here.** When it arrives it must re-check applicability
against the state of that moment, and may carry out a non-cancellation action
only if it is ``VALIDATED`` with an ``ALLOW`` decision, or ``APPROVED`` with a
``REQUIRE_APPROVAL`` decision. A ``VALIDATED`` action with no decision is never
executable.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field, field_validator

from xaytune.core.capabilities import (
    AgentRolloutCapabilities,
    AlgorithmCapabilities,
    CapabilityDocument,
    CheckpointCapabilities,
    DistributedCapabilities,
    ElasticityCapabilities,
    PrecisionCapabilities,
    ResilienceCapabilities,
)
from xaytune.core.clock import utc_now
from xaytune.core.domain.action import Action, ActionStatus, ActionTarget
from xaytune.core.domain.actions import ActionSpec, ChangeWorkerCount, MutationClass
from xaytune.core.domain.actions.builtin import (
    ChangeCheckpointInterval,
    ChangeGradientAccumulation,
    ChangeLearningRate,
    ChangeScheduler,
    ChangeWarmup,
    PromoteCandidate,
    RejectCandidate,
    ResizeMicrobatch,
)
from xaytune.core.domain.budget import BudgetStatus, DimensionStatus
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import ActionId, ExperimentId, PolicyDecisionId
from xaytune.core.immutable import FrozenDict, FrozenDomainModel
from xaytune.core.refs import Actor, ActorType
from xaytune.core.state.machines import NODE_MACHINE
from xaytune.core.state.status import ExperimentNodeStatus, ExperimentStatus, RunStatus

__all__ = [
    "GovernedAction",
    "PolicyContext",
    "PolicyDecision",
    "PolicyProposal",
    "PolicyProposer",
    "PolicyVerdict",
    "applicability_problems",
    "awaits_execution",
    "policy_input_identity_v1",
]


class PolicyVerdict(str, Enum):
    """What policy says about one proposed action."""

    ALLOW = "allow"
    """Authorized. The action stays ``VALIDATED``, with this decision on it."""

    DENY = "deny"
    """Refused. The action is ``REJECTED``."""

    REQUIRE_APPROVAL = "require_approval"
    """A human has to say yes. The action waits in ``APPROVAL_PENDING``."""


class PolicyProposer(FrozenDomainModel):
    """Who proposed the action, as policy sees it: type and id, nothing else.

    Not an :class:`~xaytune.core.refs.Actor`, whose free-form ``metadata`` is
    provenance, not identity. Policy may read only what
    :func:`policy_input_identity_v1` identifies, so metadata a policy could
    branch on would give one fingerprint two verdicts. The full actor stays on
    the Action it proposed.
    """

    type: ActorType
    id: str = Field(min_length=1)

    @classmethod
    def of(cls, actor: Actor) -> PolicyProposer:
        return cls(type=actor.type, id=actor.id)


class PolicyContext(FrozenDomainModel):
    """Everything validation and policy may depend on, as the record stands.

    Explicit and serializable, like ``DecisionContext``: an engine that read
    anything else could decide differently on inputs nobody changed.

    **What policy reads is exactly what v1 identifies.** Every nested model is
    rebuilt as its exact v1 type from the fields
    :func:`policy_input_identity_v1` projects, and every mapping
    (``parameters``, ``provider``, ``extensions``) as a base ``FrozenDict`` of
    base values, so a subclass carrying anything added later -- a newer
    ``CapabilityDocument``, ``DimensionStatus``, ``ActionTarget`` or mapping
    -- reaches no engine. A runtime-specific input that
    policy must see goes in ``CapabilityDocument.extensions``, which v1
    identifies, or in a v2.

    Attributes:
        target_status: The target's status, or ``None`` if it does not exist
            in this experiment -- which validation refuses before any policy
            is consulted.
        capabilities: What the experiment's runtime declares it can do, or
            ``None`` if it declares nothing. Recorded with the decision,
            because nothing else keeps it.
    """

    experiment_id: ExperimentId
    experiment_status: ExperimentStatus
    experiment_revision: int

    action_type: str
    action_version: str
    provider: FrozenDict | None = None
    mutation_class: MutationClass
    target: ActionTarget
    parameters: FrozenDict = Field(default_factory=FrozenDict)

    target_status: str | None = None
    target_revision: int | None = None

    proposed_by: PolicyProposer
    budget: BudgetStatus | None = None
    capabilities: CapabilityDocument | None = None

    @field_validator("parameters", mode="after")
    @classmethod
    def _parameters_v1(cls, parameters: FrozenDict) -> FrozenDict:
        return _exact_mapping(parameters)

    @field_validator("provider", mode="after")
    @classmethod
    def _provider_v1(cls, provider: FrozenDict | None) -> FrozenDict | None:
        return None if provider is None else _exact_mapping(provider)

    @field_validator("target", mode="after")
    @classmethod
    def _target_v1(cls, target: ActionTarget) -> ActionTarget:
        return ActionTarget(kind=target.kind, id=target.id)

    @field_validator("proposed_by", mode="after")
    @classmethod
    def _proposer_v1(cls, proposer: PolicyProposer) -> PolicyProposer:
        return PolicyProposer(type=proposer.type, id=proposer.id)

    @field_validator("budget", mode="after")
    @classmethod
    def _budget_view_v1(cls, budget: BudgetStatus | None) -> BudgetStatus | None:
        if budget is None:
            return None
        return BudgetStatus(
            dimensions=tuple(
                DimensionStatus(**{name: getattr(d, name) for name in _DIMENSION_FIELDS_V1})
                for d in budget.dimensions
            )
        )

    @field_validator("capabilities", mode="after")
    @classmethod
    def _capabilities_view_v1(
        cls, document: CapabilityDocument | None
    ) -> CapabilityDocument | None:
        if document is None:
            return None
        sections: dict[str, Any] = {}
        for section, fields in _CAPABILITY_FIELDS_V1.items():
            value = getattr(document, section)
            sections[section] = (
                None
                if value is None
                else _CAPABILITY_SECTIONS_V1[section](
                    **{name: getattr(value, name) for name in fields}
                )
            )
        return CapabilityDocument(
            schema_version=document.schema_version,
            extensions=_exact_mapping(document.extensions),
            **sections,
        )

    def input_fingerprint(self) -> str:
        """The identity of these inputs: :func:`policy_input_identity_v1`, hashed."""
        return fingerprint(policy_input_identity_v1(self))


def policy_input_identity_v1(context: PolicyContext) -> Mapping[str, Any]:
    """What makes two policy inputs the same, version 1. Frozen.

    An explicit projection, never a dump, for the reason
    ``decision_input_identity_v1`` is one: a field added later to
    ``BudgetStatus``, ``CapabilityDocument`` or ``Actor`` must not change the
    identity of any policy input. A new field that policy should see is a new
    version -- ``policy_input_identity_v2`` -- recorded as such; this function
    projects exactly these fields, forever:

    - the experiment: id, status, revision;
    - the action: type, schema version, provider, mutation class, and its
      parameters -- whose shape the action's own schema version fixes;
    - the target: kind, id, status, revision;
    - the proposer: type and id, not free-form metadata;
    - each budget dimension: dimension, kind, limit, reserved, committed,
      consumed, released, outstanding, remaining;
    - the capability document's schema version and extensions, and each
      section's fields as listed below.

    No time: when the question was asked does not change what was asked.
    :class:`PolicyContext` holds exactly these fields, so the recorded snapshot
    is what policy saw, and nothing policy saw is outside the identity.
    """
    return {
        "kind": "policy-input",
        "identity_version": 1,
        "experiment": {
            "id": str(context.experiment_id),
            "status": context.experiment_status.value,
            "revision": context.experiment_revision,
        },
        "action": {
            "type": context.action_type,
            "version": context.action_version,
            "provider": _plain(context.provider) if context.provider is not None else None,
            "mutation_class": context.mutation_class.value,
            "parameters": _plain(context.parameters),
        },
        "target": {
            "kind": context.target.kind,
            "id": context.target.id,
            "status": context.target_status,
            "revision": context.target_revision,
        },
        "proposed_by": {"type": context.proposed_by.type, "id": context.proposed_by.id},
        "budget": _budget_v1(context.budget),
        "capabilities": _capabilities_v1(context.capabilities),
    }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _budget_v1(budget: BudgetStatus | None) -> list[dict[str, Any]] | None:
    if budget is None:
        return None
    return [
        {
            name: (
                d.dimension.value
                if name == "dimension"
                else d.kind
                if name == "kind"
                else str(getattr(d, name))
            )
            for name in _DIMENSION_FIELDS_V1
        }
        for d in sorted(budget.dimensions, key=lambda d: d.dimension.value)
    ]


def _exact_mapping(mapping: Mapping[str, Any]) -> FrozenDict:
    """*mapping* rebuilt from its entries alone, as base types all the way down.

    ``FrozenDict`` keeps a subclass instance it is given, and so does
    ``deep_freeze`` for a nested one, a ``str`` or an ``int``: attributes a
    subclass adds would reach policy without being identified. The base
    types' own methods copy each value, so no subclass override is called.
    """
    return FrozenDict({str.__str__(key): _exact_value(value) for key, value in mapping.items()})


def _exact_value(value: Any) -> Any:
    if value is None or type(value) is bool:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int.__int__(value)
    if isinstance(value, float):
        return float.__float__(value)
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, Mapping):
        return _exact_mapping(value)
    if isinstance(value, tuple):
        return tuple(_exact_value(item) for item in value)
    raise TypeError(f"{type(value).__name__} is not a frozen domain value")


_DIMENSION_FIELDS_V1 = (
    "dimension",
    "kind",
    "limit",
    "reserved",
    "committed",
    "consumed",
    "released",
    "outstanding",
    "remaining",
)
"""Frozen v1 fields driving both exact budget normalization and identity projection."""

_CAPABILITY_SECTIONS_V1: dict[str, type[Any]] = {
    "precision": PrecisionCapabilities,
    "distributed": DistributedCapabilities,
    "checkpoint": CheckpointCapabilities,
    "elasticity": ElasticityCapabilities,
    "resilience": ResilienceCapabilities,
    "agent_rollout": AgentRolloutCapabilities,
    "algorithms": AlgorithmCapabilities,
}

_CAPABILITY_FIELDS_V1: dict[str, tuple[str, ...]] = {
    "precision": ("supported",),
    "distributed": ("strategies", "min_workers", "max_workers"),
    "checkpoint": (
        "formats",
        "asynchronous",
        "reshardable",
        "atomic_commit",
        "full_exact_restore",
    ),
    "elasticity": ("supported", "min_workers", "max_workers", "membership_change"),
    "resilience": (
        "per_step",
        "provider",
        "provider_version",
        "supports_event_replay",
        "reports_completed_operations",
    ),
    "agent_rollout": ("stateful", "asynchronous"),
    "algorithms": ("supported",),
}


def _capabilities_v1(document: CapabilityDocument | None) -> dict[str, Any] | None:
    if document is None:
        return None
    projected: dict[str, Any] = {
        "schema_version": document.schema_version,
        "extensions": _plain(document.extensions),
    }
    for section, fields in _CAPABILITY_FIELDS_V1.items():
        value = getattr(document, section)
        projected[section] = (
            None if value is None else {name: _plain(getattr(value, name)) for name in fields}
        )
    return projected


class PolicyProposal(FrozenDomainModel):
    """What an engine said: a pure function of the spec, the context and the engine.

    No id, no time, no actor: the repository adds those when it records it.
    """

    verdict: PolicyVerdict
    reasons: tuple[str, ...] = Field(min_length=1)
    rule_ids: tuple[str, ...] = Field(default_factory=tuple)
    engine_name: str = Field(min_length=1)
    engine_version: str = Field(min_length=1)
    input_fingerprint: str = Field(min_length=1)


class PolicyDecision(PolicyProposal):
    """A proposal, recorded against one action, with the snapshot it judged. Immutable."""

    id: PolicyDecisionId = Field(default_factory=PolicyDecisionId.generate)
    action_id: ActionId
    experiment_id: ExperimentId
    context: PolicyContext
    actor: Actor
    created_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def record(
        cls,
        proposal: PolicyProposal,
        *,
        action_id: ActionId,
        context: PolicyContext,
        actor: Actor,
    ) -> PolicyDecision:
        return cls(
            **dict(proposal),
            action_id=action_id,
            experiment_id=context.experiment_id,
            context=context,
            actor=actor,
        )

    def proposal(self) -> PolicyProposal:
        """What was decided, without what identifies the record of it."""
        return PolicyProposal.model_validate(
            self.model_dump(
                include={
                    "verdict",
                    "reasons",
                    "rule_ids",
                    "engine_name",
                    "engine_version",
                    "input_fingerprint",
                }
            )
        )


class GovernedAction(FrozenDomainModel):
    """A proposed action as governance left it.

    Attributes:
        decision: Policy's decision, or ``None`` when validation refused the
            action before any policy was consulted.
        problems: Why validation refused it; empty when it was valid.
    """

    action: Action
    decision: PolicyDecision | None = None
    problems: tuple[str, ...] = Field(default_factory=tuple)


def awaits_execution(action: Action, decision: PolicyDecision | None) -> bool:
    """Whether *action* is authorized and waiting for an executor. The executor's rule.

    Exactly two ways, both with the decision on record::

        VALIDATED  + ALLOW
        APPROVED   + REQUIRE_APPROVAL

    A ``VALIDATED`` action with no decision never qualifies, nor does a
    decision filed against another action. Whoever carries the action out
    must still check, then, that it applies to the state of that moment.
    """
    if decision is None or decision.action_id != action.id:
        return False
    if action.policy_decision_id != decision.id:
        return False
    return (action.status, decision.verdict) in {
        (ActionStatus.VALIDATED, PolicyVerdict.ALLOW),
        (ActionStatus.APPROVED, PolicyVerdict.REQUIRE_APPROVAL),
    }


# ---- validation --------------------------------------------------------------------------

_RUN_NOT_ENDED = frozenset({RunStatus.CREATED.value, RunStatus.ACTIVE.value})
_NODE_ENDED = frozenset(status.value for status in NODE_MACHINE.terminal_states)
_OPERATIONAL_ON_A_RUN = (
    ResizeMicrobatch,
    ChangeGradientAccumulation,
    ChangeWorkerCount,
    ChangeCheckpointInterval,
)
_INTERVENTIONS = (ChangeLearningRate, ChangeScheduler, ChangeWarmup)


def applicability_problems(spec: ActionSpec, context: PolicyContext) -> tuple[str, ...]:
    """Why *spec* cannot apply in *context*, or nothing. Pure; every reason at once.

    For every action, built-in or a plugin's: the experiment is ``ACTIVE``,
    and the target exists in it. Then, per built-in type:

    ```text
    resize-microbatch, change-gradient-accumulation,
    change-worker-count, change-checkpoint-interval    the run has not ended
    change-learning-rate, change-scheduler,
    change-warmup                                       the run is ACTIVE: a change to one
                                                        not yet started is a new candidate
    reject-candidate                                    the node has not ended
    promote-candidate                                   the node is COMPLETED
    change-worker-count                                 the runtime declares that its worker
                                                        count can change, and the count is
                                                        inside every range it declares
    ```
    """
    problems: list[str] = []
    if context.experiment_status is not ExperimentStatus.ACTIVE:
        problems.append(
            f"experiment {context.experiment_id} is {context.experiment_status.value}, not active"
        )
    target = f"{context.target.kind} {context.target.id}"
    status = context.target_status
    if status is None:
        problems.append(f"{target} does not exist in experiment {context.experiment_id}")
        return tuple(problems)

    if isinstance(spec, _OPERATIONAL_ON_A_RUN) and status not in _RUN_NOT_ENDED:
        problems.append(f"{target} has ended ({status}); there is no next attempt to change")
    if isinstance(spec, _INTERVENTIONS) and status != RunStatus.ACTIVE.value:
        problems.append(
            f"{target} is {status}, not active: an intervention changes a continuing "
            f"trajectory, and a change before training starts is a different candidate"
        )
    if isinstance(spec, RejectCandidate) and status in _NODE_ENDED:
        problems.append(f"{target} has already ended ({status})")
    if isinstance(spec, PromoteCandidate) and status != ExperimentNodeStatus.COMPLETED.value:
        problems.append(f"{target} is {status}; only a completed candidate can be promoted")
    if isinstance(spec, ChangeWorkerCount):
        problems.extend(_worker_count_problems(spec.workers, context.capabilities))
    return tuple(problems)


def _worker_count_problems(workers: int, capabilities: CapabilityDocument | None) -> list[str]:
    """Fail closed: a change needs the runtime to say, explicitly, that it can make one."""
    if capabilities is None or capabilities.elasticity is None:
        return ["the runtime does not declare whether its worker count can change"]
    elasticity = capabilities.elasticity
    if elasticity.supported is not True:
        return ["the runtime declares that its worker count cannot change"]
    problems = []
    ranges = [("elastic", elasticity.min_workers, elasticity.max_workers)]
    if capabilities.distributed is not None:
        distributed = capabilities.distributed
        ranges.append(("distributed", distributed.min_workers, distributed.max_workers))
    for name, low, high in ranges:
        if low is not None and workers < low:
            problems.append(f"{workers} workers is below the runtime's {name} minimum of {low}")
        if high is not None and workers > high:
            problems.append(f"{workers} workers is above the runtime's {name} maximum of {high}")
    return problems
