"""Validation and policy, as pure functions of a snapshot (PR-023).

```text
applicability_problems(spec, context)   validation: built in, always
PolicyEngine.evaluate(spec, context)    authorization: first matching rule, or the default
context.input_fingerprint()             the snapshot's identity; no time in it
```
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError, create_model

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
from xaytune.core.domain.action import ActionTarget
from xaytune.core.domain.actions import (
    ActionSpec,
    ChangeCheckpointInterval,
    ChangeGradientAccumulation,
    ChangeLearningRate,
    ChangeScheduler,
    ChangeWarmup,
    ChangeWorkerCount,
    MutationClass,
    PromoteCandidate,
    RejectCandidate,
    ResizeMicrobatch,
    action_descriptor,
)
from xaytune.core.domain.budget import BudgetDimension, BudgetStatus, DimensionStatus
from xaytune.core.domain.policy import (
    PolicyContext,
    PolicyProposer,
    PolicyVerdict,
    applicability_problems,
    policy_input_identity_v1,
)
from xaytune.core.ids import ExperimentId
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import Actor
from xaytune.core.state.machines import NODE_MACHINE
from xaytune.core.state.status import ExperimentNodeStatus, ExperimentStatus
from xaytune.policy import DenyAllPolicy, PolicyEngine, PolicyRule, RulePolicyEngine

EXPERIMENT = ExperimentId("exp_01J9ZQ00000000000000000EXP")
AGENT = Actor(type="llm_agent", id="planner")
RUN = ActionTarget(kind="run", id="run_1")
NODE = ActionTarget(kind="node", id="node_1")
ELASTIC = CapabilityDocument(
    distributed=DistributedCapabilities(min_workers=1, max_workers=8),
    elasticity=ElasticityCapabilities(
        supported=True, min_workers=2, max_workers=16, membership_change="restart"
    ),
)


def _context(spec: ActionSpec, **overrides: Any) -> PolicyContext:
    descriptor = action_descriptor(spec.type)  # type: ignore[attr-defined]
    values: dict[str, Any] = {
        "experiment_id": EXPERIMENT,
        "experiment_status": ExperimentStatus.ACTIVE,
        "experiment_revision": 3,
        "action_type": descriptor.type,
        "action_version": descriptor.version,
        "mutation_class": descriptor.mutation_class,
        "target": spec.target,
        "parameters": FrozenDict(spec.parameters()),
        "target_status": "active",
        "target_revision": 2,
        "proposed_by": PolicyProposer.of(AGENT),
    }
    values.update(overrides)
    return PolicyContext(**values)


# ---- validation --------------------------------------------------------------------------


def test_nothing_applies_to_an_experiment_that_is_not_active_or_a_target_it_lacks() -> None:
    spec = ChangeLearningRate(target=RUN, learning_rate=1e-4)
    problems = applicability_problems(
        spec, _context(spec, experiment_status=ExperimentStatus.PAUSED, target_status=None)
    )
    assert problems == (
        f"experiment {EXPERIMENT} is paused, not active",
        f"run run_1 does not exist in experiment {EXPERIMENT}",
    )


_OPERATIONAL = [
    ResizeMicrobatch(target=RUN, micro_batch_size=2),
    ChangeGradientAccumulation(target=RUN, gradient_accumulation=4),
    ChangeCheckpointInterval(target=RUN, every_steps=100),
]
_INTERVENTIONS = [
    ChangeLearningRate(target=RUN, learning_rate=1e-4),
    ChangeScheduler(target=RUN, name="cosine"),
    ChangeWarmup(target=RUN, warmup_steps=10),
]


@pytest.mark.parametrize("spec", _OPERATIONAL, ids=lambda s: s.type)
@pytest.mark.parametrize(
    ("status", "applies"),
    [("created", True), ("active", True), ("succeeded", False), ("failed", False)],
)
def test_an_operational_change_needs_a_run_that_has_not_ended(
    spec: ActionSpec, status: str, applies: bool
) -> None:
    assert (applicability_problems(spec, _context(spec, target_status=status)) == ()) is applies


@pytest.mark.parametrize("spec", _INTERVENTIONS, ids=lambda s: s.type)
@pytest.mark.parametrize(
    ("status", "applies"), [("created", False), ("active", True), ("succeeded", False)]
)
def test_an_intervention_needs_a_run_that_is_training(
    spec: ActionSpec, status: str, applies: bool
) -> None:
    problems = applicability_problems(spec, _context(spec, target_status=status))
    assert (problems == ()) is applies
    if not applies:
        assert "a change before training starts is a different candidate" in problems[0]


@pytest.mark.parametrize(
    ("spec", "status", "applies"),
    [
        (RejectCandidate(target=NODE), "deciding", True),
        (RejectCandidate(target=NODE), "active", True),
        (RejectCandidate(target=NODE), "completed", False),
        (RejectCandidate(target=NODE), "rejected", False),
        (PromoteCandidate(target=NODE), "completed", True),
        (PromoteCandidate(target=NODE), "deciding", False),
    ],
    ids=lambda v: v.type if isinstance(v, ActionSpec) else str(v),
)
def test_a_candidate_is_judged_only_where_that_makes_sense(
    spec: ActionSpec, status: str, applies: bool
) -> None:
    assert (applicability_problems(spec, _context(spec, target_status=status)) == ()) is applies


@pytest.mark.parametrize("status", list(ExperimentNodeStatus), ids=lambda s: s.value)
def test_a_candidate_that_has_ended_in_any_way_is_not_rejected(
    status: ExperimentNodeStatus,
) -> None:
    spec = RejectCandidate(target=NODE)
    problems = applicability_problems(spec, _context(spec, target_status=status.value))
    ended = NODE_MACHINE.is_terminal(status)
    assert (f"node node_1 has already ended ({status.value})" in problems) is ended
    assert ended is (
        status
        in {
            ExperimentNodeStatus.COMPLETED,
            ExperimentNodeStatus.REJECTED,
            ExperimentNodeStatus.CANCELLED,
            ExperimentNodeStatus.FAILED,
        }
    ), "every terminal node status, FAILED included"


@pytest.mark.parametrize(
    ("capabilities", "workers", "problem"),
    [
        (None, 4, "does not declare whether its worker count can change"),
        (CapabilityDocument(), 4, "does not declare whether its worker count can change"),
        (
            CapabilityDocument(distributed=DistributedCapabilities(min_workers=1, max_workers=8)),
            4,
            "does not declare whether its worker count can change",
        ),
        (
            CapabilityDocument(elasticity=ElasticityCapabilities(supported=False)),
            4,
            "declares that its worker count cannot change",
        ),
        (
            CapabilityDocument(elasticity=ElasticityCapabilities(supported=None)),
            4,
            "declares that its worker count cannot change",
        ),
        (ELASTIC, 1, "below the runtime's elastic minimum of 2"),
        (ELASTIC, 12, "above the runtime's distributed maximum of 8"),
        (ELASTIC, 17, "above the runtime's elastic maximum of 16"),
        (ELASTIC, 4, None),
        (CapabilityDocument(elasticity=ElasticityCapabilities(supported=True)), 64, None),
    ],
    ids=[
        "no-document",
        "no-sections",
        "distributed-only",
        "unsupported",
        "undeclared-support",
        "below-elastic",
        "above-distributed",
        "above-elastic",
        "inside-both",
        "no-bounds",
    ],
)
def test_a_worker_count_changes_only_where_the_runtime_says_it_can(
    capabilities: CapabilityDocument | None, workers: int, problem: str | None
) -> None:
    spec = ChangeWorkerCount(target=RUN, workers=workers)
    problems = applicability_problems(spec, _context(spec, capabilities=capabilities))
    if problem is None:
        assert problems == ()
    else:
        assert any(problem in p for p in problems), problems


# ---- the snapshot's identity -------------------------------------------------------------


def test_the_snapshot_is_identified_by_what_it_says_not_when_or_by_whose_metadata() -> None:
    spec = ChangeLearningRate(target=RUN, learning_rate=1e-4)
    one = _context(spec)
    tagged = Actor(type="llm_agent", id="planner", metadata={"role": "admin"})
    two = _context(spec, proposed_by=PolicyProposer.of(tagged))
    assert one == two, "the proposer's metadata never reaches policy"
    assert one.input_fingerprint() == two.input_fingerprint()
    assert "created_at" not in str(policy_input_identity_v1(one))


@pytest.mark.parametrize(
    "proposer",
    [
        Actor(type="llm_agent", id="planner", metadata={"role": "admin"}),
        {"type": "llm_agent", "id": "planner", "metadata": {"role": "admin"}},
    ],
    ids=["actor", "mapping"],
)
def test_policy_is_never_shown_the_proposers_metadata(proposer: Any) -> None:
    spec = ChangeLearningRate(target=RUN, learning_rate=1e-4)
    with pytest.raises(ValidationError):
        _context(spec, proposed_by=proposer)


# What an engine can read, field by field. Each is projected by v1; a field
# added to any of these models is readable by policy and not identified, and
# fails here -- deciding whether it is v2 input is then an explicit choice.
_READABLE = {
    PolicyContext: {
        "experiment_id",
        "experiment_status",
        "experiment_revision",
        "action_type",
        "action_version",
        "provider",
        "mutation_class",
        "target",
        "parameters",
        "target_status",
        "target_revision",
        "proposed_by",
        "budget",
        "capabilities",
    },
    ActionTarget: {"kind", "id"},
    PolicyProposer: {"type", "id"},
    BudgetStatus: {"dimensions"},
    DimensionStatus: {
        "dimension",
        "kind",
        "limit",
        "reserved",
        "committed",
        "consumed",
        "released",
        "outstanding",
        "remaining",
    },
    CapabilityDocument: {
        "schema_version",
        "precision",
        "distributed",
        "checkpoint",
        "elasticity",
        "resilience",
        "agent_rollout",
        "algorithms",
        "extensions",
    },
    PrecisionCapabilities: {"supported"},
    DistributedCapabilities: {"strategies", "min_workers", "max_workers"},
    CheckpointCapabilities: {"formats", "asynchronous", "reshardable", "atomic_commit"},
    ElasticityCapabilities: {"supported", "min_workers", "max_workers", "membership_change"},
    ResilienceCapabilities: {
        "per_step",
        "provider",
        "provider_version",
        "supports_event_replay",
        "reports_completed_operations",
    },
    AgentRolloutCapabilities: {"stateful", "asynchronous"},
    AlgorithmCapabilities: {"supported"},
}


@pytest.mark.parametrize("model", list(_READABLE), ids=lambda m: m.__name__)
def test_everything_policy_can_read_is_identified_by_v1(model: type) -> None:
    assert set(model.model_fields) == _READABLE[model], (  # type: ignore[attr-defined]
        f"{model.__name__} changed shape: policy can read what v1 does not identify"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"experiment_revision": 4},
        {"experiment_status": ExperimentStatus.PAUSED},
        {"target_status": "succeeded"},
        {"target_revision": 3},
        {"proposed_by": PolicyProposer(type="human", id="ana")},
        {"capabilities": ELASTIC},
        {
            "budget": BudgetStatus(
                dimensions=(
                    DimensionStatus(
                        dimension=BudgetDimension.RUNS,
                        kind="quota",
                        limit=Decimal(2),
                        reserved=Decimal(1),
                        committed=Decimal(1),
                        consumed=Decimal(1),
                        released=Decimal(0),
                        outstanding=Decimal(0),
                        remaining=Decimal(1),
                    ),
                )
            )
        },
        {"parameters": FrozenDict({"learning_rate": 2e-4})},
    ],
    ids=lambda c: next(iter(c)),
)
def test_anything_the_engine_could_read_is_part_of_the_snapshot(change: dict) -> None:
    spec = ChangeLearningRate(target=RUN, learning_rate=1e-4)
    assert _context(spec).input_fingerprint() != _context(spec, **change).input_fingerprint()


# ---- engines -----------------------------------------------------------------------------


def test_with_no_policy_everything_is_denied_and_says_why() -> None:
    spec = ChangeLearningRate(target=RUN, learning_rate=1e-4)
    context = _context(spec)
    proposal = DenyAllPolicy().evaluate(spec, context)

    assert proposal.verdict is PolicyVerdict.DENY
    assert "no policy is configured" in proposal.reasons[0]
    assert (proposal.rule_ids, proposal.input_fingerprint) == (
        ("deny-all",),
        context.input_fingerprint(),
    )
    assert isinstance(DenyAllPolicy(), PolicyEngine)


RULES = RulePolicyEngine(
    [
        PolicyRule(
            id="lr-needs-a-human",
            verdict=PolicyVerdict.REQUIRE_APPROVAL,
            reason="learning-rate changes are reviewed",
            action_types=("change-learning-rate",),
        ),
        PolicyRule(
            id="science-denied",
            verdict=PolicyVerdict.DENY,
            reason="no scientific interventions",
            mutation_classes=(MutationClass.SCIENTIFIC_INTERVENTION,),
        ),
        PolicyRule(
            id="operations-allowed",
            verdict=PolicyVerdict.ALLOW,
            reason="operational changes are fine",
            mutation_classes=(MutationClass.OPERATIONAL,),
        ),
    ],
    default=PolicyVerdict.DENY,
)


@pytest.mark.parametrize(
    ("spec", "verdict", "rule"),
    [
        (
            ChangeLearningRate(target=RUN, learning_rate=1e-4),
            "require_approval",
            "lr-needs-a-human",
        ),
        (ChangeWarmup(target=RUN, warmup_steps=1), "deny", "science-denied"),
        (ResizeMicrobatch(target=RUN, micro_batch_size=2), "allow", "operations-allowed"),
        (RejectCandidate(target=NODE), "deny", "default"),
    ],
    ids=["first-match-wins", "class-rule", "allow", "default"],
)
def test_the_first_matching_rule_decides_and_the_default_decides_the_rest(
    spec: ActionSpec, verdict: str, rule: str
) -> None:
    proposal = RULES.evaluate(spec, _context(spec))
    assert (proposal.verdict.value, proposal.rule_ids) == (verdict, (rule,))
    assert proposal == RULES.evaluate(spec, _context(spec)), "pure: same inputs, same answer"


def test_the_engine_version_names_its_rule_set() -> None:
    allow_all = RulePolicyEngine(default=PolicyVerdict.ALLOW)
    deny_all = RulePolicyEngine()
    assert allow_all.version != deny_all.version
    assert allow_all.version == RulePolicyEngine(default=PolicyVerdict.ALLOW).version
    assert allow_all.version.startswith("1+sha256:")


def test_a_rule_set_that_cannot_be_read_one_way_is_refused() -> None:
    with pytest.raises(ValidationError, match="names no action type and no mutation class"):
        PolicyRule(id="everything", verdict=PolicyVerdict.ALLOW, reason="all")
    rule = PolicyRule(
        id="a", verdict=PolicyVerdict.ALLOW, reason="r", action_types=("change-warmup",)
    )
    with pytest.raises(ValueError, match="repeat"):
        RulePolicyEngine([rule, rule])
    with pytest.raises(ValueError, match="engine's default"):
        RulePolicyEngine([rule.model_copy(update={"id": "default"})])


# ---- v1 is frozen ------------------------------------------------------------------------

_RUNS = DimensionStatus(
    dimension=BudgetDimension.RUNS,
    kind="quota",
    limit=Decimal(2),
    reserved=Decimal(1),
    committed=Decimal(1),
    consumed=Decimal(1),
    released=Decimal(0),
    outstanding=Decimal(0),
    remaining=Decimal(1),
)


def test_policy_input_identity_v1_is_pinned() -> None:
    """Changing what v1 projects changes this value: that is a v2, not an edit of v1."""
    spec = ChangeWorkerCount(target=RUN, workers=4)
    context = _context(spec, capabilities=ELASTIC, budget=BudgetStatus(dimensions=(_RUNS,)))
    assert context.input_fingerprint() == (
        "sha256:3505b09610f40e9af4fda27e21dcb2b932e827148c9c7eee168ea77a64dbf56e"
    )


class _LaterElasticity(ElasticityCapabilities):
    """A field a later release might add, that v1 does not know."""

    rebalance_seconds: int | None = None


class _LaterCapabilities(CapabilityDocument):
    topology: str | None = None


class _LaterDimension(DimensionStatus):
    burn_rate: Decimal | None = None


def test_a_field_added_later_is_not_part_of_v1() -> None:
    spec = ChangeWorkerCount(target=RUN, workers=4)
    now = _context(spec, capabilities=ELASTIC, budget=BudgetStatus(dimensions=(_RUNS,)))
    later = _context(
        spec,
        capabilities=_LaterCapabilities(
            distributed=ELASTIC.distributed,
            elasticity=_LaterElasticity(
                **ELASTIC.elasticity.model_dump(),  # type: ignore[union-attr]
                rebalance_seconds=30,
            ),
            topology="ring",
        ),
        budget=BudgetStatus(
            dimensions=(_LaterDimension(**_RUNS.model_dump(), burn_rate=Decimal("0.5")),)
        ),
    )
    assert later.input_fingerprint() == now.input_fingerprint()
    assert policy_input_identity_v1(later)["identity_version"] == 1


class _LaterTarget(ActionTarget):
    shard: str | None = None


class _LaterProposer(PolicyProposer):
    role: str | None = None


class _ReadsWhatItShouldNot:
    """Would allow on any later field it found; denies otherwise."""

    name = "reads-later-fields"
    version = "1"

    def __init__(self) -> None:
        self.seen: PolicyContext | None = None

    def evaluate(self, spec: ActionSpec, context: PolicyContext) -> Any:
        self.seen = context
        leaked = (
            getattr(context.capabilities, "topology", None),
            getattr(getattr(context.capabilities, "elasticity", None), "rebalance_seconds", None),
            *(getattr(d, "burn_rate", None) for d in context.budget.dimensions),  # type: ignore[union-attr]
            getattr(context.target, "shard", None),
            getattr(context.proposed_by, "role", None),
        )
        verdict = PolicyVerdict.ALLOW if any(v is not None for v in leaked) else PolicyVerdict.DENY
        return DenyAllPolicy().evaluate(spec, context).model_copy(update={"verdict": verdict})


def test_policy_never_sees_a_field_v1_does_not_identify() -> None:
    spec = ChangeWorkerCount(target=RUN, workers=4)
    now = _context(spec, capabilities=ELASTIC, budget=BudgetStatus(dimensions=(_RUNS,)))
    later = _context(
        spec,
        target=_LaterTarget(kind="run", id="run_1", shard="a"),
        proposed_by=_LaterProposer(type="llm_agent", id="planner", role="admin"),
        capabilities=_LaterCapabilities(
            distributed=ELASTIC.distributed,
            elasticity=_LaterElasticity(
                **ELASTIC.elasticity.model_dump(),  # type: ignore[union-attr]
                rebalance_seconds=30,
            ),
            topology="ring",
        ),
        budget=BudgetStatus(
            dimensions=(_LaterDimension(**_RUNS.model_dump(), burn_rate=Decimal("0.5")),)
        ),
    )
    engine = _ReadsWhatItShouldNot()

    assert engine.evaluate(spec, later).verdict is PolicyVerdict.DENY
    seen = engine.seen
    assert seen is not None and seen.capabilities is not None and seen.budget is not None
    assert type(seen.capabilities) is CapabilityDocument
    assert type(seen.capabilities.elasticity) is ElasticityCapabilities
    assert type(seen.budget.dimensions[0]) is DimensionStatus
    assert type(seen.target) is ActionTarget
    assert type(seen.proposed_by) is PolicyProposer
    assert not hasattr(seen.capabilities, "topology")
    assert not hasattr(seen.capabilities.elasticity, "rebalance_seconds")
    assert seen == now, "what policy reads is exactly the v1 view"
    assert later.input_fingerprint() == now.input_fingerprint()


# ---- the executor's rule -----------------------------------------------------------------


_SECTIONS = {
    "precision": PrecisionCapabilities,
    "distributed": DistributedCapabilities,
    "checkpoint": CheckpointCapabilities,
    "elasticity": ElasticityCapabilities,
    "resilience": ResilienceCapabilities,
    "agent_rollout": AgentRolloutCapabilities,
    "algorithms": AlgorithmCapabilities,
}


@pytest.mark.parametrize("section", list(_SECTIONS))
def test_every_capability_section_reaches_policy_as_its_v1_type(section: str) -> None:
    base = _SECTIONS[section]
    later = create_model(f"_Later{base.__name__}", __base__=base, secret=(str, "admin"))
    spec = ChangeLearningRate(target=RUN, learning_rate=1e-4)

    context = _context(spec, capabilities=CapabilityDocument(**{section: later()}))

    seen = getattr(context.capabilities, section)
    assert type(seen) is base
    assert not hasattr(seen, "secret")
    assert context.input_fingerprint() == (
        _context(spec, capabilities=CapabilityDocument(**{section: base()})).input_fingerprint()
    )


class _LaterMapping(FrozenDict):
    """A mapping with a property no entry declares."""

    @property
    def secret(self) -> str:
        return "admin"


class _LaterString(str):
    secret = "admin"


class _LaterInt(int):
    secret = "admin"


def _mapping_leaks(value: Any) -> bool:
    """Whether *value*, at any depth, is anything but base mappings, tuples and scalars."""
    if isinstance(value, Mapping):
        return type(value) is not FrozenDict or any(
            type(k) is not str or _mapping_leaks(v) for k, v in value.items()
        )
    if isinstance(value, tuple):
        return type(value) is not tuple or any(_mapping_leaks(v) for v in value)
    return value is not None and type(value) not in (bool, int, float, str)


def _hostile(entries: dict[str, Any]) -> _LaterMapping:
    return _LaterMapping(
        {
            **entries,
            "nested": _LaterMapping({"tag": _LaterString("x")}),
            "count": _LaterInt(3),
            "items": (_LaterString("a"), _LaterMapping({"k": 1})),
        }
    )


def _plainly(entries: dict[str, Any]) -> FrozenDict:
    return FrozenDict({**entries, "nested": {"tag": "x"}, "count": 3, "items": ("a", {"k": 1})})


_MAPPING_SURFACES = {
    "parameters": lambda m: {"parameters": m},
    "provider": lambda m: {"provider": m},
    "extensions": lambda m: {"capabilities": CapabilityDocument(extensions=m)},
}


def _surface(context: PolicyContext, name: str) -> Any:
    if name == "extensions":
        return context.capabilities.extensions  # type: ignore[union-attr]
    return getattr(context, name)


class _ReadsSecrets:
    name = "reads-secrets"
    version = "1"

    def __init__(self, surface: str) -> None:
        self.surface = surface

    def evaluate(self, spec: ActionSpec, context: PolicyContext) -> Any:
        mapping = _surface(context, self.surface)
        leaked = hasattr(mapping, "secret") or _mapping_leaks(mapping)
        verdict = PolicyVerdict.ALLOW if leaked else PolicyVerdict.DENY
        return DenyAllPolicy().evaluate(spec, context).model_copy(update={"verdict": verdict})


@pytest.mark.parametrize("surface", list(_MAPPING_SURFACES))
def test_a_mapping_subclass_shows_policy_only_its_entries(surface: str) -> None:
    spec = ChangeLearningRate(target=RUN, learning_rate=1e-4)
    hostile = _context(spec, **_MAPPING_SURFACES[surface](_hostile({"rack": "b"})))
    plain = _context(spec, **_MAPPING_SURFACES[surface](_plainly({"rack": "b"})))

    seen = _surface(hostile, surface)
    assert type(seen) is FrozenDict and not hasattr(seen, "secret")
    assert not _mapping_leaks(seen), "base types at every depth"
    assert seen == _surface(plain, surface)
    assert _ReadsSecrets(surface).evaluate(spec, hostile).verdict is PolicyVerdict.DENY
    assert hostile.input_fingerprint() == plain.input_fingerprint()
    assert hostile.model_dump(mode="json") == plain.model_dump(mode="json")


def test_only_an_authorized_action_awaits_execution() -> None:
    from xaytune.core.domain.action import ActionStatus
    from xaytune.core.domain.actions import action_from_spec
    from xaytune.core.domain.policy import PolicyDecision, PolicyProposal, awaits_execution

    spec = ChangeLearningRate(target=RUN, learning_rate=1e-4)
    context = _context(spec)
    proposed = action_from_spec(spec, experiment_id=EXPERIMENT, proposed_by=AGENT, reason="r")
    validated = proposed.with_status(ActionStatus.VALIDATING).with_status(ActionStatus.VALIDATED)

    def decided(verdict: PolicyVerdict, action: Any = proposed) -> PolicyDecision:
        proposal = PolicyProposal(
            verdict=verdict,
            reasons=("r",),
            engine_name="e",
            engine_version="1",
            input_fingerprint=context.input_fingerprint(),
        )
        return PolicyDecision.record(proposal, action_id=action.id, context=context, actor=AGENT)

    allow, approval = decided(PolicyVerdict.ALLOW), decided(PolicyVerdict.REQUIRE_APPROVAL)
    authorized = validated.governed_by(str(allow.id), ActionStatus.VALIDATED)
    pending = validated.governed_by(str(approval.id), ActionStatus.APPROVAL_PENDING)
    approved = pending.with_status(ActionStatus.APPROVED)

    assert awaits_execution(authorized, allow)
    assert awaits_execution(approved, approval)
    assert not awaits_execution(validated, None), "validated with no decision: never"
    assert not awaits_execution(pending, approval), "not until a human approves"
    assert not awaits_execution(authorized, approval), "a decision that is not its own"
    other = decided(PolicyVerdict.ALLOW, action=validated.model_copy(update={}))
    assert not awaits_execution(authorized, other)
