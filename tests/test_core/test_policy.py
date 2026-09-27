"""Validation and policy, as pure functions of a snapshot (PR-023).

```text
applicability_problems(spec, context)   validation: built in, always
PolicyEngine.evaluate(spec, context)    authorization: first matching rule, or the default
context.input_fingerprint()             the snapshot's identity; no time in it
```
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from xaytune.core.capabilities import (
    CapabilityDocument,
    DistributedCapabilities,
    ElasticityCapabilities,
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
    PolicyVerdict,
    applicability_problems,
    policy_input_identity_v1,
)
from xaytune.core.ids import ExperimentId
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import Actor
from xaytune.core.state.status import ExperimentStatus
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
        "proposed_by": AGENT,
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
    two = _context(spec, proposed_by=Actor(type="llm_agent", id="planner", metadata={"x": 1}))
    assert one.input_fingerprint() == two.input_fingerprint()
    assert "created_at" not in str(policy_input_identity_v1(one))


@pytest.mark.parametrize(
    "change",
    [
        {"experiment_revision": 4},
        {"experiment_status": ExperimentStatus.PAUSED},
        {"target_status": "succeeded"},
        {"target_revision": 3},
        {"proposed_by": Actor(type="human", id="ana")},
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
