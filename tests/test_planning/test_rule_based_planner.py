"""RuleBasedPlanner and NoOpPlanner: a context in, proposals (or nothing) out.

Pure tests. What the repository projects into a context is
``test_planning_context.py``; binding a planner at submission is
``tests/test_experiment/test_planner_binding.py``.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from xaytune.core.domain.budget import BudgetDimension, BudgetStatus, DimensionStatus
from xaytune.core.domain.candidate import (
    AdapterSpec,
    CandidateSpec,
    DataSpec,
    ModelSpec,
    OptimizationSpec,
    TrainingKind,
    TrainingSpec,
)
from xaytune.core.domain.decision import DecisionOutcome
from xaytune.core.domain.objective import Objective, ObjectiveMetric
from xaytune.core.domain.planning import (
    PLANNING_CONTEXT_IDENTITY_VERSION,
    CandidateProposal,
    DecisionSummary,
    EvaluationSummary,
    MetricSummary,
    NodeSummary,
    PlanningContext,
    planning_context_identity_v1,
)
from xaytune.core.domain.specs import PlannerSpec
from xaytune.core.ids import DecisionId, EvaluationId, ExperimentId, ExperimentNodeId
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import DatasetRef, ModelRef
from xaytune.core.state.status import ExperimentNodeStatus, ExperimentStatus
from xaytune.planning import (
    PLANNERS,
    IncreaseLoRARank,
    NoOpPlanner,
    Planner,
    PlannerConfigurationError,
    RuleBasedPlanner,
    RuleBasedPlannerConfig,
    bind_planner,
)

EXPERIMENT = ExperimentId.generate()
COMPLETED, REJECTED = ExperimentNodeStatus.COMPLETED, ExperimentNodeStatus.REJECTED
GROW = {"kind": "increase-lora-rank", "factor": 2, "max_rank": 64}


def candidate(rank: int | None = 16, adapter_type: str | None = "lora") -> CandidateSpec:
    return CandidateSpec(
        model=ModelSpec(model=ModelRef(uri="Qwen/Qwen3-8B")),
        data=DataSpec(dataset=DatasetRef(uri="./data/support-v4.jsonl", revision="sha256:v4")),
        training=TrainingSpec(
            kind=TrainingKind.SFT,
            adapter=None
            if adapter_type is None
            else AdapterSpec(
                type=adapter_type,
                rank=rank,
                alpha=32.0,
                target_modules=("q_proj", "v_proj"),
                metadata=FrozenDict({"note": "kept"}),
            ),
            optimization=OptimizationSpec(
                learning_rate=2e-5, micro_batch_size=4, gradient_accumulation=8, epochs=2
            ),
        ),
    )


def node(
    value: float | None = 0.79,
    *,
    rank: int | None = 16,
    status: ExperimentNodeStatus = COMPLETED,
    outcome: DecisionOutcome | None = DecisionOutcome.BRANCH,
    node_id: ExperimentNodeId | None = None,
    spec: CandidateSpec | None = None,
    metrics: tuple[MetricSummary, ...] | None = None,
) -> NodeSummary:
    result = EvaluationId.generate()
    if metrics is None:
        metrics = (
            ()
            if value is None
            else (MetricSummary(name="task_success", value=value, evaluator_name="support"),)
        )
    spec = spec or candidate(rank)
    return NodeSummary(
        node_id=node_id or ExperimentNodeId.generate(),
        status=status,
        candidate=spec,
        candidate_fingerprint=spec.candidate_fingerprint(),
        decisions=()
        if outcome is None
        else (
            DecisionSummary(
                decision_id=DecisionId.generate(),
                evaluation_cycle=1,
                outcome=outcome,
                engine_name="adaptive-threshold",
                engine_version="1.0.0",
                input_fingerprint="sha256:input",
                evaluation_result_ids=(result,),
            ),
        ),
        evaluations=(
            EvaluationSummary(evaluation_result_id=result, evaluation_cycle=1, metrics=metrics),
        ),
    )


def context(
    *nodes: NodeSummary,
    direction: str = "maximize",
    status: ExperimentStatus = ExperimentStatus.ACTIVE,
    budget: BudgetStatus | None = None,
) -> PlanningContext:
    return PlanningContext(
        experiment_id=EXPERIMENT,
        experiment_status=status,
        objective=Objective(
            primary=ObjectiveMetric(name="task_success", direction=direction),  # type: ignore[arg-type]
            target=0.82 if direction == "maximize" else 0.1,
        ),
        nodes=nodes,
        budget=budget,
    )


def planner(*rules: dict) -> RuleBasedPlanner:
    return RuleBasedPlanner.from_spec(
        PlannerSpec(kind="rule-based", config=FrozenDict({"rules": list(rules or (GROW,))}))
    )


def propose(p: Planner, ctx: PlanningContext):
    return asyncio.run(p.propose(ctx))


def rank_of(proposal: CandidateProposal) -> int | None:
    adapter = proposal.candidate.training.adapter
    assert adapter is not None
    return adapter.rank


# ---- the contract ------------------------------------------------------------------------


def test_both_built_in_planners_satisfy_the_protocol() -> None:
    assert isinstance(planner(), Planner)
    assert isinstance(NoOpPlanner(), Planner)
    assert set(PLANNERS) == {"rule-based", "no-op"}
    assert (RuleBasedPlanner.descriptor.provider, RuleBasedPlanner.descriptor.name) == (
        "xaytune",
        "rule-based",
    )


def test_the_no_op_planner_proposes_nothing() -> None:
    assert propose(NoOpPlanner(), context(node())) == ()
    assert propose(bind_planner(PlannerSpec(kind="no-op")), context(node())) == ()


def test_the_mvp_step_d_proposal() -> None:
    """node_A COMPLETED after BRANCH at 0.79, LoRA 16 → a novel LoRA-32 candidate."""
    parent = node(0.79)
    ctx = context(parent)
    (proposal,) = propose(planner(), ctx)

    assert isinstance(proposal, CandidateProposal)
    assert proposal.parent_ids == (parent.node_id,)
    assert rank_of(proposal) == 32
    assert proposal.candidate_fingerprint == proposal.candidate.candidate_fingerprint()
    assert proposal.candidate_fingerprint not in ctx.candidate_fingerprints
    assert proposal.hypothesis == "Adapter capacity may be limiting task performance."
    assert dict(proposal.mutation) == {
        "rule": "increase-lora-rank",
        "field": "training.adapter.rank",
        "from": 16,
        "to": 32,
    }
    (decision,) = parent.decisions
    assert proposal.evidence_refs == (
        f"decision:{decision.decision_id}",
        f"evaluation-result:{parent.evaluations[0].evaluation_result_id}",
    )
    provenance = proposal.provenance
    assert (provenance.planner_provider, provenance.planner_name) == ("xaytune", "rule-based")
    assert provenance.planner_version == RuleBasedPlanner.descriptor.plugin_version
    assert (provenance.planner_spec_kind, provenance.planner_spec_version) == (
        "rule-based",
        RuleBasedPlanner.descriptor.plugin_version,
    )
    assert provenance.context_fingerprint == ctx.input_fingerprint()
    assert provenance.context_identity_version == PLANNING_CONTEXT_IDENTITY_VERSION
    assert "task_success = 0.79" in proposal.reason


def test_a_proposal_mints_nothing_and_is_deterministic(monkeypatch) -> None:
    """No clock, no id, no randomness: the same context gives the same proposal."""
    import datetime as datetime_module
    import time
    import uuid

    ctx = context(node(0.79))
    first = propose(planner(), ctx)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a planner read the clock or minted an id")

    monkeypatch.setattr(time, "time", forbidden)
    monkeypatch.setattr(uuid, "uuid4", forbidden)
    monkeypatch.setattr(datetime_module, "datetime", None)
    second = propose(planner(), ctx)
    assert first == second
    dumped = first[0].model_dump(mode="json")
    assert not {"id", "node_id", "created_at", "action_id", "run_id"} & set(dumped)


# ---- the LoRA rule -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rank", "max_rank", "expected"),
    [(16, 64, 32), (32, 64, 64), (48, 64, 64), (64, 64, None), (128, 64, None)],
    ids=["16-to-32", "32-to-64", "capped-at-max", "at-max", "above-max-never-shrinks"],
)
def test_lora_rank_grows_up_to_the_cap(rank, max_rank, expected) -> None:
    proposals = propose(
        planner({"kind": "increase-lora-rank", "factor": 2, "max_rank": max_rank}),
        context(node(0.79, rank=rank)),
    )
    if expected is None:
        assert proposals == ()
    else:
        (proposal,) = proposals
        assert rank_of(proposal) == expected


@pytest.mark.parametrize(
    "spec",
    [candidate(adapter_type=None), candidate(adapter_type="ia3"), candidate(rank=None)],
    ids=["no-adapter", "not-lora", "rank-undeclared"],
)
def test_the_rule_does_not_apply_where_its_preconditions_fail(spec) -> None:
    assert propose(planner(), context(node(0.79, spec=spec))) == ()


def test_every_other_candidate_field_is_kept_and_the_parent_is_untouched() -> None:
    parent = node(0.79)
    before = parent.candidate.model_dump(mode="json")
    (proposal,) = propose(planner(), context(parent))

    assert parent.candidate.model_dump(mode="json") == before
    after = proposal.candidate.model_dump(mode="json")
    assert after["training"]["adapter"]["rank"] == 32
    after["training"]["adapter"]["rank"] = 16
    assert after == before


@pytest.mark.parametrize(
    "rules",
    [
        {"rules": []},
        {"rules": [{"kind": "unknown-rule"}]},
        {"rules": [{"kind": "increase-lora-rank", "factor": 1, "max_rank": 64}]},
        {"rules": [{"kind": "increase-lora-rank", "factor": 2, "max_rank": 0}]},
        {"rules": [{"kind": "increase-lora-rank", "factor": 2.0, "max_rank": 64}]},
        {"rules": [{**GROW, "path": "training.adapter.rank"}]},
        {"rules": [GROW], "extra": True},
        {},
    ],
    ids=[
        "no-rules",
        "unknown-kind",
        "factor-one",
        "max-rank-zero",
        "float-factor",
        "unknown-rule-key",
        "unknown-config-key",
        "missing-rules",
    ],
)
def test_malformed_configuration_is_refused(rules) -> None:
    with pytest.raises(PlannerConfigurationError):
        bind_planner(PlannerSpec(kind="rule-based", config=FrozenDict(rules)))


def test_binding_refuses_a_wrong_kind_or_version_and_records_the_bound_spec() -> None:
    with pytest.raises(PlannerConfigurationError, match="no planner of kind"):
        bind_planner(PlannerSpec(kind="llm"))
    with pytest.raises(PlannerConfigurationError, match="version 9.9.9"):
        bind_planner(
            PlannerSpec(kind="rule-based", version="9.9.9", config=FrozenDict({"rules": [GROW]}))
        )
    with pytest.raises(PlannerConfigurationError, match="takes no config"):
        bind_planner(PlannerSpec(kind="no-op", config=FrozenDict({"rules": [GROW]})))
    bound = bind_planner(PlannerSpec(kind="rule-based", config=FrozenDict({"rules": [GROW]}))).spec
    assert bound.version == RuleBasedPlanner.descriptor.plugin_version
    assert (
        bound
        == RuleBasedPlanner(
            RuleBasedPlannerConfig(rules=(IncreaseLoRARank(factor=2, max_rank=64),))
        ).spec
    )
    assert bind_planner(bound).spec == bound, "a bound spec binds to itself"


# ---- parent selection --------------------------------------------------------------------


def test_the_best_completed_candidate_is_the_parent_maximize() -> None:
    low, high = node(0.70), node(0.79, rank=32)
    (proposal,) = propose(planner(), context(low, high))
    assert proposal.parent_ids == (high.node_id,)
    assert rank_of(proposal) == 64


def test_the_best_completed_candidate_is_the_parent_minimize() -> None:
    low, high = node(0.2, rank=32), node(0.5)
    (proposal,) = propose(planner(), context(low, high, direction="minimize"))
    assert proposal.parent_ids == (low.node_id,)


def test_ties_are_broken_by_node_id() -> None:
    first, second = sorted((ExperimentNodeId.generate(), ExperimentNodeId.generate()), key=str)
    a = node(0.79, node_id=first, rank=16)
    b = node(0.79, node_id=second, rank=48)
    (proposal,) = propose(planner(), context(b, a))
    assert proposal.parent_ids == (a.node_id,)


def test_a_rejected_candidate_is_never_a_parent() -> None:
    rejected = node(0.95, status=REJECTED, outcome=DecisionOutcome.REJECT)
    completed = node(0.70, rank=32)
    (proposal,) = propose(planner(), context(rejected, completed))
    assert proposal.parent_ids == (completed.node_id,)
    assert propose(planner(), context(rejected)) == (), "rejected alone: nothing to build on"


@pytest.mark.parametrize(
    "metrics",
    [
        (),
        (
            MetricSummary(name="task_success", value=0.79, evaluator_name="support"),
            MetricSummary(name="task_success", value=0.80, evaluator_name="other"),
        ),
        (MetricSummary(name="task_success", value=0.79, slice="hard", evaluator_name="support"),),
    ],
    ids=["missing", "measured-twice", "sliced-only"],
)
def test_a_missing_or_ambiguous_metric_is_not_guessed(metrics) -> None:
    assert propose(planner(), context(node(metrics=metrics))) == ()


# ---- when the planner does not plan ------------------------------------------------------


@pytest.mark.parametrize(
    ("nodes", "status"),
    [
        ((), ExperimentStatus.ACTIVE),
        ((("deciding",),), ExperimentStatus.ACTIVE),
        ((("failed",),), ExperimentStatus.ACTIVE),
        ((("completed",),), ExperimentStatus.SUCCEEDED),
        ((("completed",),), ExperimentStatus.PAUSED),
    ],
    ids=["no-candidates", "a-candidate-deciding", "a-candidate-failed", "terminal", "paused"],
)
def test_outside_the_planning_stage_nothing_is_proposed(nodes, status) -> None:
    built = []
    for (kind,) in nodes:
        if kind == "completed":
            built.append(node(0.79))
        elif kind == "deciding":
            built.append(node(0.79, status=ExperimentNodeStatus.DECIDING, outcome=None))
        else:
            built.append(node(None, status=ExperimentNodeStatus.FAILED, outcome=None))
    if nodes and nodes[0][0] != "completed":
        built.append(node(0.79))
    assert propose(planner(), context(*built, status=status)) == ()


def test_a_completed_candidate_without_a_branch_decision_is_no_parent() -> None:
    """COMPLETED by STOP_SUCCEEDED ends the experiment; only a BRANCH leaves it to build on."""
    stopped = node(0.9, outcome=DecisionOutcome.STOP_SUCCEEDED)
    assert propose(planner(), context(stopped)) == ()


def test_an_exhausted_quota_means_no_new_work_and_nothing_is_reserved() -> None:
    def runs(remaining: str) -> BudgetStatus:
        return BudgetStatus(
            dimensions=(
                DimensionStatus(
                    dimension=BudgetDimension.RUNS,
                    kind="quota",
                    limit=Decimal(4),
                    reserved=Decimal(4) - Decimal(remaining),
                    committed=Decimal(0),
                    consumed=Decimal(0),
                    released=Decimal(0),
                    outstanding=Decimal(4) - Decimal(remaining),
                    remaining=Decimal(remaining),
                ),
            )
        )

    assert propose(planner(), context(node(0.79), budget=runs("0"))) == ()
    (proposal,) = propose(planner(), context(node(0.79), budget=runs("3")))
    assert rank_of(proposal) == 32


# ---- deduplication -----------------------------------------------------------------------


def test_a_candidate_the_experiment_already_has_is_skipped() -> None:
    parent = node(0.79)
    existing = node(0.60, spec=candidate(32), status=REJECTED, outcome=DecisionOutcome.REJECT)
    assert propose(planner(), context(parent, existing)) == ()


def test_after_a_duplicate_the_next_rule_is_tried() -> None:
    parent = node(0.79)
    existing = node(0.60, spec=candidate(32), status=REJECTED, outcome=DecisionOutcome.REJECT)
    (proposal,) = propose(
        planner(GROW, {"kind": "increase-lora-rank", "factor": 4, "max_rank": 64}),
        context(parent, existing),
    )
    assert rank_of(proposal) == 64
    assert proposal.mutation["to"] == 64


def test_rules_are_tried_in_declared_order() -> None:
    (proposal,) = propose(
        planner({"kind": "increase-lora-rank", "factor": 4, "max_rank": 64}, GROW),
        context(node(0.79)),
    )
    assert rank_of(proposal) == 64


# ---- the context's identity --------------------------------------------------------------


def test_the_context_fingerprint_is_stable_and_order_independent() -> None:
    a, b = node(0.70), node(0.79, rank=32)
    assert context(a, b).input_fingerprint() == context(b, a).input_fingerprint()
    assert context(a, b).input_fingerprint() == context(a, b).input_fingerprint()
    assert context(a).input_fingerprint() != context(a, b).input_fingerprint()


def test_the_identity_projection_is_explicit_and_versioned() -> None:
    identity = planning_context_identity_v1(context(node(0.79)))
    assert (identity["kind"], identity["identity_version"]) == ("planning-context", 1)
    assert set(identity) == {
        "kind",
        "identity_version",
        "experiment_id",
        "experiment_status",
        "objective",
        "nodes",
        "budget",
    }
    (projected,) = identity["nodes"]
    assert "candidate" not in projected, "identity is the candidate fingerprint, not its bytes"


def test_a_stale_candidate_fingerprint_is_refused() -> None:
    spec = candidate(16)
    with pytest.raises(ValueError, match="not the current identity"):
        NodeSummary(
            node_id=ExperimentNodeId.generate(),
            status=COMPLETED,
            candidate=spec,
            candidate_fingerprint=spec.candidate_fingerprint_v1(),
        )
