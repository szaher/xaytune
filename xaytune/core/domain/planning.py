"""What a planner sees, and what it may say back (spec 09, PR-024).

```text
durable record ──(repository projection)──▶ PlanningContext
      PlanningContext ──Planner.propose()──▶ CandidateProposal | ActionProposal
```

**A planner proposes; it decides and applies nothing.** Terminal judgements on
a candidate belong to the decision engine (PR-015/015b), failures of an
attempt to recovery (PR-019--021), and turning a proposed candidate into a
node to branching (PR-025). A proposal is intent: it carries no node, run or
action id and no timestamp. Those belong to the record that accepts it.

**Same context, same proposals.** :class:`PlanningContext` is a curated,
serializable projection of durable state -- never a repository, a runtime or
anything with a handle. Its identity is :func:`planning_context_identity_v1`,
an explicit versioned projection, so a field added to a summary later does
not silently change the identity of contexts already planned on. Every
proposal carries that fingerprint with the planner's identity
(:class:`ProposalProvenance`), which is how a later reader shows what a
proposal was based on. Nothing here is persisted: durable planner audit
arrives when proposals are consumed (PR-025/026), not before.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any, Literal

from pydantic import Field, SerializeAsAny, field_validator, model_validator

from xaytune.core.domain.actions.contract import ActionSpec
from xaytune.core.domain.budget import BudgetStatus
from xaytune.core.domain.candidate import CandidateSpec
from xaytune.core.domain.decision import DecisionOutcome
from xaytune.core.domain.objective import Objective
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import DecisionId, EvaluationId, ExperimentId, ExperimentNodeId
from xaytune.core.immutable import FrozenDict, FrozenDomainModel
from xaytune.core.observability import Finite
from xaytune.core.state.status import ExperimentNodeStatus, ExperimentStatus

__all__ = [
    "PLANNING_CONTEXT_IDENTITY_VERSION",
    "ActionProposal",
    "CandidateProposal",
    "DecisionSummary",
    "EvaluationSummary",
    "MetricSummary",
    "NodeSummary",
    "PlanningContext",
    "Proposal",
    "ProposalProvenance",
    "planning_context_identity_v1",
]

PLANNING_CONTEXT_IDENTITY_VERSION = 1


class MetricSummary(FrozenDomainModel):
    """One recorded metric, as far as planning reads it."""

    name: str
    value: Finite
    slice: str | None = None
    evaluator_name: str
    evaluator_version: str | None = None


class EvaluationSummary(FrozenDomainModel):
    """One recorded evaluation result of a node: which, for which cycle, what it measured."""

    evaluation_result_id: EvaluationId
    evaluation_cycle: int = Field(ge=1)
    metrics: tuple[MetricSummary, ...] = ()


class DecisionSummary(FrozenDomainModel):
    """One recorded decision about a node: what it established, on which evidence."""

    decision_id: DecisionId
    evaluation_cycle: int = Field(ge=1)
    outcome: DecisionOutcome
    engine_name: str
    engine_version: str
    input_fingerprint: str
    evaluation_result_ids: tuple[EvaluationId, ...] = ()


class NodeSummary(FrozenDomainModel):
    """One candidate of the experiment, from its node's durable record.

    ``candidate_fingerprint`` is recomputed under the **current** projection
    from the stored snapshot, never copied from the node's stored digest:
    that digest may be a v1 fingerprint, and v1 cannot see fields v2 can, so
    a v1 match would be weaker evidence of equivalence than it looks.
    """

    node_id: ExperimentNodeId
    status: ExperimentNodeStatus
    parent_ids: tuple[ExperimentNodeId, ...] = ()
    candidate: CandidateSpec
    candidate_fingerprint: str
    decisions: tuple[DecisionSummary, ...] = ()
    evaluations: tuple[EvaluationSummary, ...] = ()

    @model_validator(mode="after")
    def _fingerprint_is_current(self) -> NodeSummary:
        expected = self.candidate.candidate_fingerprint()
        if self.candidate_fingerprint != expected:
            raise ValueError(
                f"candidate_fingerprint {self.candidate_fingerprint!r} is not the current "
                f"identity of the candidate, {expected!r}"
            )
        return self

    @property
    def latest_decision(self) -> DecisionSummary | None:
        """The decision of the node's latest decided cycle, if it was decided."""
        if not self.decisions:
            return None
        return max(self.decisions, key=lambda decision: decision.evaluation_cycle)


class PlanningContext(FrozenDomainModel):
    """Everything a planner may depend on, assembled from the durable record.

    Explicit and serializable for the reason ``DecisionContext`` is: a planner
    that read the clock, the database or the environment could plan
    differently on inputs nobody changed. Topology comes from the
    repository's relationships (open question 15); nodes are ordered by id.
    """

    experiment_id: ExperimentId
    experiment_status: ExperimentStatus
    objective: Objective
    nodes: tuple[NodeSummary, ...] = ()
    budget: BudgetStatus | None = None

    @field_validator("nodes")
    @classmethod
    def _ordered_and_unique(cls, nodes: tuple[NodeSummary, ...]) -> tuple[NodeSummary, ...]:
        ids = [str(node.node_id) for node in nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("a node appears more than once in the planning context")
        return tuple(sorted(nodes, key=lambda node: str(node.node_id)))

    @property
    def candidate_fingerprints(self) -> frozenset[str]:
        """Every candidate the experiment already has, under the current identity."""
        return frozenset(node.candidate_fingerprint for node in self.nodes)

    def input_fingerprint(self) -> str:
        """The identity of this context: :func:`planning_context_identity_v1`, hashed."""
        return fingerprint(planning_context_identity_v1(self))


def planning_context_identity_v1(context: PlanningContext) -> Mapping[str, Any]:
    """What makes two planning contexts the same, version 1.

    An explicit projection rather than a dump, so a field added to a summary
    later does not change the identity of every context already planned on.
    It names what a planner plans from:

    - the experiment, its status and its objective;
    - each node: id, status, sorted parents and **current** candidate
      fingerprint (the candidate's identity, not its bytes);
    - each decision: id, cycle, outcome, engine and version, input
      fingerprint and the results it decided on;
    - each evaluation: result id, cycle and its metrics (name, value, slice,
      evaluator and version);
    - each enforced budget dimension: kind, limit and remaining.

    Collections are sorted, so the order they were read in is not identity.
    """

    def metric(m: MetricSummary) -> dict[str, Any]:
        return {
            "name": m.name,
            "value": m.value,
            "slice": m.slice,
            "evaluator_name": m.evaluator_name,
            "evaluator_version": m.evaluator_version,
        }

    def node(n: NodeSummary) -> dict[str, Any]:
        return {
            "node_id": str(n.node_id),
            "status": n.status.value,
            "parent_ids": sorted(str(parent) for parent in n.parent_ids),
            "candidate_fingerprint": n.candidate_fingerprint,
            "decisions": [
                {
                    "decision_id": str(d.decision_id),
                    "evaluation_cycle": d.evaluation_cycle,
                    "outcome": d.outcome.value,
                    "engine": [d.engine_name, d.engine_version],
                    "input_fingerprint": d.input_fingerprint,
                    "evaluation_result_ids": sorted(str(r) for r in d.evaluation_result_ids),
                }
                for d in sorted(n.decisions, key=lambda d: (d.evaluation_cycle, str(d.decision_id)))
            ],
            "evaluations": [
                {
                    "evaluation_result_id": str(e.evaluation_result_id),
                    "evaluation_cycle": e.evaluation_cycle,
                    "metrics": sorted(
                        (metric(m) for m in e.metrics),
                        key=lambda m: (m["name"], m["slice"] or "", m["evaluator_name"]),
                    ),
                }
                for e in sorted(
                    n.evaluations, key=lambda e: (e.evaluation_cycle, str(e.evaluation_result_id))
                )
            ],
        }

    objective = context.objective
    budget = context.budget
    return {
        "kind": "planning-context",
        "identity_version": PLANNING_CONTEXT_IDENTITY_VERSION,
        "experiment_id": str(context.experiment_id),
        "experiment_status": context.experiment_status.value,
        "objective": {
            "primary": {"name": objective.primary.name, "direction": objective.primary.direction},
            "target": objective.target,
            "constraints": sorted(
                (
                    {"name": c.name, "operator": c.operator, "value": c.value}
                    for c in objective.constraints
                ),
                key=lambda c: (c["name"], c["operator"], c["value"]),
            ),
        },
        "nodes": [node(n) for n in sorted(context.nodes, key=lambda n: str(n.node_id))],
        "budget": None
        if budget is None
        else [
            {
                "dimension": d.dimension.value,
                "kind": d.kind,
                "limit": _decimal(d.limit),
                "remaining": _decimal(d.remaining),
            }
            for d in sorted(budget.dimensions, key=lambda d: d.dimension.value)
        ],
    }


def _decimal(value: Decimal) -> str:
    """One spelling per amount: ``Decimal("1.0")`` and ``Decimal("1")`` are the same budget."""
    return format(value.normalize(), "f")


class ProposalProvenance(FrozenDomainModel):
    """Which planner proposed this, configured how, from which context.

    Carried on every proposal, so a proposal can be traced without a planning
    record: the planner's descriptor identity, the bound ``PlannerSpec`` it
    ran under, and the fingerprint of the context it saw.
    """

    planner_provider: str = Field(min_length=1)
    planner_name: str = Field(min_length=1)
    planner_version: str = Field(min_length=1)
    planner_spec_kind: str = Field(min_length=1)
    planner_spec_version: str = Field(min_length=1)
    context_identity_version: int = Field(ge=1)
    context_fingerprint: str = Field(min_length=1)


class CandidateProposal(FrozenDomainModel):
    """A new scientific candidate a planner proposes to explore next.

    Everything branching needs to materialize it later -- the full candidate,
    its parents, why -- and nothing it would have to mint: no node id, no
    timestamp. ``mutation`` describes the declared change in canonical JSON
    (which rule, from what, to what); it is rationale, not identity, and the
    candidate's identity is ``candidate_fingerprint``, checked against the
    candidate itself.
    """

    kind: Literal["candidate"] = "candidate"
    candidate: CandidateSpec
    candidate_fingerprint: str
    parent_ids: tuple[ExperimentNodeId, ...] = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    mutation: FrozenDict = Field(default_factory=FrozenDict)
    evidence_refs: tuple[str, ...] = ()
    provenance: ProposalProvenance

    @model_validator(mode="after")
    def _fingerprint_describes_candidate(self) -> CandidateProposal:
        expected = self.candidate.candidate_fingerprint()
        if self.candidate_fingerprint != expected:
            raise ValueError(
                f"candidate_fingerprint {self.candidate_fingerprint!r} does not describe the "
                f"candidate, which fingerprints as {expected!r}"
            )
        return self


class ActionProposal(FrozenDomainModel):
    """A typed action a planner proposes (spec 09 §5). Intent only, never executed here.

    ``action`` must be an instance of a registered :class:`ActionSpec` type:
    a dictionary, or the bare base class, is refused, so a planner cannot
    propose an action nothing can validate. Whether it is allowed is policy's
    to decide when it is proposed as a durable Action.
    """

    kind: Literal["action"] = "action"
    action: SerializeAsAny[ActionSpec]
    reason: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = ()
    provenance: ProposalProvenance

    @field_validator("action", mode="before")
    @classmethod
    def _a_typed_action(cls, value: Any) -> Any:
        if not isinstance(value, ActionSpec) or type(value) is ActionSpec:
            raise ValueError(
                "an action proposal carries a typed ActionSpec instance, never a mapping"
            )
        return value


Proposal = CandidateProposal | ActionProposal
"""What a planner may return: a candidate to explore, or a typed action."""
