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

from xaytune.core.domain.action import UnknownActionTypeError
from xaytune.core.domain.actions.contract import ActionSpec, action_descriptor
from xaytune.core.domain.budget import BudgetStatus
from xaytune.core.domain.candidate import CandidateSpec, candidate_identity_v2
from xaytune.core.domain.decision import DecisionOutcome
from xaytune.core.domain.objective import Objective
from xaytune.core.domain.specs import PlannerSpec
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import (
    AgentInvocationId,
    DecisionId,
    EvaluationId,
    ExperimentId,
    ExperimentNodeId,
)
from xaytune.core.immutable import FrozenDict, FrozenDomainModel, thaw
from xaytune.core.observability import Finite
from xaytune.core.state.status import ExperimentNodeStatus, ExperimentStatus

__all__ = [
    "ACTION_PROPOSAL_IDENTITY_VERSION",
    "CANDIDATE_PROPOSAL_IDENTITY_VERSION",
    "CandidateBranchOrigin",
    "EvidenceRef",
    "action_proposal_identity_v1",
    "candidate_proposal_identity_v1",
    "PLANNER_SPEC_IDENTITY_VERSION",
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
    "planning_candidate_projection_v1",
    "settled_for_planning",
    "planner_spec_identity_v1",
    "planning_context_identity_v1",
]

PLANNING_CONTEXT_IDENTITY_VERSION = 1


def planning_candidate_projection_v1(candidate: CandidateSpec) -> Mapping[str, Any]:
    """Everything a planner can see of a candidate, as an explicit versioned projection.

    The candidate's identity (``candidate_identity_v2``) deliberately leaves
    out fields that do not bear scientific identity -- ``metadata`` at every
    level, the training spec's ``api_version``, a scheduled intervention's
    ``rationale``. A planner still sees them, and a mutation carries them
    into the candidate it proposes, so two contexts differing only there are
    different planner inputs. This projection is the identity plus each of
    those fields, named one by one: a field added to the candidate later is
    a deliberate choice to project, not a silent change of every context's
    identity. A test pins the candidate's schema against this list.
    """
    model, data, training = candidate.model, candidate.data, candidate.training
    schedule = candidate.schedule
    return {
        "identity": candidate_identity_v2(candidate),
        "beyond_identity": {
            "metadata": thaw(candidate.metadata),
            "model": {"metadata": thaw(model.metadata), "ref_metadata": thaw(model.model.metadata)},
            "data": {
                "metadata": thaw(data.metadata),
                "dataset_metadata": thaw(data.dataset.metadata),
            },
            "training": {
                "api_version": training.api_version,
                "metadata": thaw(training.metadata),
                "adapter_metadata": None
                if training.adapter is None
                else thaw(training.adapter.metadata),
                "optimization_metadata": thaw(training.optimization.metadata),
            },
            "reward_metadata": None
            if candidate.reward is None
            else thaw(candidate.reward.metadata),
            "environment_metadata": None
            if candidate.environment is None
            else thaw(candidate.environment.metadata),
            "schedule_rationales": None
            if schedule is None
            else [{"id": item.id, "rationale": item.rationale} for item in schedule.interventions],
        },
    }


_SETTLED_FOR_PLANNING: frozenset[tuple[ExperimentNodeStatus, DecisionOutcome]] = frozenset(
    {
        (ExperimentNodeStatus.COMPLETED, DecisionOutcome.BRANCH),
        (ExperimentNodeStatus.REJECTED, DecisionOutcome.REJECT),
    }
)


def settled_for_planning(
    status: ExperimentNodeStatus, latest_outcome: DecisionOutcome | None
) -> bool:
    """Whether a candidate was decided on its merits in a way that leaves room for another.

    The status alone does not say: ``COMPLETED`` is also what ``STOP_SUCCEEDED``
    leaves, and ``REJECTED`` what ``STOP_FAILED`` does. So the latest decision
    must agree -- ``COMPLETED`` by ``BRANCH``, ``REJECTED`` by ``REJECT``. A
    ``STOP_*`` decision says the experiment should end (one decided while it
    was paused is not applied, and a resume does not undo that), and a settled
    node with no decision was not settled by one; neither invites planning.
    """
    return (status, latest_outcome) in _SETTLED_FOR_PLANNING


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
    - each node: id, status, sorted parents, its **current** candidate
      fingerprint, and everything the planner can see of the candidate
      (:func:`planning_candidate_projection_v1`);
    - each decision: id, cycle, outcome, engine and version, input
      fingerprint and the results it decided on;
    - each evaluation: result id, cycle and its metrics (name, value, slice,
      evaluator and version);
    - each enforced budget dimension, every balance it carries: limit,
      reserved, committed, consumed, released, outstanding and remaining.

    Every field a planner can read is here; a test pins the context's
    schema against it, so a field added later cannot escape the identity.

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
            "candidate": planning_candidate_projection_v1(n.candidate),
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
                "reserved": _decimal(d.reserved),
                "committed": _decimal(d.committed),
                "consumed": _decimal(d.consumed),
                "released": _decimal(d.released),
                "outstanding": _decimal(d.outstanding),
                "remaining": _decimal(d.remaining),
            }
            for d in sorted(budget.dimensions, key=lambda d: d.dimension.value)
        ],
    }


def _decimal(value: Decimal) -> str:
    """One spelling per amount: ``Decimal("1.0")`` and ``Decimal("1")`` are the same budget."""
    return format(value.normalize(), "f")


class EvidenceRef(FrozenDomainModel):
    """A durable record a proposal rests on, by kind and id -- checkable, not decorative.

    ``decision`` names a :class:`~xaytune.core.domain.decision.Decision`,
    ``evaluation-result`` an
    :class:`~xaytune.core.domain.evaluation.EvaluationResult`. Branching
    verifies each one exists and belongs to the proposal's parents, so
    provenance can never cite evidence from another experiment.
    """

    kind: Literal["decision", "evaluation-result"]
    id: str = Field(min_length=1)


PLANNER_SPEC_IDENTITY_VERSION = 1


def planner_spec_identity_v1(
    spec: PlannerSpec, *, provider: str, name: str, plugin_version: str, api_version: str
) -> Mapping[str, Any]:
    """What makes two bound planners the same, version 1.

    The bound spec -- kind, resolved version and canonical configuration --
    and the descriptor contract it was bound to: provider, name, plugin
    version and plugin API version. Two planners of one kind and version
    configured differently behave differently, so the configuration is
    identity (ADR-016: the spec is what appears in provenance).
    """
    return {
        "kind": "planner-spec",
        "identity_version": PLANNER_SPEC_IDENTITY_VERSION,
        "spec": {"kind": spec.kind, "version": spec.version, "config": thaw(spec.config)},
        "descriptor": {
            "provider": provider,
            "name": name,
            "plugin_version": plugin_version,
            "api_version": api_version,
        },
    }


class ProposalProvenance(FrozenDomainModel):
    """Which planner proposed this, configured how, from which context.

    Carried on every proposal, so a proposal can be traced without a planning
    record: the planner's descriptor contract, the bound ``PlannerSpec`` it ran
    under -- configuration included, by its fingerprint
    (:func:`planner_spec_identity_v1`) -- and the fingerprint of the context
    it saw (:func:`planning_context_identity_v1`).
    """

    planner_provider: str = Field(min_length=1)
    planner_name: str = Field(min_length=1)
    planner_version: str = Field(min_length=1)
    planner_api_version: str = Field(min_length=1)
    planner_spec_kind: str = Field(min_length=1)
    planner_spec_version: str = Field(min_length=1)
    planner_spec_identity_version: int = Field(ge=1)
    planner_spec_fingerprint: str = Field(min_length=1)
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
    evidence_refs: tuple[EvidenceRef, ...] = ()
    provenance: ProposalProvenance

    def proposal_fingerprint(self) -> str:
        """The identity of this proposal: :func:`candidate_proposal_identity_v1`, hashed."""
        return fingerprint(candidate_proposal_identity_v1(self))

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

    ``agent_invocation_id`` names the recorded model call a model-backed
    planner derived the proposal from (PR-032); ``None`` for a planner that
    asks no model. It is the record's identity, not the proposal's, so it is
    not in :func:`action_proposal_identity_v1`.
    """

    kind: Literal["action"] = "action"
    action: SerializeAsAny[ActionSpec]
    reason: str = Field(min_length=1)
    evidence_refs: tuple[EvidenceRef, ...] = ()
    provenance: ProposalProvenance
    agent_invocation_id: AgentInvocationId | None = None

    def proposal_fingerprint(self) -> str:
        """The identity of this proposal: :func:`action_proposal_identity_v1`, hashed."""
        return fingerprint(action_proposal_identity_v1(self))

    @field_validator("action", mode="before")
    @classmethod
    def _a_registered_action(cls, value: Any) -> Any:
        if not isinstance(value, ActionSpec) or type(value) is ActionSpec:
            raise ValueError(
                "an action proposal carries a typed ActionSpec instance, never a mapping"
            )
        action_type = type(value).model_fields["type"].default
        try:
            descriptor = action_descriptor(action_type, value.version)
        except UnknownActionTypeError as unknown:
            raise ValueError(
                f"{type(value).__name__} ({action_type} v{value.version}) is not a registered "
                f"action type"
            ) from unknown
        if descriptor.spec is not type(value):
            raise ValueError(
                f"{type(value).__name__} is not the schema registered for {action_type} "
                f"v{value.version} ({descriptor.spec.__name__})"
            )
        return value


Proposal = CandidateProposal | ActionProposal
"""What a planner may return: a candidate to explore, or a typed action."""


CANDIDATE_PROPOSAL_IDENTITY_VERSION = 1
ACTION_PROPOSAL_IDENTITY_VERSION = 1


def action_proposal_identity_v1(proposal: ActionProposal) -> Mapping[str, Any]:
    """What makes two action proposals the same, version 1.

    The action -- type, schema version, target and canonical parameters --
    the reason, the evidence sorted by kind and id, and the full provenance.
    Not ``agent_invocation_id``: a recorded invocation stores this
    fingerprint of the proposal it produced, and the proposal names the
    invocation, so the reference cannot be part of what it refers to.
    """
    action = proposal.action
    provenance = proposal.provenance
    return {
        "kind": "action-proposal",
        "identity_version": ACTION_PROPOSAL_IDENTITY_VERSION,
        "action": {
            "type": type(action).model_fields["type"].default,
            "version": action.version,
            "target": {"kind": action.target.kind, "id": action.target.id},
            "parameters": action.parameters(),
        },
        "reason": proposal.reason,
        "evidence_refs": sorted(
            ({"kind": ref.kind, "id": ref.id} for ref in proposal.evidence_refs),
            key=lambda ref: (ref["kind"], ref["id"]),
        ),
        "provenance": {
            "planner_provider": provenance.planner_provider,
            "planner_name": provenance.planner_name,
            "planner_version": provenance.planner_version,
            "planner_api_version": provenance.planner_api_version,
            "planner_spec_kind": provenance.planner_spec_kind,
            "planner_spec_version": provenance.planner_spec_version,
            "planner_spec_identity_version": provenance.planner_spec_identity_version,
            "planner_spec_fingerprint": provenance.planner_spec_fingerprint,
            "context_identity_version": provenance.context_identity_version,
            "context_fingerprint": provenance.context_fingerprint,
        },
    }


def candidate_proposal_identity_v1(proposal: CandidateProposal) -> Mapping[str, Any]:
    """What makes two candidate proposals the same, version 1.

    The durable idempotency and provenance key of a branch: materializing the
    same proposal twice returns the node it made, and a different proposal
    for a candidate the experiment already has is a conflict. It binds:

    - the candidate -- its current fingerprint and everything a planner can
      see of it (:func:`planning_candidate_projection_v1`), since the node
      stores the candidate whole;
    - the parents, sorted;
    - hypothesis, reason and the mutation description;
    - the evidence, sorted by kind and id;
    - the full provenance: planner identity, bound spec fingerprint and the
      context fingerprint it was planned against.
    """
    provenance = proposal.provenance
    return {
        "kind": "candidate-proposal",
        "identity_version": CANDIDATE_PROPOSAL_IDENTITY_VERSION,
        "candidate_fingerprint": proposal.candidate_fingerprint,
        "candidate": planning_candidate_projection_v1(proposal.candidate),
        "parent_ids": sorted(str(parent) for parent in proposal.parent_ids),
        "hypothesis": proposal.hypothesis,
        "reason": proposal.reason,
        "mutation": thaw(proposal.mutation),
        "evidence_refs": sorted(
            ({"kind": ref.kind, "id": ref.id} for ref in proposal.evidence_refs),
            key=lambda ref: (ref["kind"], ref["id"]),
        ),
        "provenance": {
            "planner_provider": provenance.planner_provider,
            "planner_name": provenance.planner_name,
            "planner_version": provenance.planner_version,
            "planner_api_version": provenance.planner_api_version,
            "planner_spec_kind": provenance.planner_spec_kind,
            "planner_spec_version": provenance.planner_spec_version,
            "planner_spec_identity_version": provenance.planner_spec_identity_version,
            "planner_spec_fingerprint": provenance.planner_spec_fingerprint,
            "context_identity_version": provenance.context_identity_version,
            "context_fingerprint": provenance.context_fingerprint,
        },
    }


class CandidateBranchOrigin(FrozenDomainModel):
    """Why a branched node exists: the proposal it materializes, durably (PR-025).

    Kept on the node, so it survives the in-memory proposal. Fields the node
    already holds authoritatively -- candidate, parents, hypothesis, reason,
    creator -- are not repeated; this adds what only the proposal knew: its
    identity, who proposed it under which configuration from which context
    (``provenance``, whose ``context_fingerprint`` names that context), what
    changed (``mutation``) and on what evidence.
    """

    proposal_identity_version: int = Field(ge=1)
    proposal_fingerprint: str = Field(min_length=1)
    provenance: ProposalProvenance
    mutation: FrozenDict = Field(default_factory=FrozenDict)
    evidence_refs: tuple[EvidenceRef, ...] = ()

    @classmethod
    def of(cls, proposal: CandidateProposal) -> CandidateBranchOrigin:
        return cls(
            proposal_identity_version=CANDIDATE_PROPOSAL_IDENTITY_VERSION,
            proposal_fingerprint=proposal.proposal_fingerprint(),
            provenance=proposal.provenance,
            mutation=proposal.mutation,
            evidence_refs=proposal.evidence_refs,
        )
