"""What becomes of an evaluated candidate, and why -- recorded, never re-derived.

```text
DecisionContext     the objective and one evaluation cycle's results, from the record
      ↓ DecisionEngine.decide()   pure: no clock, no id, no database, no runtime
DecisionProposal    the outcome, the evidence it rests on, and which engine said so
      ↓ recorded -- given an id, a time and an actor -- in one commit
Decision            with the transitions the outcome causes:
                      STOP_SUCCEEDED   node COMPLETED, experiment SUCCEEDED
                      STOP_FAILED      node REJECTED,  experiment FAILED
                      REJECT           node REJECTED   (the experiment goes on)
```

A decision is a historical fact, like an evaluation result. It is written
once, in the same transaction that applies it, and is never edited: a node
that is evaluated again is decided again, in a new cycle, by a new decision.

``input_fingerprint`` is what makes a decision auditable. It hashes the
objective and the evidence-bearing fields of the exact results the engine
saw (:func:`decision_input_identity_v1`), so it can be shown later that two
executions of an engine had the same inputs -- and a controller that decides
a cycle twice, after a restart, can tell the second attempt apart from a
*different* decision about the same cycle.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import Field

from xaytune.core.clock import utc_now
from xaytune.core.domain.evaluation import EvaluationResult
from xaytune.core.domain.objective import ConstraintOperator, Objective
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import DecisionId, EvaluationId, ExperimentId, ExperimentNodeId
from xaytune.core.immutable import FrozenDomainModel
from xaytune.core.observability import Finite
from xaytune.core.refs import Actor
from xaytune.core.state.status import ExperimentNodeStatus

__all__ = [
    "Decision",
    "DecisionContext",
    "DecisionOutcome",
    "DecisionProposal",
    "MetricEvidence",
    "decision_input_identity_v1",
]


class DecisionOutcome(str, Enum):
    """What a decision does with a candidate. The vocabulary the engine uses so far.

    The specification names more -- ``CONTINUE_CURRENT``, ``EVALUATE_MORE``,
    ``BRANCH``, ``PROMOTE``, ``PAUSE``, ``STOP_BUDGET`` -- and each arrives
    with the machinery that can act on it. Until then an outcome nothing
    could carry out would be a record of an intention, not a decision.
    """

    STOP_SUCCEEDED = "stop_succeeded"
    """The objective's target is met and no constraint is violated."""

    STOP_FAILED = "stop_failed"
    """No constraint is violated, but the target is not met: the experiment ends unsuccessfully."""

    REJECT = "reject"
    """A constraint is violated: this candidate is unacceptable, whatever its objective.

    A judgement on the candidate, not the experiment. The experiment stays
    ``ACTIVE``: another candidate may still be proposed and succeed.
    """

    @property
    def node_status(self) -> ExperimentNodeStatus:
        """Where the decision leaves the candidate."""
        if self is DecisionOutcome.STOP_SUCCEEDED:
            return ExperimentNodeStatus.COMPLETED
        return ExperimentNodeStatus.REJECTED


class DecisionContext(FrozenDomainModel):
    """Everything a decision may depend on, assembled from the durable record.

    Explicit and serializable, for the reason ``CompilationContext`` is: an
    engine that read anything else -- the clock, the database, the
    environment -- could decide differently on inputs nobody changed, and
    ``input_fingerprint`` would no longer describe what was decided on.

    ``results`` are the node's results **for this evaluation cycle only**.
    Results of an earlier cycle describe an earlier round and never decide
    this one (ADR-015 §5).
    """

    experiment_id: ExperimentId
    node_id: ExperimentNodeId
    evaluation_cycle: int = Field(ge=1)
    objective: Objective
    results: tuple[EvaluationResult, ...] = Field(default_factory=tuple)

    def input_fingerprint(self) -> str:
        """The identity of these inputs: :func:`decision_input_identity_v1`, hashed."""
        return fingerprint(decision_input_identity_v1(self))


def decision_input_identity_v1(context: DecisionContext) -> Mapping[str, Any]:
    """What makes two decision inputs the same, version 1.

    An explicit projection, not a dump, for the reason
    ``candidate_identity_v1`` and ``evaluation_identity_v1`` are: a field
    added to ``Objective``, ``EvaluationResult`` or ``MetricResult`` later
    must not silently change the identity of every decision already recorded.
    So it names the fields that are evidence -- and only those:

    - the candidate and the cycle;
    - the objective: primary metric and direction, target, and constraints;
    - each result: its id, run, evaluation fingerprint and subject (id and
      digest);
    - each metric: name, value, slice, evaluator and version, seed, sample
      count, confidence interval and standard error.

    Not a result's timestamp, its report artifacts or a metric's free-form
    metadata, which describe rather than decide. Results, metrics and
    constraints are sorted, so the order they were read in is not identity.
    """

    def metric(m: Any) -> dict[str, Any]:
        return {
            "name": m.name,
            "value": m.value,
            "slice": m.slice,
            "evaluator_name": m.evaluator_name,
            "evaluator_version": m.evaluator_version,
            "seed": m.seed,
            "sample_count": m.sample_count,
            "confidence_interval": (
                list(m.confidence_interval) if m.confidence_interval is not None else None
            ),
            "standard_error": m.standard_error,
        }

    def result(r: EvaluationResult) -> dict[str, Any]:
        metrics = [metric(m) for m in r.metrics]
        return {
            "id": str(r.id),
            "evaluation_run_id": str(r.evaluation_run_id),
            "evaluation_fingerprint": r.evaluation_fingerprint,
            "subject": {"id": str(r.subject.id), "digest": r.subject.digest},
            "metrics": sorted(metrics, key=_canonical),
        }

    objective = context.objective
    constraints = [
        {"name": c.name, "operator": c.operator, "value": c.value} for c in objective.constraints
    ]
    return {
        "kind": "decision-input",
        "identity_version": 1,
        "experiment_id": str(context.experiment_id),
        "node_id": str(context.node_id),
        "evaluation_cycle": context.evaluation_cycle,
        "objective": {
            "primary": {"name": objective.primary.name, "direction": objective.primary.direction},
            "target": objective.target,
            "constraints": sorted(constraints, key=_canonical),
        },
        "results": [result(r) for r in sorted(context.results, key=lambda r: str(r.id))],
    }


def _canonical(value: Mapping[str, Any]) -> str:
    """A total order for mappings of plain values: their canonical JSON."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class MetricEvidence(FrozenDomainModel):
    """One comparison a decision rests on: which value, against what, and whether it held."""

    metric: str
    role: Literal["objective", "constraint"]
    value: Finite
    operator: ConstraintOperator
    threshold: Finite
    satisfied: bool
    evaluation_result_id: EvaluationId
    evaluator_name: str
    evaluator_version: str | None = None


class DecisionProposal(FrozenDomainModel):
    """What an engine decided: a pure function of its context and the engine.

    No id, no timestamp, no actor: those belong to the record of a decision,
    and an engine that minted them would give two different answers to one
    question. The repository adds them when it records the proposal, so
    deciding the same context twice yields byte-identical proposals.
    """

    experiment_id: ExperimentId
    node_id: ExperimentNodeId
    evaluation_cycle: int = Field(ge=1)

    outcome: DecisionOutcome
    reason: str = Field(min_length=1)
    evidence: tuple[MetricEvidence, ...] = Field(min_length=1)

    engine_name: str
    engine_version: str
    evaluation_result_ids: tuple[EvaluationId, ...] = Field(min_length=1)
    input_fingerprint: str


class Decision(DecisionProposal):
    """A proposal, recorded: one decision about one candidate for one cycle. Immutable.

    Whether a cycle was already decided is answered by the node, the cycle
    and ``input_fingerprint`` -- never by ``id``, which is the record's.
    """

    id: DecisionId = Field(default_factory=DecisionId.generate)
    actor: Actor
    created_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def record(cls, proposal: DecisionProposal, *, actor: Actor) -> Decision:
        """The durable decision for *proposal*: its content, with an id, an actor and a time."""
        return cls(**dict(proposal), actor=actor)

    def proposal(self) -> DecisionProposal:
        """What was decided, without what identifies the record of it."""
        return DecisionProposal.model_validate(
            self.model_dump(exclude={"id", "actor", "created_at"})
        )
