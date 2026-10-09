"""What a search provider sees, and what it is told back (spec 09 §3, PR-034).

```text
PlanningContext ──(search planner)──▶ SearchContext ──SearchProvider.suggest()──▶ CandidateProposal
     node outcomes ──observation_of()──▶ CandidateObservation ──SearchProvider.observe()
```

**A search provider searches; Xaytune owns everything else** -- experiments,
candidate identity, lineage, execution, decisions and durability. A provider
proposes full candidates through the same :class:`CandidateProposal` every
planner uses, so branching (PR-025) governs them unchanged, and it never
mints an id, reads a clock, opens a database or holds a runtime handle.

Everything here is data with an explicit, versioned identity:

- :class:`SearchSpace` -- the typed parameters a search varies, each naming
  the field of the candidate it sets. Parameters are kept in name order, so
  the order they were declared in is not identity, and a search algorithm
  sees them in one order everywhere.
- :class:`SearchContext` -- the experiment as a search sees it: the base
  candidate the search varies, every candidate the experiment already has,
  and the provenance its proposals carry. Assembled from the durable record;
  it holds no handle.
- :class:`CandidateObservation` -- what became of one candidate the search
  proposed, derived from durable outcomes alone (:func:`observation_of`).
  ``measured`` is the only outcome that carries a value; a candidate with no
  unambiguous measurement of the objective's primary metric is
  ``unmeasured``, never zero.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from xaytune.core.domain.candidate import CandidateSpec
from xaytune.core.domain.objective import Objective
from xaytune.core.domain.planning import (
    CandidateBranchOrigin,
    EvidenceRef,
    NodeSummary,
    ProposalProvenance,
    branch_origin_identity,
    planning_candidate_projection_v1,
    provenance_identity,
)
from xaytune.core.errors import DomainError
from xaytune.core.fingerprint import canonical_encode, fingerprint
from xaytune.core.ids import ExperimentId, ExperimentNodeId
from xaytune.core.immutable import FrozenDict, FrozenDomainModel, thaw
from xaytune.core.observability import Finite
from xaytune.core.state.status import ExperimentNodeStatus

__all__ = [
    "CANDIDATE_OBSERVATION_IDENTITY_VERSION",
    "SEARCH_CONTEXT_IDENTITY_VERSION",
    "SEARCH_PROVIDER_SPEC_IDENTITY_VERSION",
    "SEARCH_SPACE_IDENTITY_VERSION",
    "CandidateObservation",
    "ChoiceParameter",
    "FloatParameter",
    "IntParameter",
    "ObservationOutcome",
    "Parameter",
    "SearchCandidate",
    "SearchContext",
    "SearchProviderSpec",
    "SearchSpace",
    "SearchSpaceError",
    "apply_parameters",
    "candidate_observation_identity_v1",
    "observation_of",
    "search_context_identity_v1",
    "search_provider_spec_identity_v1",
    "search_space_identity_v1",
]


class SearchSpaceError(DomainError, ValueError):
    """Parameter values do not make a valid candidate from the base, or are outside the space."""


# ---- the search space ---------------------------------------------------------------------

_SEGMENT = re.compile(r"^[a-z_][a-z0-9_]*$")
_NOT_IDENTITY = frozenset({"metadata"})
_NOT_IDENTITY_PATHS = frozenset({"training.api_version"})
"""Addressable fields candidate identity leaves out, besides any ``metadata``.

Varying one would make every value the same candidate. (A scheduled
intervention's ``rationale`` is not identity either, but sits inside a list,
which a path cannot address.)"""


class _Parameter(FrozenDomainModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)
    """How the search and its record call the parameter."""
    path: str = Field(min_length=1)
    """The field of the candidate it sets, dotted: ``training.optimization.learning_rate``."""

    @field_validator("path")
    @classmethod
    def _a_field_path(cls, path: str) -> str:
        segments = path.split(".")
        bad = [segment for segment in segments if not _SEGMENT.match(segment)]
        if bad:
            raise ValueError(f"path {path!r} has segments that are not field names: {bad}")
        if _NOT_IDENTITY.intersection(segments):
            raise ValueError(
                f"path {path!r} sets metadata, which is not candidate identity: every value "
                f"would be the same candidate"
            )
        if path in _NOT_IDENTITY_PATHS:
            raise ValueError(
                f"path {path!r} is not candidate identity: every value would be the same candidate"
            )
        return path


class FloatParameter(_Parameter):
    """A real number in ``[low, high]``, sampled uniformly or, with ``log``, log-uniformly."""

    type: Literal["float"] = "float"
    low: Finite
    high: Finite
    log: StrictBool = False

    @model_validator(mode="after")
    def _a_range(self) -> FloatParameter:
        if not self.low < self.high:
            raise ValueError(f"low {self.low!r} must be below high {self.high!r}")
        if self.log and self.low <= 0:
            raise ValueError("a log-scale range must be positive")
        return self

    def contains(self, value: Any) -> bool:
        return isinstance(value, float) and math.isfinite(value) and self.low <= value <= self.high


class IntParameter(_Parameter):
    """An integer in ``[low, high]``, both included; with ``log``, sampled log-uniformly."""

    type: Literal["int"] = "int"
    low: StrictInt
    high: StrictInt
    log: StrictBool = False

    @model_validator(mode="after")
    def _a_range(self) -> IntParameter:
        if not self.low < self.high:
            raise ValueError(f"low {self.low} must be below high {self.high}")
        if self.log and self.low <= 0:
            raise ValueError("a log-scale range must be positive")
        return self

    def contains(self, value: Any) -> bool:
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and self.low <= value <= self.high
        )


ChoiceValue = StrictBool | StrictInt | StrictFloat | StrictStr


class ChoiceParameter(_Parameter):
    """One of ``values``, each a JSON scalar of its own type: ``1`` and ``1.0`` are different."""

    type: Literal["choice"] = "choice"
    values: tuple[ChoiceValue, ...] = Field(min_length=2)

    @field_validator("values")
    @classmethod
    def _distinct_and_finite(cls, values: tuple[Any, ...]) -> tuple[Any, ...]:
        if any(isinstance(value, float) and not math.isfinite(value) for value in values):
            raise ValueError("a choice must be finite")
        encoded = [canonical_encode(value) for value in values]
        if len(set(encoded)) != len(encoded):
            raise ValueError("a choice appears more than once")
        return values

    def contains(self, value: Any) -> bool:
        if isinstance(value, float) and not math.isfinite(value):
            return False
        try:
            encoded = canonical_encode(value)
        except Exception:
            return False
        return any(canonical_encode(choice) == encoded for choice in self.values)


Parameter = Annotated[FloatParameter | IntParameter | ChoiceParameter, Field(discriminator="type")]
"""One searched parameter, by ``type``: ``float``, ``int`` or ``choice``."""


class SearchSpace(FrozenDomainModel):
    """The parameters a search varies, in name order.

    Each sets one field of the base candidate; no two set the same field, or
    one a field inside another's. Whether a path really is a field is checked
    against the base candidate when values are applied
    (:func:`apply_parameters`), since a ``params`` mapping has no schema.
    """

    parameters: tuple[Parameter, ...] = Field(min_length=1)

    @field_validator("parameters")
    @classmethod
    def _ordered_and_distinct(cls, parameters: tuple[Any, ...]) -> tuple[Any, ...]:
        names = [parameter.name for parameter in parameters]
        if len(set(names)) != len(names):
            raise ValueError("a parameter name appears more than once")
        paths = sorted(parameter.path for parameter in parameters)
        for shorter, longer in zip(paths, paths[1:], strict=False):
            if longer == shorter or longer.startswith(shorter + "."):
                raise ValueError(f"parameters set overlapping fields: {shorter!r} and {longer!r}")
        return tuple(sorted(parameters, key=lambda parameter: parameter.name))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    def fingerprint(self) -> str:
        """The identity of this space: :func:`search_space_identity_v1`, hashed."""
        return fingerprint(search_space_identity_v1(self))


SEARCH_SPACE_IDENTITY_VERSION = 1


def search_space_identity_v1(space: SearchSpace) -> Mapping[str, Any]:
    """What makes two search spaces the same, version 1: every parameter, in name order."""

    def parameter(p: FloatParameter | IntParameter | ChoiceParameter) -> dict[str, Any]:
        common = {"name": p.name, "path": p.path, "type": p.type}
        if isinstance(p, ChoiceParameter):
            return {**common, "values": list(p.values)}
        return {**common, "low": p.low, "high": p.high, "log": p.log}

    return {
        "kind": "search-space",
        "identity_version": SEARCH_SPACE_IDENTITY_VERSION,
        "parameters": [parameter(p) for p in space.parameters],
    }


def apply_parameters(
    base: CandidateSpec, space: SearchSpace, values: Mapping[str, Any]
) -> CandidateSpec:
    """*base* with each parameter's field set to its value: a full, validated candidate.

    Never trusts where the values came from. Exactly the space's parameters
    must be given, each inside its range or choice and of its own type; each
    path must name a field the base candidate has -- a key it already
    carries, so a ``params`` entry is searched only if the base declares it
    -- below fields that are set; and the result must validate as a
    :class:`CandidateSpec`. Everything else of the base is kept exactly.

    Raises:
        SearchSpaceError: Naming every problem.
    """
    problems = []
    if set(values) != set(space.names):
        problems.append(
            f"values are given for {sorted(values)}, but the space's parameters are "
            f"{sorted(space.names)}"
        )
        raise SearchSpaceError("; ".join(problems))
    document = base.model_dump(mode="json")
    for parameter in space.parameters:
        value = values[parameter.name]
        if not parameter.contains(value):
            problems.append(f"{parameter.name}: {value!r} is outside the parameter's domain")
            continue
        *parents, last = parameter.path.split(".")
        container: Any = document
        for segment in parents:
            container = container.get(segment) if isinstance(container, dict) else None
            if container is None:
                break
        if not isinstance(container, dict) or last not in container:
            problems.append(
                f"{parameter.name}: the base candidate has no field {parameter.path!r} to set"
            )
            continue
        container[last] = value
    if problems:
        raise SearchSpaceError("; ".join(problems))
    try:
        return CandidateSpec.model_validate(document)
    except ValidationError as invalid:
        raise SearchSpaceError(
            "the values do not make a valid candidate: "
            + "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in invalid.errors()
            )
        ) from None


# ---- provider specs -----------------------------------------------------------------------


class SearchProviderSpec(FrozenDomainModel):
    """Which search provider, configured how -- as a :class:`PlannerSpec` names a planner.

    ``version`` is ``None`` as a request and set once bound to the provider's
    ``plugin_version``. ``config`` is canonical JSON, validated into the
    provider's typed configuration when it is bound; the search space, the
    algorithm and its seed all live there, so they are all identity.
    """

    kind: str = Field(min_length=1)
    version: str | None = None
    config: FrozenDict = Field(default_factory=FrozenDict)


SEARCH_PROVIDER_SPEC_IDENTITY_VERSION = 1


def search_provider_spec_identity_v1(
    spec: SearchProviderSpec, *, provider: str, name: str, plugin_version: str, api_version: str
) -> Mapping[str, Any]:
    """What makes two bound search providers the same, version 1.

    The bound spec -- kind, resolved version, canonical configuration (search
    space, algorithm, seed) -- and the descriptor contract it was bound to.
    """
    return {
        "kind": "search-provider-spec",
        "identity_version": SEARCH_PROVIDER_SPEC_IDENTITY_VERSION,
        "spec": {"kind": spec.kind, "version": spec.version, "config": thaw(spec.config)},
        "descriptor": {
            "provider": provider,
            "name": name,
            "plugin_version": plugin_version,
            "api_version": api_version,
        },
    }


# ---- the context ---------------------------------------------------------------------------


class SearchCandidate(FrozenDomainModel):
    """A candidate the experiment already has: its node, its current identity, and why it exists.

    ``branch_origin`` is the proposal branching recorded for the node (PR-025),
    ``None`` for one that was not branched. A search accepts a candidate as one
    of its trials only if this proves the search proposed it -- a candidate
    that merely equals what the search would suggest is not its history.
    """

    node_id: ExperimentNodeId
    candidate_fingerprint: str = Field(min_length=1)
    branch_origin: CandidateBranchOrigin | None = None


class SearchContext(FrozenDomainModel):
    """Everything a search provider may depend on, assembled from the durable record.

    - ``base`` is the candidate the search varies, and the parent of every
      candidate it proposes; its fingerprint is recomputed, never trusted.
    - ``candidates`` is every candidate the experiment has, the base among
      them, in node order, each with its branch origin: what must never be
      proposed as new, and the record a search proves its history from.
    - ``provenance`` is what every proposal made from this context carries,
      issued by whoever assembled it -- the search planner, under the
      experiment's recorded planner spec. A provider copies it; it never
      makes one.
    """

    experiment_id: ExperimentId
    objective: Objective
    base_node_id: ExperimentNodeId
    base: CandidateSpec
    candidates: tuple[SearchCandidate, ...]
    provenance: ProposalProvenance

    @field_validator("candidates")
    @classmethod
    def _ordered_and_unique(
        cls, candidates: tuple[SearchCandidate, ...]
    ) -> tuple[SearchCandidate, ...]:
        ids = [str(candidate.node_id) for candidate in candidates]
        fingerprints = [candidate.candidate_fingerprint for candidate in candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("a node appears more than once in the search context")
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("a candidate appears more than once in the search context")
        return tuple(sorted(candidates, key=lambda candidate: str(candidate.node_id)))

    @model_validator(mode="after")
    def _the_base_is_a_candidate(self) -> SearchContext:
        expected = (str(self.base_node_id), self.base.candidate_fingerprint())
        if expected not in {(str(c.node_id), c.candidate_fingerprint) for c in self.candidates}:
            raise ValueError(
                f"the base node {self.base_node_id}, as {expected[1]}, is not among the "
                f"experiment's candidates"
            )
        return self

    @property
    def origins_by_fingerprint(self) -> Mapping[str, CandidateBranchOrigin | None]:
        return {c.candidate_fingerprint: c.branch_origin for c in self.candidates}

    @property
    def base_fingerprint(self) -> str:
        return self.base.candidate_fingerprint()

    @property
    def nodes_by_fingerprint(self) -> Mapping[str, ExperimentNodeId]:
        return {c.candidate_fingerprint: c.node_id for c in self.candidates}

    def input_fingerprint(self) -> str:
        """The identity of this context: :func:`search_context_identity_v1`, hashed."""
        return fingerprint(search_context_identity_v1(self))


SEARCH_CONTEXT_IDENTITY_VERSION = 1


def search_context_identity_v1(context: SearchContext) -> Mapping[str, Any]:
    """What makes two search contexts the same, version 1.

    The experiment and its objective; the base node and everything a planner
    can see of its candidate (:func:`planning_candidate_projection_v1`, since
    every proposal carries the base's fields beyond identity too); every
    candidate by node, current fingerprint and branch origin, in node order;
    and the full
    provenance proposals from it carry. A test pins the context's schema
    against this list.
    """
    objective = context.objective
    return {
        "kind": "search-context",
        "identity_version": SEARCH_CONTEXT_IDENTITY_VERSION,
        "experiment_id": str(context.experiment_id),
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
        "base": {
            "node_id": str(context.base_node_id),
            "candidate_fingerprint": context.base_fingerprint,
            "candidate": planning_candidate_projection_v1(context.base),
        },
        "candidates": [
            {
                "node_id": str(c.node_id),
                "candidate_fingerprint": c.candidate_fingerprint,
                "branch_origin": branch_origin_identity(c.branch_origin),
            }
            for c in context.candidates
        ],
        "provenance": provenance_identity(context.provenance),
    }


# ---- observations -------------------------------------------------------------------------

ObservationOutcome = Literal["measured", "unmeasured", "rejected", "failed", "cancelled"]
"""What became of a proposed candidate, as far as a search is concerned.

```text
measured     COMPLETED, its latest decision decided on exactly one unsliced
             measurement of the objective's primary metric -- the value
unmeasured   COMPLETED without that: no decision, no measurement, or more than
             one -- no value, and never zero
rejected     REJECTED on its merits: it violated a constraint
failed       FAILED: it could not be trained or evaluated
cancelled    CANCELLED: stopped before it could be judged
```

Only ``measured`` carries a value. The rest are explicit so an algorithm can
tell "this point is bad" from "nothing was learned about this point"; each
adapter documents how it reports them.
"""

_ENDED: Mapping[ExperimentNodeStatus, ObservationOutcome] = {
    ExperimentNodeStatus.REJECTED: "rejected",
    ExperimentNodeStatus.FAILED: "failed",
    ExperimentNodeStatus.CANCELLED: "cancelled",
}


class CandidateObservation(FrozenDomainModel):
    """What became of one candidate a search proposed, from the durable record.

    ``metric`` is the objective's primary metric the observation is about;
    ``value`` is set exactly when ``outcome`` is ``measured``. Evidence names
    the decision and evaluation result the outcome rests on, sorted.
    """

    node_id: ExperimentNodeId
    candidate_fingerprint: str = Field(min_length=1)
    outcome: ObservationOutcome
    metric: str = Field(min_length=1)
    value: Finite | None = None
    evidence_refs: tuple[EvidenceRef, ...] = ()

    @field_validator("evidence_refs")
    @classmethod
    def _sorted(cls, refs: tuple[EvidenceRef, ...]) -> tuple[EvidenceRef, ...]:
        return tuple(sorted(set(refs), key=lambda ref: (ref.kind, ref.id)))

    @model_validator(mode="after")
    def _a_value_only_when_measured(self) -> CandidateObservation:
        if (self.outcome == "measured") != (self.value is not None):
            raise ValueError("an observation carries a value exactly when it was measured")
        return self

    def fingerprint(self) -> str:
        """The identity of this observation: :func:`candidate_observation_identity_v1`, hashed."""
        return fingerprint(candidate_observation_identity_v1(self))


CANDIDATE_OBSERVATION_IDENTITY_VERSION = 1


def candidate_observation_identity_v1(observation: CandidateObservation) -> Mapping[str, Any]:
    """What makes two observations the same, version 1: every field."""
    return {
        "kind": "candidate-observation",
        "identity_version": CANDIDATE_OBSERVATION_IDENTITY_VERSION,
        "node_id": str(observation.node_id),
        "candidate_fingerprint": observation.candidate_fingerprint,
        "outcome": observation.outcome,
        "metric": observation.metric,
        "value": observation.value,
        "evidence_refs": [{"kind": r.kind, "id": r.id} for r in observation.evidence_refs],
    }


def observation_of(node: NodeSummary, objective: Objective) -> CandidateObservation | None:
    """What *node*'s durable record says became of it; ``None`` while it is still in flight.

    Pure: the node's status, its latest decision and the evaluation results
    that decision decided on. The value is read as the planner reads a
    parent's (exactly one unsliced measurement of the primary metric among
    the decided results); anything else is ``unmeasured``.
    """
    primary = objective.primary.name
    decision = node.latest_decision
    evidence: list[EvidenceRef] = []
    if decision is not None:
        evidence.append(EvidenceRef(kind="decision", id=str(decision.decision_id)))

    def observed(outcome: ObservationOutcome, value: float | None = None) -> CandidateObservation:
        return CandidateObservation(
            node_id=node.node_id,
            candidate_fingerprint=node.candidate_fingerprint,
            outcome=outcome,
            metric=primary,
            value=value,
            evidence_refs=tuple(evidence),
        )

    if node.status in _ENDED:
        return observed(_ENDED[node.status])
    if node.status is not ExperimentNodeStatus.COMPLETED:
        return None
    if decision is None:
        return observed("unmeasured")
    decided = set(decision.evaluation_result_ids)
    measured = [
        (metric.value, str(evaluation.evaluation_result_id))
        for evaluation in node.evaluations
        if evaluation.evaluation_result_id in decided
        for metric in evaluation.metrics
        if metric.name == primary and metric.slice is None
    ]
    if len(measured) != 1:
        return observed("unmeasured")
    ((value, result_id),) = measured
    evidence.append(EvidenceRef(kind="evaluation-result", id=result_id))
    return observed("measured", value)
