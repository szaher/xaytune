"""Harness search and mutation: the contracts (H07, ADR-024 §P2–§P12a).

```text
HarnessPatch (typed v1 operations on a pinned base) ─apply_patch→ child AgentHarnessSpec
HarnessSearchPolicy   autonomous_search: authorized, tools, operations, ceilings, envelope
HarnessSearchContext  what a search provider may see: provider feedback and provider-safe
                      comparisons only (§P4, §P6)
MutationRequest       what a mutation provider may see about one base (§P4)
HarnessCandidateProposal  the child, its patch, both providers as bound, every feedback ref
verify_harness_proposal   §P7's four checks, plus duplicates
SpendingLedger        §P12a: every unit of work reserved before it starts
```

**Everything here is pure.** Nothing runs a model, reads a store or mints an
id; the plugins that select and mutate, and the local loop that runs
candidates, live in :mod:`xaytune.harness.search`.

**A patch grants nothing.** ``set-enabled-tools`` only narrows: the child's
enabled set is a subset of the base's enabled set *and* of the experiment's
authorized tools (``child.enabled ⊆ parent.enabled ∩ authorized_tools``). A
tool the base defines but has disabled stays disabled; enabling it is a
grant, and a grant is an Action (§P7).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Annotated, Any, Final, Literal

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from xaytune.core.domain.harness import (
    AgentHarnessSpec,
    ContextSelection,
    HarnessContent,
    PromptTemplate,
    harness_fingerprint,
)
from xaytune.core.domain.planning import EvidenceRef, ProposalProvenance
from xaytune.core.fingerprint import fingerprint
from xaytune.core.harness_evaluation import Digest, ProviderFeedback
from xaytune.core.immutable import FrozenDict, FrozenDomainModel, thaw

__all__ = [
    "HARNESS_PATCH_SCHEMA",
    "HARNESS_SEARCH_POLICY_SCHEMA",
    "OPERATIONS",
    "FeedbackRef",
    "HarnessCandidateProposal",
    "HarnessCeilings",
    "HarnessFeedback",
    "HarnessMutationCapabilities",
    "HarnessNodeView",
    "HarnessOperation",
    "HarnessPatch",
    "HarnessProposalRefusedError",
    "HarnessProviderSpec",
    "HarnessSearchCapabilities",
    "HarnessSearchContext",
    "HarnessSearchPolicy",
    "LocalSearchEnvelope",
    "MutationRequest",
    "NoChoice",
    "OperationName",
    "PatchRefusedError",
    "ProviderComparison",
    "ProviderMetricComparison",
    "ProviderProvenance",
    "Reservation",
    "SearchChoice",
    "SetContextPolicy",
    "SetEnabledTools",
    "SetStoppingLimits",
    "SetSystemInstructions",
    "SetTaskTemplate",
    "SetToolDescription",
    "Spend",
    "SpendingExceededError",
    "SpendingLedger",
    "SpendingLimitError",
    "SpendingUnenforceableError",
    "apply_patch",
    "attempt_reservation",
    "ceiling_problems",
    "envelope_refusals",
    "verify_harness_proposal",
]

HARNESS_PATCH_SCHEMA: Final = "xaytune.harness-patch/v1"
HARNESS_SEARCH_POLICY_SCHEMA: Final = "xaytune.harness-search-policy/v1"

OperationName = Literal[
    "set-context-policy",
    "set-enabled-tools",
    "set-stopping-limits",
    "set-system-instructions",
    "set-task-template",
    "set-tool-description",
]
OPERATIONS: Final[tuple[str, ...]] = (
    "set-context-policy",
    "set-enabled-tools",
    "set-stopping-limits",
    "set-system-instructions",
    "set-task-template",
    "set-tool-description",
)
"""The v1 operations, sorted. A surface is an operation: allowed surfaces name them."""


class _Closed(FrozenDomainModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


def _sorted_unique(values: Iterable[str], what: str) -> tuple[str, ...]:
    values = tuple(values)
    if list(values) != sorted(set(values)):
        raise ValueError(f"{what} are listed once each, sorted")
    return values


def _identity(kind: str, value: Any) -> str:
    return fingerprint({"type": kind, "version": 1, "content": value})


# ---- §P3: the patch -------------------------------------------------------------------------


class SetSystemInstructions(_Closed):
    op: Literal["set-system-instructions"] = "set-system-instructions"
    system: HarnessContent


class SetTaskTemplate(_Closed):
    """A new task template; its variables must be the base's (the task binding fixes them)."""

    op: Literal["set-task-template"] = "set-task-template"
    template: PromptTemplate


class SetToolDescription(_Closed):
    """A new description for one tool the base defines."""

    op: Literal["set-tool-description"] = "set-tool-description"
    tool: str = Field(min_length=1)
    description: HarnessContent


class SetEnabledTools(_Closed):
    """The child's enabled set: a subset of the base's enabled set, never more (§P7.2)."""

    op: Literal["set-enabled-tools"] = "set-enabled-tools"
    enabled: tuple[str, ...]

    @field_validator("enabled")
    @classmethod
    def _canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _sorted_unique(value, "enabled tools")


class SetContextPolicy(_Closed):
    """Context fields to set. A limit can be set, never removed: removing one widens."""

    op: Literal["set-context-policy"] = "set-context-policy"
    max_context_tokens: int | None = Field(default=None, gt=0, strict=True)
    selection: ContextSelection | None = None
    overflow: Literal["fail", "truncate-oldest"] | None = None
    max_tool_result_tokens: int | None = Field(default=None, gt=0, strict=True)

    @model_validator(mode="after")
    def _sets_something(self) -> SetContextPolicy:
        if all(getattr(self, name) is None for name in _CONTEXT_FIELDS):
            raise ValueError("set-context-policy sets at least one field")
        return self


class SetStoppingLimits(_Closed):
    """Stopping limits to set. A limit can be set, never removed: removing one widens."""

    op: Literal["set-stopping-limits"] = "set-stopping-limits"
    max_turns: int | None = Field(default=None, gt=0, strict=True)
    max_total_tokens: int | None = Field(default=None, gt=0, strict=True)
    max_tool_calls: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def _sets_something(self) -> SetStoppingLimits:
        if all(getattr(self, name) is None for name in _STOPPING_FIELDS):
            raise ValueError("set-stopping-limits sets at least one field")
        return self


_CONTEXT_FIELDS: Final = ("max_context_tokens", "selection", "overflow", "max_tool_result_tokens")
_STOPPING_FIELDS: Final = ("max_turns", "max_total_tokens", "max_tool_calls")

HarnessOperation = Annotated[
    SetSystemInstructions
    | SetTaskTemplate
    | SetToolDescription
    | SetEnabledTools
    | SetContextPolicy
    | SetStoppingLimits,
    Field(discriminator="op"),
]


def _target(operation: Any) -> str:
    """Which surface field an operation writes: at most one operation per target."""
    if isinstance(operation, SetToolDescription):
        return f"set-tool-description:{operation.tool}"
    return str(operation.op)


class HarnessPatch(_Closed):
    """A declared change to one pinned base harness (ADR-024 §P3).

    ``rationale`` is not identity. The patch is identified by its base and
    its operations, in order.
    """

    schema_version: Literal["xaytune.harness-patch/v1"] = "xaytune.harness-patch/v1"
    base_fingerprint: Digest
    operations: tuple[HarnessOperation, ...] = Field(min_length=1)
    rationale: str = ""

    @model_validator(mode="after")
    def _one_per_field(self) -> HarnessPatch:
        targets = [_target(operation) for operation in self.operations]
        if len(set(targets)) != len(targets):
            raise ValueError("a patch writes each surface field at most once")
        return self

    def patch_fingerprint(self) -> str:
        return _identity(
            "harness-patch",
            {
                "schema_version": self.schema_version,
                "base_fingerprint": self.base_fingerprint,
                "operations": [operation.model_dump(mode="json") for operation in self.operations],
            },
        )


class PatchRefusedError(ValueError):
    """A patch cannot be applied. Every reason, never the content it carried."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(reasons)
        super().__init__("the patch is refused: " + "; ".join(self.reasons))


class HarnessCeilings(_Closed):
    """The experiment's limits on any candidate it searches (§P7.3, policy, not science).

    All finite: they bound every attempt the search can start (§P12a).
    """

    max_context_tokens: int = Field(gt=0, strict=True)
    max_tool_result_tokens: int = Field(gt=0, strict=True)
    max_turns: int = Field(gt=0, strict=True)
    max_total_tokens: int = Field(gt=0, strict=True)
    max_tool_calls: int = Field(ge=0, strict=True)
    max_output_tokens: int = Field(gt=0, strict=True)


def ceiling_problems(harness: AgentHarnessSpec, ceilings: HarnessCeilings) -> list[str]:
    """Every way *harness* exceeds *ceilings*, or leaves a limit the ceilings require unset."""
    problems: list[str] = []
    context, stopping = harness.context, harness.stopping
    pairs: list[tuple[str, int | None, int]] = [
        ("context.max_context_tokens", context.max_context_tokens, ceilings.max_context_tokens),
        (
            "context.max_tool_result_tokens",
            context.max_tool_result_tokens,
            ceilings.max_tool_result_tokens,
        ),
        ("stopping.max_turns", stopping.max_turns, ceilings.max_turns),
        ("stopping.max_total_tokens", stopping.max_total_tokens, ceilings.max_total_tokens),
        ("stopping.max_tool_calls", stopping.max_tool_calls, ceilings.max_tool_calls),
        (
            "model.generation.max_output_tokens",
            harness.model.generation.max_output_tokens,
            ceilings.max_output_tokens,
        ),
    ]
    for name, value, ceiling in pairs:
        if value is None:
            problems.append(f"{name} is unset, so it cannot stay within the ceiling {ceiling}")
        elif value > ceiling:
            problems.append(f"{name} {value} exceeds the experiment's ceiling {ceiling}")
    return problems


def apply_patch(
    base: AgentHarnessSpec,
    patch: HarnessPatch,
    *,
    allowed: Collection[str] | None = None,
    authorized_tools: Collection[str] | None = None,
    ceilings: HarnessCeilings | None = None,
) -> AgentHarnessSpec:
    """The child harness *patch* describes, applied to *base*; refused with every reason.

    Refuses a stale base, an operation outside *allowed*, a tool the base does
    not define (descriptions) or does not enable (the enabled set), a tool
    outside *authorized_tools*, a template whose variables are not the
    base's, a result that is not a valid harness, a result outside
    *ceilings*, and a no-op.

    Raises:
        PatchRefusedError: With every reason.
    """
    reasons: list[str] = []
    base_fingerprint = harness_fingerprint(base)
    if patch.base_fingerprint != base_fingerprint:
        reasons.append("the patch was written against another base (stale base)")
    record: dict[str, Any] = thaw(base.model_dump(mode="python"))
    defined = {tool.id for tool in base.tools.definitions}
    enabled = set(base.tools.enabled)
    for operation in patch.operations:
        if allowed is not None and operation.op not in allowed:
            reasons.append(f"{operation.op} is outside the allowed surfaces")
            continue
        if isinstance(operation, SetSystemInstructions):
            record["instructions"]["system"] = operation.system.model_dump(mode="python")
        elif isinstance(operation, SetTaskTemplate):
            current = base.instructions.task_template
            if current is None:
                reasons.append("the base has no task template to replace")
            elif operation.template.variables != current.variables:
                reasons.append("a task template's variables must be the base's")
            else:
                record["instructions"]["task_template"] = operation.template.model_dump(
                    mode="python"
                )
        elif isinstance(operation, SetToolDescription):
            if operation.tool not in defined:
                reasons.append(f"tool {operation.tool} is not defined in the base")
            else:
                for tool in record["tools"]["definitions"]:
                    if tool["id"] == operation.tool:
                        tool["description"] = operation.description.model_dump(mode="python")
        elif isinstance(operation, SetEnabledTools):
            widened = sorted(set(operation.enabled) - enabled)
            if widened:
                reasons.append(
                    "tools not enabled in the base cannot be enabled by a patch "
                    f"(enabling is a grant): {', '.join(widened)}"
                )
            if authorized_tools is not None:
                outside = sorted(set(operation.enabled) - set(authorized_tools))
                if outside:
                    reasons.append(f"tools outside the authorized set: {', '.join(outside)}")
            record["tools"]["enabled"] = list(operation.enabled)
        elif isinstance(operation, SetContextPolicy):
            for name in _CONTEXT_FIELDS:
                value = getattr(operation, name)
                if value is not None:
                    record["context"][name] = (
                        value.model_dump(mode="python") if name == "selection" else value
                    )
        else:
            for name in _STOPPING_FIELDS:
                value = getattr(operation, name)
                if value is not None:
                    record["stopping"][name] = value
    child: AgentHarnessSpec | None = None
    if not reasons:
        try:
            child = AgentHarnessSpec.model_validate(record)
        except ValidationError as invalid:
            reasons.append(
                "the result is not a valid harness: "
                + "; ".join(
                    f"{'.'.join(str(part) for part in error['loc']) or 'harness'}: {error['msg']}"
                    for error in invalid.errors()
                )
            )
    if child is not None:
        if ceilings is not None:
            reasons.extend(ceiling_problems(child, ceilings))
        if harness_fingerprint(child) == base_fingerprint:
            reasons.append("the patch changes nothing the harness identity covers (no-op)")
    if reasons or child is None:
        raise PatchRefusedError(reasons)
    return child


# ---- §P12a: the spending envelope -----------------------------------------------------------


ModelRole = Literal["agent", "judge", "provider"]


class LocalSearchEnvelope(_Closed):
    """What the local search may spend, in total (ADR-024 §P12a).

    ``token_limits`` has one limit per token unit; units are never added
    together. ``cost`` is refused in v1 -- cost is not metered -- so a search
    declaring one refuses to start rather than pretending to bound it.
    """

    max_rounds: int = Field(gt=0, strict=True)
    max_candidate_evaluations: int = Field(gt=0, strict=True)
    """Evaluation runs: one per execution replicate (each seed of each candidate), not
    one per candidate. Three candidates on three seeds are nine."""
    max_task_attempts: int = Field(gt=0, strict=True)
    max_model_calls: FrozenDict = Field(default_factory=FrozenDict)
    """``{role: calls}`` for ``agent``, ``judge`` and ``provider``; a missing role may make none."""
    token_limits: FrozenDict = Field(default_factory=FrozenDict)
    """``{unit: tokens}``, e.g. ``{"tokens(whitespace-v1)": 50000}``."""
    judge_prompt_tokens: int | None = Field(default=None, gt=0, strict=True)
    """The declared bound on one judge call's prompt, in the judge's token unit."""
    cost: FrozenDict | None = None

    @field_validator("max_model_calls")
    @classmethod
    def _roles(cls, value: FrozenDict) -> FrozenDict:
        for role, calls in value.items():
            if role not in ("agent", "judge", "provider"):
                raise ValueError(
                    f"model calls are limited per role (agent, judge, provider), not {role!r}"
                )
            if not isinstance(calls, int) or isinstance(calls, bool) or calls < 0:
                raise ValueError("a model call limit is a non-negative integer")
        return value

    @field_validator("token_limits")
    @classmethod
    def _limits(cls, value: FrozenDict) -> FrozenDict:
        for unit, limit in value.items():
            if not unit:
                raise ValueError("a token limit names its unit")
            if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
                raise ValueError("a token limit is a positive integer")
        return value


class Spend(_Closed):
    """An amount of work: reserved before it starts, or measured after it ends."""

    rounds: int = Field(default=0, ge=0, strict=True)
    evaluations: int = Field(default=0, ge=0, strict=True)
    attempts: int = Field(default=0, ge=0, strict=True)
    model_calls: FrozenDict = Field(default_factory=FrozenDict)
    tokens: FrozenDict = Field(default_factory=FrozenDict)

    def plus(self, other: Spend) -> Spend:
        return Spend(
            rounds=self.rounds + other.rounds,
            evaluations=self.evaluations + other.evaluations,
            attempts=self.attempts + other.attempts,
            model_calls=FrozenDict(_add(self.model_calls, other.model_calls)),
            tokens=FrozenDict(_add(self.tokens, other.tokens)),
        )

    def exceeds(self, other: Spend) -> list[str]:
        """Every dimension where this is more than *other*."""
        found = [
            name
            for name in ("rounds", "evaluations", "attempts")
            if getattr(self, name) > getattr(other, name)
        ]
        found += [
            f"model_calls[{role}]"
            for role, calls in self.model_calls.items()
            if calls > other.model_calls.get(role, 0)
        ]
        found += [
            f"tokens[{unit}]"
            for unit, tokens in self.tokens.items()
            if tokens > other.tokens.get(unit, 0)
        ]
        return found


def _add(a: Mapping[str, int], b: Mapping[str, int]) -> dict[str, int]:
    return {key: a.get(key, 0) + b.get(key, 0) for key in sorted(set(a) | set(b)) if key}


class Reservation(_Closed):
    """One unit of work and its worst case, held from before it starts until it settles."""

    unit: Literal["round", "evaluation", "attempt", "judge-call", "provider-call"]
    label: str
    worst_case: Spend


class SpendingLimitError(RuntimeError):
    """The next unit of work would cross a limit, so it was not started."""

    def __init__(self, reservation: Reservation, limits: Sequence[str]) -> None:
        self.reservation = reservation
        self.limits = tuple(limits)
        super().__init__(
            f"spending-limit: {reservation.unit} {reservation.label} was not started; "
            f"it could cross {', '.join(self.limits)}"
        )


class SpendingExceededError(RuntimeError):
    """A unit measured more than its reservation: the bound was wrong, and the search stops."""


class SpendingUnenforceableError(ValueError):
    """The envelope cannot be enforced, so the search does not start. Every reason."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(reasons)
        super().__init__("the spending envelope cannot be enforced: " + "; ".join(self.reasons))


def _limits_of(envelope: LocalSearchEnvelope) -> Spend:
    return Spend(
        rounds=envelope.max_rounds,
        evaluations=envelope.max_candidate_evaluations,
        attempts=envelope.max_task_attempts,
        model_calls=envelope.max_model_calls,
        tokens=envelope.token_limits,
    )


class SpendingLedger:
    """Reserve-before-start accounting for one local search (§P12a).

    A unit starts only if, for every limit it touches, what is spent plus its
    worst case fits. After it ends, its reservation becomes what it measured;
    unaccounted work keeps its full reservation charged, and a measurement
    above the reservation is an error. One unit at a time (``count = 1``).
    """

    def __init__(self, envelope: LocalSearchEnvelope) -> None:
        self.envelope = envelope
        self.limits = _limits_of(envelope)
        self.spent = Spend()
        self.entries: list[dict[str, Any]] = []
        self._open: Reservation | None = None

    def reserve(self, reservation: Reservation) -> Reservation:
        """Hold *reservation*, or refuse it without starting anything.

        Raises:
            SpendingLimitError: Naming every limit it could cross.
        """
        if self._open is not None:
            raise RuntimeError("one unit of work at a time: settle the open reservation first")
        self.admit(reservation)
        self._open = reservation
        return reservation

    def admit(self, reservation: Reservation) -> None:
        """Check that *reservation* would fit now, without holding it.

        For work whose admission must be known before other work starts: a
        replicate's evaluation is admitted before any of its task attempts
        runs, so no task runs for an evaluation that could never start.

        Raises:
            SpendingLimitError: Naming every limit it could cross (recorded as
                ``not-started``).
        """
        undeclared = {
            f"tokens[{unit}]"
            for unit in reservation.worst_case.tokens
            if unit not in self.limits.tokens
        } | {
            f"model_calls[{role}]"
            for role, calls in reservation.worst_case.model_calls.items()
            if calls and role not in self.limits.model_calls
        }
        crossed = [
            name
            for name in self.spent.plus(reservation.worst_case).exceeds(self.limits)
            if name not in undeclared
        ]
        unlimited = [
            f"tokens[{unit}] (no limit declared)"
            for unit in reservation.worst_case.tokens
            if unit not in self.limits.tokens
        ] + [
            f"model_calls[{role}] (no limit declared)"
            for role, calls in reservation.worst_case.model_calls.items()
            if calls and role not in self.limits.model_calls
        ]
        if crossed or unlimited:
            self.entries.append(
                {
                    "unit": reservation.unit,
                    "label": reservation.label,
                    "status": "not-started",
                    "reserved": reservation.worst_case.model_dump(mode="json"),
                    "limits": sorted(set(crossed) | set(unlimited)),
                }
            )
            raise SpendingLimitError(reservation, sorted(set(crossed) | set(unlimited)))

    def abandon(self, *, note: str) -> Reservation | None:
        """Settle the open reservation, if any, as unaccounted: charged in full.

        For work that ended in an error nothing measured. *note* says why
        (the error type, never its message). The reservation, or ``None``.
        """
        reservation = self._open
        if reservation is not None:
            self.settle(reservation, None, note=note)
        return reservation

    def settle(
        self, reservation: Reservation, measured: Spend | None, *, note: str | None = None
    ) -> Spend:
        """Close *reservation*: charge what was measured, or all of it if nothing was.

        *note* says why, when the work is unaccounted (the error type, never its message).

        Raises:
            SpendingExceededError: If *measured* is above the reservation.
        """
        if self._open is not reservation:
            raise RuntimeError("settle the reservation that is open")
        self._open = None
        over = [] if measured is None else measured.exceeds(reservation.worst_case)
        charged = reservation.worst_case if measured is None or over else measured
        self.spent = self.spent.plus(charged)
        self.entries.append(
            {
                "unit": reservation.unit,
                "label": reservation.label,
                "status": "exceeded"
                if over
                else ("unaccounted" if measured is None else "settled"),
                "reserved": reservation.worst_case.model_dump(mode="json"),
                "charged": charged.model_dump(mode="json"),
                **({"note": note} if note else {}),
            }
        )
        if over:
            raise SpendingExceededError(
                f"{reservation.unit} {reservation.label} measured more than its reservation "
                f"({', '.join(over)}): the bound is wrong, and the search stops"
            )
        return charged

    def remaining(self) -> FrozenDict:
        """What is left under every limit: part of the search context."""
        limits, spent = self.limits, self.spent
        return FrozenDict(
            {
                "rounds": limits.rounds - spent.rounds,
                "evaluations": limits.evaluations - spent.evaluations,
                "attempts": limits.attempts - spent.attempts,
                "model_calls": {
                    role: calls - spent.model_calls.get(role, 0)
                    for role, calls in limits.model_calls.items()
                },
                "tokens": {
                    unit: tokens - spent.tokens.get(unit, 0)
                    for unit, tokens in limits.tokens.items()
                },
            }
        )


def attempt_reservation(harness: AgentHarnessSpec, token_unit: str, label: str) -> Reservation:
    """One task attempt's worst case: the runner checks its total after each reply.

    So the last reply can overshoot ``max_total_tokens`` by one call: at most
    one prompt (``max_context_tokens``) and one reply (``max_output_tokens``).

    Raises:
        SpendingUnenforceableError: If the harness leaves the attempt unbounded.
    """
    stopping = harness.stopping
    if stopping.max_total_tokens is None:
        raise SpendingUnenforceableError(
            ("a harness without stopping.max_total_tokens leaves its attempts unbounded",)
        )
    tokens = (
        stopping.max_total_tokens
        + harness.context.max_context_tokens
        + harness.model.generation.max_output_tokens
    )
    return Reservation(
        unit="attempt",
        label=label,
        worst_case=Spend(
            attempts=1,
            model_calls=FrozenDict({"agent": stopping.max_turns}),
            tokens=FrozenDict({token_unit: tokens}),
        ),
    )


# ---- policy ---------------------------------------------------------------------------------


class HarnessSearchPolicy(_Closed):
    """What an experiment authorizes a harness search to do, fixed for the whole search.

    ``autonomous_search`` has no default: without the experiment saying so
    explicitly, no candidate is generated without a human (§P7, Q3). The
    permissions (tools, operations, ceilings) and the budget (the envelope)
    are part of the fingerprint; changing any of them is a new search.
    """

    schema_version: Literal["xaytune.harness-search-policy/v1"] = "xaytune.harness-search-policy/v1"
    autonomous_search: Literal["authorized"]
    authorized_tools: tuple[str, ...]
    allowed_operations: tuple[OperationName, ...] = Field(min_length=1)
    ceilings: HarnessCeilings
    max_retries: int = Field(default=2, ge=0, le=10, strict=True)
    envelope: LocalSearchEnvelope

    @field_validator("authorized_tools")
    @classmethod
    def _tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _sorted_unique(value, "authorized tools")

    @field_validator("allowed_operations")
    @classmethod
    def _operations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _sorted_unique(value, "allowed operations")

    def policy_fingerprint(self) -> str:
        return _identity("harness-search-policy", self.model_dump(mode="json"))


def envelope_refusals(
    policy: HarnessSearchPolicy,
    root: AgentHarnessSpec,
    *,
    token_unit: str,
    judge_unit: str | None = None,
    provider_calls: bool = False,
) -> list[str]:
    """Why *policy*'s envelope cannot be enforced for a search from *root*; ``[]`` if it can."""
    envelope = policy.envelope
    reasons: list[str] = []
    if envelope.cost is not None:
        reasons.append("cost-not-metered: v1 records no cost, so a cost limit cannot be enforced")
    reasons.extend(
        f"the root harness: {problem}" for problem in ceiling_problems(root, policy.ceilings)
    )
    if token_unit not in envelope.token_limits:
        reasons.append(
            f"no token limit for {token_unit}, the unit every task attempt is charged in"
        )
    if "agent" not in envelope.max_model_calls:
        reasons.append("no agent model call limit")
    if judge_unit is not None:
        if judge_unit not in envelope.token_limits:
            reasons.append(f"no token limit for {judge_unit}, the unit judge calls are charged in")
        if "judge" not in envelope.max_model_calls:
            reasons.append("no judge model call limit")
        if envelope.judge_prompt_tokens is None:
            reasons.append("no judge_prompt_tokens: a judge call's worst case cannot be reserved")
    if provider_calls:
        reasons.append(
            "a provider that calls a model cannot be reserved in v1: stochastic providers are "
            "deferred (ADR-024 §P13)"
        )
    return reasons


# ---- §P8: provenance and feedback references ------------------------------------------------


class HarnessProviderSpec(FrozenDomainModel):
    """Which provider, configured how. ``version`` is set once bound; ``config`` is identity."""

    kind: str = Field(min_length=1)
    version: str | None = None
    config: FrozenDict = Field(default_factory=FrozenDict)


class ProviderProvenance(_Closed):
    """One provider as bound: descriptor, configuration, capabilities, engine and seed."""

    provider: str
    name: str
    plugin_version: str
    api_version: str
    spec_kind: str
    spec_version: str
    spec_identity_version: int = 1
    spec_fingerprint: Digest
    capabilities_fingerprint: Digest
    engine: FrozenDict = Field(default_factory=FrozenDict)
    seed: int | None = None
    agent_invocation_ids: tuple[str, ...] = ()


class FeedbackRef(_Closed):
    """One evaluation a provider learned from: what a reviewer follows back to the record."""

    evaluation_fingerprint: Digest
    evidence_digest: Digest
    evaluation_result_id: str = Field(min_length=1)


class HarnessFeedback(_Closed):
    """A provider's view of one evaluation (``ProviderFeedback``) and where it came from."""

    ref: FeedbackRef
    feedback: ProviderFeedback

    @model_validator(mode="after")
    def _same_evaluation(self) -> HarnessFeedback:
        if (self.ref.evaluation_fingerprint, self.ref.evidence_digest) != (
            self.feedback.evaluation_fingerprint,
            self.feedback.evidence_digest,
        ):
            raise ValueError("a feedback reference names another evaluation than its feedback")
        return self


# ---- §P5: capabilities ----------------------------------------------------------------------


class HarnessMutationCapabilities(_Closed):
    operations: tuple[OperationName, ...] = Field(min_length=1)
    determinism: Literal["deterministic", "seeded", "stochastic"]
    model: FrozenDict | None = None
    feedback: Literal["none", "run-values", "run-values+task-execution"]

    @field_validator("operations")
    @classmethod
    def _operations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _sorted_unique(value, "operations")

    @model_validator(mode="after")
    def _model_when_stochastic(self) -> HarnessMutationCapabilities:
        if (self.determinism == "stochastic") != (self.model is not None):
            raise ValueError("a stochastic provider names its model; no other provider does")
        return self


class HarnessSearchCapabilities(_Closed):
    objectives: Literal["single", "vector"]
    determinism: Literal["deterministic", "seeded"]
    needs: Literal["evaluated-nodes", "comparisons"]


# ---- §P4, §P6: what providers see -----------------------------------------------------------


ProviderVerdict = Literal[
    "favors-a", "favors-b", "inconclusive", "not-comparable", "withheld-from-provider"
]
ProviderRelation = Literal[
    "favors-a", "favors-b", "trade-off", "inconclusive", "not-comparable", "withheld-from-provider"
]


class ProviderMetricComparison(_Closed):
    """One objective metric of a provider-safe comparison (§P6).

    ``withheld-from-provider`` when either side's value was withheld from
    providers: never a verdict, a tie or a rank.
    """

    metric: str
    slice: str | None = None
    direction: Literal["maximize", "minimize"]
    verdict: ProviderVerdict
    mean_difference: float | None = None
    standard_error: float | None = None
    threshold: float | None = None
    replicates: int = Field(default=0, ge=0, strict=True)
    reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _withheld_carries_nothing(self) -> ProviderMetricComparison:
        if self.verdict == "withheld-from-provider" and (
            self.mean_difference is not None or self.standard_error is not None
        ):
            raise ValueError("a withheld comparison carries no number")
        return self


class ProviderComparison(_Closed):
    """H06's ``compare()`` of two nodes, over provider-visible values only (§P4, §P6)."""

    a: str
    b: str
    evaluation_fingerprint: Digest
    comparable: bool
    reasons: tuple[str, ...] = ()
    metrics: tuple[ProviderMetricComparison, ...]
    relation: ProviderRelation


class HarnessNodeView(_Closed):
    """One harness node as a provider sees it: identity, lineage, status, provider feedback."""

    node_id: str = Field(min_length=1)
    harness_fingerprint: Digest
    parent_ids: tuple[str, ...] = ()
    status: Literal["evaluated", "unevaluated", "failed"]
    feedback: tuple[HarnessFeedback, ...] = ()

    @field_validator("feedback")
    @classmethod
    def _ordered(cls, value: tuple[HarnessFeedback, ...]) -> tuple[HarnessFeedback, ...]:
        return tuple(
            sorted(
                value,
                key=lambda f: (
                    f.ref.evaluation_fingerprint,
                    f.ref.evidence_digest,
                    f.ref.evaluation_result_id,
                ),
            )
        )


class ObjectiveView(_Closed):
    """One objective metric the search optimizes: direction declared, never inferred."""

    metric: str
    slice: str | None = None
    direction: Literal["maximize", "minimize"]
    min_effect: float = Field(default=0.0, ge=0, allow_inf_nan=False)


class HarnessSearchContext(_Closed):
    """Everything a search provider may depend on (§P4): built from the record, fingerprinted.

    Carries no released report, released summary or recorded comparison:
    only provider feedback and comparisons computed from it (§P6, H7-F12).
    """

    objectives: tuple[ObjectiveView, ...] = Field(min_length=1)
    scalarization: FrozenDict | None = None
    """Declared weights, when a provider is bound to one (§P11); ``None`` otherwise."""
    evaluation_fingerprint: Digest
    """The evaluation the search compares under; nothing is compared across fingerprints."""
    nodes: tuple[HarnessNodeView, ...]
    comparisons: tuple[ProviderComparison, ...] = ()
    exhausted: tuple[str, ...] = ()
    """Bases the mutation provider had nothing more for."""
    remaining: FrozenDict = Field(default_factory=FrozenDict)
    """What the spending envelope has left (§P12a)."""
    history: tuple[str, ...] = ()
    """The search's proposals so far, by proposal fingerprint, in order."""

    @field_validator("nodes")
    @classmethod
    def _nodes(cls, value: tuple[HarnessNodeView, ...]) -> tuple[HarnessNodeView, ...]:
        ids = [node.node_id for node in value]
        if len(set(ids)) != len(ids):
            raise ValueError("a node appears more than once in the search context")
        return tuple(sorted(value, key=lambda node: node.node_id))

    @field_validator("comparisons")
    @classmethod
    def _comparisons(cls, value: tuple[ProviderComparison, ...]) -> tuple[ProviderComparison, ...]:
        return tuple(sorted(value, key=lambda c: (c.a, c.b, c.evaluation_fingerprint)))

    @field_validator("exhausted")
    @classmethod
    def _exhausted(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(set(value)))

    def node(self, node_id: str) -> HarnessNodeView | None:
        return next((node for node in self.nodes if node.node_id == node_id), None)

    def context_fingerprint(self) -> str:
        return _identity("harness-search-context", self.model_dump(mode="json"))


class SearchChoice(_Closed):
    """Where to search next: a base node, why, and the objective view the choice used."""

    kind: Literal["choice"] = "choice"
    base_node_id: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    objective_view: FrozenDict = Field(default_factory=FrozenDict)


class NoChoice(_Closed):
    kind: Literal["none"] = "none"
    reason: Literal["exhausted", "nothing-comparable", "waiting-for-evaluation"]
    detail: str = ""


class MutationRequest(_Closed):
    """Everything a mutation provider may depend on about one base (§P4): fingerprinted."""

    base: AgentHarnessSpec
    base_fingerprint: Digest
    base_node_id: str = Field(min_length=1)
    allowed_operations: tuple[OperationName, ...] = Field(min_length=1)
    ceilings: HarnessCeilings
    feedback: tuple[HarnessFeedback, ...] = ()
    """The base's provider feedback, and its ancestors'."""
    existing: tuple[Digest, ...] = ()
    """Every harness fingerprint the search already has."""
    refused: tuple[Digest, ...] = ()
    """Patches refused earlier in this round, by patch fingerprint."""
    seed: int = Field(strict=True)
    round: int = Field(ge=0, strict=True)
    attempt: int = Field(default=0, ge=0, strict=True)
    count: Literal[1] = 1

    @model_validator(mode="after")
    def _base(self) -> MutationRequest:
        if harness_fingerprint(self.base) != self.base_fingerprint:
            raise ValueError("the base fingerprint is not the base's")
        return self

    def request_fingerprint(self) -> str:
        content = self.model_dump(mode="json", exclude={"base"})
        return _identity("harness-mutation-request", content)


# ---- §P2: the proposal ----------------------------------------------------------------------


class HarnessCandidateProposal(FrozenDomainModel):
    """A new harness candidate to explore: the full child, its patch, and why (ADR-024 §P2).

    Distinct from the training ``CandidateProposal``, whose identity is
    unchanged. The candidate is still identified by ``harness_fingerprint``
    alone; this is the proposal's identity.
    """

    kind: Literal["harness-candidate"] = "harness-candidate"
    harness: AgentHarnessSpec
    harness_fingerprint: Digest
    parent_ids: tuple[str, ...] = Field(min_length=1, max_length=1)
    patch: HarnessPatch
    hypothesis: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence_refs: tuple[EvidenceRef, ...] = ()
    provenance: ProposalProvenance
    search: ProviderProvenance
    mutation: ProviderProvenance
    feedback: tuple[FeedbackRef, ...] = ()

    @model_validator(mode="after")
    def _identified(self) -> HarnessCandidateProposal:
        if harness_fingerprint(self.harness) != self.harness_fingerprint:
            raise ValueError("harness_fingerprint is not the harness's")
        return self

    def proposal_fingerprint(self) -> str:
        return _identity(
            "harness-candidate-proposal",
            {
                "harness_fingerprint": self.harness_fingerprint,
                "parent_ids": list(self.parent_ids),
                "patch": self.patch.patch_fingerprint(),
                "hypothesis": self.hypothesis,
                "reason": self.reason,
                "evidence_refs": sorted(
                    ({"kind": ref.kind, "id": ref.id} for ref in self.evidence_refs),
                    key=lambda ref: (ref["kind"], ref["id"]),
                ),
                "provenance": self.provenance.model_dump(mode="json"),
                "search": self.search.model_dump(mode="json"),
                "mutation": self.mutation.model_dump(mode="json"),
                "feedback": sorted(
                    (ref.model_dump(mode="json") for ref in self.feedback),
                    key=lambda ref: (ref["evaluation_fingerprint"], ref["evaluation_result_id"]),
                ),
            },
        )


class HarnessProposalRefusedError(ValueError):
    """A harness proposal fails §P7's checks. Every reason; nothing is branched or run."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(reasons)
        super().__init__("the harness proposal is refused: " + "; ".join(self.reasons))


def verify_harness_proposal(
    proposal: HarnessCandidateProposal,
    parent: AgentHarnessSpec,
    policy: HarnessSearchPolicy,
    existing: Collection[str],
    *,
    provider_operations: Collection[str] | None = None,
) -> None:
    """ADR-024 §P7's four checks, and duplicates, against the *recorded* parent.

    1. Patch integrity: the patch is for this parent, and applying it gives
       exactly ``proposal.harness``.
    2. No capability escalation: ``child.enabled ⊆ parent.enabled ∩ authorized_tools``.
    3. Ceilings: the child is within the experiment's ceilings.
    4. Surfaces: every operation is allowed by the policy (and the provider).

    Run in the planner, and again wherever the proposal is acted on.

    Raises:
        HarnessProposalRefusedError: With every reason.
    """
    reasons: list[str] = []
    allowed = set(policy.allowed_operations)
    if provider_operations is not None:
        allowed &= set(provider_operations)
    parent_fingerprint = harness_fingerprint(parent)
    if proposal.patch.base_fingerprint != parent_fingerprint:
        reasons.append("patch integrity: the patch was written against another parent")
    try:
        child = apply_patch(parent, proposal.patch)
    except PatchRefusedError as refused:
        reasons.extend(f"patch integrity: {reason}" for reason in refused.reasons)
    else:
        if child.model_dump(mode="json") != proposal.harness.model_dump(mode="json"):
            reasons.append(
                "patch integrity: the proposed harness is not the patch applied to the parent"
            )
    granted = sorted(
        set(proposal.harness.tools.enabled)
        - (set(parent.tools.enabled) & set(policy.authorized_tools))
    )
    if granted:
        reasons.append(
            "capability escalation: enabled tools outside parent.enabled ∩ authorized_tools: "
            + ", ".join(granted)
        )
    reasons.extend(
        f"ceilings: {problem}" for problem in ceiling_problems(proposal.harness, policy.ceilings)
    )
    outside = sorted({operation.op for operation in proposal.patch.operations} - allowed)
    if outside:
        reasons.append(f"surfaces: operations outside the allowed surfaces: {', '.join(outside)}")
    if proposal.harness_fingerprint in existing:
        reasons.append("duplicate: the search already has this harness")
    if reasons:
        raise HarnessProposalRefusedError(reasons)
