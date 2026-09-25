"""Deciding what becomes of an evaluated candidate (PR-015).

```text
DecisionContext  ──decide──>  DecisionProposal  ──recorded──>  Decision
```

A :class:`DecisionEngine` is the sibling of a compiler and an evaluator, held
to the same rule in its own terms: **it decides; it applies nothing.** The
decision it returns is written by the repository, in one commit with the
transitions it causes, so there is no moment at which a candidate has been
decided but not moved -- or moved with no decision on record.

And it decides **only from its context**. No clock, no id, no database, no
environment, no runtime: the same context always yields the same proposal,
byte for byte, which is what lets a restarted controller decide a cycle again
and have the repository recognize the answer it already recorded. The id,
time and actor of the durable decision are the repository's to add.

:class:`ThresholdDecisionEngine` is the one built in.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterator
from typing import Protocol, runtime_checkable

from xaytune.core.domain.decision import (
    DecisionContext,
    DecisionOutcome,
    DecisionProposal,
    MetricEvidence,
)
from xaytune.core.domain.evaluation import EvaluationResult, MetricResult
from xaytune.core.domain.objective import ConstraintOperator

__all__ = ["DecisionEngine", "ThresholdDecisionEngine", "UndecidableError"]


class UndecidableError(ValueError):
    """The context does not determine a decision, so none is made.

    Carries every reason. The candidate stays ``DECIDING``: guessing -- a
    missing metric read as passing, an objective with no target read as met
    -- would record an outcome nothing in the evidence supports.
    """

    def __init__(self, engine: str, reasons: tuple[str, ...]) -> None:
        self.engine = engine
        self.reasons = reasons
        super().__init__(f"{engine} cannot decide: " + "; ".join(reasons))


@runtime_checkable
class DecisionEngine(Protocol):
    """Decides one candidate's evaluation cycle from its context alone."""

    name: str
    version: str

    def decide(self, context: DecisionContext) -> DecisionProposal:
        """The decision *context* determines.

        Pure: nothing but *context* is read, and nothing is minted -- no id,
        no timestamp. The same context gives an identical proposal.

        Raises:
            UndecidableError: With every reason, if the context does not
                determine a decision.
        """
        ...


_COMPARE: dict[ConstraintOperator, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


class ThresholdDecisionEngine:
    """Decides by comparing recorded point estimates with the objective's thresholds.

    ```text
    a metric the objective or a constraint names is missing,
      or reported more than once                  → undecidable
    any constraint violated                       → REJECT
    no target                                     → undecidable
    target met (maximize: ≥, minimize: ≤)         → STOP_SUCCEEDED
    target not met                                → STOP_FAILED
    ```

    **Point estimates, not statistics.** It compares the values recorded,
    exactly, and claims nothing about uncertainty: it does not say that 0.83
    is better than 0.81, only whether 0.83 meets a stated threshold. It reads
    no sample count, since a count means different things for different
    evaluators (texts for the native evaluator, whose metrics are over
    tokens). Noise-aware comparison belongs to a decision policy that can
    see replicates and candidates side by side.

    **A target is required to succeed or fail.** An objective without one
    says "optimize this", not "this is good enough", and with one candidate
    there is nothing to compare it against. Such a node stays ``DECIDING``.

    **Only unsliced metrics decide.** A metric measured on a slice describes
    that slice; the objective names the metric, not a slice of it.
    """

    name = "threshold"
    version = "1.0.0"

    def decide(self, context: DecisionContext) -> DecisionProposal:
        objective = context.objective
        found: dict[str, tuple[EvaluationResult, MetricResult]] = {}
        reasons: list[str] = []

        wanted = [objective.primary.name, *(c.name for c in objective.constraints)]
        for name in dict.fromkeys(wanted):
            located = list(_measurements(context.results, name))
            if not located:
                role = "objective" if name == objective.primary.name else "constraint"
                reasons.append(
                    f"the {role} metric {name!r} is not among this cycle's results; "
                    f"a missing value is not read as passing"
                )
            elif len(located) > 1:
                reasons.append(
                    f"the metric {name!r} is reported {len(located)} times this cycle; "
                    f"which one decides is not defined"
                )
            else:
                found[name] = located[0]
        if reasons:
            raise UndecidableError(self.name, tuple(reasons))

        constraints = tuple(
            _evidence("constraint", c.name, c.operator, c.value, *found[c.name])
            for c in objective.constraints
        )
        violated = [e for e in constraints if not e.satisfied]

        target: MetricEvidence | None = None
        if objective.target is not None:
            primary = objective.primary
            comparison: ConstraintOperator = ">=" if primary.direction == "maximize" else "<="
            target = _evidence(
                "objective", primary.name, comparison, objective.target, *found[primary.name]
            )

        if violated:
            outcome = DecisionOutcome.REJECT
            reason = "constraint violated: " + "; ".join(_describe(e) for e in violated)
        elif target is None:
            raise UndecidableError(
                self.name,
                (
                    f"the objective {objective.primary.direction}s "
                    f"{objective.primary.name!r} with no target; with one candidate "
                    f"there is nothing to compare it against, so neither success nor "
                    f"failure follows",
                ),
            )
        elif target.satisfied:
            outcome = DecisionOutcome.STOP_SUCCEEDED
            reason = f"target met: {_describe(target)}"
        else:
            outcome = DecisionOutcome.STOP_FAILED
            reason = f"target not met: {_describe(target)}"
        if constraints and not violated:
            reason += "; constraints held: " + "; ".join(_describe(e) for e in constraints)

        return DecisionProposal(
            experiment_id=context.experiment_id,
            node_id=context.node_id,
            evaluation_cycle=context.evaluation_cycle,
            outcome=outcome,
            reason=reason,
            evidence=((target,) if target else ()) + constraints,
            engine_name=self.name,
            engine_version=self.version,
            evaluation_result_ids=tuple(sorted(r.id for r in context.results)),
            input_fingerprint=context.input_fingerprint(),
        )


def _measurements(
    results: tuple[EvaluationResult, ...], name: str
) -> Iterator[tuple[EvaluationResult, MetricResult]]:
    for result in results:
        for metric in result.metrics:
            if metric.name == name and metric.slice is None:
                yield result, metric


def _evidence(
    role: str,
    name: str,
    comparison: ConstraintOperator,
    threshold: float,
    result: EvaluationResult,
    metric: MetricResult,
) -> MetricEvidence:
    return MetricEvidence(
        metric=name,
        role=role,  # type: ignore[arg-type]
        value=metric.value,
        operator=comparison,
        threshold=threshold,
        satisfied=_COMPARE[comparison](metric.value, threshold),
        evaluation_result_id=result.id,
        evaluator_name=metric.evaluator_name,
        evaluator_version=metric.evaluator_version,
    )


def _describe(evidence: MetricEvidence) -> str:
    held = "" if evidence.satisfied else "not "
    return (
        f"{evidence.metric} {evidence.value!r} is {held}{evidence.operator} {evidence.threshold!r}"
    )
