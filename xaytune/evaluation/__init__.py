"""The prepare half of evaluation's prepare/execute boundary (ADR-007, ADR-015).

An :class:`Evaluator` turns *what to measure, on what* into *how to run it*:

```text
EvaluationSpec + subject ArtifactRef  ──prepare──>  EvaluationExecutionSpec
```

The sibling of :class:`~xaytune.compilation.TrainerCompiler`, and held to the
same rule: **an evaluator prepares; it never executes.** The spec it returns
crosses the same boundary a training spec does -- resolved against a runtime,
submitted through the operation journal, observed through the telemetry
envelope -- so an evaluation in flight survives a controller restart exactly
as training does.

Evaluation is **not** a trainer callback and never runs inside training
(ADR-007). It consumes a finished artifact and mutates no training state.

This module is the contract. The built-in evaluator is
:class:`~xaytune.evaluation.native.NativeEvaluator`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from xaytune.compilation import SupportResult
from xaytune.core.capabilities import CapabilityDocument, PluginDescriptor
from xaytune.core.domain.evaluation import EvaluationSpec, EvaluatorDeterminism
from xaytune.core.execution import EvaluationExecutionSpec
from xaytune.core.immutable import FrozenDomainModel
from xaytune.core.refs import ArtifactRef

__all__ = ["EvaluationContext", "Evaluator", "UnsupportedEvaluationError"]


class UnsupportedEvaluationError(ValueError):
    """An evaluator was asked for an evaluation it cannot run as declared.

    Carries every reason, not the first, for the reason
    :class:`~xaytune.compilation.UnsupportedCandidateError` does.
    """

    def __init__(self, evaluator: str, reasons: tuple[str, ...]) -> None:
        self.evaluator = evaluator
        self.reasons = reasons
        super().__init__(f"{evaluator} cannot run this evaluation: " + "; ".join(reasons))


class EvaluationContext(FrozenDomainModel):
    """Everything preparing an evaluation may depend on besides the spec and subject.

    Explicit and serializable for the reason :class:`CompilationContext` is:
    preparation must be **deterministic**, because a submission re-issued
    after a restart is rebuilt from the record and its digest checked
    against the one recorded. An evaluator that read the clock or the
    environment would make that check fail on a request nobody changed.

    ``seed`` and ``replicate`` are the run's, not the spec's (ADR-015 §3).
    """

    experiment_id: str
    node_id: str
    evaluation_run_id: str
    seed: int | None = None
    replicate: int | None = None
    output_uri: str | None = None


@runtime_checkable
class Evaluator(Protocol):
    """Prepares one evaluation of one subject, runtime-neutrally.

    ``determinism`` is the evaluator's own statement of whether its result
    can stand in for running it again (ADR-015 §3). It is recorded with the
    bound spec when an experiment is submitted, so reuse is decided by what
    the evaluator declared, never by an assumption.
    """

    descriptor: PluginDescriptor
    determinism: EvaluatorDeterminism

    def capabilities(self) -> CapabilityDocument:
        """What this evaluator can measure, and what it needs to run."""
        ...

    def supports(self, spec: EvaluationSpec) -> SupportResult:
        """Whether this evaluator can run *spec* exactly as declared, and if not, why.

        Asked when an experiment is submitted, before anything is recorded:
        an evaluation it would refuse must be refused then, not after hours
        of training. Only what the spec says is judged -- the subject does not
        exist yet -- so :meth:`prepare` may still refuse a subject it cannot
        read. Inspects nothing outside the spec.
        """
        ...

    def prepare(
        self, subject: ArtifactRef, spec: EvaluationSpec, context: EvaluationContext
    ) -> EvaluationExecutionSpec:
        """Return how to run *spec* against *subject*.

        Deterministic for the same inputs, and **must not start anything**:
        an evaluator that ran the evaluation itself would produce an effect
        with no operation record behind it (ADR-013).

        Raises:
            UnsupportedEvaluationError: If it cannot run the evaluation as
                declared.
        """
        ...
