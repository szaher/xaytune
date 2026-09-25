"""A small, scripted evaluator for the evaluation lifecycle tests (PR-013).

Not an evaluator of anything real -- PR-014 wraps Xaytune's metrics and
lm-eval. This one exists to drive the **substrate**: a real worker process,
started by LocalRuntime from a real ``EvaluationExecutionSpec``, speaking
telemetry v1alpha3, whose behaviour the test chooses through its config:

```text
mode     "complete"   EvaluationCompleted(metrics) and exit 0   (default)
         "no-result"  exit 0 without a completion
         "fail"       EvaluationFailed and exit 1
         "complete-then-fail"   EvaluationCompleted(metrics), then exit 1
hold     a path: wait for it to exist before finishing, so a test can
         crash or cancel the controller while the evaluation is live
hold_after_completion
         a path: report the completion, then wait for it before exiting --
         the window between a result reported and the workload's end
value    the accuracy it reports
evaluator_name          the evaluator its metric claims (default "scripted")
report_names_producer   the report names an EvaluationId of the worker's own
```

The worker is ``tests/evaluation_worker.py``, found through ``PYTHONPATH``
set in the spec's environment -- the one thing a test worker needs that a
packaged one would not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xaytune.core.capabilities import PLUGIN_API_VERSIONS, CapabilityDocument, PluginDescriptor
from xaytune.core.domain.evaluation import EvaluationSpec, EvaluatorDeterminism, EvaluatorSpec
from xaytune.core.execution import (
    EvaluationExecutionSpec,
    EvaluatorIdentity,
    PythonModuleEntrypoint,
)
from xaytune.core.immutable import thaw
from xaytune.core.refs import ArtifactRef
from xaytune.evaluation import EvaluationContext

_REPOSITORY = Path(__file__).resolve().parents[1]

EVALUATOR = "scripted"


class ScriptedEvaluator:
    """Prepares the scripted worker; declares itself seeded, as a local sampler is."""

    descriptor = PluginDescriptor(
        api_version=PLUGIN_API_VERSIONS[0],
        name=EVALUATOR,
        plugin_version="1.0.0",
        provider="tests",
        xaytune_version="0.6.0",
    )
    determinism = EvaluatorDeterminism.SEEDED

    def capabilities(self) -> CapabilityDocument:
        return CapabilityDocument()

    def prepare(
        self, subject: ArtifactRef, spec: EvaluationSpec, context: EvaluationContext
    ) -> EvaluationExecutionSpec:
        return EvaluationExecutionSpec(
            evaluator=EvaluatorIdentity(
                name=EVALUATOR,
                version=self.descriptor.plugin_version,
                descriptor=self.descriptor,
            ),
            evaluation_fingerprint=spec.evaluation_fingerprint(),
            subject=subject,
            entrypoint=PythonModuleEntrypoint(module="tests.evaluation_worker"),
            config={
                **thaw(spec.evaluator.config),
                "subject_uri": subject.uri,
                "seed": context.seed,
                "output_uri": context.output_uri,
                "evaluator_version": self.descriptor.plugin_version,
            },
            environment={"PYTHONPATH": str(_REPOSITORY)},
        )


class RenumberedEvaluator(ScriptedEvaluator):
    """The scripted evaluator, claiming a version other than the one recorded.

    It prepares exactly what 1.0.0 prepares, so the request it rebuilds has
    the recorded digest: only the version check can refuse it, which is what
    a test of the version check needs.
    """

    descriptor = ScriptedEvaluator.descriptor.model_copy(update={"plugin_version": "9.9.9"})

    def prepare(
        self, subject: ArtifactRef, spec: EvaluationSpec, context: EvaluationContext
    ) -> EvaluationExecutionSpec:
        return ScriptedEvaluator().prepare(subject, spec, context)


EVALUATORS: dict[str, Any] = {EVALUATOR: ScriptedEvaluator}


def evaluation(**config: object) -> EvaluationSpec:
    """An evaluation of the trained model by the scripted evaluator, unbound."""
    return EvaluationSpec(evaluator=EvaluatorSpec(name=EVALUATOR, config=config))
