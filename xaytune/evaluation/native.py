"""NativeEvaluator: loss, perplexity and token accuracy of a trained model on held-out text.

The first built-in :class:`~xaytune.evaluation.Evaluator`. It prepares; the
measuring happens in :mod:`xaytune.workers.eval_native`, a worker the runtime
starts like any other, reporting under telemetry v1alpha3.

**Narrow on purpose, and exact within it.** One local model artifact, one
local JSONL file of plain text pinned by its content digest, next-token
metrics over it. Anything it cannot represent precisely is refused -- with
every reason, at submission where the spec alone decides it -- rather than
approximated. Two evaluations that share an ``EvaluationFingerprint`` must
have measured the same thing, so everything that changes the numbers is in
the spec: the dataset's content, how its text becomes tokens (format,
truncation length), how batches are formed, the precision, and which metrics.
The tokenizer is the model's own, saved with it, so it is part of the subject.

**SEEDED, not deterministic.** Scoring fixed text with a fixed model involves
no sampling, but floating-point reductions depend on the device, the kernel
and the library versions. ``DETERMINISTIC`` would promise the same number
anywhere, forever, and a reuse lookup would take it at its word; the run's
seed is applied and recorded instead, and the report names the environment.

**What the metrics mean.** Causal-LM metrics, computed the way the model is
trained: the logits at position *i* are scored against the token at *i + 1*.
Aggregated over tokens, not batches, so the batch size decides how work is
grouped, not what is measured -- although it stays in the spec, because
padding can still move the last digits.

- ``loss``: mean next-token cross-entropy over every predicted token.
- ``perplexity``: ``exp(loss)``.
- ``token_accuracy``: the fraction of predicted tokens whose argmax is the
  next token.

The same definitions :func:`xaytune.eval.evaluate` uses since issue #36,
computed here in the worker so that an evaluation depends on nothing but
its own schema.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator

from xaytune._version import __version__
from xaytune.compilation import SupportResult
from xaytune.compilation._sft import local_path
from xaytune.core.capabilities import PLUGIN_API_VERSIONS, CapabilityDocument, PluginDescriptor
from xaytune.core.domain.evaluation import EvaluationSpec, EvaluatorDeterminism
from xaytune.core.execution import (
    ArtifactOutput,
    EvaluationExecutionSpec,
    EvaluatorIdentity,
    PythonModuleEntrypoint,
    ResourceRequirements,
)
from xaytune.core.immutable import FrozenDict, FrozenDomainModel
from xaytune.core.refs import ArtifactRef, DatasetRef
from xaytune.evaluation import EvaluationContext, UnsupportedEvaluationError
from xaytune.workers.eval_native_schema import (
    NATIVE_EVALUATION_API_VERSION,
    NativeEvaluationData,
    NativeEvaluationMeasure,
    NativeEvaluationRealization,
    NativeEvaluationWorkerConfig,
    NativeMetric,
    content_digest,
)

__all__ = ["NativeEvaluationConfig", "NativeEvaluator", "local_dataset"]

_WORKER_MODULE = "xaytune.workers.eval_native"

_UNVERIFIABLE_DATASET_FIELDS = (
    "transform_fingerprint",
    "tokenizer_fingerprint",
    "template_fingerprint",
)


class NativeEvaluationConfig(FrozenDomainModel):
    """``EvaluatorSpec.config`` for the native evaluator. Every field required.

    Attributes:
        format: How each JSONL record becomes text. Only ``"text"``: the
            record's ``text`` field, scored in full.
        max_seq_length: Where each text is truncated, in tokens.
        batch_size: How many texts share a forward pass.
        metrics: Which of ``loss``, ``perplexity`` and ``token_accuracy``.
        precision: The dtype the model is evaluated in. Only ``"fp32"``.
    """

    format: Literal["text"]
    max_seq_length: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    metrics: tuple[NativeMetric, ...] = Field(min_length=1)
    precision: Literal["fp32"]

    @field_validator("metrics")
    @classmethod
    def _once_each(cls, value: tuple[NativeMetric, ...]) -> tuple[NativeMetric, ...]:
        if len(set(value)) != len(value):
            raise ValueError(f"metrics {value} names one more than once")
        return value

    def data(self, path: str, content_digest: str) -> NativeEvaluationData:
        return NativeEvaluationData(
            path=path,
            content_digest=content_digest,
            format=self.format,
            max_seq_length=self.max_seq_length,
        )

    def measure(self) -> NativeEvaluationMeasure:
        return NativeEvaluationMeasure(
            metrics=self.metrics, batch_size=self.batch_size, precision=self.precision
        )


def local_dataset(path: str | Path) -> DatasetRef:
    """A local dataset file, pinned by the digest of its contents.

    What the native evaluator requires of its data, resolved **before**
    anything is recorded: the evaluation names these bytes, the worker checks
    the file still holds them, and a file that changed afterwards fails the
    evaluation instead of silently measuring something else.
    """
    location = Path(path).resolve()
    return DatasetRef(uri=str(location), content_digest=content_digest(location))


class NativeEvaluator:
    """Prepares next-token loss, perplexity and token accuracy over held-out text."""

    descriptor = PluginDescriptor(
        api_version=PLUGIN_API_VERSIONS[0],
        name="native",
        plugin_version="0.1.0",
        provider="xaytune",
        xaytune_version=__version__,
    )
    determinism = EvaluatorDeterminism.SEEDED

    def capabilities(self) -> CapabilityDocument:
        """Nothing beyond one local process: no distribution, no accelerator required."""
        return CapabilityDocument()

    def supports(self, spec: EvaluationSpec) -> SupportResult:
        """Whether *spec* can be measured exactly as declared. Every reason, not the first."""
        reasons = tuple(_refusals(spec))
        return SupportResult(supported=not reasons, reasons=reasons)

    def prepare(
        self, subject: ArtifactRef, spec: EvaluationSpec, context: EvaluationContext
    ) -> EvaluationExecutionSpec:
        """How to measure *subject*. Mechanical and deterministic; reads no file.

        Raises:
            UnsupportedEvaluationError: With every reason, if the spec, the
                subject or the run's realization cannot be honoured exactly.
        """
        reasons = [*_refusals(spec), *_subject_refusals(subject)]
        if context.seed is None:
            reasons.append(
                "the run has no seed; a seeded evaluation is reproducible only under "
                "the seed it records, so none is invented"
            )
        output_dir = local_path(context.output_uri) if context.output_uri else None
        if output_dir is None:
            reasons.append(
                f"output_uri {context.output_uri!r} is not an absolute local path to "
                f"write the report to"
            )
        if reasons:
            raise UnsupportedEvaluationError(self.descriptor.name, tuple(reasons))

        # _refusals() has proved all of these.
        assert spec.dataset is not None and spec.dataset.content_digest is not None
        assert context.seed is not None and output_dir is not None
        config = NativeEvaluationConfig.model_validate(dict(spec.evaluator.config))
        dataset_path = local_path(spec.dataset.uri)
        model_path = local_path(subject.uri)
        assert dataset_path is not None and model_path is not None

        worker_config = NativeEvaluationWorkerConfig(
            api_version=NATIVE_EVALUATION_API_VERSION,
            model_uri=model_path,
            data=config.data(dataset_path, spec.dataset.content_digest),
            measure=config.measure(),
            realization=NativeEvaluationRealization(
                evaluator_name=self.descriptor.name,
                evaluator_version=self.descriptor.plugin_version,
                seed=context.seed,
                output_dir=output_dir,
            ),
        )
        return EvaluationExecutionSpec(
            evaluator=EvaluatorIdentity(
                name=self.descriptor.name,
                version=self.descriptor.plugin_version,
                descriptor=self.descriptor,
            ),
            evaluation_fingerprint=spec.evaluation_fingerprint(),
            subject=subject,
            entrypoint=PythonModuleEntrypoint(module=_WORKER_MODULE, function="main"),
            config=FrozenDict(worker_config.model_dump(mode="json")),
            outputs=(
                ArtifactOutput(
                    name="report",
                    uri=str(Path(output_dir) / "report.json"),
                    kind="evaluation_report",
                ),
            ),
            resources=ResourceRequirements(workers=1),
        )


def _refusals(spec: EvaluationSpec) -> Iterator[str]:
    """Every reason the spec alone rules out an exact native evaluation."""
    try:
        NativeEvaluationConfig.model_validate(dict(spec.evaluator.config))
    except ValidationError as exc:
        for error in exc.errors():
            where = ".".join(str(part) for part in error["loc"]) or "config"
            yield f"evaluator.config.{where}: {error['msg']}"

    if spec.slices:
        yield (
            f"slices {spec.slices} are declared; the native evaluator scores the whole "
            f"file and has no slices to select"
        )

    dataset = spec.dataset
    if dataset is None:
        yield "dataset is undeclared; there is nothing to evaluate on"
        return
    if local_path(dataset.uri) is None:
        yield (
            f"dataset.uri {dataset.uri!r} is not an absolute local path; the worker "
            f"reads local JSONL, and a relative path would depend on its working directory"
        )
    if dataset.content_digest is None:
        yield (
            "dataset.content_digest is undeclared; without it the evaluation names a "
            "path, not data, and the same fingerprint could measure different text "
            "(xaytune.evaluation.native.local_dataset() pins a file)"
        )
    elif not _is_sha256(dataset.content_digest):
        yield (
            f"dataset.content_digest {dataset.content_digest!r} is not 'sha256:' and 64 "
            f"lowercase hex digits, which is what the worker can verify"
        )
    if dataset.revision is not None:
        yield "dataset.revision is declared; a local file has no revisions to select"
    if dataset.split is not None:
        yield "dataset.split is declared; the worker reads the whole file"
    for field in _UNVERIFIABLE_DATASET_FIELDS:
        if getattr(dataset, field) is not None:
            yield (
                f"dataset.{field} is declared; it is part of the evaluation's identity, "
                f"but the worker cannot verify it and would measure whatever its own "
                f"preprocessing produces"
            )


def _subject_refusals(subject: ArtifactRef) -> Iterator[str]:
    if subject.kind != "model":
        yield f"the subject is a {subject.kind!r} artifact, not a model"
    if local_path(subject.uri) is None:
        yield (
            f"the subject {subject.uri!r} is not an absolute local path; the worker "
            f"loads a local model directory"
        )


def _is_sha256(value: str) -> bool:
    prefix, _, digest = value.partition(":")
    return prefix == "sha256" and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
