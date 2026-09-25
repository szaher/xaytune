"""The native evaluation worker: scores a model on held-out text, and reports the result.

Run by a runtime from what :class:`~xaytune.evaluation.native.NativeEvaluator`
prepared. It reports under telemetry v1alpha3::

    EvaluationStarted
    EvaluationProgress        after each batch
    EvaluationCompleted       the final metrics, inline, and the report
  or EvaluationFailed         with the reason, then a non-zero exit

**It checks before it measures.** The data file must still hold the bytes
the evaluation names -- its content digest -- and the model directory must
hold a model and its tokenizer. Either failing is a failed evaluation, not a
measurement of something else.

The text pipeline is the native trainer's own -- ``load_dataset`` with the
``text`` format, then ``tokenize_dataset`` and ``collate_tokenized`` -- so an
evaluation reads a file the way training reads one. The metrics are defined
in :mod:`xaytune.evaluation.native`.
"""

from __future__ import annotations

import json
import math
import platform
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xaytune._version import __version__
from xaytune.core.domain.evaluation import MetricResult
from xaytune.core.ids import ArtifactId
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import ArtifactRef, DatasetRef
from xaytune.core.telemetry import (
    EvaluationCompletedPayload,
    EvaluationFailedPayload,
    EvaluationProgressPayload,
    EvaluationStartedPayload,
)
from xaytune.runtimes.worker import (
    OBSERVATIONS_PATH_ENV,
    WORKER_CONFIG_PATH_ENV,
    ObservationWriter,
)
from xaytune.workers.common import failure_reason, require_environment, require_model_artifact
from xaytune.workers.eval_native_schema import NativeEvaluationWorkerConfig, content_digest

if TYPE_CHECKING:
    import torch

__all__ = [
    "DatasetChangedError",
    "NextTokenTally",
    "NothingToMeasureError",
    "main",
    "measure",
]


class DatasetChangedError(RuntimeError):
    """The data file no longer holds the bytes the evaluation names."""


class NothingToMeasureError(RuntimeError):
    """No token in the data could be scored: a metric over nothing is not a number."""


class NextTokenTally:
    """Running totals for next-token metrics, aggregated over tokens, not batches.

    The logits at position *i* are scored against the token at *i + 1*, and
    only where that label is not the ignore index. A batch's contribution is
    its sums, so how the texts were batched decides how the work is grouped,
    not what is measured.
    """

    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.tokens = 0
        self.correct = 0

    def add(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        import torch.nn.functional as F

        from xaytune.data.tokenizer import IGNORE_INDEX

        predicted = logits[:, :-1, :].float()
        target = labels[:, 1:]
        scored = target != IGNORE_INDEX
        if not bool(scored.any()):
            return
        loss = F.cross_entropy(
            predicted.reshape(-1, predicted.size(-1)),
            target.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        self.loss_sum += float(loss)
        self.tokens += int(scored.sum())
        self.correct += int(((predicted.argmax(dim=-1) == target) & scored).sum())

    def metrics(self, names: Iterable[str]) -> dict[str, float]:
        if self.tokens == 0:
            raise NothingToMeasureError("the data produced no token that could be scored")
        loss = self.loss_sum / self.tokens
        values = {
            "loss": loss,
            "perplexity": math.exp(loss),
            "token_accuracy": self.correct / self.tokens,
        }
        return {name: values[name] for name in names}


def measure(
    config: NativeEvaluationWorkerConfig, writer: ObservationWriter | None = None
) -> tuple[dict[str, float], dict[str, Any]]:
    """Score the model on the data; return the metrics and what the report records.

    Raises:
        DatasetChangedError: If the file's digest is not the one named.
        FileNotFoundError: If the model directory lacks a model or tokenizer.
        NothingToMeasureError: If no token could be scored.
    """
    data_path = Path(config.data.path)
    found = content_digest(data_path)
    if found != config.data.content_digest:
        raise DatasetChangedError(
            f"{data_path} has digest {found}, but the evaluation names "
            f"{config.data.content_digest}; it would measure different data"
        )
    require_model_artifact(Path(config.model_uri), with_tokenizer=True)

    import torch
    import transformers

    from xaytune.data.loader import load_dataset
    from xaytune.data.tokenizer import collate_tokenized, tokenize_dataset
    from xaytune.models import load_model
    from xaytune.trainer.device import get_device, seed_all

    seed_all(config.realization.seed)
    loaded = load_model(config.model_uri, dtype=config.measure.precision, device_map="cpu")
    model, tokenizer = loaded.model, loaded.tokenizer
    if next(model.parameters()).dtype is not torch.float32:
        raise TypeError(f"the model loaded as {next(model.parameters()).dtype}, not fp32")
    device = get_device()
    model.to(device)
    model.eval()

    samples = load_dataset(str(data_path), format=config.data.format)
    assert isinstance(samples, list)
    sequences = tokenize_dataset(samples, tokenizer, config.data.max_seq_length)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    tally = NextTokenTally()
    size = config.measure.batch_size
    with torch.no_grad():
        for start in range(0, len(sequences), size):
            batch = collate_tokenized(sequences[start : start + size], pad_token_id=pad_id)
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            tally.add(logits, batch["labels"].to(device))
            if writer is not None:
                writer.write(
                    EvaluationProgressPayload(
                        examples_completed=min(start + size, len(sequences)),
                        examples_total=len(sequences),
                    )
                )

    values = tally.metrics(config.measure.metrics)
    record = {
        "records": len(samples),
        "sequences": len(sequences),
        "skipped_empty": len(samples) - len(sequences),
        "tokens_scored": tally.tokens,
        "environment": {
            "device": str(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "xaytune": __version__,
        },
    }
    return values, record


def main(*_arguments: str) -> int:
    """Run one prepared native evaluation, reporting as it goes."""
    config = NativeEvaluationWorkerConfig.model_validate_json(
        Path(require_environment(WORKER_CONFIG_PATH_ENV)).read_text(encoding="utf-8")
    )
    writer = ObservationWriter(Path(require_environment(OBSERVATIONS_PATH_ENV)))
    writer.verify()
    writer.write(EvaluationStartedPayload())

    try:
        values, record = measure(config, writer)
        report_path = _write_report(config, values, record)
    except Exception as exc:
        try:
            writer.write(EvaluationFailedPayload(reason=failure_reason(exc), detail=str(exc)))
        except OSError:
            pass  # a broken channel must not replace the error that matters
        raise

    realization = config.realization
    dataset = DatasetRef(uri=config.data.path, content_digest=config.data.content_digest)
    writer.write(
        EvaluationCompletedPayload(
            metrics=tuple(
                MetricResult(
                    name=name,
                    value=value,
                    sample_count=record["sequences"],
                    dataset_ref=dataset,
                    evaluator_name=realization.evaluator_name,
                    evaluator_version=realization.evaluator_version,
                    seed=realization.seed,
                    metadata=FrozenDict(
                        {
                            "tokens_scored": record["tokens_scored"],
                            "aggregation": "token-weighted",
                            "prediction": "next-token",
                        }
                    ),
                )
                for name, value in values.items()
            ),
            result_ref=ArtifactRef(
                id=ArtifactId.generate(), kind="evaluation_report", uri=str(report_path)
            ),
        )
    )
    return 0


def _write_report(
    config: NativeEvaluationWorkerConfig, values: dict[str, float], record: dict[str, Any]
) -> Path:
    """The full account of the evaluation: what was measured, on what, where, and how."""
    realization = config.realization
    report_path = Path(realization.output_dir) / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "api_version": config.api_version,
                "evaluator": {
                    "name": realization.evaluator_name,
                    "version": realization.evaluator_version,
                },
                "model_uri": config.model_uri,
                "data": config.data.model_dump(mode="json"),
                "measure": config.measure.model_dump(mode="json"),
                "seed": realization.seed,
                "metrics": values,
                **record,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return report_path


if __name__ == "__main__":
    sys.exit(main())
