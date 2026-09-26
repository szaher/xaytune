"""The lm-eval evaluation worker: runs one bound benchmark task on a model, and reports.

Run by a runtime from what :class:`~xaytune.evaluation.lmeval.LMEvalEvaluator`
prepared. It reports under telemetry v1alpha3::

    EvaluationStarted
    EvaluationCompleted       the task's metrics, inline, and the report
  or EvaluationFailed         with the reason, then a non-zero exit

**It checks before it measures.** The installed lm-eval must be the release
the task was bound under, and its definition of the task must digest to the
one recorded; otherwise the same name would run another benchmark. The model
directory must hold a model and its tokenizer. The dataset is loaded at the
recorded commit, never at whatever its branch points to now.

The run's seed is lm-eval's Python, NumPy, Torch and few-shot seed. Nothing is
generated: the task is scored by log-likelihood, in fp32.
"""

from __future__ import annotations

import json
import math
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

from xaytune._version import __version__
from xaytune.core.domain.evaluation import MetricResult
from xaytune.core.ids import ArtifactId
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import ArtifactRef, DatasetRef
from xaytune.core.telemetry import (
    EvaluationCompletedPayload,
    EvaluationFailedPayload,
    EvaluationStartedPayload,
)
from xaytune.runtimes.worker import (
    OBSERVATIONS_PATH_ENV,
    WORKER_CONFIG_PATH_ENV,
    ObservationWriter,
)
from xaytune.workers.common import failure_reason, require_environment, require_model_artifact
from xaytune.workers.eval_lmeval_schema import LMEvalTaskBinding, LMEvalWorkerConfig

__all__ = [
    "LMEvalVersionError",
    "Measurement",
    "TaskChangedError",
    "main",
    "measure",
    "measurement_from",
]


class LMEvalVersionError(RuntimeError):
    """The installed lm-eval is not the release the task was bound under."""


class TaskChangedError(RuntimeError):
    """The installed lm-eval defines the task differently from the binding."""


class Measurement:
    """One metric of the bound task, as lm-eval reported it."""

    def __init__(
        self, name: str, value: float, standard_error: float | None, sample_count: int
    ) -> None:
        self.name = name
        self.value = value
        self.standard_error = standard_error
        self.sample_count = sample_count


def measurement_from(
    results: dict[str, Any], binding: LMEvalTaskBinding
) -> tuple[list[Measurement], dict[str, int]]:
    """The bound metrics out of lm-eval's results, and the task's sample counts.

    ``sample_count`` is lm-eval's **effective** count: the documents it
    scored, after ``limit`` -- not the dataset's size. A standard error lm-eval
    could not compute (it reports ``"N/A"``) is left out rather than invented.

    Raises:
        KeyError: If lm-eval reported no value for a bound metric.
    """
    scored = results["results"][binding.task]
    samples = results["n-samples"][binding.task]
    counts = {"original": int(samples["original"]), "effective": int(samples["effective"])}
    measurements = []
    for name in binding.metrics:
        value = float(scored[f"{name},none"])
        error = scored.get(f"{name}_stderr,none")
        standard_error = (
            float(error)
            if isinstance(error, (int, float)) and math.isfinite(error) and error >= 0
            else None
        )
        measurements.append(Measurement(name, value, standard_error, counts["effective"]))
    return measurements, counts


def measure(config: LMEvalWorkerConfig) -> tuple[list[Measurement], dict[str, Any]]:
    """Run the bound task on the model; return the metrics and what the report records.

    Raises:
        LMEvalVersionError: If lm-eval is not the bound release.
        TaskChangedError: If the task's definition is not the bound one.
        FileNotFoundError: If the model directory lacks a model or tokenizer.
    """
    binding = config.binding
    installed = version("lm-eval")
    if installed != binding.lm_eval_version:
        raise LMEvalVersionError(
            f"lm-eval {installed} is installed, but the task was bound under "
            f"{binding.lm_eval_version}; it would run another definition of {binding.task!r}"
        )
    from xaytune.evaluation.lmeval import load_task_definition, task_manager

    definition = load_task_definition(binding.task)
    if definition.digest != binding.task_config_digest:
        raise TaskChangedError(
            f"lm-eval defines {binding.task!r} as {definition.digest}, but it was bound as "
            f"{binding.task_config_digest}; it would measure another benchmark"
        )
    require_model_artifact(Path(config.model_uri), with_tokenizer=True)

    import datasets
    import torch
    import transformers
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    loadable = definition.loadable()
    loadable["dataset_kwargs"] = {
        **(loadable.get("dataset_kwargs") or {}),
        "revision": binding.dataset_revision,
    }
    tasks = task_manager().load_task_or_group([loadable])
    task = tasks[binding.task]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HFLM(
        pretrained=config.model_uri,
        dtype="float32",
        batch_size=config.measure.batch_size,
        device=device,
    )
    seed = config.realization.seed
    results = simple_evaluate(
        model=model,
        tasks=[task],
        num_fewshot=binding.num_fewshot,
        limit=config.measure.limit,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
        log_samples=False,
    )
    assert results is not None, "simple_evaluate returns results on the main process"
    measurements, counts = measurement_from(results, binding)
    split = task.config.test_split or task.config.validation_split
    record = {
        "split": split,
        "samples": counts,
        "lm_eval": results["results"][binding.task],
        "environment": {
            "device": device,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "lm_eval": installed,
            "xaytune": __version__,
        },
    }
    return measurements, record


def main(*_arguments: str) -> int:
    """Run one prepared lm-eval evaluation, reporting as it goes."""
    config = LMEvalWorkerConfig.model_validate_json(
        Path(require_environment(WORKER_CONFIG_PATH_ENV)).read_text(encoding="utf-8")
    )
    writer = ObservationWriter(Path(require_environment(OBSERVATIONS_PATH_ENV)))
    writer.verify()
    writer.write(EvaluationStartedPayload())

    try:
        measurements, record = measure(config)
        report_path = _write_report(config, measurements, record)
    except Exception as exc:
        try:
            writer.write(EvaluationFailedPayload(reason=failure_reason(exc), detail=str(exc)))
        except OSError:
            pass  # a broken channel must not replace the error that matters
        raise

    binding = config.binding
    realization = config.realization
    dataset = DatasetRef(
        uri=f"hf://datasets/{binding.dataset_path}",
        revision=binding.dataset_revision,
        split=record["split"],
        metadata=FrozenDict({"name": binding.dataset_name}),
    )
    writer.write(
        EvaluationCompletedPayload(
            metrics=tuple(
                MetricResult(
                    name=measurement.name,
                    value=measurement.value,
                    sample_count=measurement.sample_count,
                    dataset_ref=dataset,
                    evaluator_name=realization.evaluator_name,
                    evaluator_version=realization.evaluator_version,
                    seed=realization.seed,
                    standard_error=measurement.standard_error,
                    metadata=FrozenDict(
                        {
                            "task": binding.task,
                            "task_version": binding.task_version,
                            "num_fewshot": binding.num_fewshot,
                            "limit": config.measure.limit,
                            "documents_available": record["samples"]["original"],
                            "lm_eval_version": binding.lm_eval_version,
                        }
                    ),
                )
                for measurement in measurements
            ),
            result_ref=ArtifactRef(
                id=ArtifactId.generate(), kind="evaluation_report", uri=str(report_path)
            ),
        )
    )
    return 0


def _write_report(
    config: LMEvalWorkerConfig, measurements: list[Measurement], record: dict[str, Any]
) -> Path:
    """The full account of the evaluation: which task, bound how, on what, where, and how."""
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
                "binding": config.binding.model_dump(mode="json"),
                "measure": config.measure.model_dump(mode="json"),
                "seed": realization.seed,
                "metrics": {
                    m.name: {"value": m.value, "standard_error": m.standard_error}
                    for m in measurements
                },
                **record,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    return report_path


if __name__ == "__main__":
    sys.exit(main())
