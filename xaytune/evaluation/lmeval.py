"""LMEvalEvaluator: an lm-eval-harness benchmark task, on a trained model, pinned.

The second built-in :class:`~xaytune.evaluation.Evaluator`. It prepares; the
measuring happens in :mod:`xaytune.workers.eval_lmeval`, a worker the runtime
starts like any other, reporting under telemetry v1alpha3.

**A task name is not an evaluation.** ``arc_easy`` names a definition that
ships inside lm-eval and a dataset on the Hugging Face Hub, and both change:
a new lm-eval release can redefine the task, and the dataset's default branch
moves. Two evaluations recorded as "``arc_easy``, 0-shot" could have measured
different benchmarks. So an lm-eval evaluation is **resolved** at submission
(:meth:`LMEvalEvaluator.resolve`) into an :class:`LMEvalTaskBinding` -- the
task definition by digest under the one lm-eval release it supports, the
dataset by commit -- and the binding is what is recorded, fingerprinted and
run. The worker re-derives the definition and refuses to run if it differs.

**Narrow on purpose.** One registered task at a time, not a group or a tag;
defined in YAML, not in custom Python; reading a hub dataset that can be
pinned to a commit; scored by log-likelihood (``multiple_choice`` or
``loglikelihood``); reporting only ``acc`` and ``acc_norm``. Anything else is
refused with the reason. See
:data:`~xaytune.workers.eval_lmeval_schema.LMEVAL_OUTPUT_TYPES` for why
generation tasks are refused.

**SEEDED, not deterministic**, for the reasons the native evaluator is. The
run's seed is lm-eval's Python, NumPy, Torch and few-shot seed alike, and is
recorded on every metric.

**What is recorded per metric**: the value, lm-eval's standard error of it,
and the number of documents lm-eval actually scored -- after ``limit`` --
not the dataset's size.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field, ValidationError

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
from xaytune.core.immutable import FrozenDict, FrozenDomainModel, thaw
from xaytune.core.refs import ArtifactRef
from xaytune.evaluation import EvaluationContext, UnsupportedEvaluationError
from xaytune.workers.eval_lmeval_schema import (
    LM_EVAL_VERSION,
    LMEVAL_EVALUATION_API_VERSION,
    LMEVAL_METRICS,
    LMEVAL_OUTPUT_TYPES,
    LMEvalMeasure,
    LMEvalRealization,
    LMEvalTaskBinding,
    LMEvalWorkerConfig,
    task_config_digest,
)

__all__ = [
    "HubTaskResolver",
    "LMEvalConfig",
    "LMEvalEvaluator",
    "TaskDefinition",
    "TaskResolver",
    "UnresolvableTaskError",
    "load_task_definition",
]

_WORKER_MODULE = "xaytune.workers.eval_lmeval"


class UnresolvableTaskError(ValueError):
    """An lm-eval task cannot be pinned, or cannot be loaded as pinned. Carries every reason."""

    def __init__(self, task: str, reasons: tuple[str, ...]) -> None:
        self.task = task
        self.reasons = reasons
        super().__init__(f"lm-eval task {task!r} cannot be bound: " + "; ".join(reasons))


class LMEvalConfig(FrozenDomainModel):
    """``EvaluatorSpec.config`` for the lm-eval evaluator.

    Attributes:
        task: A registered lm-eval task, such as ``"arc_easy"``.
        num_fewshot: Examples placed before each question; ``0`` for none.
        batch_size: How many requests share a forward pass.
        precision: The dtype the model is evaluated in. Only ``"fp32"``.
        limit: Score only the first *limit* documents, or all of them.
        binding: What the task resolved to. Set by
            :meth:`LMEvalEvaluator.resolve` at submission, never by hand.
    """

    task: str = Field(min_length=1)
    num_fewshot: int = Field(ge=0)
    batch_size: int = Field(gt=0)
    precision: Literal["fp32"]
    limit: int | None = Field(default=None, gt=0)
    binding: LMEvalTaskBinding | None = None


class TaskResolver(Protocol):
    """Pins an lm-eval task and its dataset. The one thing allowed to read the network."""

    def bind(self, task: str, num_fewshot: int) -> LMEvalTaskBinding:
        """The binding for *task*, as it is now.

        Raises:
            UnresolvableTaskError: With every reason, if it cannot be pinned.
        """
        ...


@dataclass(frozen=True)
class TaskDefinition:
    """One lm-eval task's definition, as the installed lm-eval defines it.

    ``config`` is its YAML, includes merged and function references left
    unresolved, made independent of where lm-eval is installed; ``digest`` is
    what a binding records of it. Reading it imports none of the task's
    Python: binding a task runs no task code.
    """

    task: str
    config: Mapping[str, Any]
    digest: str
    yaml_path: str

    def loadable(self) -> dict[str, Any]:
        """The same YAML with its functions resolved: what lm-eval builds the task from.

        Imports the task's Python, so only the worker, about to run it, calls this.
        """
        from lm_eval.tasks._yaml_loader import load_yaml

        return dict(load_yaml(self.yaml_path, resolve_func=True))


def load_task_definition(task: str) -> TaskDefinition:
    """The installed lm-eval's definition of *task*. Reads no network.

    Used at submission, to bind a task, and by the worker, to check that the
    definition it is about to run is the one that was bound.

    Raises:
        UnresolvableTaskError: If lm-eval is missing or another release, or
            *task* is not one registered, YAML-defined task.
    """
    lm_eval = _lm_eval(task)
    from lm_eval.tasks._yaml_loader import load_yaml

    entry = task_manager().task_index.get(task)
    if entry is None:
        raise UnresolvableTaskError(
            task, (f"{task!r} is not a task registered in lm-eval {LM_EVAL_VERSION}",)
        )
    kind = getattr(entry.kind, "name", str(entry.kind))
    if kind != "TASK":
        raise UnresolvableTaskError(
            task,
            (
                f"{task!r} is a {kind.lower()}, not a task; a group or tag aggregates "
                f"several tasks, and this evaluator binds one",
            ),
        )
    if not entry.yaml_path:
        raise UnresolvableTaskError(
            task,
            (f"{task!r} is defined in Python, not YAML; its definition cannot be pinned",),
        )
    root = Path(lm_eval.__file__).resolve().parent
    config = _portable(load_yaml(entry.yaml_path, resolve_func=False), str(root) + os.sep)
    try:
        digest = task_config_digest(config)
    except (TypeError, ValueError) as exc:
        raise UnresolvableTaskError(
            task, (f"its definition does not serialize canonically: {exc}",)
        ) from exc
    return TaskDefinition(task=task, config=config, digest=digest, yaml_path=str(entry.yaml_path))


class HubTaskResolver:
    """Binds a task through the installed lm-eval and the Hugging Face Hub.

    Args:
        dataset_commit: ``(repository, revision or None) -> commit``. The
            default asks the Hub; a test passes one that does not.
    """

    def __init__(self, dataset_commit: Callable[[str, str | None], str] | None = None) -> None:
        self._dataset_commit = dataset_commit or _hub_dataset_commit

    def bind(self, task: str, num_fewshot: int) -> LMEvalTaskBinding:
        definition = load_task_definition(task)
        config = definition.config
        reasons = list(_definition_refusals(config))
        dataset_path = config.get("dataset_path")
        version = (config.get("metadata") or {}).get("version")
        if version is None:
            reasons.append("the task declares no metadata.version to record")
        metric_list = config.get("metric_list")
        if not metric_list:
            reasons.append(
                "the task declares no metric_list; lm-eval would apply defaults for its "
                "output_type, which the binding could not name"
            )
        commit = None
        if isinstance(dataset_path, str) and not reasons:
            revision = (config.get("dataset_kwargs") or {}).get("revision")
            try:
                commit = self._dataset_commit(dataset_path, revision)
            except Exception as exc:  # the Hub's errors are many; each is a refusal
                reasons.append(
                    f"dataset {dataset_path!r} at revision {revision or 'default'} cannot "
                    f"be pinned to a commit: {exc}"
                )
        if reasons:
            raise UnresolvableTaskError(task, tuple(reasons))
        assert isinstance(dataset_path, str) and commit is not None and metric_list
        return LMEvalTaskBinding(
            task=task,
            task_version=str(version),
            task_config_digest=definition.digest,
            lm_eval_version=LM_EVAL_VERSION,
            dataset_path=dataset_path,
            dataset_name=config.get("dataset_name"),
            dataset_revision=commit,
            output_type=str(config.get("output_type")),
            num_fewshot=num_fewshot,
            metrics=tuple(str(metric.get("metric")) for metric in metric_list),
        )


class LMEvalEvaluator:
    """Prepares one lm-eval benchmark task, bound at submission, on a trained model."""

    descriptor = PluginDescriptor(
        api_version=PLUGIN_API_VERSIONS[0],
        name="lm-eval",
        plugin_version="0.1.0",
        provider="xaytune",
        xaytune_version=__version__,
    )
    determinism = EvaluatorDeterminism.SEEDED

    def __init__(self, resolver: TaskResolver | None = None) -> None:
        self._resolver = HubTaskResolver() if resolver is None else resolver

    def capabilities(self) -> CapabilityDocument:
        """Nothing beyond one local process: no distribution, no accelerator required."""
        return CapabilityDocument()

    def supports(self, spec: EvaluationSpec) -> SupportResult:
        """Whether *spec* can be run exactly as declared -- or, once resolved, as bound.

        A spec with no binding yet is judged on what it declares; one with a
        binding is judged on what the task turned out to be too.
        """
        reasons = tuple(_refusals(spec))
        return SupportResult(supported=not reasons, reasons=reasons)

    def resolve(self, spec: EvaluationSpec) -> EvaluationSpec:
        """*spec*, with the task and its dataset bound. Reads the Hub; called once.

        Raises:
            UnsupportedEvaluationError: If the spec already carries a
                binding, or the task cannot be bound.
        """
        config = LMEvalConfig.model_validate(dict(spec.evaluator.config))
        if config.binding is not None:
            raise UnsupportedEvaluationError(
                self.descriptor.name,
                (
                    "evaluator.config.binding is declared; it records what the task resolved "
                    "to at submission, and is not taken from the caller",
                ),
            )
        try:
            binding = self._resolver.bind(config.task, config.num_fewshot)
        except UnresolvableTaskError as exc:
            raise UnsupportedEvaluationError(self.descriptor.name, exc.reasons) from exc
        resolved = {**thaw(spec.evaluator.config), "binding": binding.model_dump(mode="json")}
        evaluator = spec.evaluator.model_copy(update={"config": resolved})
        return spec.model_copy(update={"evaluator": evaluator})

    def prepare(
        self, subject: ArtifactRef, spec: EvaluationSpec, context: EvaluationContext
    ) -> EvaluationExecutionSpec:
        """How to run the bound task on *subject*. Mechanical and deterministic; reads nothing.

        Raises:
            UnsupportedEvaluationError: With every reason, if the spec is not
                bound, or the spec, subject or run cannot be honoured exactly.
        """
        reasons = list(_refusals(spec))
        config = _config(spec)
        if config is not None and config.binding is None:
            reasons.append(
                "evaluator.config.binding is absent; the task was never resolved, so "
                "nothing says which definition and dataset it means"
            )
        if subject.kind != "model":
            reasons.append(f"the subject is a {subject.kind!r} artifact, not a model")
        model_path = local_path(subject.uri)
        if model_path is None:
            reasons.append(
                f"the subject {subject.uri!r} is not an absolute local path; the worker "
                f"loads a local model directory"
            )
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

        assert config is not None and config.binding is not None
        assert model_path is not None and output_dir is not None and context.seed is not None
        worker_config = LMEvalWorkerConfig(
            api_version=LMEVAL_EVALUATION_API_VERSION,
            model_uri=model_path,
            binding=config.binding,
            measure=LMEvalMeasure(
                batch_size=config.batch_size, precision=config.precision, limit=config.limit
            ),
            realization=LMEvalRealization(
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


def _config(spec: EvaluationSpec) -> LMEvalConfig | None:
    try:
        return LMEvalConfig.model_validate(dict(spec.evaluator.config))
    except ValidationError:
        return None


def _refusals(spec: EvaluationSpec) -> Iterator[str]:
    """Every reason the spec -- and its binding, once it has one -- rules out an exact run."""
    try:
        config = LMEvalConfig.model_validate(dict(spec.evaluator.config))
    except ValidationError as exc:
        for error in exc.errors():
            where = ".".join(str(part) for part in error["loc"]) or "config"
            yield f"evaluator.config.{where}: {error['msg']}"
        config = None

    if spec.dataset is not None:
        yield (
            "dataset is declared; an lm-eval task names its own dataset, which is pinned "
            "when the task is resolved"
        )
    if spec.slices:
        yield (
            f"slices {spec.slices} are declared; the lm-eval evaluator scores one task and "
            f"has no slices to select"
        )
    if config is None or config.binding is None:
        return

    binding = config.binding
    if binding.task != config.task:
        yield f"the binding is for task {binding.task!r}, but the config names {config.task!r}"
    if binding.num_fewshot != config.num_fewshot:
        yield (
            f"the binding is for {binding.num_fewshot}-shot, but the config declares "
            f"{config.num_fewshot}-shot"
        )
    if binding.lm_eval_version != LM_EVAL_VERSION:
        yield (
            f"the task was bound under lm-eval {binding.lm_eval_version}; this evaluator "
            f"runs {LM_EVAL_VERSION} only, whose definition of the task may differ"
        )
    if binding.output_type not in LMEVAL_OUTPUT_TYPES:
        yield (
            f"task {binding.task!r} is {binding.output_type!r}; only "
            f"{', '.join(LMEVAL_OUTPUT_TYPES)} are supported. Generation adds decoding "
            f"settings, stop sequences, filters and answer extraction that decide the "
            f"score and that no binding pins yet"
        )
    unsupported = [metric for metric in binding.metrics if metric not in LMEVAL_METRICS]
    if unsupported:
        yield (
            f"task {binding.task!r} reports {unsupported}; only {', '.join(LMEVAL_METRICS)} "
            f"are supported"
        )


def _definition_refusals(config: Mapping[str, Any]) -> Iterator[str]:
    """What rules a task definition out before its dataset is even looked up."""
    if "class" in config:
        yield "the task is implemented by a custom Python class, whose behaviour no digest pins"
    if config.get("unsafe_code"):
        yield "the task is marked unsafe_code: scoring it executes model-generated code"
    if "custom_dataset" in config:
        yield "the task builds its dataset in custom code, which cannot be pinned to a commit"
    kwargs = config.get("dataset_kwargs") or {}
    if kwargs.get("trust_remote_code"):
        yield "the task's dataset needs trust_remote_code: loading it runs repository code"
    if not isinstance(config.get("dataset_path"), str):
        yield "the task names no dataset_path to pin"


def _portable(value: Any, root: str) -> Any:
    """*value* with the lm-eval install location removed from every string.

    Unresolved ``!function`` references come back as absolute paths into the
    installed package; the same release installed elsewhere must digest the
    same, so the prefix becomes ``lm_eval/``.
    """
    if isinstance(value, str):
        return "lm_eval/" + value[len(root) :] if value.startswith(root) else value
    if isinstance(value, Mapping):
        return {key: _portable(item, root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item, root) for item in value]
    return value


@functools.cache
def task_manager() -> Any:
    """lm-eval's index of its installed tasks, built once per process.

    Building it reads every task file lm-eval ships; the files are part of
    the installed release, which cannot change under a running process.
    """
    from lm_eval.tasks import TaskManager

    return TaskManager()


def _lm_eval(task: str) -> Any:
    try:
        import lm_eval
    except ImportError as exc:
        raise UnresolvableTaskError(
            task,
            (
                "lm-eval is not installed; it is needed to bind the task "
                "(pip install 'xaytune[eval]')",
            ),
        ) from exc
    from importlib.metadata import version

    installed = version("lm-eval")
    if installed != LM_EVAL_VERSION:
        raise UnresolvableTaskError(
            task,
            (
                f"lm-eval {installed} is installed; tasks are bound under {LM_EVAL_VERSION} "
                f"exactly, because the release defines the task",
            ),
        )
    return lm_eval


def _hub_dataset_commit(repository: str, revision: str | None) -> str:
    from huggingface_hub import HfApi

    sha = HfApi().dataset_info(repository, revision=revision).sha
    if not sha:
        raise ValueError("the Hub reported no commit")
    return str(sha)
