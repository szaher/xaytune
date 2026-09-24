"""TRLWorker: runs a compiled TRL SFT config on TRL's ``SFTTrainer``.

```text
TRLSftConfig         the wire contract, from TRLCompiler
      ↓ sft_arguments
SFTConfig            TRL's own configuration (TrainingArguments underneath)
      ↓
SFTTrainer.train()   TRL, unmodified
```

The translation is the only place ``SFTConfig`` appears, so it stays an
implementation detail of this worker rather than part of the execution
contract.

**Every ``SFTConfig`` field is classified, and the classification is
enforced.** ``SFTConfig`` has well over a hundred fields, each with a default
that TRL or transformers may change in any release -- and several of today's
defaults change training outright: ``bf16=True``, ``gradient_checkpointing=True``,
a checkpoint every 500 steps, a fused optimizer, a chunked loss, a 1024-token
truncation, and ``logging_nan_inf_filter=True``, which replaces a NaN loss with
the running average so a diverging run reports as healthy. So each field is
either

- **controlled** -- set explicitly by :func:`sft_arguments`, from the wire
  config or as a deliberate constant; or
- **inert** -- listed in :data:`INERT_FIELDS` with its expected default and the
  reason it cannot affect this run.

:func:`verify_classification` checks both before training starts: a field
that is neither, an inert default that has moved, or a controlled value that
``SFTConfig`` rewrote after construction stops the run. That makes a TRL
upgrade fail loudly until someone has decided what its new or changed fields
mean -- a tripwire rather than a test that happened to pass once.

**The classification is of particular releases**, so the worker checks them
first. :data:`SUPPORTED_VERSIONS` names the TRL and transformers minors it was
made against -- the same ranges the ``trl`` extra pins, so the locked
environment is the classified one -- and :func:`verify_versions` refuses any
other before ``trl`` is imported, naming what is installed and what is
supported. The classification check stays behind it, for a patch release
that changes a default.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xaytune.core.telemetry import (
    NumericalInstabilityObserved,
    TrainingCompletedPayload,
    TrainingFailedPayload,
    TrainingMetricObserved,
    TrainingStartedPayload,
)
from xaytune.runtimes.worker import (
    OBSERVATIONS_PATH_ENV,
    WORKER_CONFIG_PATH_ENV,
    ObservationWriter,
)
from xaytune.workers.common import (
    failure_reason,
    nonfinite,
    publish_model,
    report,
    require_environment,
)
from xaytune.workers.trl_schema import TRLSftConfig

if TYPE_CHECKING:
    from trl import SFTConfig

__all__ = [
    "INERT_FIELDS",
    "SUPPORTED_VERSIONS",
    "TRLObservations",
    "UnclassifiedBehaviourError",
    "UnsupportedTrainerVersionError",
    "main",
    "sft_arguments",
    "verify_classification",
    "verify_versions",
]


class UnclassifiedBehaviourError(RuntimeError):
    """``SFTConfig`` has behaviour this worker has not decided about.

    Raised before training, never during it. The usual cause is a TRL or
    transformers release that added a field or changed a default; the fix is
    to classify it -- control it, or record why it is inert -- not to relax
    the check.
    """


class UnsupportedTrainerVersionError(RuntimeError):
    """The installed TRL or transformers is not a release this worker classified.

    Raised before training, and before ``trl`` is imported. The fix is an
    environment with the supported versions -- ``uv sync --locked --extra
    trl`` gives one -- or, to support a new release, classifying it and then
    widening :data:`SUPPORTED_VERSIONS` and the ``trl`` extra together.
    """


# ---- the supported releases -------------------------------------------------

# The releases INERT_FIELDS and sft_arguments were classified against: TRL's
# SFTConfig fields, and transformers' TrainingArguments underneath them. Equal
# to the ``trl`` extra in pyproject.toml, which a test enforces.
SUPPORTED_VERSIONS: Mapping[str, str] = {
    "trl": ">=1.13,<1.14",
    "transformers": ">=5.17,<5.18",
}


def verify_versions(installed: Mapping[str, str | None] | None = None) -> None:
    """Refuse to train on a TRL or transformers release nobody classified.

    Args:
        installed: Package name to installed version, ``None`` when absent.
            Read from the environment when omitted.

    Raises:
        UnsupportedTrainerVersionError: Naming each package whose installed
            version is missing or outside :data:`SUPPORTED_VERSIONS`.
    """
    from importlib.metadata import PackageNotFoundError, version

    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    if installed is None:
        found: dict[str, str | None] = {}
        for name in SUPPORTED_VERSIONS:
            try:
                found[name] = version(name)
            except PackageNotFoundError:
                found[name] = None
        installed = found

    problems: list[str] = []
    for name, supported in SUPPORTED_VERSIONS.items():
        actual = installed.get(name)
        if actual is None:
            problems.append(f"{name} is not installed (supported: {supported})")
        # A prerelease is refused explicitly: whether a range admits one by
        # default has changed between packaging releases.
        elif not SpecifierSet(supported).contains(Version(actual), prereleases=False):
            problems.append(f"{name} {actual} is installed (supported: {supported})")
    if problems:
        raise UnsupportedTrainerVersionError(
            "this worker's SFTConfig classification was made for other releases: "
            + "; ".join(problems)
            + ". Install the supported versions, e.g. `uv sync --locked --extra trl`"
        )


# ---- the classification ---------------------------------------------------

_EVALUATION = "evaluation never runs: eval_strategy='no' and no eval dataset is passed (ADR-007)"
_SAVING = (
    "nothing is saved by the trainer: save_strategy='no', and the model is published separately"
)
_HUB = "nothing is pushed: push_to_hub=False"
_REPORTING = "no external reporting: report_to=[]; these only label or format logs"
_SINGLE_PROCESS = "one process, one device: LocalRuntime runs a single worker without WORLD_SIZE"
_LOADER_PERFORMANCE = "data-loader performance only, with dataloader_num_workers=0"
_GATED = "read only when a controlled switch that is off turns it on"
_SCRIPT_FLAGS = "flags for HF example scripts; Trainer.train() does not read them"
_PLACEMENT = "device placement belongs to the runtime (CUDA_VISIBLE_DEVICES), not to the candidate"

INERT_FIELDS: Mapping[str, tuple[Any, str]] = {
    # evaluation
    "eval_steps": (None, _EVALUATION),
    "eval_delay": (0, _EVALUATION),
    "per_device_eval_batch_size": (8, _EVALUATION),
    "prediction_loss_only": (False, _EVALUATION),
    "eval_on_start": (False, _EVALUATION),
    "eval_do_concat_batches": (True, _EVALUATION),
    "eval_use_gather_object": (False, _EVALUATION),
    "eval_accumulation_steps": (None, _EVALUATION),
    "include_for_metrics": ([], _EVALUATION),
    "batch_eval_metrics": (False, _EVALUATION),
    "bf16_full_eval": (False, _EVALUATION),
    "fp16_full_eval": (False, _EVALUATION),
    "eval_packing": (None, _EVALUATION),
    "metric_for_best_model": (None, _EVALUATION),
    "greater_is_better": (None, _EVALUATION),
    # saving and resuming
    "save_only_model": (False, _SAVING),
    "save_steps": (500, _SAVING),
    "save_on_each_node": (False, _SAVING),
    "save_total_limit": (None, _SAVING),
    "restore_callback_states_from_checkpoint": (False, _SAVING),
    "ignore_data_skip": (False, "read only when resuming, and train() is never given a checkpoint"),
    "resume_from_checkpoint": (
        None,
        "read by HF example scripts; train() is never given a checkpoint",
    ),
    # hub
    "hub_token": (None, _HUB),
    "hub_private_repo": (None, _HUB),
    "hub_model_id": (None, _HUB),
    "hub_strategy": ("every_save", _HUB),
    "hub_always_push": (False, _HUB),
    "hub_revision": (None, _HUB),
    # logging presentation
    "log_on_each_node": (True, _REPORTING),
    "log_level": ("passive", _REPORTING),
    "log_level_replica": ("warning", _REPORTING),
    "run_name": (None, _REPORTING),
    "project": ("huggingface", _REPORTING),
    "trackio_space_id": (None, _REPORTING),
    "trackio_bucket_id": (None, _REPORTING),
    "trackio_static_space_id": (None, _REPORTING),
    "include_num_input_tokens_seen": ("no", _REPORTING),
    "skip_memory_metrics": (True, _REPORTING),
    # one process
    "local_rank": (-1, _SINGLE_PROCESS),
    "ddp_find_unused_parameters": (None, _SINGLE_PROCESS),
    "ddp_bucket_cap_mb": (None, _SINGLE_PROCESS),
    "ddp_broadcast_buffers": (None, _SINGLE_PROCESS),
    "ddp_static_graph": (None, _SINGLE_PROCESS),
    "ddp_backend": (None, _SINGLE_PROCESS),
    "ddp_timeout": (1800, _SINGLE_PROCESS),
    "fsdp": (None, _SINGLE_PROCESS),
    "fsdp_config": (None, _SINGLE_PROCESS),
    "deepspeed": (None, _SINGLE_PROCESS),
    "parallelism_config": (None, _SINGLE_PROCESS),
    "average_tokens_across_devices": (True, _SINGLE_PROCESS),
    "accelerator_config": (
        {
            "split_batches": False,
            "dispatch_batches": None,
            "even_batches": True,
            "use_seedable_sampler": True,
            "non_blocking": False,
            "gradient_accumulation_kwargs": None,
            "use_configured_state": False,
        },
        _SINGLE_PROCESS,
    ),
    # loader performance
    "dataloader_pin_memory": (True, _LOADER_PERFORMANCE),
    "dataloader_persistent_workers": (False, _LOADER_PERFORMANCE),
    "dataloader_prefetch_factor": (None, _LOADER_PERFORMANCE),
    "dataloader_multiprocessing_context": (None, _LOADER_PERFORMANCE),
    "dataloader_in_order": (True, _LOADER_PERFORMANCE),
    "dataset_num_proc": (None, "parallelism of dataset preparation; the result is the same"),
    # gated by a controlled switch
    "torch_compile_backend": (None, _GATED + " (torch_compile)"),
    "torch_compile_mode": (None, _GATED + " (torch_compile)"),
    "liger_kernel_config": (None, _GATED + " (use_liger_kernel)"),
    "gradient_checkpointing_kwargs": (None, _GATED + " (gradient_checkpointing)"),
    "optim_target_modules": (None, _GATED + " (optim: only GaLore-style optimizers read it)"),
    "router_aux_loss_coef": (
        0.001,
        "on transformers >= 5 the MoE aux-loss coefficient is read from the model's own "
        "config, as it is by the model's forward on the native trainer",
    ),
    "length_column_name": ("length", "read only by length-grouped sampling, which is not used"),
    "label_names": (None, "the SFT collator produces 'labels', which Trainer finds by default"),
    "torch_empty_cache_steps": (None, "memory housekeeping; does not change what is computed"),
    "use_cache": (False, "the KV cache serves generation, and training computes no generation"),
    # script flags
    "do_train": (False, _SCRIPT_FLAGS),
    "do_eval": (False, _SCRIPT_FLAGS),
    "do_predict": (False, _SCRIPT_FLAGS),
    "debug": ([], "debug instrumentation; off"),
    # placement
    "use_cpu": (False, _PLACEMENT),
}


def sft_arguments(spec: TRLSftConfig, *, trainer_dir: str) -> dict[str, Any]:
    """Every controlled ``SFTConfig`` field, and its value for *spec*.

    Pure and TRL-free, so the mapping is testable without the training stack.
    *trainer_dir* is where ``Trainer`` would put its own files; nothing should
    land there, because saving is off -- it is kept apart from the declared
    output so that a trainer writing anything cannot be mistaken for, or
    corrupt, the published model.
    """
    optimization = spec.optimization
    precision = optimization.mixed_precision
    return {
        "output_dir": trainer_dir,
        # --- what the candidate declared
        "learning_rate": optimization.learning_rate,
        "per_device_train_batch_size": optimization.micro_batch_size,
        "gradient_accumulation_steps": optimization.gradient_accumulation,
        "num_train_epochs": float(optimization.epochs),
        "max_steps": -1 if optimization.max_steps is None else optimization.max_steps,
        "max_grad_norm": optimization.max_grad_norm,
        "weight_decay": optimization.weight_decay,
        "adam_beta1": optimization.adam_beta1,
        "adam_beta2": optimization.adam_beta2,
        "adam_epsilon": optimization.adam_epsilon,
        "lr_scheduler_type": optimization.scheduler,
        "lr_scheduler_kwargs": {},
        # A count, never a ratio: below 1 TrainingArguments reads it as one.
        "warmup_steps": optimization.warmup_steps,
        "bf16": precision == "bf16",
        "fp16": precision == "fp16",
        # torch's own default. Unset, transformers may switch TF32 on.
        "tf32": False,
        "max_length": spec.data.max_length,
        # --- realization
        "seed": spec.realization.seed,
        "data_seed": spec.realization.seed,
        # --- what "adamw" means on both trainers: torch.optim.AdamW, unfused
        "optim": "adamw_torch",
        "optim_args": None,
        # --- data semantics: plain text, the whole sequence is the target
        "dataset_text_field": "text",
        "dataset_kwargs": None,
        "packing": False,
        "packing_strategy": "bfd",
        "padding_free": False,
        "pad_to_multiple_of": None,
        "truncation_mode": "keep_start",
        "completion_only_loss": False,
        "assistant_only_loss": False,
        "shuffle_dataset": False,
        "chat_template_path": None,
        # None means the tokenizer's own tokens, which is what the native
        # trainer uses too.
        "eos_token": None,
        "pad_token": None,
        "remove_unused_columns": True,
        "train_sampling_strategy": "random",
        "dataloader_drop_last": False,
        "dataloader_num_workers": 0,
        # --- loss and numerics: nothing that changes what is optimized
        "loss_type": "nll",
        "label_smoothing_factor": 0.0,
        "neftune_noise_alpha": None,
        "use_liger_kernel": False,
        "torch_compile": False,
        "gradient_checkpointing": False,
        "activation_offloading": False,
        "full_determinism": False,
        "auto_find_batch_size": False,
        # --- the model is loaded by this worker, not by SFTTrainer
        "model_init_kwargs": None,
        "trust_remote_code": False,
        # --- no evaluation, no saving, no pushing, no external reporting
        "eval_strategy": "no",
        "load_best_model_at_end": False,
        "save_strategy": "no",
        "enable_jit_checkpoint": False,
        "push_to_hub": False,
        # An empty list is "no integrations"; "none" is rewritten to it.
        "report_to": [],
        # --- observation: every optimizer step, and a NaN reported as a NaN
        "logging_strategy": "steps",
        "logging_steps": 1,
        "logging_first_step": False,
        "logging_nan_inf_filter": False,
        "disable_tqdm": True,
    }


def _normalized(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value


def verify_classification(config: SFTConfig, controlled: Mapping[str, Any]) -> None:
    """Refuse to train on behaviour nobody has decided about.

    Raises:
        UnclassifiedBehaviourError: Naming every field that is unclassified,
            whose inert default has moved, or whose controlled value
            ``SFTConfig`` changed after construction.
    """
    fields = {field.name for field in dataclasses.fields(config) if field.init}
    problems: list[str] = []

    unclassified = sorted(fields - set(controlled) - set(INERT_FIELDS))
    if unclassified:
        problems.append(f"unclassified fields: {', '.join(unclassified)}")
    vanished = sorted((set(controlled) | set(INERT_FIELDS)) - fields)
    if vanished:
        problems.append(f"classified fields SFTConfig no longer has: {', '.join(vanished)}")

    for name, (expected, _reason) in INERT_FIELDS.items():
        if name in fields and _normalized(getattr(config, name)) != expected:
            problems.append(
                f"{name} defaults to {_normalized(getattr(config, name))!r}, "
                f"not the {expected!r} it was classified as inert at"
            )

    for name, wanted in controlled.items():
        if name not in fields:
            continue
        actual = _normalized(getattr(config, name))
        if actual != wanted:
            problems.append(f"{name} was set to {wanted!r} but SFTConfig holds {actual!r}")

    if problems:
        raise UnclassifiedBehaviourError(
            "SFTConfig has behaviour this worker has not classified; classify it in "
            "xaytune.workers.trl before running: " + "; ".join(problems)
        )


# ---- observations ---------------------------------------------------------


class TRLObservations:
    """Turns ``Trainer`` callbacks into observations, and does nothing else.

    **Observational only**: it never returns or mutates ``TrainerControl``,
    so it cannot stop, save, evaluate or log on the trainer's behalf (ADR-011).
    A ``TrainerCallback`` subclass is built from it in :func:`_callback`, so
    this class stays importable without transformers.

    **Reports the trainer's numbers as the trainer means them.** With
    ``logging_steps=1`` each ``on_log`` covers exactly one optimizer step:
    ``loss`` is that step's loss, normalized across gradient accumulation by
    ``Trainer`` itself -- so unlike the native trainer, it is reported as
    ``loss`` under accumulation too. ``learning_rate`` is the rate the step
    used: ``Trainer`` reads it before the scheduler steps.
    """

    def __init__(self, writer: ObservationWriter) -> None:
        self._writer = writer

    def train_begin(self, global_step: int) -> None:
        report(self._writer, lambda: TrainingStartedPayload(optimizer_step=global_step))

    def log(self, global_step: int, logs: Mapping[str, Any]) -> None:
        # The end-of-training summary is logged too, as train_loss; only
        # per-step logs carry "loss".
        if "loss" not in logs:
            return
        loss = float(logs["loss"])
        if not math.isfinite(loss):
            report(
                self._writer,
                lambda: NumericalInstabilityObserved(
                    optimizer_step=global_step, quantity="loss", observation=nonfinite(loss)
                ),
            )
            return

        grad_norm = logs.get("grad_norm")
        if grad_norm is not None and not math.isfinite(grad_norm):
            report(
                self._writer,
                lambda: NumericalInstabilityObserved(
                    optimizer_step=global_step,
                    quantity="gradient_norm",
                    observation=nonfinite(grad_norm),
                ),
            )
            grad_norm = None

        report(
            self._writer,
            lambda: TrainingMetricObserved(
                optimizer_step=global_step,
                loss=loss,
                learning_rate=logs.get("learning_rate"),
                gradient_norm=grad_norm,
            ),
        )

    def train_end(self, global_step: int) -> None:
        """The loop ended normally, which includes reaching ``max_steps``.

        ``Trainer`` fires this only on the normal path, never after an
        exception, so it means what ``TrainingCompleted`` means.
        """
        report(self._writer, lambda: TrainingCompletedPayload(optimizer_step=global_step))


def _callback(observations: TRLObservations) -> Any:
    from transformers import TrainerCallback

    class _Observe(TrainerCallback):  # type: ignore[misc]
        def on_train_begin(self, args: Any, state: Any, control: Any, **_: Any) -> None:
            observations.train_begin(state.global_step)

        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **_: Any) -> None:
            if logs:
                observations.log(state.global_step, logs)

        def on_train_end(self, args: Any, state: Any, control: Any, **_: Any) -> None:
            observations.train_end(state.global_step)

    return _Observe()


# ---- the run --------------------------------------------------------------


def _read_text_dataset(path: str) -> list[dict[str, str]]:
    """The file's ``{"text": ...}`` records, or an error naming the bad line.

    Read here rather than by ``datasets.load_dataset``, which caches under the
    user's home directory and resolves paths its own way.
    """
    records: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"{path}:{number} is not a {{'text': str}} record")
            records.append({"text": record["text"]})
    if not records:
        raise ValueError(f"{path} contains no records")
    return records


def main(*_arguments: str) -> int:
    """Run one compiled TRL SFT config, reporting as it goes."""
    config_path = Path(require_environment(WORKER_CONFIG_PATH_ENV))
    writer = ObservationWriter(Path(require_environment(OBSERVATIONS_PATH_ENV)))
    writer.verify()

    spec = TRLSftConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    observations = TRLObservations(writer)

    with tempfile.TemporaryDirectory(prefix="xaytune-trl-") as trainer_dir:
        try:
            model, tokenizer = _train(spec, trainer_dir, observations)
        except Exception as exc:
            # Trainer fires no callback on an exception, so this boundary is
            # the only place that can report one (see NativeWorker).
            try:
                writer.write(TrainingFailedPayload(reason=failure_reason(exc), detail=str(exc)))
            except OSError:
                pass
            raise

    publish_model(model, tokenizer, Path(spec.realization.output_dir), writer)
    return 0


def _train(spec: TRLSftConfig, trainer_dir: str, observations: TRLObservations) -> Any:
    # Before importing trl: an unsupported release may not import cleanly,
    # and its error would not say that the release is the problem.
    verify_versions()
    # Heavy imports here, so the classification above stays importable.
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    controlled = sft_arguments(spec, trainer_dir=trainer_dir)
    sft_config = SFTConfig(**controlled)
    verify_classification(sft_config, controlled)

    dataset = Dataset.from_list(_read_text_dataset(spec.data.path))
    model = AutoModelForCausalLM.from_pretrained(
        spec.model.uri, dtype="auto", trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained(spec.model.uri, trust_remote_code=False)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[_callback(observations)],
    )
    trainer.train()
    return trainer.model, tokenizer
