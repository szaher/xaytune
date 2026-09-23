"""TRLCompiler: the candidate says everything, or it is refused -- on TRL's terms.

The native compiler's rule, applied to a trainer Xaytune does not own. Two
kinds of test:

- **the value reaches the trainer**: a sentinel candidate, every value distinct
  from any ``SFTConfig`` default, checked in the arguments the worker hands to
  ``SFTConfig``. No TRL needed: :func:`~xaytune.workers.trl.sft_arguments` is
  pure.
- **the value TRL would honour differently is refused**: the places where the
  same declared number means something else on TRL than on the native trainer.

And one about capability: TRL can do what the native compiler must refuse, so
the two compilers accepting different candidates is tested as intended.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from xaytune.core.domain.candidate import (
    AdapterSpec,
    CandidateSpec,
    CheckpointIntent,
    DataSpec,
    LRScheduleSpec,
    ModelSpec,
    OptimizationSpec,
    OptimizerSpec,
    PrecisionSpec,
    TrainingKind,
    TrainingSpec,
)
from xaytune.core.refs import DatasetRef, ModelRef

MODEL = "/models/tiny"
DATASET = "/data/train.jsonl"


def _candidate(**overrides: object) -> CandidateSpec:
    """A fully declared text SFT candidate, every value distinct from an SFTConfig default."""
    optimization = OptimizationSpec(
        optimizer=OptimizerSpec(name="adamw", weight_decay=0.0, betas=(0.85, 0.95)),
        lr_schedule=LRScheduleSpec(name="cosine", warmup_steps=13),
        learning_rate=1.234e-5,
        micro_batch_size=7,
        gradient_accumulation=3,
        epochs=5,
        max_steps=11,
        max_grad_norm=0.456,
    )
    training = TrainingSpec(
        kind=TrainingKind.SFT,
        optimization=optimization,
        precision=PrecisionSpec(dtype="fp16"),
    )
    fields: dict[str, object] = {
        "model": ModelSpec(model=ModelRef(uri=MODEL)),
        "data": DataSpec(
            dataset=DatasetRef(uri=DATASET), format="text", max_seq_length=333, packing=False
        ),
        "training": training,
    }
    fields.update(overrides)
    return CandidateSpec(**fields)  # type: ignore[arg-type]


def _context(**overrides: object):
    from xaytune.compilation import CompilationContext

    fields: dict[str, object] = {
        "run_id": "run_1",
        "seed": 1234,
        "output_uri": "/out/run_1",
        "checkpoint_store_uri": "/ckpt/run_1",
    }
    fields.update(overrides)
    return CompilationContext(**fields)  # type: ignore[arg-type]


def _compile(candidate: CandidateSpec | None = None, **context: object):
    from xaytune.compilation.trl import TRLCompiler

    return TRLCompiler().compile(candidate or _candidate(), _context(**context))


def _arguments(candidate: CandidateSpec | None = None) -> dict:
    """What ``SFTConfig`` is constructed from -- the only thing that matters."""
    from xaytune.workers.trl import sft_arguments
    from xaytune.workers.trl_schema import TRLSftConfig

    spec = TRLSftConfig.model_validate(dict(_compile(candidate).config))
    return sft_arguments(spec, trainer_dir="/scratch")


def _with_optimization(**changes: object) -> CandidateSpec:
    base = _candidate()
    optimization = base.training.optimization.model_copy(update=changes)
    return base.model_copy(
        update={"training": base.training.model_copy(update={"optimization": optimization})}
    )


def _with_data(**changes: object) -> CandidateSpec:
    base = _candidate()
    return base.model_copy(update={"data": base.data.model_copy(update=changes)})


def _with_training(**changes: object) -> CandidateSpec:
    base = _candidate()
    return base.model_copy(update={"training": base.training.model_copy(update=changes)})


def _supports(candidate: CandidateSpec):
    from xaytune.compilation.trl import TRLCompiler

    return TRLCompiler().supports(candidate)


# ---- every declared value reaches SFTConfig -------------------------------


def test_every_scientific_value_reaches_sft_config_from_the_candidate() -> None:
    arguments = _arguments()

    assert arguments["learning_rate"] == 1.234e-5, "not TRL's 2e-5"
    assert arguments["per_device_train_batch_size"] == 7, "not TRL's 8"
    assert arguments["gradient_accumulation_steps"] == 3, "not TRL's 1"
    assert arguments["num_train_epochs"] == 5.0, "not TRL's 3.0"
    assert arguments["max_steps"] == 11, "not TRL's -1"
    assert arguments["lr_scheduler_type"] == "cosine", "not TRL's linear"
    assert arguments["warmup_steps"] == 13, "not TRL's 0"
    assert arguments["max_grad_norm"] == 0.456, "not TRL's 1.0"
    assert (arguments["adam_beta1"], arguments["adam_beta2"]) == (0.85, 0.95)
    assert arguments["fp16"] is True and arguments["bf16"] is False, "not TRL's bf16=True"
    assert arguments["max_length"] == 333, "not TRL's 1024"
    assert arguments["packing"] is False


def test_realization_comes_from_the_context_not_the_candidate() -> None:
    arguments = _arguments()

    assert arguments["seed"] == 1234, "not TRL's 42"
    assert arguments["data_seed"] == 1234


@pytest.mark.parametrize(("dtype", "bf16", "fp16"), [("fp32", False, False), ("bf16", True, False)])
def test_precision_is_exactly_what_was_declared(dtype: str, bf16: bool, fp16: bool) -> None:
    """``bf16=True`` is TRL's default; an fp32 candidate must not inherit it."""
    arguments = _arguments(_with_training(precision=PrecisionSpec(dtype=dtype)))

    assert (arguments["bf16"], arguments["fp16"]) == (bf16, fp16)
    assert arguments["tf32"] is False, "and TF32 is not switched on behind it"


def test_undeclared_betas_mean_the_same_numbers_on_both_trainers() -> None:
    """Absent betas are AdamW's defaults -- torch's, not TrainingArguments'."""
    arguments = _arguments(
        _with_optimization(optimizer=OptimizerSpec(name="adamw", weight_decay=0.0))
    )

    assert (arguments["adam_beta1"], arguments["adam_beta2"]) == (0.9, 0.999)
    assert arguments["adam_epsilon"] == 1e-8
    assert arguments["optim"] == "adamw_torch", "the unfused torch AdamW, not TRL's fused default"


def test_a_declared_warmup_is_honoured_under_a_constant_schedule() -> None:
    """transformers' ``constant`` ignores warmup; the candidate declared one.

    The native trainer warms up under this declaration too, so both trainers
    do what was declared.
    """
    arguments = _arguments(
        _with_optimization(lr_schedule=LRScheduleSpec(name="constant", warmup_steps=5))
    )

    assert arguments["lr_scheduler_type"] == "constant_with_warmup"
    assert arguments["warmup_steps"] == 5


def test_a_constant_schedule_without_warmup_stays_constant() -> None:
    arguments = _arguments(_with_optimization(lr_schedule=LRScheduleSpec(name="constant")))

    assert arguments["lr_scheduler_type"] == "constant"
    assert arguments["warmup_steps"] == 0


def test_nothing_is_saved_evaluated_pushed_or_reported() -> None:
    arguments = _arguments()

    assert arguments["save_strategy"] == "no", "not TRL's checkpoint every 500 steps"
    assert arguments["eval_strategy"] == "no"
    assert arguments["push_to_hub"] is False
    assert arguments["report_to"] == []


def test_a_nan_is_reported_as_a_nan() -> None:
    """``logging_nan_inf_filter=True`` replaces a NaN loss with the running mean."""
    arguments = _arguments()

    assert arguments["logging_nan_inf_filter"] is False
    assert arguments["logging_steps"] == 1, "one observation per optimizer step"


def test_nothing_changes_what_is_optimized_behind_the_candidate() -> None:
    arguments = _arguments()

    assert arguments["loss_type"] == "nll", "not TRL's chunked_nll"
    assert arguments["gradient_checkpointing"] is False, "not TRL's True"
    assert arguments["neftune_noise_alpha"] is None
    assert arguments["label_smoothing_factor"] == 0.0
    assert arguments["auto_find_batch_size"] is False, "which would change the batch size"
    assert arguments["completion_only_loss"] is False
    assert arguments["assistant_only_loss"] is False


def test_the_trainer_never_writes_where_the_model_is_published() -> None:
    """The declared output is written once, by publication, and by nothing else."""
    arguments = _arguments()

    assert arguments["output_dir"] == "/scratch"
    assert arguments["output_dir"] != "/out/run_1"


# ---- what TRL would honour differently is refused -------------------------


@pytest.mark.parametrize(
    ("label", "candidate"),
    [
        ("alpaca data", lambda: _with_data(format="alpaca")),
        ("chat data", lambda: _with_data(format="chat")),
        ("packing", lambda: _with_data(packing=True)),
        (
            "weight decay applied to a different parameter set",
            lambda: _with_optimization(optimizer=OptimizerSpec(name="adamw", weight_decay=0.1)),
        ),
        (
            "a warmup ratio rounded differently",
            lambda: _with_optimization(lr_schedule=LRScheduleSpec(name="linear", warmup_ratio=0.1)),
        ),
        (
            "betas that are not a pair",
            lambda: _with_optimization(
                optimizer=OptimizerSpec(name="adamw", weight_decay=0.0, betas=(0.9, 0.99, 0.999))
            ),
        ),
        # The shared boundary rules hold here as they do for the native compiler.
        ("dpo", lambda: _with_training(kind=TrainingKind.DPO)),
        ("an adapter", lambda: _with_training(adapter=AdapterSpec(type="lora", rank=8))),
        (
            "another optimizer",
            lambda: _with_optimization(optimizer=OptimizerSpec(name="sgd", weight_decay=0.0)),
        ),
        (
            "periodic checkpoints",
            lambda: _with_training(checkpoint=CheckpointIntent(every_optimizer_steps=100)),
        ),
        (
            "a model digest it cannot verify",
            lambda: _candidate(model=ModelSpec(model=ModelRef(uri=MODEL, digest="sha256:abc"))),
        ),
        (
            "a dataset fingerprint it cannot verify",
            lambda: _with_data(dataset=DatasetRef(uri=DATASET, tokenizer_fingerprint="sha256:t")),
        ),
        ("a relative dataset path", lambda: _with_data(dataset=DatasetRef(uri="data.jsonl"))),
    ],
)
def test_what_trl_would_honour_differently_is_refused(label: str, candidate) -> None:
    result = _supports(candidate())

    assert not result, f"{label} should be refused"
    assert result.reasons


def test_every_reason_is_reported_not_the_first() -> None:
    result = _supports(
        _with_data(format="alpaca", packing=True).model_copy(
            update={
                "training": _with_optimization(
                    optimizer=OptimizerSpec(name="adamw", weight_decay=0.1)
                ).training
            }
        )
    )

    joined = " ".join(result.reasons)
    assert "data.format" in joined
    assert "data.packing" in joined
    assert "weight_decay" in joined


def test_a_fully_declared_text_candidate_is_supported() -> None:
    result = _supports(_candidate())
    assert result, result.reasons


# ---- two compilers, two capability sets -----------------------------------


def test_trl_honours_betas_the_native_trainer_cannot_set() -> None:
    """Capability resolution, not inconsistency: the same candidate, two answers."""
    from xaytune.compilation.native import NativeCompiler

    candidate = _candidate()

    assert _supports(candidate), "TRL passes AdamW's betas through"
    native = NativeCompiler().supports(candidate)
    assert not native
    assert any("betas" in reason for reason in native.reasons)


def test_both_compilers_refuse_under_their_own_name() -> None:
    """A planner reading a refusal can tell which compiler said no."""
    result = _supports(_with_training(adapter=AdapterSpec(type="lora", rank=8)))

    assert any("the TRL trainer" in reason for reason in result.reasons)


# ---- compile() is mechanical ----------------------------------------------


def test_the_plan_names_the_trl_worker_and_its_producer() -> None:
    from xaytune.compilation.trl import TRLCompiler

    spec = _compile()

    assert spec.compiler.name == "trl"
    assert spec.compiler.descriptor == TRLCompiler.descriptor
    assert spec.entrypoint.module == "xaytune.workers.trl"
    assert spec.telemetry.protocol_version == "xaytune.telemetry/v1alpha2"
    assert spec.candidate_fingerprint == _candidate().candidate_fingerprint()


def test_compilation_is_deterministic() -> None:
    assert _compile() == _compile()


def test_a_missing_seed_is_an_error_not_a_default() -> None:
    with pytest.raises(ValueError, match="seed"):
        _compile(seed=None)


@pytest.mark.parametrize(
    "output_uri", ["out/run_1", "s3://bucket/models/run_1", "https://host/run_1"]
)
def test_an_output_the_worker_cannot_write_to_is_an_error(output_uri: str) -> None:
    with pytest.raises(ValueError, match="absolute local path"):
        _compile(output_uri=output_uri)


def test_the_plan_and_the_worker_agree_on_one_local_output() -> None:
    spec = _compile(output_uri="file:///out/run_1")

    assert spec.outputs[0].uri == "/out/run_1"
    assert spec.config["realization"]["output_dir"] == "/out/run_1"


def test_compiling_needs_no_trl() -> None:
    """A controller host compiles plans without the worker's training stack.

    Run in a fresh interpreter with ``trl`` made unimportable, because in this
    one it may already have been imported by another test.
    """
    program = (
        "import sys; sys.modules['trl'] = None\n"
        "from xaytune.compilation.trl import TRLCompiler\n"
        "import xaytune.workers.trl_schema\n"
        "assert 'transformers' not in sys.modules, 'the compiler imported transformers'\n"
        "assert 'torch' not in sys.modules, 'the compiler imported torch'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


# ---- the model is one the worker can pin ----------------------------------


@pytest.mark.parametrize("compiler", ["native", "trl"])
@pytest.mark.parametrize("uri", ["Qwen/Qwen3-8B", "gpt2", "models/tiny", "hf://org/model"])
def test_a_model_the_worker_cannot_pin_is_refused_by_both(compiler: str, uri: str) -> None:
    """A hub name loads whatever it points at the day the worker starts.

    One rule in ``_sft``, so the two compilers cannot disagree about it.
    """
    from xaytune.compilation.native import NativeCompiler
    from xaytune.compilation.trl import TRLCompiler

    instance = NativeCompiler() if compiler == "native" else TRLCompiler()
    result = instance.supports(_candidate(model=ModelSpec(model=ModelRef(uri=uri))))

    assert not result
    assert any("model.model.uri" in reason for reason in result.reasons)


@pytest.mark.parametrize("compiler", ["native", "trl"])
def test_a_file_uri_model_reaches_the_worker_as_a_path(compiler: str) -> None:
    """``from_pretrained`` knows no schemes, as ``Path`` does not."""
    from xaytune.compilation.native import NativeCompiler
    from xaytune.compilation.trl import TRLCompiler

    instance = NativeCompiler() if compiler == "native" else TRLCompiler()
    candidate = _candidate(model=ModelSpec(model=ModelRef(uri=f"file://{MODEL}")))
    if compiler == "native":
        # The native trainer cannot set betas; this candidate declares some.
        candidate = candidate.model_copy(
            update={
                "training": candidate.training.model_copy(
                    update={
                        "optimization": candidate.training.optimization.model_copy(
                            update={"optimizer": OptimizerSpec(name="adamw", weight_decay=0.0)}
                        )
                    }
                )
            }
        )

    spec = instance.compile(candidate, _context())

    assert spec.config["model"]["uri"] == MODEL
