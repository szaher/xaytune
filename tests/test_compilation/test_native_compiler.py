"""NativeCompiler: a candidate says everything, or it is refused.

The native trainer predates the control plane, and its ``TrainConfig`` has a
default for nearly every field -- ``learning_rate=2e-4``, ``batch_size=4``,
``num_epochs=3``, ``packing=True``, ``max_seq_length=2048``. A compiler that
left a field unset would not produce an error. It would produce a run whose
behaviour the candidate never described, under a fingerprint that claims to
describe it completely. Two candidates differing only in a value the compiler
filled in would share a fingerprint and not share a result.

So the rule is binary: **a value that changes what the model learns either
comes from the candidate, or** ``supports()`` **refuses the candidate.**

The completeness test works by sentinel rather than by a mapping table written
here. Every scientific value is set to something no legacy default equals; if
any default leaked through, the ``TrainConfig`` the worker builds would show it
instead. A table would only restate what the implementation does. A sentinel
checks what the trainer actually receives.
"""

from __future__ import annotations

import os

import pytest

from xaytune.core.capabilities import require_supported_plugin
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
    """A fully declared SFT candidate, every value distinct from a legacy default."""
    optimization = OptimizationSpec(
        optimizer=OptimizerSpec(name="adamw", weight_decay=0.123),
        lr_schedule=LRScheduleSpec(name="linear", warmup_steps=13),
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
    from xaytune.compilation.native import NativeCompiler

    return NativeCompiler().compile(candidate or _candidate(), _context(**context))


def _train_config(candidate: CandidateSpec | None = None):
    """What the trainer actually receives -- the only thing that matters."""
    from xaytune.workers.native import train_config_from

    return train_config_from(_compile(candidate).config)


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


# ---- no legacy default leaks into the trainer ----------------------------


def test_every_scientific_value_reaches_the_trainer_from_the_candidate() -> None:
    config = _train_config()

    assert config.trainer.learning_rate == 1.234e-5, "not the legacy 2e-4"
    assert config.trainer.batch_size == 7, "not the legacy 4"
    assert config.trainer.gradient_accumulation == 3, "not the legacy 1"
    assert config.trainer.num_epochs == 5, "not the legacy 3"
    assert config.trainer.max_steps == 11, "not the legacy -1"
    assert config.trainer.scheduler == "linear", "not the legacy cosine"
    assert config.trainer.warmup_steps == 13, "not the legacy 0"
    assert config.trainer.weight_decay == 0.123, "not the legacy 0.01"
    assert config.trainer.max_grad_norm == 0.456, "not the legacy 1.0"
    assert config.trainer.mixed_precision == "fp16", "not the legacy bf16"

    assert config.data.format == "text"
    assert config.data.max_seq_length == 333, "not the legacy 2048"
    assert config.data.packing is False, "not the legacy True"


def test_realization_comes_from_the_context_not_the_candidate() -> None:
    """Seed and locations belong to the run, so replicates share a fingerprint."""
    config = _train_config()

    assert config.trainer.seed == 1234, "not the legacy 42"
    assert config.output.dir == "/out/run_1"


def test_a_missing_seed_is_an_error_not_a_default() -> None:
    """A run with no declared seed is not reproducible, whatever it defaults to."""
    with pytest.raises(ValueError, match="seed"):
        _compile(seed=None)


def test_an_absent_step_cap_means_no_cap() -> None:
    """``None`` becomes ``-1`` because that is what "no cap" means to the loop.

    Translation, not a default: the candidate declared no step cap, and the
    trainer's word for no step cap is ``-1``. Training length is still
    bounded by the declared epochs.
    """
    assert _train_config(_with_optimization(max_steps=None)).trainer.max_steps == -1


def test_an_absent_checkpoint_intent_means_no_checkpoints_at_all() -> None:
    """Neither periodic ones nor a final one the candidate never asked for.

    ``0`` rather than the legacy ``500``; and no ``save_last``, because the
    run's output is the model published deliberately after training, and a raw
    state_dict at the end would be a second artifact with a different meaning.
    """
    config = _train_config(_with_training(checkpoint=CheckpointIntent()))

    assert config.trainer.checkpoint_every_n_steps == 0, "not the legacy 500"
    assert config.trainer.save_last is False, "not the legacy True"


def test_every_behaviour_bearing_field_is_chosen_not_inherited() -> None:
    """Checked by what was *set*, not by what the values happen to be.

    A value-equality test passes as long as the translator's choice and the
    legacy default agree today. ``model_fields_set`` asks the question that
    matters: did the translator decide this, or did ``TrainConfig`` decide it
    for us? Only the first survives a change to a default.
    """
    config = _train_config()

    required = [
        (config, _REQUIRED_TOP_LEVEL),
        (config.model, {"name", "quantization", "dtype", "trust_remote_code"}),
        (config.data, _REQUIRED_DATA),
        (config.trainer, _REQUIRED_TRAINER),
        (config.output, {"dir", "merge_on_complete"}),
        (config.eval, {"every_n_steps", "metrics", "benchmarks", "early_stopping_patience"}),
        (config.logging, {"backends", "log_every_n_steps"}),
        (config.online_rl, {"enabled"}),
    ]

    inherited = {
        type(section).__name__: sorted(fields - section.model_fields_set)
        for section, fields in required
        if fields - section.model_fields_set
    }
    assert inherited == {}, f"left to a legacy default: {inherited}"


_REQUIRED_TOP_LEVEL = frozenset(
    "recipe method model data trainer output eval logging online_rl "
    "data_prep method_params base".split()
)
_REQUIRED_DATA = frozenset(
    "path format source eval_split eval_path packing max_seq_length streaming".split()
)
_REQUIRED_TRAINER = frozenset(
    "strategy mixed_precision batch_size gradient_accumulation learning_rate "
    "num_epochs max_steps warmup_steps warmup_ratio scheduler weight_decay "
    "max_grad_norm seed checkpoint_every_n_steps save_last "
    "activation_checkpointing async_checkpoint".split()
)


def test_evaluation_can_never_run_inside_training() -> None:
    """Set to never, not trusted to stay idle.

    It currently stays idle because no evaluation data is supplied. That is an
    argument about today's inputs; ``every_n_steps=0`` is a guarantee.
    """
    config = _train_config()
    assert config.eval.every_n_steps == 0
    assert config.eval.early_stopping_patience == 0
    assert config.online_rl.enabled is False
    assert config.data_prep == []


# ---- a candidate that says too little is refused -------------------------


@pytest.mark.parametrize(
    ("field", "candidate"),
    [
        ("learning_rate", lambda: _with_optimization(learning_rate=None)),
        ("micro_batch_size", lambda: _with_optimization(micro_batch_size=None)),
        ("gradient_accumulation", lambda: _with_optimization(gradient_accumulation=None)),
        ("epochs", lambda: _with_optimization(epochs=None)),
        ("max_grad_norm", lambda: _with_optimization(max_grad_norm=None)),
        ("lr_schedule", lambda: _with_optimization(lr_schedule=None)),
        ("optimizer", lambda: _with_optimization(optimizer=None)),
        # Non-optional with a default factory, so "undeclared" is an empty
        # PrecisionSpec rather than None.
        ("precision", lambda: _with_training(precision=PrecisionSpec())),
        ("format", lambda: _with_data(format=None)),
        ("max_seq_length", lambda: _with_data(max_seq_length=None)),
        ("packing", lambda: _with_data(packing=None)),
    ],
)
def test_an_undeclared_scientific_value_is_refused(field: str, candidate) -> None:
    """The candidate declares it, or the compiler says no -- by name."""
    from xaytune.compilation.native import NativeCompiler

    result = NativeCompiler().supports(candidate())

    assert not result
    assert any(field in reason for reason in result.reasons), result.reasons


# ---- what the native loop cannot honour is refused, not ignored ----------


@pytest.mark.parametrize(
    ("label", "candidate"),
    [
        ("continued pretraining", lambda: _with_training(kind=TrainingKind.CONTINUED_PRETRAIN)),
        ("dpo", lambda: _with_training(kind=TrainingKind.DPO)),
        ("grpo", lambda: _with_training(kind=TrainingKind.GRPO)),
        ("an adapter", lambda: _with_training(adapter=AdapterSpec(type="lora", rank=8))),
        (
            "an optimizer it does not build",
            lambda: _with_optimization(optimizer=OptimizerSpec(name="sgd", weight_decay=0.0)),
        ),
        (
            "betas it cannot set",
            lambda: _with_optimization(
                optimizer=OptimizerSpec(name="adamw", weight_decay=0.1, betas=(0.8, 0.95))
            ),
        ),
        (
            "optimizer parameters it cannot pass",
            lambda: _with_optimization(
                optimizer=OptimizerSpec(name="adamw", weight_decay=0.1, params={"eps": 1e-6})
            ),
        ),
        (
            "periodic checkpoints it cannot report",
            lambda: _with_training(checkpoint=CheckpointIntent(every_optimizer_steps=100)),
        ),
        (
            "a retention count it cannot enforce",
            lambda: _with_training(checkpoint=CheckpointIntent(keep_last=3)),
        ),
        ("a relative dataset path", lambda: _with_data(dataset=DatasetRef(uri="data.jsonl"))),
        ("a remote dataset", lambda: _with_data(dataset=DatasetRef(uri="hf://org/set"))),
        (
            "a model digest it cannot verify",
            lambda: _candidate(model=ModelSpec(model=ModelRef(uri=MODEL, digest="sha256:abc"))),
        ),
        *[
            (
                f"a dataset {field} it cannot verify",
                lambda field=field: _with_data(
                    dataset=DatasetRef(uri=DATASET, **{field: "sha256:abc"})
                ),
            )
            for field in (
                "content_digest",
                "transform_fingerprint",
                "tokenizer_fingerprint",
                "template_fingerprint",
            )
        ],
        (
            "both warmup forms, one of which it would ignore",
            lambda: _with_optimization(
                lr_schedule=LRScheduleSpec(name="linear", warmup_steps=13, warmup_ratio=0.1)
            ),
        ),
    ],
)
def test_what_the_native_loop_cannot_honour_is_refused(label: str, candidate) -> None:
    """The loop builds ``AdamW(lr, weight_decay)`` and nothing else.

    A candidate naming another optimizer, or AdamW with other betas, would
    train on exactly that AdamW regardless -- the declared value silently
    replaced by the loop's. Refusing is the only honest answer.
    """
    from xaytune.compilation.native import NativeCompiler

    result = NativeCompiler().supports(candidate())

    assert not result, f"{label} should be refused"
    assert result.reasons


def test_default_adamw_betas_are_honourable() -> None:
    """The boundary is what the loop can do, not a blanket refusal of betas."""
    from xaytune.compilation.native import NativeCompiler

    explicit = _with_optimization(
        optimizer=OptimizerSpec(name="adamw", weight_decay=0.123, betas=(0.9, 0.999))
    )
    assert NativeCompiler().supports(explicit)


def test_a_fully_declared_sft_candidate_is_supported() -> None:
    from xaytune.compilation.native import NativeCompiler

    result = NativeCompiler().supports(_candidate())
    assert result, result.reasons


# ---- compile() is mechanical ---------------------------------------------


def test_compiling_inspects_nothing() -> None:
    """Paths are opaque strings to a compiler.

    None of these exist. A compiler that checked would be one whose output
    depended on the machine it ran on, and a plan compiled on a laptop could
    not be trusted on the cluster that executes it.
    """
    assert not os.path.exists(MODEL)
    assert not os.path.exists(DATASET)

    spec = _compile()

    assert spec.config["model"]["uri"] == MODEL


def test_compilation_is_deterministic() -> None:
    assert _compile() == _compile()


def test_the_plan_names_the_compiler_that_made_it() -> None:
    """ADR-008, exercised by the first real compiler rather than a synthetic one."""
    from xaytune.compilation.native import NativeCompiler

    compiler = NativeCompiler()
    spec = compiler.compile(_candidate(), _context())

    assert spec.compiler.descriptor == compiler.descriptor
    require_supported_plugin(spec.compiler.descriptor)


def test_the_plan_carries_the_candidates_identity() -> None:
    candidate = _candidate()
    assert _compile(candidate).candidate_fingerprint == candidate.candidate_fingerprint()


def test_the_config_is_not_train_config() -> None:
    """``TrainConfig`` is the worker's detail, not the boundary.

    Exposing it would make the control plane inherit every legacy field and
    default permanently. The config is a versioned native-worker schema that
    the worker translates.
    """
    config = _compile().config

    assert config["api_version"] == "xaytune.native-sft/v1alpha1"
    assert "recipe" not in config, "a TrainConfig field leaked into the boundary"


# ---- identity it cannot verify, and locations it cannot reach -----------


def test_a_declared_identity_constraint_is_named_in_the_refusal() -> None:
    """The fingerprint would carry the digest while the worker ignored it.

    Each identity field is refused by name, so a planner learns which
    constraint the native path cannot keep rather than that "something" failed.
    """
    from xaytune.compilation.native import NativeCompiler

    candidate = _candidate(
        model=ModelSpec(model=ModelRef(uri=MODEL, digest="sha256:m")),
        data=DataSpec(
            dataset=DatasetRef(
                uri=DATASET,
                content_digest="sha256:c",
                transform_fingerprint="sha256:t",
                tokenizer_fingerprint="sha256:k",
                template_fingerprint="sha256:p",
            ),
            format="text",
            max_seq_length=333,
            packing=False,
        ),
    )

    reasons = " ".join(NativeCompiler().supports(candidate).reasons)

    for field in (
        "model.model.digest",
        "data.dataset.content_digest",
        "data.dataset.transform_fingerprint",
        "data.dataset.tokenizer_fingerprint",
        "data.dataset.template_fingerprint",
    ):
        assert field in reasons, field


@pytest.mark.parametrize(
    ("steps", "ratio"),
    [(13, None), (None, 0.1), (13, 0.0), (0, 0.1), (None, None)],
)
def test_one_warmup_form_is_honourable(steps: int | None, ratio: float | None) -> None:
    """The boundary is two *effective* warmups, not two fields being present."""
    from xaytune.compilation.native import NativeCompiler

    candidate = _with_optimization(
        lr_schedule=LRScheduleSpec(name="linear", warmup_steps=steps, warmup_ratio=ratio)
    )

    assert NativeCompiler().supports(candidate)


@pytest.mark.parametrize(
    ("output_uri", "expected"),
    [("/out/run_1", "/out/run_1"), ("file:///out/run_1", "/out/run_1")],
)
def test_the_plan_and_the_worker_agree_on_one_local_output(output_uri: str, expected: str) -> None:
    """The plan's declared output and where the worker writes are one path."""
    plan = _compile(output_uri=output_uri)

    assert plan.outputs[0].uri == expected
    assert plan.config["realization"]["output_dir"] == expected
    assert _train_config_for(plan).output.dir == expected


@pytest.mark.parametrize(
    "output_uri",
    ["out/run_1", "file://out/run_1", "s3://bucket/models/run_1", "https://host/run_1"],
)
def test_an_output_the_worker_cannot_write_to_is_an_error(output_uri: str) -> None:
    """``Path("s3://b/k")`` is the local path ``s3:/b/k``, not an S3 location.

    Accepting it would make the plan, the worker's actual write and the
    ArtifactProduced event name three different places.
    """
    with pytest.raises(ValueError, match="absolute local path"):
        _compile(output_uri=output_uri)


def test_a_file_uri_dataset_reaches_the_worker_as_a_path() -> None:
    """The worker hands the dataset location to ``Path``, which knows no schemes."""
    plan = _compile(_with_data(dataset=DatasetRef(uri=f"file://{DATASET}")))

    assert _train_config_for(plan).data.path == DATASET


def _train_config_for(plan):
    from xaytune.workers.native import train_config_from

    return train_config_from(plan.config)
