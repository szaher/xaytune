import pytest
from pydantic import ValidationError

from trainlib.config.schema import (
    DataConfig,
    DeepSpeedConfig,
    EvalConfig,
    FSDPConfig,
    LoggingConfig,
    LoraConfig,
    ModelConfig,
    OutputConfig,
    TrainConfig,
    TrainerConfig,
)


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig(name="meta-llama/Llama-3.1-8B")
        assert cfg.name == "meta-llama/Llama-3.1-8B"
        assert cfg.quantization is None
        assert cfg.dtype == "auto"
        assert cfg.trust_remote_code is False

    def test_with_quantization(self):
        cfg = ModelConfig(name="my-model", quantization="4bit")
        assert cfg.quantization == "4bit"

    def test_invalid_quantization(self):
        with pytest.raises(ValueError):
            ModelConfig(name="my-model", quantization="3bit")


class TestLoraConfig:
    def test_defaults(self):
        cfg = LoraConfig()
        assert cfg.rank == 16
        assert cfg.alpha == 32
        assert cfg.dropout == 0.05
        assert cfg.target_modules == "auto"

    def test_custom(self):
        cfg = LoraConfig(rank=64, alpha=128, target_modules=["q_proj", "v_proj"])
        assert cfg.rank == 64
        assert cfg.target_modules == ["q_proj", "v_proj"]


class TestDataConfig:
    def test_minimal(self):
        cfg = DataConfig(path="data.jsonl", format="alpaca")
        assert cfg.path == "data.jsonl"
        assert cfg.format == "alpaca"
        assert cfg.eval_split == 0.0
        assert cfg.packing is True
        assert cfg.max_seq_length == 2048

    def test_streaming(self):
        cfg = DataConfig(path="corpus/", format="text", streaming=True)
        assert cfg.streaming is True


class TestTrainerConfig:
    def test_defaults(self):
        cfg = TrainerConfig()
        assert cfg.strategy == "auto"
        assert cfg.mixed_precision == "bf16"
        assert cfg.batch_size == 4
        assert cfg.gradient_accumulation == 1
        assert cfg.learning_rate == 2e-4
        assert cfg.num_epochs == 3
        assert cfg.max_steps == -1
        assert cfg.warmup_steps == 0
        assert cfg.weight_decay == 0.01
        assert cfg.max_grad_norm == 1.0
        assert cfg.seed == 42

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            TrainerConfig(strategy="invalid")

    def test_checkpoint_defaults(self):
        cfg = TrainerConfig()
        assert cfg.checkpoint_every_n_steps == 500
        assert cfg.save_last is True

    def test_activation_checkpointing_default_false(self):
        cfg = TrainerConfig()
        assert cfg.activation_checkpointing is False

    def test_activation_checkpointing_enabled(self):
        cfg = TrainerConfig(activation_checkpointing=True)
        assert cfg.activation_checkpointing is True

    def test_async_checkpoint_default_false(self):
        cfg = TrainerConfig()
        assert cfg.async_checkpoint is False

    def test_async_checkpoint_enabled(self):
        cfg = TrainerConfig(async_checkpoint=True)
        assert cfg.async_checkpoint is True

    def test_scheduler_default_cosine(self):
        cfg = TrainerConfig()
        assert cfg.scheduler == "cosine"

    def test_scheduler_valid_values(self):
        for sched in ("cosine", "linear", "constant", "constant_with_warmup"):
            cfg = TrainerConfig(scheduler=sched)
            assert cfg.scheduler == sched

    def test_scheduler_invalid_value(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TrainerConfig(scheduler="invalid")


class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.every_n_steps == 500
        assert cfg.metrics == ["loss", "perplexity"]
        assert cfg.benchmarks == []


class TestLoggingConfig:
    def test_defaults(self):
        cfg = LoggingConfig()
        assert cfg.backends == ["console"]
        assert cfg.project is None
        assert cfg.log_every_n_steps == 10

    def test_with_wandb(self):
        cfg = LoggingConfig(backends=["console", "wandb"], project="my-run")
        assert "wandb" in cfg.backends


class TestOutputConfig:
    def test_defaults(self):
        cfg = OutputConfig()
        assert cfg.dir == "output"
        assert cfg.merge_on_complete is False

    def test_custom(self):
        cfg = OutputConfig(dir="my-output", merge_on_complete=True)
        assert cfg.dir == "my-output"


class TestFSDPConfig:
    def test_defaults(self):
        cfg = FSDPConfig()
        assert cfg.sharding_strategy == "full_shard"
        assert cfg.cpu_offload is False
        assert cfg.backward_prefetch is None
        assert cfg.mixed_precision is True

    def test_custom(self):
        cfg = FSDPConfig(
            sharding_strategy="shard_grad_op",
            cpu_offload=True,
            backward_prefetch="backward_pre",
            mixed_precision=False,
        )
        assert cfg.sharding_strategy == "shard_grad_op"
        assert cfg.cpu_offload is True
        assert cfg.backward_prefetch == "backward_pre"
        assert cfg.mixed_precision is False

    def test_invalid_sharding_strategy(self):
        with pytest.raises(ValidationError):
            FSDPConfig(sharding_strategy="invalid")


class TestDeepSpeedConfig:
    def test_defaults(self):
        cfg = DeepSpeedConfig()
        assert cfg.zero_stage == 2
        assert cfg.config_file is None

    def test_custom(self):
        cfg = DeepSpeedConfig(config_file="ds_config.json", zero_stage=3)
        assert cfg.config_file == "ds_config.json"
        assert cfg.zero_stage == 3


class TestTrainConfig:
    def test_minimal(self):
        cfg = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="my-model"),
            data=DataConfig(path="data.jsonl", format="alpaca"),
        )
        assert cfg.recipe == "finetune"
        assert cfg.method == "full"
        assert cfg.model.name == "my-model"
        assert cfg.trainer.batch_size == 4
        assert cfg.output.dir == "output"

    def test_full_config(self):
        cfg = TrainConfig(
            recipe="finetune",
            method="lora",
            model=ModelConfig(name="my-model", quantization="4bit"),
            data=DataConfig(path="data.jsonl", format="alpaca"),
            lora=LoraConfig(rank=32),
            trainer=TrainerConfig(num_epochs=5, strategy="ddp"),
            eval=EvalConfig(every_n_steps=100),
            logging=LoggingConfig(backends=["console", "wandb"]),
            output=OutputConfig(dir="my-output"),
        )
        assert cfg.method == "lora"
        assert cfg.lora.rank == 32
        assert cfg.trainer.num_epochs == 5

    def test_custom_recipe_accepted(self):
        cfg = TrainConfig(
            recipe="distill",
            model=ModelConfig(name="m"),
            data=DataConfig(path="d", format="alpaca"),
        )
        assert cfg.recipe == "distill"

    def test_empty_recipe_rejected(self):
        with pytest.raises(ValueError):
            TrainConfig(
                recipe="",
                model=ModelConfig(name="m"),
                data=DataConfig(path="d", format="alpaca"),
            )

    def test_custom_method_accepted(self):
        cfg = TrainConfig(
            recipe="finetune",
            method="distill",
            model=ModelConfig(name="m"),
            data=DataConfig(path="d", format="alpaca"),
        )
        assert cfg.method == "distill"

    def test_empty_method_rejected(self):
        with pytest.raises(ValueError):
            TrainConfig(
                recipe="finetune",
                method="",
                model=ModelConfig(name="m"),
                data=DataConfig(path="d", format="alpaca"),
            )

    def test_includes_fsdp_and_deepspeed_defaults(self):
        cfg = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="my-model"),
            data=DataConfig(path="data.jsonl", format="alpaca"),
        )
        assert cfg.fsdp.sharding_strategy == "full_shard"
        assert cfg.fsdp.cpu_offload is False
        assert cfg.deepspeed_config.zero_stage == 2
        assert cfg.deepspeed_config.config_file is None

    def test_reinforce_method(self):
        cfg = TrainConfig(
            recipe="align",
            method="reinforce",
            model=ModelConfig(name="my-model"),
            data=DataConfig(path="data.jsonl", format="alpaca"),
        )
        assert cfg.method == "reinforce"

    def test_method_params_default_empty(self):
        cfg = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="m"),
            data=DataConfig(path="d", format="alpaca"),
        )
        assert cfg.method_params == {}

    def test_method_params_set(self):
        cfg = TrainConfig(
            recipe="align",
            method="dpo",
            model=ModelConfig(name="m"),
            data=DataConfig(path="d", format="alpaca"),
            method_params={"beta": 0.2},
        )
        assert cfg.method_params == {"beta": 0.2}

    def test_method_params_roundtrip(self):
        cfg = TrainConfig(
            recipe="align",
            method="grpo",
            model=ModelConfig(name="m"),
            data=DataConfig(path="d", format="alpaca"),
            method_params={"kl_coeff": 0.1},
        )
        d = cfg.model_dump()
        cfg2 = TrainConfig(**d)
        assert cfg2.method_params == {"kl_coeff": 0.1}

    def test_to_dict_roundtrip(self):
        cfg = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="my-model"),
            data=DataConfig(path="data.jsonl", format="alpaca"),
        )
        d = cfg.model_dump()
        cfg2 = TrainConfig(**d)
        assert cfg2.recipe == cfg.recipe
        assert cfg2.model.name == cfg.model.name
