import time
from unittest.mock import patch

import pytest

gr = pytest.importorskip("gradio")

from xaytune.cli import _build_parser  # noqa: E402
from xaytune.studio.app import _poll, build_config, create_app, validate_form  # noqa: E402
from xaytune.studio.jobs import JobInfo, JobManager, JobStatus  # noqa: E402


class TestValidateForm:
    def test_valid_inputs(self):
        errors = validate_form(
            recipe="finetune",
            method="full",
            model_name="m",
            data_path="d",
            quantization=None,
            learning_rate=2e-4,
            batch_size=4,
            warmup_steps=0,
            warmup_ratio=0.0,
            eval_split=0.1,
        )
        assert errors == []

    def test_empty_model_name(self):
        errors = validate_form(
            recipe="finetune",
            method="full",
            model_name="",
            data_path="d",
            quantization=None,
            learning_rate=2e-4,
            batch_size=4,
            warmup_steps=0,
            warmup_ratio=0.0,
            eval_split=0.0,
        )
        assert any("Model Name" in e for e in errors)

    def test_empty_data_path(self):
        errors = validate_form(
            recipe="finetune",
            method="full",
            model_name="m",
            data_path="",
            quantization=None,
            learning_rate=2e-4,
            batch_size=4,
            warmup_steps=0,
            warmup_ratio=0.0,
            eval_split=0.0,
        )
        assert any("Data Path" in e for e in errors)

    def test_recipe_method_mismatch_align(self):
        errors = validate_form(
            recipe="align",
            method="full",
            model_name="m",
            data_path="d",
            quantization=None,
            learning_rate=2e-4,
            batch_size=4,
            warmup_steps=0,
            warmup_ratio=0.0,
            eval_split=0.0,
        )
        assert any("align" in e for e in errors)

    def test_recipe_method_mismatch_finetune(self):
        errors = validate_form(
            recipe="finetune",
            method="dpo",
            model_name="m",
            data_path="d",
            quantization=None,
            learning_rate=2e-4,
            batch_size=4,
            warmup_steps=0,
            warmup_ratio=0.0,
            eval_split=0.0,
        )
        assert any("finetune" in e for e in errors)

    def test_recipe_method_mismatch_pretrain(self):
        errors = validate_form(
            recipe="pretrain",
            method="lora",
            model_name="m",
            data_path="d",
            quantization=None,
            learning_rate=2e-4,
            batch_size=4,
            warmup_steps=0,
            warmup_ratio=0.0,
            eval_split=0.0,
        )
        assert any("pretrain" in e for e in errors)

    def test_qlora_requires_4bit(self):
        errors = validate_form(
            recipe="finetune",
            method="qlora",
            model_name="m",
            data_path="d",
            quantization=None,
            learning_rate=2e-4,
            batch_size=4,
            warmup_steps=0,
            warmup_ratio=0.0,
            eval_split=0.0,
        )
        assert any("QLoRA" in e for e in errors)

    def test_qlora_with_4bit_passes(self):
        errors = validate_form(
            recipe="finetune",
            method="qlora",
            model_name="m",
            data_path="d",
            quantization="4bit",
            learning_rate=2e-4,
            batch_size=4,
            warmup_steps=0,
            warmup_ratio=0.0,
            eval_split=0.0,
        )
        assert not any("QLoRA" in e for e in errors)

    def test_warmup_conflict(self):
        errors = validate_form(
            recipe="finetune",
            method="full",
            model_name="m",
            data_path="d",
            quantization=None,
            learning_rate=2e-4,
            batch_size=4,
            warmup_steps=100,
            warmup_ratio=0.03,
            eval_split=0.0,
        )
        assert any("Warmup" in e for e in errors)

    def test_negative_learning_rate(self):
        errors = validate_form(
            recipe="finetune",
            method="full",
            model_name="m",
            data_path="d",
            quantization=None,
            learning_rate=-1e-4,
            batch_size=4,
            warmup_steps=0,
            warmup_ratio=0.0,
            eval_split=0.0,
        )
        assert any("Learning Rate" in e for e in errors)

    def test_zero_batch_size(self):
        errors = validate_form(
            recipe="finetune",
            method="full",
            model_name="m",
            data_path="d",
            quantization=None,
            learning_rate=2e-4,
            batch_size=0,
            warmup_steps=0,
            warmup_ratio=0.0,
            eval_split=0.0,
        )
        assert any("Batch Size" in e for e in errors)

    def test_eval_split_out_of_range(self):
        errors = validate_form(
            recipe="finetune",
            method="full",
            model_name="m",
            data_path="d",
            quantization=None,
            learning_rate=2e-4,
            batch_size=4,
            warmup_steps=0,
            warmup_ratio=0.0,
            eval_split=1.5,
        )
        assert any("Eval Split" in e for e in errors)

    def test_multiple_errors_at_once(self):
        errors = validate_form(
            recipe="align",
            method="full",
            model_name="",
            data_path="",
            quantization=None,
            learning_rate=-1,
            batch_size=0,
            warmup_steps=10,
            warmup_ratio=0.1,
            eval_split=2.0,
        )
        assert len(errors) >= 5


class TestBuildConfig:
    def test_minimal(self):
        config = build_config(
            recipe="finetune",
            method="full",
            model_name="test-model",
            data_path="data.jsonl",
            data_format="alpaca",
        )
        assert config.recipe == "finetune"
        assert config.method == "full"
        assert config.model.name == "test-model"
        assert config.data.path == "data.jsonl"

    def test_lora_fields(self):
        config = build_config(
            recipe="finetune",
            method="lora",
            model_name="m",
            data_path="d",
            data_format="alpaca",
            lora_rank=32,
            lora_alpha=64,
            lora_dropout=0.1,
        )
        assert config.lora.rank == 32
        assert config.lora.alpha == 64
        assert config.lora.dropout == 0.1

    def test_lora_defaults_for_non_lora(self):
        config = build_config(
            recipe="finetune",
            method="full",
            model_name="m",
            data_path="d",
            data_format="alpaca",
            lora_rank=999,
        )
        assert config.lora.rank == 16

    def test_full_fields(self):
        config = build_config(
            recipe="pretrain",
            method="full",
            model_name="my-model",
            data_path="train.jsonl",
            data_format="pretrain",
            max_seq_length=4096,
            packing=False,
            batch_size=8,
            gradient_accumulation=4,
            learning_rate=1e-5,
            num_epochs=2,
            max_steps=1000,
            warmup_steps=100,
            scheduler="linear",
            weight_decay=0.1,
            max_grad_norm=0.5,
            mixed_precision="fp16",
            seed=123,
            eval_every_n_steps=200,
            early_stopping_patience=3,
            log_every_n_steps=5,
            output_dir="my-output",
        )
        assert config.data.max_seq_length == 4096
        assert config.trainer.batch_size == 8
        assert config.trainer.scheduler == "linear"
        assert config.trainer.mixed_precision == "fp16"
        assert config.eval.every_n_steps == 200
        assert config.output.dir == "my-output"

    def test_new_fields(self):
        config = build_config(
            recipe="finetune",
            method="qlora",
            model_name="m",
            data_path="d",
            data_format="alpaca",
            quantization="4bit",
            dtype="float16",
            trust_remote_code=True,
            source="huggingface",
            streaming=True,
            eval_split=0.1,
            warmup_ratio=0.03,
            checkpoint_every_n_steps=100,
            save_last=False,
            activation_checkpointing=True,
            async_checkpoint=True,
            early_stopping_min_delta=0.001,
            merge_on_complete=True,
        )
        assert config.model.quantization == "4bit"
        assert config.model.dtype == "float16"
        assert config.model.trust_remote_code is True
        assert config.data.source == "huggingface"
        assert config.data.streaming is True
        assert config.data.eval_split == 0.1
        assert config.trainer.warmup_ratio == 0.03
        assert config.trainer.checkpoint_every_n_steps == 100
        assert config.trainer.save_last is False
        assert config.trainer.activation_checkpointing is True
        assert config.trainer.async_checkpoint is True
        assert config.eval.early_stopping_min_delta == 0.001
        assert config.output.merge_on_complete is True

    def test_method_params_passed(self):
        config = build_config(
            recipe="align",
            method="dpo",
            model_name="m",
            data_path="d",
            data_format="preference",
            method_params={"beta": 0.2},
        )
        assert config.method_params == {"beta": 0.2}

    def test_method_params_default_empty(self):
        config = build_config(
            recipe="finetune",
            method="full",
            model_name="m",
            data_path="d",
            data_format="alpaca",
        )
        assert config.method_params == {}

    def test_validation_error(self):
        with pytest.raises(Exception):
            build_config(
                recipe="",
                method="full",
                model_name="m",
                data_path="d",
                data_format="alpaca",
            )


class TestCreateApp:
    def test_returns_blocks(self):
        app = create_app()
        assert isinstance(app, gr.Blocks)

    def test_accepts_job_manager(self):
        mgr = JobManager()
        app = create_app(job_manager=mgr)
        assert isinstance(app, gr.Blocks)


class TestPollJob:
    def test_no_job_selected(self):
        mgr = JobManager()
        status, fig, metrics, history = _poll(mgr, None, [])
        assert "Select" in status

    def test_unknown_job(self):
        mgr = JobManager()
        status, fig, metrics, history = _poll(mgr, "nonexistent", [])
        assert "Unknown job" in status

    def test_accumulates_metrics(self):
        mgr = JobManager()
        job = JobInfo(
            job_id="j1",
            status=JobStatus.RUNNING,
            recipe="finetune",
            created_at=0.0,
            started_at=time.time(),
            state={"global_step": 5, "metrics": {"loss": 0.5}},
        )
        mgr._jobs["j1"] = job
        status, fig, metrics, history = _poll(mgr, "j1", [])
        assert len(history) == 1
        assert history[0]["step"] == 5

    def test_deduplicates_by_step(self):
        mgr = JobManager()
        job = JobInfo(
            job_id="j1",
            status=JobStatus.RUNNING,
            recipe="finetune",
            created_at=0.0,
            started_at=time.time(),
            state={"global_step": 5, "metrics": {"loss": 0.5}},
        )
        mgr._jobs["j1"] = job
        existing = [{"step": 5, "loss": 0.5}]
        status, fig, metrics, history = _poll(mgr, "j1", existing)
        assert len(history) == 1

    def test_shows_error(self):
        mgr = JobManager()
        job = JobInfo(
            job_id="j1",
            status=JobStatus.FAILED,
            recipe="finetune",
            created_at=0.0,
            error="boom",
        )
        mgr._jobs["j1"] = job
        status, fig, metrics, history = _poll(mgr, "j1", [])
        assert "boom" in status

    def test_shows_status_badge(self):
        mgr = JobManager()
        job = JobInfo(
            job_id="j1",
            status=JobStatus.COMPLETED,
            recipe="finetune",
            created_at=0.0,
        )
        mgr._jobs["j1"] = job
        status, fig, metrics, history = _poll(mgr, "j1", [])
        assert "COMPLETED" in status


class TestStudioCLI:
    def test_parser(self):
        parser = _build_parser()
        args = parser.parse_args(["studio"])
        assert args.command == "studio"
        assert args.host == "0.0.0.0"
        assert args.port == 7860
        assert args.share is False

    def test_parser_custom_args(self):
        parser = _build_parser()
        args = parser.parse_args(["studio", "--host", "127.0.0.1", "--port", "8080", "--share"])
        assert args.host == "127.0.0.1"
        assert args.port == 8080
        assert args.share is True

    @patch.dict("sys.modules", {"xaytune.studio.server": None})
    def test_missing_gradio_error(self):
        from xaytune.cli import main

        result = main(["studio"])
        assert result == 1
