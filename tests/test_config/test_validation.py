from unittest.mock import patch

import pytest

from trainlib.config.schema import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    TrainerConfig,
)
from trainlib.config.validation import ConfigValidationError, preflight_check, validate_config


class TestValidateConfig:
    def _make_config(self, **kwargs) -> TrainConfig:
        defaults = {
            "recipe": "finetune",
            "model": ModelConfig(name="test-model"),
            "data": DataConfig(path="data.jsonl", format="alpaca"),
        }
        defaults.update(kwargs)
        return TrainConfig(**defaults)

    def test_valid_config_passes(self):
        cfg = self._make_config(method="lora")
        validate_config(cfg)  # should not raise

    def test_qlora_without_4bit(self):
        cfg = self._make_config(
            method="qlora",
            model=ModelConfig(name="m", quantization=None),
        )
        with pytest.raises(ConfigValidationError, match="4bit quantization"):
            validate_config(cfg)

    def test_qlora_with_8bit(self):
        cfg = self._make_config(
            method="qlora",
            model=ModelConfig(name="m", quantization="8bit"),
        )
        with pytest.raises(ConfigValidationError, match="4bit quantization"):
            validate_config(cfg)

    def test_qlora_with_4bit_passes(self):
        cfg = self._make_config(
            method="qlora",
            model=ModelConfig(name="m", quantization="4bit"),
        )
        validate_config(cfg)  # should not raise

    def test_eval_split_too_large(self):
        cfg = self._make_config(
            data=DataConfig(path="d", format="alpaca", eval_split=1.5),
        )
        with pytest.raises(ConfigValidationError, match="eval_split"):
            validate_config(cfg)

    def test_eval_split_negative(self):
        cfg = self._make_config(
            data=DataConfig(path="d", format="alpaca", eval_split=-0.1),
        )
        with pytest.raises(ConfigValidationError, match="eval_split"):
            validate_config(cfg)

    def test_batch_size_zero(self):
        cfg = self._make_config(
            trainer=TrainerConfig(batch_size=0),
        )
        with pytest.raises(ConfigValidationError, match="batch_size"):
            validate_config(cfg)

    def test_learning_rate_negative(self):
        cfg = self._make_config(
            trainer=TrainerConfig(learning_rate=-1e-4),
        )
        with pytest.raises(ConfigValidationError, match="learning_rate"):
            validate_config(cfg)

    def test_align_recipe_requires_align_method(self):
        cfg = self._make_config(recipe="align", method="lora")
        with pytest.raises(ConfigValidationError, match="alignment method"):
            validate_config(cfg)

    def test_align_recipe_with_dpo_passes(self):
        cfg = self._make_config(recipe="align", method="dpo")
        validate_config(cfg)

    def test_finetune_with_align_method(self):
        cfg = self._make_config(recipe="finetune", method="dpo")
        with pytest.raises(ConfigValidationError, match="fine-tuning method"):
            validate_config(cfg)

    def test_error_includes_suggestion(self):
        cfg = self._make_config(
            method="qlora",
            model=ModelConfig(name="m"),
        )
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(cfg)
        assert "suggestion" in str(exc_info.value).lower() or "set" in str(exc_info.value).lower()

    def test_warmup_steps_and_ratio_both_set(self):
        cfg = self._make_config(
            trainer=TrainerConfig(warmup_steps=100, warmup_ratio=0.1),
        )
        with pytest.raises(ConfigValidationError, match="mutually exclusive"):
            validate_config(cfg)

    def test_warmup_steps_only_passes(self):
        cfg = self._make_config(
            trainer=TrainerConfig(warmup_steps=100, warmup_ratio=0.0),
        )
        validate_config(cfg)

    def test_warmup_ratio_only_passes(self):
        cfg = self._make_config(
            trainer=TrainerConfig(warmup_steps=0, warmup_ratio=0.1),
        )
        validate_config(cfg)

    def test_warmup_both_zero_passes(self):
        cfg = self._make_config(
            trainer=TrainerConfig(warmup_steps=0, warmup_ratio=0.0),
        )
        validate_config(cfg)

    def test_method_params_valid_dpo_beta(self):
        cfg = self._make_config(recipe="align", method="dpo", method_params={"beta": 0.2})
        validate_config(cfg)

    def test_method_params_unknown_for_dpo(self):
        cfg = self._make_config(recipe="align", method="dpo", method_params={"kl_coeff": 0.1})
        with pytest.raises(ConfigValidationError, match="Unknown method_params"):
            validate_config(cfg)

    def test_method_params_on_finetune_rejected(self):
        cfg = self._make_config(recipe="finetune", method="full", method_params={"beta": 0.1})
        with pytest.raises(ConfigValidationError, match="only supported for alignment"):
            validate_config(cfg)

    def test_method_params_simpo_two_params(self):
        cfg = self._make_config(
            recipe="align", method="simpo",
            method_params={"beta": 2.0, "gamma": 0.5},
        )
        validate_config(cfg)

    def test_method_params_empty_passes(self):
        cfg = self._make_config(recipe="align", method="dpo", method_params={})
        validate_config(cfg)


class TestPreflightCheck:
    def _make_config(self, **kwargs) -> TrainConfig:
        defaults = {
            "recipe": "finetune",
            "model": ModelConfig(name="test-model"),
            "data": DataConfig(path="data.jsonl", format="alpaca"),
        }
        defaults.update(kwargs)
        return TrainConfig(**defaults)

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_quantization_without_cuda(self, _mock_mps, _mock_cuda):
        cfg = self._make_config(model=ModelConfig(name="m", quantization="4bit"))
        issues = preflight_check(cfg)
        assert any("CUDA" in i for i in issues)

    @patch("torch.cuda.is_available", return_value=True)
    def test_quantization_with_cuda_ok(self, _mock_cuda):
        cfg = self._make_config(model=ModelConfig(name="m", quantization="4bit"))
        issues = preflight_check(cfg)
        assert not any("Quantization" in i for i in issues)

    def test_data_path_not_found(self):
        cfg = self._make_config(
            data=DataConfig(path="/nonexistent/path/data.jsonl", format="alpaca"),
        )
        issues = preflight_check(cfg)
        assert any("not found" in i for i in issues)

    def test_data_path_exists(self, tmp_path):
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"text": "hello"}\n')
        cfg = self._make_config(
            data=DataConfig(path=str(data_file), format="alpaca"),
        )
        issues = preflight_check(cfg)
        assert not any("not found" in i for i in issues)

    def test_hf_source_skips_path_check(self):
        cfg = self._make_config(
            data=DataConfig(
                path="org/nonexistent-dataset", format="alpaca", source="huggingface",
            ),
        )
        issues = preflight_check(cfg)
        assert not any("not found" in i for i in issues)
