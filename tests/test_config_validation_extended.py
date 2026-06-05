import pytest

from xaytune.config.schema import DataConfig, ModelConfig, TrainConfig, TrainerConfig
from xaytune.config.validation import (
    ConfigValidationError,
    _ALIGN_METHODS,
    _FINETUNE_METHODS,
    validate_config,
)


class TestReinforcValidation:
    def _make_config(self, **kwargs) -> TrainConfig:
        defaults = {
            "recipe": "align",
            "method": "dpo",
            "model": ModelConfig(name="test"),
            "data": DataConfig(path="data.jsonl", format="preference"),
        }
        defaults.update(kwargs)
        return TrainConfig(**defaults)

    def test_reinforce_accepted(self):
        config = self._make_config(method="reinforce")
        validate_config(config)  # should not raise

    def test_reinforce_in_align_methods(self):
        assert "reinforce" in _ALIGN_METHODS

    def test_all_align_methods_accepted(self):
        for method in ["dpo", "grpo", "ppo", "orpo", "simpo", "reinforce"]:
            config = self._make_config(method=method)
            validate_config(config)  # should not raise

    def test_invalid_method_rejected(self):
        config = self._make_config(method="invalid")
        with pytest.raises(ConfigValidationError, match="alignment method"):
            validate_config(config)

    def test_reinforce_no_method_params(self):
        config = self._make_config(method="reinforce", method_params={})
        validate_config(config)  # should not raise

    def test_reinforce_unknown_method_param_rejected(self):
        config = self._make_config(method="reinforce", method_params={"beta": 0.1})
        with pytest.raises(ConfigValidationError, match="Unknown method_params"):
            validate_config(config)


class TestFinetuneMethodValidation:
    def _make_config(self, **kwargs) -> TrainConfig:
        defaults = {
            "recipe": "finetune",
            "method": "full",
            "model": ModelConfig(name="test"),
            "data": DataConfig(path="data.jsonl", format="alpaca"),
        }
        defaults.update(kwargs)
        return TrainConfig(**defaults)

    def test_all_finetune_methods_accepted(self):
        for method in ["full", "lora", "qlora"]:
            if method == "qlora":
                config = self._make_config(
                    method=method,
                    model=ModelConfig(name="test", quantization="4bit"),
                )
            else:
                config = self._make_config(method=method)
            validate_config(config)  # should not raise

    def test_finetune_with_align_method_rejected(self):
        config = self._make_config(method="dpo")
        with pytest.raises(ConfigValidationError, match="fine-tuning method"):
            validate_config(config)

    def test_finetune_method_params_rejected(self):
        config = self._make_config(method="full", method_params={"beta": 0.1})
        with pytest.raises(ConfigValidationError, match="only supported for alignment"):
            validate_config(config)


class TestMultipleValidationErrors:
    def test_multiple_errors_collected(self):
        config = TrainConfig(
            recipe="align",
            method="invalid",
            model=ModelConfig(name="test"),
            data=DataConfig(path="data.jsonl", format="preference", eval_split=2.0),
            trainer=TrainerConfig(batch_size=0, learning_rate=-1.0),
        )
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        msg = str(exc_info.value)
        assert "eval_split" in msg
        assert "batch_size" in msg
        assert "learning_rate" in msg
        assert "alignment method" in msg

    def test_error_count_in_message(self):
        config = TrainConfig(
            recipe="align",
            method="invalid",
            model=ModelConfig(name="test"),
            data=DataConfig(path="data.jsonl", format="preference", eval_split=2.0),
        )
        with pytest.raises(ConfigValidationError, match=r"2 error\(s\)"):
            validate_config(config)
