"""Tests for the multi-stage training pipeline runner."""

from unittest.mock import MagicMock, patch

import pytest

from xaytune.pipeline import _run_train_stage
from xaytune.pipeline_schema import StageConfig


def _fake_state(metrics=None):
    state = MagicMock()
    state.metrics = metrics or {"loss": 0.5}
    return state


class TestRunTrainStage:
    def test_defaults_trainer_and_lora_when_stage_omits_them(self, tmp_path):
        """A stage may omit ``trainer:`` and ``lora:``; defaults must be built.

        Regression test: the ``TrainerConfig``/``LoraConfig`` import used to sit
        below the call that constructs them, so this path raised ``NameError``.
        ``stage.trainer or TrainerConfig()`` short-circuits whenever a stage
        *does* supply them, which is why it went unnoticed.
        """
        stage = StageConfig(name="sft", recipe="finetune")
        assert stage.trainer is None
        assert stage.lora is None

        with patch("xaytune.recipes.finetune.finetune") as mock_finetune:
            mock_finetune.return_value = _fake_state({"loss": 0.25})
            result = _run_train_stage(stage, "base-model", str(tmp_path))

        assert mock_finetune.call_count == 1
        config = mock_finetune.call_args.kwargs["config"]
        assert config.trainer is not None
        assert config.lora is not None
        assert config.model.name == "base-model"

        assert result.type == "train"
        assert result.output == str(tmp_path)
        assert result.metrics == {"loss": 0.25}

    def test_explicit_trainer_and_lora_are_preserved(self, tmp_path):
        from xaytune.config.schema import LoraConfig, TrainerConfig

        stage = StageConfig(
            name="sft",
            recipe="finetune",
            trainer=TrainerConfig(learning_rate=1e-5),
            lora=LoraConfig(rank=8),
        )

        with patch("xaytune.recipes.finetune.finetune") as mock_finetune:
            mock_finetune.return_value = _fake_state()
            _run_train_stage(stage, "base-model", str(tmp_path))

        config = mock_finetune.call_args.kwargs["config"]
        assert config.trainer.learning_rate == 1e-5
        assert config.lora.rank == 8

    def test_missing_model_path_raises(self, tmp_path):
        stage = StageConfig(name="sft", recipe="finetune")

        with pytest.raises(ValueError, match="no model specified"):
            _run_train_stage(stage, None, str(tmp_path))

    def test_unknown_recipe_raises(self, tmp_path):
        stage = StageConfig(name="sft", recipe="nope")

        with pytest.raises(ValueError, match="Unknown recipe"):
            _run_train_stage(stage, "base-model", str(tmp_path))
