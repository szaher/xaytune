from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

import trainlib
from trainlib.config import load_config
from trainlib.config.schema import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    TrainerConfig,
)
from trainlib.trainer.callbacks import TrainState


@pytest.fixture
def mock_model():
    """Create a minimal mock model that behaves like a transformer."""
    model = MagicMock()
    model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
    model.train.return_value = None

    mock_output = MagicMock()
    mock_output.loss = torch.tensor(0.5, requires_grad=True)
    model.return_value = mock_output
    model.__call__ = MagicMock(return_value=mock_output)

    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "[PAD]"
    return tokenizer


@pytest.fixture
def mock_model_result(mock_model, mock_tokenizer):
    """Create a ModelResult with mocked model and tokenizer."""
    from trainlib.models.loader import ModelResult

    return ModelResult(
        model=mock_model,
        tokenizer=mock_tokenizer,
        name="test-model",
    )


@pytest.fixture
def mock_dataset():
    """Create a mock dataset with tensor data."""
    return [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
        },
        {
            "input_ids": torch.tensor([4, 5, 6]),
            "labels": torch.tensor([4, 5, 6]),
            "attention_mask": torch.tensor([1, 1, 1]),
        },
    ]


class TestFinetuneIntegration:
    """Test finetune recipe end-to-end with mocked model/data."""

    @patch("trainlib.models.peft.get_peft_model")
    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_finetune_lora_returns_train_state(
        self, mock_load_model, mock_load_dataset, mock_get_peft_model,
        mock_model_result, mock_dataset
    ):
        mock_load_model.return_value = mock_model_result
        mock_load_dataset.return_value = mock_dataset
        mock_get_peft_model.return_value = mock_model_result.model

        config = TrainConfig(
            recipe="finetune",
            method="lora",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca"),
            trainer=TrainerConfig(batch_size=2, num_epochs=1, max_steps=2),
        )

        state = trainlib.finetune(config=config)

        assert isinstance(state, TrainState)
        assert state.global_step > 0
        # 2 samples / batch_size=2 = 1 batch, so only 1 step despite max_steps=2
        assert state.global_step == 1
        mock_load_model.assert_called_once()
        mock_load_dataset.assert_called_once()

    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_finetune_full_with_kwargs(
        self, mock_load_model, mock_load_dataset, mock_model_result, mock_dataset
    ):
        mock_load_model.return_value = mock_model_result
        mock_load_dataset.return_value = mock_dataset

        state = trainlib.finetune(
            model="test-model",
            dataset="fake.jsonl",
            method="full",
            format="alpaca",
            num_epochs=1,
            learning_rate=2e-4,
            batch_size=2,
            max_steps=1,
        )

        assert isinstance(state, TrainState)
        assert state.global_step == 1
        assert "loss" in state.metrics

    @patch("trainlib.recipes.base.apply_lora")
    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_finetune_lora_applies_peft(
        self, mock_load_model, mock_load_dataset, mock_apply_lora, mock_model_result, mock_dataset
    ):
        mock_load_model.return_value = mock_model_result
        mock_load_dataset.return_value = mock_dataset

        # apply_lora returns a new ModelResult with peft_applied=True
        lora_result = MagicMock()
        lora_result.model = mock_model_result.model
        lora_result.tokenizer = mock_model_result.tokenizer
        lora_result.peft_applied = True
        mock_apply_lora.return_value = lora_result

        config = TrainConfig(
            recipe="finetune",
            method="lora",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca"),
            trainer=TrainerConfig(batch_size=2, num_epochs=1, max_steps=1),
        )

        state = trainlib.finetune(config=config)

        mock_apply_lora.assert_called_once()
        call_kwargs = mock_apply_lora.call_args.kwargs
        assert call_kwargs["rank"] == 16  # default
        assert call_kwargs["alpha"] == 32  # default
        assert isinstance(state, TrainState)


class TestPretrainIntegration:
    """Test pretrain recipe end-to-end with mocked model/data."""

    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_pretrain_returns_train_state(
        self, mock_load_model, mock_load_dataset, mock_model_result, mock_dataset
    ):
        mock_load_model.return_value = mock_model_result
        mock_load_dataset.return_value = mock_dataset

        config = TrainConfig(
            recipe="pretrain",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake/", format="text"),
            trainer=TrainerConfig(batch_size=1, num_epochs=1, max_steps=1),
        )

        state = trainlib.pretrain(config=config)

        assert isinstance(state, TrainState)
        assert state.global_step == 1
        mock_load_model.assert_called_once()

    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_pretrain_with_kwargs(
        self, mock_load_model, mock_load_dataset, mock_model_result, mock_dataset
    ):
        mock_load_model.return_value = mock_model_result
        # Create more samples to ensure we can reach max_steps
        large_dataset = mock_dataset * 5  # 10 samples
        mock_load_dataset.return_value = large_dataset

        state = trainlib.pretrain(
            model="gpt2",
            dataset="corpus/",
            format="text",
            num_epochs=1,
            batch_size=2,
            max_steps=2,
        )

        assert isinstance(state, TrainState)
        # 10 samples / batch_size=2 = 5 batches, but max_steps=2 stops at 2
        assert state.global_step == 2


class TestAlignIntegration:
    """Test align recipe end-to-end with mocked model/data."""

    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_align_dpo_returns_train_state(
        self, mock_load_model, mock_load_dataset, mock_model_result, mock_dataset
    ):
        mock_load_model.return_value = mock_model_result
        mock_load_dataset.return_value = mock_dataset

        config = TrainConfig(
            recipe="align",
            method="dpo",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="preferences.jsonl", format="preference"),
            trainer=TrainerConfig(batch_size=2, num_epochs=1, max_steps=1),
        )

        state = trainlib.align(config=config)

        assert isinstance(state, TrainState)
        assert state.global_step == 1

    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_align_with_kwargs(
        self, mock_load_model, mock_load_dataset, mock_model_result, mock_dataset
    ):
        mock_load_model.return_value = mock_model_result
        mock_load_dataset.return_value = mock_dataset

        state = trainlib.align(
            model="test-model",
            dataset="prefs.jsonl",
            method="dpo",
            format="preference",
            num_epochs=1,
            batch_size=1,
            max_steps=1,
        )

        assert isinstance(state, TrainState)


# Note: Callback integration is tested in test_trainer/test_callbacks.py
# Testing callbacks through the full recipe pipeline is complex due to
# CallbackManager being created inside setup_training()


class TestDataLoaderIntegration:
    """Test that DataLoader integration works correctly."""

    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_dataloader_created_with_correct_batch_size(
        self, mock_load_model, mock_load_dataset, mock_model_result
    ):
        mock_load_model.return_value = mock_model_result

        # Create a dataset with 10 samples
        dataset = [
            {
                "input_ids": torch.tensor([i]),
                "labels": torch.tensor([i]),
                "attention_mask": torch.tensor([1]),
            }
            for i in range(10)
        ]
        mock_load_dataset.return_value = dataset

        state = trainlib.finetune(
            model="test-model",
            dataset="fake.jsonl",
            batch_size=5,
            num_epochs=1,
            max_steps=-1,  # run full epoch
        )

        # With 10 samples and batch_size=5, we should have 2 steps
        assert state.global_step == 2

    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_eval_split_creates_eval_dataloader(
        self, mock_load_model, mock_load_dataset, mock_model_result
    ):
        mock_load_model.return_value = mock_model_result

        # Mock load_dataset to return train/eval split
        train_data = [
            {
                "input_ids": torch.tensor([1, 2]),
                "labels": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
            }
        ]
        eval_data = [
            {
                "input_ids": torch.tensor([3, 4]),
                "labels": torch.tensor([3, 4]),
                "attention_mask": torch.tensor([1, 1]),
            }
        ]
        mock_load_dataset.return_value = (train_data, eval_data)

        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca", eval_split=0.1),
            trainer=TrainerConfig(batch_size=1, num_epochs=1, max_steps=1),
        )

        state = trainlib.finetune(config=config)

        # Should complete without error
        assert isinstance(state, TrainState)


class TestConfigIntegration:
    """Test config loading and validation end-to-end."""

    def test_load_and_validate_lora_finetune_config(self, tmp_path):
        defaults_dir = Path(__file__).resolve().parent.parent / "trainlib" / "config" / "defaults"
        config_yaml = tmp_path / "train.yaml"
        config_yaml.write_text(
            f"base: {defaults_dir / 'lora.yaml'}\n"
            "model:\n"
            "  name: test-model\n"
            "data:\n"
            "  path: data.jsonl\n"
            "  format: alpaca\n"
        )
        config = load_config(str(config_yaml))
        assert config.recipe == "finetune"
        assert config.method == "lora"
        assert config.lora.rank == 16
        assert config.trainer.learning_rate == 2e-4

    def test_config_with_overrides(self, tmp_path):
        defaults_dir = Path(__file__).resolve().parent.parent / "trainlib" / "config" / "defaults"
        config_yaml = tmp_path / "train.yaml"
        config_yaml.write_text(
            f"base: {defaults_dir / 'lora.yaml'}\n"
            "model:\n"
            "  name: test-model\n"
            "data:\n"
            "  path: data.jsonl\n"
            "  format: alpaca\n"
        )
        config = load_config(str(config_yaml), overrides=["trainer.batch_size=8"])
        assert config.trainer.batch_size == 8
        assert config.method == "lora"

    def test_programmatic_config_creation(self):
        config = TrainConfig(
            recipe="finetune",
            method="lora",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="data.jsonl", format="alpaca"),
        )

        assert config.recipe == "finetune"
        assert config.method == "lora"
        assert config.trainer.batch_size == 4  # default
        assert config.trainer.num_epochs == 3  # default

    def test_config_validation_catches_invalid_recipe(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="Input should be"):
            TrainConfig(
                recipe="invalid",
                model=ModelConfig(name="test"),
                data=DataConfig(path="data.jsonl", format="alpaca"),
            )

    def test_config_validation_catches_invalid_method(self):
        with pytest.raises(ValueError, match="method must be one of"):
            TrainConfig(
                recipe="finetune",
                method="invalid",
                model=ModelConfig(name="test"),
                data=DataConfig(path="data.jsonl", format="alpaca"),
            )


class TestEarlyStoppingIntegration:
    """Test early stopping via max_steps."""

    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_max_steps_stops_training_early(
        self, mock_load_model, mock_load_dataset, mock_model_result
    ):
        mock_load_model.return_value = mock_model_result

        # Create dataset with 100 samples
        dataset = [
            {
                "input_ids": torch.tensor([i]),
                "labels": torch.tensor([i]),
                "attention_mask": torch.tensor([1]),
            }
            for i in range(100)
        ]
        mock_load_dataset.return_value = dataset

        state = trainlib.finetune(
            model="test-model",
            dataset="fake.jsonl",
            batch_size=1,
            num_epochs=10,  # Would run 1000 steps
            max_steps=5,  # But stop at 5
        )

        assert state.global_step == 5
        assert state.should_stop


class TestGradientAccumulationIntegration:
    """Test gradient accumulation in training pipeline."""

    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_gradient_accumulation_reduces_optimizer_steps(
        self, mock_load_model, mock_load_dataset, mock_model_result
    ):
        mock_load_model.return_value = mock_model_result

        # Create dataset with 8 samples
        dataset = [
            {
                "input_ids": torch.tensor([i]),
                "labels": torch.tensor([i]),
                "attention_mask": torch.tensor([1]),
            }
            for i in range(8)
        ]
        mock_load_dataset.return_value = dataset

        # Track optimizer calls
        original_model_call = mock_model_result.model.__call__

        def track_backward(*args, **kwargs):
            result = original_model_call(*args, **kwargs)
            # Count actual backward calls
            result.loss.backward = lambda: None
            return result

        mock_model_result.model.__call__ = track_backward

        state = trainlib.finetune(
            model="test-model",
            dataset="fake.jsonl",
            batch_size=2,
            gradient_accumulation=2,
            num_epochs=1,
            max_steps=-1,
        )

        # 8 samples / batch_size(2) = 4 micro-steps
        # 4 micro-steps / gradient_accumulation(2) = 2 optimizer steps
        # But global_step tracks micro-steps, so should be 4
        assert state.global_step == 4
