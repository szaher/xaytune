from unittest.mock import MagicMock, patch

import pytest
import torch

from trainlib.config.schema import DataConfig, ModelConfig, TrainConfig
from trainlib.recipes.align.align import align
from trainlib.trainer.callbacks import TrainState

_SETUP = "trainlib.recipes.base.setup_training"


class TestAlign:
    @patch(_SETUP)
    def test_align_with_config(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState(global_step=100)
        mock_components.model = MagicMock()
        mock_setup.return_value = mock_components

        config = TrainConfig(
            recipe="align",
            method="dpo",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="prefs.jsonl", format="preference"),
        )

        state = align(config=config)

        mock_setup.assert_called_once()
        mock_components.trainer.train.assert_called_once()
        assert state.global_step == 100

    @patch(_SETUP)
    def test_align_with_kwargs(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState(global_step=50)
        mock_components.model = MagicMock()
        mock_setup.return_value = mock_components

        state = align(
            model="my-sft-model",
            dataset="prefs.jsonl",
            method="grpo",
        )

        config = mock_setup.call_args[0][0]
        assert config.model.name == "my-sft-model"
        assert config.data.path == "prefs.jsonl"
        assert config.method == "grpo"
        assert config.recipe == "align"
        assert state.global_step == 50

    @patch(_SETUP)
    def test_align_default_method_is_dpo(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
        mock_components.model = MagicMock()
        mock_setup.return_value = mock_components

        align(model="test-model", dataset="prefs.jsonl")

        config = mock_setup.call_args[0][0]
        assert config.method == "dpo"

    @patch(_SETUP)
    def test_align_default_format_is_preference(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
        mock_components.model = MagicMock()
        mock_setup.return_value = mock_components

        align(model="test-model", dataset="prefs.jsonl")

        config = mock_setup.call_args[0][0]
        assert config.data.format == "preference"

    @patch(_SETUP)
    def test_align_accepts_trainer_kwargs(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
        mock_components.model = MagicMock()
        mock_setup.return_value = mock_components

        align(
            model="test-model",
            dataset="prefs.jsonl",
            num_epochs=2,
            learning_rate=5e-6,
        )

        config = mock_setup.call_args[0][0]
        assert config.trainer.num_epochs == 2
        assert config.trainer.learning_rate == 5e-6

    @patch(_SETUP)
    def test_align_returns_train_state(self, mock_setup):
        expected_state = TrainState(global_step=200, metrics={"loss": 0.4})
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = expected_state
        mock_components.model = MagicMock()
        mock_setup.return_value = mock_components

        result = align(model="test-model", dataset="prefs.jsonl")

        assert isinstance(result, TrainState)
        assert result.metrics["loss"] == 0.4

    def test_align_requires_model_and_dataset(self):
        with pytest.raises(ValueError, match="required"):
            align(model="test-model")

    def test_align_requires_model_and_dataset_2(self):
        with pytest.raises(ValueError, match="required"):
            align(dataset="prefs.jsonl")

    @patch(_SETUP)
    def test_alignment_method_passes_loss_fn(self, mock_setup):
        mock_model = torch.nn.Linear(1, 1)
        mock_components = MagicMock()
        mock_components.model = mock_model
        mock_components.trainer.train.return_value = TrainState()
        mock_setup.return_value = mock_components

        config = TrainConfig(
            recipe="align",
            method="dpo",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="prefs.jsonl", format="preference"),
        )
        align(config=config)

        call_kwargs = mock_components.trainer.train.call_args.kwargs
        assert call_kwargs["loss_fn"] is not None
        assert callable(call_kwargs["loss_fn"])

    @patch(_SETUP)
    def test_non_alignment_method_no_loss_fn(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
        mock_components.model = MagicMock()
        mock_setup.return_value = mock_components

        config = TrainConfig(
            recipe="align",
            method="full",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="prefs.jsonl", format="preference"),
        )
        align(config=config)

        call_kwargs = mock_components.trainer.train.call_args.kwargs
        assert call_kwargs["loss_fn"] is None
