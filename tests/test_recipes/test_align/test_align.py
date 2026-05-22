import pytest
from unittest.mock import patch, MagicMock
from trainlib.recipes.align.align import align
from trainlib.config.schema import TrainConfig, ModelConfig, DataConfig, TrainerConfig
from trainlib.trainer.callbacks import TrainState


class TestAlign:
    @patch("trainlib.recipes.align.align.setup_training")
    def test_align_with_config(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState(global_step=100)
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

    @patch("trainlib.recipes.align.align.setup_training")
    def test_align_with_kwargs(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState(global_step=50)
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

    @patch("trainlib.recipes.align.align.setup_training")
    def test_align_default_method_is_dpo(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
        mock_setup.return_value = mock_components

        align(model="test-model", dataset="prefs.jsonl")

        config = mock_setup.call_args[0][0]
        assert config.method == "dpo"

    @patch("trainlib.recipes.align.align.setup_training")
    def test_align_default_format_is_preference(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
        mock_setup.return_value = mock_components

        align(model="test-model", dataset="prefs.jsonl")

        config = mock_setup.call_args[0][0]
        assert config.data.format == "preference"

    @patch("trainlib.recipes.align.align.setup_training")
    def test_align_accepts_trainer_kwargs(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
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

    @patch("trainlib.recipes.align.align.setup_training")
    def test_align_returns_train_state(self, mock_setup):
        expected_state = TrainState(global_step=200, metrics={"loss": 0.4})
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = expected_state
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
