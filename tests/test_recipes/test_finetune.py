from unittest.mock import MagicMock, patch

from trainlib.config.schema import DataConfig, ModelConfig, TrainConfig
from trainlib.recipes.finetune import finetune
from trainlib.trainer.callbacks import TrainState


class TestFinetune:
    @patch("trainlib.recipes.base.setup_training")
    def test_finetune_with_config(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState(global_step=100)
        mock_setup.return_value = mock_components

        config = TrainConfig(
            recipe="finetune",
            method="lora",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="data.jsonl", format="alpaca"),
        )

        state = finetune(config=config)

        mock_setup.assert_called_once()
        mock_components.trainer.train.assert_called_once()
        assert state.global_step == 100

    @patch("trainlib.recipes.base.setup_training")
    def test_finetune_with_kwargs(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState(global_step=50)
        mock_setup.return_value = mock_components

        state = finetune(
            model="meta-llama/Llama-3.1-8B",
            dataset="train.jsonl",
            method="lora",
        )

        call_args = mock_setup.call_args
        config = call_args[0][0] if call_args[0] else call_args[1].get("config")
        assert config.model.name == "meta-llama/Llama-3.1-8B"
        assert config.data.path == "train.jsonl"
        assert config.method == "lora"
        assert state.global_step == 50

    @patch("trainlib.recipes.base.setup_training")
    def test_finetune_default_method_is_full(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
        mock_setup.return_value = mock_components

        finetune(model="test-model", dataset="data.jsonl")

        config = mock_setup.call_args[0][0]
        assert config.method == "full"

    @patch("trainlib.recipes.base.setup_training")
    def test_finetune_passes_model_and_dataloader_to_trainer(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
        mock_setup.return_value = mock_components

        finetune(model="test-model", dataset="data.jsonl")

        train_call = mock_components.trainer.train.call_args
        assert train_call.kwargs["model"] is mock_components.model
        assert train_call.kwargs["train_dataloader"] is mock_components.train_dataloader

    @patch("trainlib.recipes.base.setup_training")
    def test_finetune_returns_train_state(self, mock_setup):
        expected_state = TrainState(global_step=75, metrics={"loss": 0.3})
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = expected_state
        mock_setup.return_value = mock_components

        result = finetune(model="test-model", dataset="data.jsonl")

        assert isinstance(result, TrainState)
        assert result.global_step == 75
        assert result.metrics["loss"] == 0.3

    @patch("trainlib.recipes.base.setup_training")
    def test_finetune_accepts_trainer_kwargs(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
        mock_setup.return_value = mock_components

        finetune(
            model="test-model",
            dataset="data.jsonl",
            num_epochs=5,
            learning_rate=1e-5,
            batch_size=8,
        )

        config = mock_setup.call_args[0][0]
        assert config.trainer.num_epochs == 5
        assert config.trainer.learning_rate == 1e-5
        assert config.trainer.batch_size == 8
