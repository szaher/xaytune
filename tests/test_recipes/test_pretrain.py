from unittest.mock import MagicMock, patch

from xaytune.config.schema import DataConfig, ModelConfig, TrainConfig
from xaytune.recipes.pretrain import pretrain
from xaytune.trainer.callbacks import TrainState


class TestPretrain:
    @patch("xaytune.recipes.base.setup_training")
    def test_pretrain_with_config(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState(global_step=1000)
        mock_setup.return_value = mock_components

        config = TrainConfig(
            recipe="pretrain",
            method="full",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="corpus.jsonl", format="text"),
        )

        state = pretrain(config=config)

        mock_setup.assert_called_once()
        assert state.global_step == 1000

    @patch("xaytune.recipes.base.setup_training")
    def test_pretrain_with_kwargs(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState(global_step=500)
        mock_setup.return_value = mock_components

        state = pretrain(
            model="my-model",
            dataset="corpus.jsonl",
        )

        config = mock_setup.call_args[0][0]
        assert config.model.name == "my-model"
        assert config.data.path == "corpus.jsonl"
        assert config.recipe == "pretrain"
        assert state.global_step == 500

    @patch("xaytune.recipes.base.setup_training")
    def test_pretrain_default_format_is_text(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
        mock_setup.return_value = mock_components

        pretrain(model="my-model", dataset="corpus.jsonl")

        config = mock_setup.call_args[0][0]
        assert config.data.format == "text"

    @patch("xaytune.recipes.base.setup_training")
    def test_pretrain_method_is_always_full(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
        mock_setup.return_value = mock_components

        pretrain(model="my-model", dataset="corpus.jsonl")

        config = mock_setup.call_args[0][0]
        assert config.method == "full"

    @patch("xaytune.recipes.base.setup_training")
    def test_pretrain_accepts_trainer_kwargs(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
        mock_setup.return_value = mock_components

        pretrain(
            model="my-model",
            dataset="corpus.jsonl",
            num_epochs=1,
            learning_rate=3e-4,
            max_steps=10000,
        )

        config = mock_setup.call_args[0][0]
        assert config.trainer.num_epochs == 1
        assert config.trainer.learning_rate == 3e-4
        assert config.trainer.max_steps == 10000

    @patch("xaytune.recipes.base.setup_training")
    def test_pretrain_returns_train_state(self, mock_setup):
        expected_state = TrainState(global_step=200, metrics={"loss": 2.1})
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = expected_state
        mock_setup.return_value = mock_components

        result = pretrain(model="my-model", dataset="corpus.jsonl")

        assert isinstance(result, TrainState)
        assert result.global_step == 200
