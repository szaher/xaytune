from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

import trainlib
from trainlib.config.schema import (
    DataConfig,
    LoggingConfig,
    ModelConfig,
    OutputConfig,
    TrainConfig,
    TrainerConfig,
)


def _mock_model():
    model = MagicMock()
    params = [torch.randn(10, 10, requires_grad=True)]
    model.parameters.return_value = params
    model.train.return_value = None
    model.state_dict.return_value = {"weight": torch.randn(10, 10)}
    model.load_state_dict = MagicMock()

    mock_output = MagicMock()
    mock_output.loss = torch.tensor(0.5, requires_grad=True)
    model.return_value = mock_output
    model.__call__ = MagicMock(return_value=mock_output)
    return model


def _mock_model_result(model=None):
    from trainlib.models.loader import ModelResult

    return ModelResult(
        model=model or _mock_model(),
        tokenizer=MagicMock(),
        name="test-model",
    )


def _make_dataset(n=10):
    return [
        {
            "input_ids": torch.tensor([i, i + 1]),
            "labels": torch.tensor([i, i + 1]),
            "attention_mask": torch.tensor([1, 1]),
        }
        for i in range(n)
    ]


class TestLoggingIntegration:
    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_training_completes_with_logging(
        self, mock_load_model, mock_load_dataset, tmp_path
    ):
        mock_load_model.return_value = _mock_model_result()
        mock_load_dataset.return_value = _make_dataset(4)

        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca"),
            trainer=TrainerConfig(
                batch_size=1,
                num_epochs=1,
                max_steps=4,
                save_last=False,
            ),
            logging=LoggingConfig(backends=["console"], log_every_n_steps=2),
            output=OutputConfig(dir=str(tmp_path / "output")),
        )

        state = trainlib.finetune(config=config)
        assert state.global_step == 4

    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_log_scalar_called_with_loss(
        self, mock_load_model, mock_load_dataset, tmp_path
    ):
        mock_load_model.return_value = _mock_model_result()
        mock_load_dataset.return_value = _make_dataset(4)

        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca"),
            trainer=TrainerConfig(
                batch_size=1,
                num_epochs=1,
                max_steps=4,
                save_last=False,
            ),
            logging=LoggingConfig(backends=["console"], log_every_n_steps=2),
            output=OutputConfig(dir=str(tmp_path / "output")),
        )

        with patch("trainlib.logging.base.LoggingManager.log_scalar") as mock_log:
            state = trainlib.finetune(config=config)

            assert state.global_step == 4
            logged_keys = [call.args[0] for call in mock_log.call_args_list]
            assert "loss" in logged_keys

    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_log_config_called_at_train_start(
        self, mock_load_model, mock_load_dataset, tmp_path
    ):
        mock_load_model.return_value = _mock_model_result()
        mock_load_dataset.return_value = _make_dataset(2)

        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca"),
            trainer=TrainerConfig(
                batch_size=1,
                num_epochs=1,
                max_steps=2,
                save_last=False,
            ),
            output=OutputConfig(dir=str(tmp_path / "output")),
        )

        with patch("trainlib.logging.base.LoggingManager.log_config") as mock_cfg:
            trainlib.finetune(config=config)

            mock_cfg.assert_called_once()
            logged_config = mock_cfg.call_args[0][0]
            assert "recipe" in logged_config
            assert logged_config["recipe"] == "finetune"

    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_learning_rate_in_metrics(
        self, mock_load_model, mock_load_dataset, tmp_path
    ):
        mock_load_model.return_value = _mock_model_result()
        mock_load_dataset.return_value = _make_dataset(4)

        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca"),
            trainer=TrainerConfig(
                batch_size=1,
                num_epochs=1,
                max_steps=4,
                save_last=False,
                scheduler="cosine",
            ),
            output=OutputConfig(dir=str(tmp_path / "output")),
        )

        state = trainlib.finetune(config=config)

        assert state.global_step == 4
        assert "learning_rate" in state.metrics
        assert isinstance(state.metrics["learning_rate"], float)

    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_close_called_at_train_end(
        self, mock_load_model, mock_load_dataset, tmp_path
    ):
        mock_load_model.return_value = _mock_model_result()
        mock_load_dataset.return_value = _make_dataset(2)

        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca"),
            trainer=TrainerConfig(
                batch_size=1,
                num_epochs=1,
                max_steps=2,
                save_last=False,
            ),
            output=OutputConfig(dir=str(tmp_path / "output")),
        )

        with patch("trainlib.logging.base.LoggingManager.close") as mock_close:
            trainlib.finetune(config=config)
            mock_close.assert_called_once()
