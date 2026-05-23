from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

import trainlib
from trainlib.config.schema import (
    DataConfig,
    EvalConfig,
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
    model.eval.return_value = None
    model.training = True
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


class TestEvalDuringTraining:
    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_eval_metrics_appear_in_state(
        self, mock_load_model, mock_load_dataset, tmp_path
    ):
        mock_load_model.return_value = _mock_model_result()
        train_data = _make_dataset(6)
        eval_data = _make_dataset(3)
        mock_load_dataset.return_value = (train_data, eval_data)

        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca", eval_split=0.3),
            trainer=TrainerConfig(
                batch_size=1,
                num_epochs=1,
                max_steps=4,
                save_last=False,
            ),
            eval=EvalConfig(every_n_steps=2, metrics=["loss", "perplexity"]),
            output=OutputConfig(dir=str(tmp_path / "output")),
        )

        state = trainlib.finetune(config=config)

        assert state.global_step == 4
        assert "eval_loss" in state.metrics
        assert "eval_perplexity" in state.metrics
        assert state.metrics["eval_loss"] > 0

    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_no_eval_without_eval_split(
        self, mock_load_model, mock_load_dataset, tmp_path
    ):
        mock_load_model.return_value = _mock_model_result()
        mock_load_dataset.return_value = _make_dataset(4)

        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca", eval_split=0.0),
            trainer=TrainerConfig(
                batch_size=1,
                num_epochs=1,
                max_steps=4,
                save_last=False,
            ),
            eval=EvalConfig(every_n_steps=1),
            output=OutputConfig(dir=str(tmp_path / "output")),
        )

        state = trainlib.finetune(config=config)

        assert state.global_step == 4
        assert "eval_loss" not in state.metrics

    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_custom_metrics_list(
        self, mock_load_model, mock_load_dataset, tmp_path
    ):
        mock_load_model.return_value = _mock_model_result()
        train_data = _make_dataset(4)
        eval_data = _make_dataset(2)
        mock_load_dataset.return_value = (train_data, eval_data)

        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca", eval_split=0.3),
            trainer=TrainerConfig(
                batch_size=1,
                num_epochs=1,
                max_steps=2,
                save_last=False,
            ),
            eval=EvalConfig(every_n_steps=2, metrics=["loss"]),
            output=OutputConfig(dir=str(tmp_path / "output")),
        )

        state = trainlib.finetune(config=config)

        assert "eval_loss" in state.metrics
        assert "eval_perplexity" not in state.metrics

    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_eval_callbacks_fire_correct_count(
        self, mock_load_model, mock_load_dataset, tmp_path
    ):
        mock_load_model.return_value = _mock_model_result()
        train_data = _make_dataset(6)
        eval_data = _make_dataset(2)
        mock_load_dataset.return_value = (train_data, eval_data)

        from trainlib.trainer.callbacks import CallbackManager

        cb = CallbackManager()
        eval_events = []

        @cb.on("eval_start")
        def _on_start(state):
            eval_events.append(("start", state.global_step))

        @cb.on("eval_end")
        def _on_end(state):
            eval_events.append(("end", state.global_step))

        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca", eval_split=0.3),
            trainer=TrainerConfig(
                batch_size=1,
                num_epochs=1,
                max_steps=6,
                save_last=False,
            ),
            eval=EvalConfig(every_n_steps=3),
            output=OutputConfig(dir=str(tmp_path / "output")),
        )

        from trainlib.recipes import base as _base

        components = _base.setup_training(config, callback_manager=cb)
        state = components.trainer.train(
            model=components.model,
            train_dataloader=components.train_dataloader,
            resume_state=components.resume_state,
        )

        assert state.global_step == 6
        starts = [e for e in eval_events if e[0] == "start"]
        ends = [e for e in eval_events if e[0] == "end"]
        assert len(starts) == 2
        assert len(ends) == 2
        assert starts[0][1] == 3
        assert starts[1][1] == 6
