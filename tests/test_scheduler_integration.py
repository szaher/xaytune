from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

import trainlib
from trainlib.config.schema import (
    DataConfig,
    ModelConfig,
    OutputConfig,
    TrainConfig,
    TrainerConfig,
)
from trainlib.trainer.checkpointing import save_checkpoint


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


@pytest.fixture()
def real_checkpoint_io():
    """Override conftest's autouse _no_checkpoint_io to allow real saves."""
    with patch("trainlib.trainer.checkpoint_callback.save_checkpoint", save_checkpoint):
        yield


class TestSchedulerEndToEnd:
    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_cosine_scheduler_with_warmup(
        self, mock_load_model, mock_load_dataset, tmp_path
    ):
        mock_load_model.return_value = _mock_model_result()
        mock_load_dataset.return_value = _make_dataset(6)

        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca"),
            trainer=TrainerConfig(
                batch_size=1,
                num_epochs=1,
                max_steps=6,
                scheduler="cosine",
                warmup_steps=2,
                save_last=False,
            ),
            output=OutputConfig(dir=str(tmp_path / "output")),
        )

        state = trainlib.finetune(config=config)
        assert state.global_step == 6

    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_linear_scheduler(
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
                scheduler="linear",
                save_last=False,
            ),
            output=OutputConfig(dir=str(tmp_path / "output")),
        )

        state = trainlib.finetune(config=config)
        assert state.global_step == 4

    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_constant_with_warmup_scheduler(
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
                scheduler="constant_with_warmup",
                warmup_steps=2,
                save_last=False,
            ),
            output=OutputConfig(dir=str(tmp_path / "output")),
        )

        state = trainlib.finetune(config=config)
        assert state.global_step == 4

    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_checkpoint_includes_scheduler_state(
        self, mock_load_model, mock_load_dataset, tmp_path, real_checkpoint_io
    ):
        mock_load_model.return_value = _mock_model_result()
        mock_load_dataset.return_value = _make_dataset(4)

        output_dir = str(tmp_path / "output")

        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca"),
            trainer=TrainerConfig(
                batch_size=1,
                num_epochs=1,
                max_steps=4,
                scheduler="cosine",
                checkpoint_every_n_steps=2,
                save_last=False,
            ),
            output=OutputConfig(dir=output_dir),
        )

        state = trainlib.finetune(config=config)

        assert state.global_step == 4
        ckpt_2 = Path(output_dir) / "checkpoint-2"
        assert ckpt_2.exists()
        assert (ckpt_2 / "scheduler.pt").exists()

    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_warmup_ratio_works(
        self, mock_load_model, mock_load_dataset, tmp_path
    ):
        mock_load_model.return_value = _mock_model_result()
        mock_load_dataset.return_value = _make_dataset(10)

        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca"),
            trainer=TrainerConfig(
                batch_size=1,
                num_epochs=1,
                max_steps=10,
                scheduler="cosine",
                warmup_ratio=0.2,
                save_last=False,
            ),
            output=OutputConfig(dir=str(tmp_path / "output")),
        )

        state = trainlib.finetune(config=config)
        assert state.global_step == 10
