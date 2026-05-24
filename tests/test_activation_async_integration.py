from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

import xaytune
from xaytune.config.schema import (
    DataConfig,
    ModelConfig,
    OutputConfig,
    TrainConfig,
    TrainerConfig,
)
from xaytune.trainer.checkpointing import save_checkpoint


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
    from xaytune.models.loader import ModelResult

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
    with patch("xaytune.trainer.checkpoint_callback.save_checkpoint", save_checkpoint):
        yield


class TestActivationCheckpointingIntegration:
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.load_model")
    def test_activation_checkpointing_calls_enable(
        self, mock_load_model, mock_load_dataset, tmp_path
    ):
        model = _mock_model()
        mock_load_model.return_value = _mock_model_result(model)
        mock_load_dataset.return_value = _make_dataset(4)

        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca"),
            trainer=TrainerConfig(
                batch_size=1,
                num_epochs=1,
                max_steps=4,
                activation_checkpointing=True,
                save_last=False,
            ),
            output=OutputConfig(dir=str(tmp_path / "output")),
        )

        state = xaytune.finetune(config=config)

        assert state.global_step == 4
        model.gradient_checkpointing_enable.assert_called_once_with(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.load_model")
    def test_training_completes_without_activation_checkpointing(
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
                activation_checkpointing=False,
                save_last=False,
            ),
            output=OutputConfig(dir=str(tmp_path / "output")),
        )

        state = xaytune.finetune(config=config)
        assert state.global_step == 4


class TestAsyncCheckpointIntegration:
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.load_model")
    def test_async_checkpoint_writes_files(
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
                checkpoint_every_n_steps=2,
                save_last=False,
                async_checkpoint=True,
            ),
            output=OutputConfig(dir=output_dir),
        )

        state = xaytune.finetune(config=config)

        assert state.global_step == 4
        ckpt_2 = Path(output_dir) / "checkpoint-2"
        ckpt_4 = Path(output_dir) / "checkpoint-4"
        assert ckpt_2.exists()
        assert ckpt_4.exists()
        assert (ckpt_2 / "metadata.json").exists()
        assert (ckpt_2 / "model.pt").exists()

        meta = json.loads((ckpt_2 / "metadata.json").read_text())
        assert meta["global_step"] == 2

    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.load_model")
    def test_async_save_last_writes_final(
        self, mock_load_model, mock_load_dataset, tmp_path, real_checkpoint_io
    ):
        mock_load_model.return_value = _mock_model_result()
        mock_load_dataset.return_value = _make_dataset(3)

        output_dir = str(tmp_path / "output")

        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca"),
            trainer=TrainerConfig(
                batch_size=1,
                num_epochs=1,
                max_steps=3,
                checkpoint_every_n_steps=0,
                save_last=True,
                async_checkpoint=True,
            ),
            output=OutputConfig(dir=output_dir),
        )

        state = xaytune.finetune(config=config)

        assert state.global_step == 3
        ckpt_final = Path(output_dir) / "checkpoint-3"
        assert ckpt_final.exists()


class TestBothFeaturesIntegration:
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.load_model")
    def test_both_features_together(
        self, mock_load_model, mock_load_dataset, tmp_path, real_checkpoint_io
    ):
        model = _mock_model()
        mock_load_model.return_value = _mock_model_result(model)
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
                checkpoint_every_n_steps=2,
                save_last=True,
                activation_checkpointing=True,
                async_checkpoint=True,
            ),
            output=OutputConfig(dir=output_dir),
        )

        state = xaytune.finetune(config=config)

        assert state.global_step == 4
        model.gradient_checkpointing_enable.assert_called_once()

        ckpt_2 = Path(output_dir) / "checkpoint-2"
        ckpt_4 = Path(output_dir) / "checkpoint-4"
        assert ckpt_2.exists()
        assert ckpt_4.exists()
