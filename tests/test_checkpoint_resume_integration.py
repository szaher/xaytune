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


class TestCheckpointResumeCycle:
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.load_model")
    def test_full_checkpoint_cycle(
        self, mock_load_model, mock_load_dataset, tmp_path, real_checkpoint_io
    ):
        model = _mock_model()
        mock_load_model.return_value = _mock_model_result(model)
        mock_load_dataset.return_value = _make_dataset(6)

        output_dir = str(tmp_path / "output")

        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca"),
            trainer=TrainerConfig(
                batch_size=1,
                num_epochs=1,
                max_steps=6,
                checkpoint_every_n_steps=3,
                save_last=False,
            ),
            output=OutputConfig(dir=output_dir),
        )

        state = xaytune.finetune(config=config)

        assert state.global_step == 6

        ckpt_3 = Path(output_dir) / "checkpoint-3"
        ckpt_6 = Path(output_dir) / "checkpoint-6"
        assert ckpt_3.exists()
        assert ckpt_6.exists()
        assert (ckpt_3 / "metadata.json").exists()
        assert (ckpt_3 / "model.pt").exists()
        assert (ckpt_3 / "optimizer.pt").exists()

        meta_3 = json.loads((ckpt_3 / "metadata.json").read_text())
        assert meta_3["global_step"] == 3

    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.load_model")
    def test_save_last_creates_final_checkpoint(
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
                checkpoint_every_n_steps=0,
                save_last=True,
            ),
            output=OutputConfig(dir=output_dir),
        )

        state = xaytune.finetune(config=config)

        assert state.global_step == 4
        ckpt_final = Path(output_dir) / "checkpoint-4"
        assert ckpt_final.exists()
        assert (ckpt_final / "metadata.json").exists()

    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.load_model")
    def test_save_last_skips_duplicate(
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
                checkpoint_every_n_steps=3,
                save_last=True,
            ),
            output=OutputConfig(dir=output_dir),
        )

        state = xaytune.finetune(config=config)

        assert state.global_step == 3
        ckpt_dirs = [d for d in Path(output_dir).iterdir() if d.is_dir()]
        assert len(ckpt_dirs) == 1

    @patch("xaytune.recipes.base.wrap_model_distributed", side_effect=lambda m, **kw: m)
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.load_model")
    def test_distributed_only_rank0_saves(
        self, mock_load_model, mock_load_dataset, mock_wrap, tmp_path, real_checkpoint_io
    ):
        from xaytune.trainer.distributed import DistributedContext

        mock_load_model.return_value = _mock_model_result()
        mock_load_dataset.return_value = _make_dataset(6)

        output_dir = str(tmp_path / "output")

        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca"),
            trainer=TrainerConfig(
                batch_size=1,
                num_epochs=1,
                max_steps=3,
                checkpoint_every_n_steps=1,
                save_last=True,
            ),
            output=OutputConfig(dir=output_dir),
        )

        ctx = DistributedContext(rank=1, world_size=2, local_rank=1)
        with patch("xaytune.recipes.base.init_distributed", return_value=ctx):
            state = xaytune.finetune(config=config)

        assert state.global_step == 3
        output_path = Path(output_dir)
        if output_path.exists():
            ckpt_dirs = [d for d in output_path.iterdir() if d.is_dir()]
            assert len(ckpt_dirs) == 0

    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.load_model")
    def test_resume_restores_global_step(
        self, mock_load_model, mock_load_dataset, tmp_path, real_checkpoint_io
    ):
        mock_load_model.return_value = _mock_model_result()
        mock_load_dataset.return_value = _make_dataset(10)

        output_dir = str(tmp_path / "output")

        # Phase 1: train 5 steps with checkpoint at step 5
        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca"),
            trainer=TrainerConfig(
                batch_size=1,
                num_epochs=1,
                max_steps=5,
                checkpoint_every_n_steps=5,
                save_last=False,
            ),
            output=OutputConfig(dir=output_dir),
        )

        state1 = xaytune.finetune(config=config)
        assert state1.global_step == 5

        ckpt_path = str(Path(output_dir) / "checkpoint-5")
        assert Path(ckpt_path).exists()

        # Phase 2: resume from checkpoint-5, train to step 10
        mock_load_model.return_value = _mock_model_result()
        mock_load_dataset.return_value = _make_dataset(10)

        config2 = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca"),
            trainer=TrainerConfig(
                batch_size=1,
                num_epochs=1,
                max_steps=10,
                checkpoint_every_n_steps=0,
                save_last=False,
            ),
            output=OutputConfig(dir=output_dir),
        )

        state2 = xaytune.finetune(config=config2, resume_from=ckpt_path)

        assert state2.global_step == 10
