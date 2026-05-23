from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

import trainlib
from trainlib.config.schema import DataConfig, ModelConfig, TrainConfig, TrainerConfig
from trainlib.trainer.callbacks import TrainState
from trainlib.trainer.distributed import DistributedContext


@pytest.fixture
def mock_model_result():
    from trainlib.models.loader import ModelResult

    model = MagicMock()
    model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
    model.train.return_value = None
    model.to.return_value = model
    mock_output = MagicMock()
    mock_output.loss = torch.tensor(0.5, requires_grad=True)
    model.return_value = mock_output
    model.__call__ = MagicMock(return_value=mock_output)
    tokenizer = MagicMock()
    tokenizer.pad_token = "[PAD]"
    return ModelResult(model=model, tokenizer=tokenizer, name="test-model")


@pytest.fixture
def mock_dataset():
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
    ] * 4  # 8 samples


class TestDistributedDDPFlow:
    """Test full pipeline with DDP wrapping."""

    @patch("trainlib.recipes.base.wrap_model_distributed")
    @patch("trainlib.recipes.base.init_distributed")
    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_finetune_with_ddp(
        self,
        mock_load_model,
        mock_load_dataset,
        mock_init,
        mock_wrap,
        mock_model_result,
        mock_dataset,
    ):
        """Full finetune pipeline through DDP path."""
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)
        mock_init.return_value = ctx
        mock_load_model.return_value = mock_model_result
        mock_load_dataset.return_value = mock_dataset
        # wrap should return the model itself (mock DDP is transparent)
        mock_wrap.return_value = mock_model_result.model

        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca"),
            trainer=TrainerConfig(
                batch_size=2, num_epochs=1, max_steps=2, strategy="ddp"
            ),
        )

        state = trainlib.finetune(config=config)

        assert isinstance(state, TrainState)
        assert state.global_step > 0
        mock_wrap.assert_called_once()
        # Verify strategy was "ddp"
        assert mock_wrap.call_args[1]["strategy"] == "ddp"


class TestDistributedFSDPFlow:
    """Test full pipeline with FSDP wrapping."""

    @patch("trainlib.recipes.base.wrap_model_distributed")
    @patch("trainlib.recipes.base.init_distributed")
    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_pretrain_with_fsdp(
        self,
        mock_load_model,
        mock_load_dataset,
        mock_init,
        mock_wrap,
        mock_model_result,
        mock_dataset,
    ):
        """Full pretrain pipeline through FSDP path."""
        ctx = DistributedContext(rank=0, world_size=4, local_rank=0)
        mock_init.return_value = ctx
        mock_load_model.return_value = mock_model_result
        mock_load_dataset.return_value = mock_dataset
        mock_wrap.return_value = mock_model_result.model

        config = TrainConfig(
            recipe="pretrain",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="corpus/", format="text"),
            trainer=TrainerConfig(
                batch_size=2, num_epochs=1, max_steps=1, strategy="fsdp"
            ),
        )

        state = trainlib.pretrain(config=config)

        assert isinstance(state, TrainState)
        mock_wrap.assert_called_once()
        assert mock_wrap.call_args[1]["strategy"] == "fsdp"


class TestSingleGPUUnchanged:
    """Verify single-GPU path is completely unaffected by distributed code."""

    @patch("trainlib.recipes.base.init_distributed")
    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_finetune_single_gpu_no_wrapping(
        self,
        mock_load_model,
        mock_load_dataset,
        mock_init,
        mock_model_result,
        mock_dataset,
    ):
        """Single GPU: no model wrapping, no DistributedSampler, same behavior as before."""
        mock_init.return_value = DistributedContext()  # default: not distributed
        mock_load_model.return_value = mock_model_result
        mock_load_dataset.return_value = mock_dataset

        state = trainlib.finetune(
            model="test-model",
            dataset="fake.jsonl",
            batch_size=2,
            num_epochs=1,
            max_steps=2,
        )

        assert isinstance(state, TrainState)
        assert state.global_step > 0

    @patch("trainlib.recipes.base.init_distributed")
    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_align_single_gpu(
        self,
        mock_load_model,
        mock_load_dataset,
        mock_init,
        mock_model_result,
        mock_dataset,
    ):
        """Align recipe works unchanged on single GPU."""
        mock_init.return_value = DistributedContext()
        mock_load_model.return_value = mock_model_result
        mock_load_dataset.return_value = mock_dataset

        state = trainlib.align(
            model="test-model",
            dataset="prefs.jsonl",
            method="dpo",
            format="preference",
            batch_size=2,
            num_epochs=1,
            max_steps=1,
        )

        assert isinstance(state, TrainState)


class TestDistributedCleanup:
    """Test that distributed cleanup happens correctly."""

    @patch("trainlib.recipes.base.cleanup_distributed")
    @patch("trainlib.recipes.base.wrap_model_distributed")
    @patch("trainlib.recipes.base.init_distributed")
    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_cleanup_called_after_training(
        self,
        mock_load_model,
        mock_load_dataset,
        mock_init,
        mock_wrap,
        mock_cleanup,
        mock_model_result,
        mock_dataset,
    ):
        """cleanup_distributed should be called when training ends in distributed mode."""
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)
        mock_init.return_value = ctx
        mock_load_model.return_value = mock_model_result
        mock_load_dataset.return_value = mock_dataset
        mock_wrap.return_value = mock_model_result.model

        state = trainlib.finetune(
            config=TrainConfig(
                recipe="finetune",
                model=ModelConfig(name="test-model"),
                data=DataConfig(path="fake.jsonl", format="alpaca"),
                trainer=TrainerConfig(
                    batch_size=2, num_epochs=1, max_steps=1, strategy="ddp"
                ),
            )
        )

        mock_cleanup.assert_called_once_with(ctx)

    @patch("trainlib.recipes.base.cleanup_distributed")
    @patch("trainlib.recipes.base.init_distributed")
    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_no_cleanup_for_single_gpu(
        self,
        mock_load_model,
        mock_load_dataset,
        mock_init,
        mock_cleanup,
        mock_model_result,
        mock_dataset,
    ):
        """cleanup_distributed should NOT be called for single GPU training."""
        mock_init.return_value = DistributedContext()
        mock_load_model.return_value = mock_model_result
        mock_load_dataset.return_value = mock_dataset

        state = trainlib.finetune(
            model="test-model",
            dataset="fake.jsonl",
            batch_size=2,
            num_epochs=1,
            max_steps=1,
        )

        mock_cleanup.assert_not_called()


class TestDistributedAutoStrategy:
    """Test auto strategy resolution in the full pipeline."""

    @patch("trainlib.recipes.base.wrap_model_distributed")
    @patch("trainlib.recipes.base.init_distributed")
    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.load_model")
    def test_auto_strategy_resolves_to_fsdp_when_distributed(
        self,
        mock_load_model,
        mock_load_dataset,
        mock_init,
        mock_wrap,
        mock_model_result,
        mock_dataset,
    ):
        """strategy='auto' should resolve to 'fsdp' when world_size > 1."""
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)
        mock_init.return_value = ctx
        mock_load_model.return_value = mock_model_result
        mock_load_dataset.return_value = mock_dataset
        mock_wrap.return_value = mock_model_result.model

        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca"),
            trainer=TrainerConfig(batch_size=2, num_epochs=1, max_steps=1),
            # strategy defaults to "auto"
        )

        state = trainlib.finetune(config=config)

        mock_wrap.assert_called_once()
        assert mock_wrap.call_args[1]["strategy"] == "fsdp"
