from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from trainlib.trainer.distributed import (
    DistributedContext,
    cleanup_distributed,
    get_strategy,
    init_distributed,
    wrap_model_distributed,
)


class TestDistributedContext:
    def test_defaults(self):
        ctx = DistributedContext()
        assert ctx.rank == 0
        assert ctx.world_size == 1
        assert ctx.local_rank == 0
        assert ctx.is_main_process is True

    def test_non_main_process(self):
        ctx = DistributedContext(rank=1, world_size=4, local_rank=1)
        assert ctx.is_main_process is False


class TestDistributedContextDevice:
    def test_device_cpu_when_no_cuda(self):
        ctx = DistributedContext()
        with patch("torch.cuda.is_available", return_value=False):
            assert ctx.device == torch.device("cpu")

    def test_device_cuda_when_available_single_gpu(self):
        ctx = DistributedContext()
        with patch("torch.cuda.is_available", return_value=True):
            assert ctx.device == torch.device("cuda")

    def test_device_cuda_local_rank_when_distributed(self):
        ctx = DistributedContext(rank=2, world_size=4, local_rank=2)
        assert ctx.device == torch.device("cuda:2")


class TestGetStrategy:
    def test_auto_single_gpu(self):
        strategy = get_strategy("auto", world_size=1)
        assert strategy == "none"

    def test_auto_multi_gpu(self):
        strategy = get_strategy("auto", world_size=4)
        assert strategy == "fsdp"

    def test_explicit_ddp(self):
        strategy = get_strategy("ddp", world_size=4)
        assert strategy == "ddp"

    def test_explicit_fsdp(self):
        strategy = get_strategy("fsdp", world_size=1)
        assert strategy == "fsdp"


class TestInitDistributed:
    def test_single_gpu_returns_default_context(self):
        with patch.dict(os.environ, {}, clear=True):
            ctx = init_distributed()
        assert not ctx.is_distributed
        assert ctx.rank == 0

    @patch("torch.cuda.set_device")
    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.init_process_group")
    def test_multi_gpu_initializes_process_group(
        self, mock_init, mock_is_init, mock_set_device
    ):
        env = {"RANK": "1", "WORLD_SIZE": "4", "LOCAL_RANK": "1"}
        with patch.dict(os.environ, env, clear=True):
            ctx = init_distributed()
        assert ctx.is_distributed
        assert ctx.rank == 1
        assert ctx.world_size == 4
        assert ctx.local_rank == 1
        mock_init.assert_called_once_with(backend="nccl")
        mock_set_device.assert_called_once_with(1)

    @patch("torch.cuda.set_device")
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.init_process_group")
    def test_skips_init_if_already_initialized(
        self, mock_init, mock_is_init, mock_set_device
    ):
        env = {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"}
        with patch.dict(os.environ, env, clear=True):
            ctx = init_distributed()
        mock_init.assert_not_called()


class TestCleanupDistributed:
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.destroy_process_group")
    def test_cleanup_destroys_process_group(self, mock_destroy, mock_is_init):
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)
        cleanup_distributed(ctx)
        mock_destroy.assert_called_once()

    @patch("torch.distributed.destroy_process_group")
    def test_cleanup_noop_for_single_gpu(self, mock_destroy):
        ctx = DistributedContext()
        cleanup_distributed(ctx)
        mock_destroy.assert_not_called()


class TestWrapModelDistributed:
    def test_none_strategy_returns_model(self):
        mock_model = MagicMock()
        ctx = DistributedContext()
        result = wrap_model_distributed(mock_model, strategy="none", ctx=ctx)
        assert result is mock_model

    @patch("torch.nn.parallel.DistributedDataParallel")
    def test_ddp_wraps_model(self, mock_ddp_cls):
        mock_model = MagicMock()
        mock_ddp_cls.return_value = MagicMock()
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)
        wrap_model_distributed(mock_model, strategy="ddp", ctx=ctx)
        mock_ddp_cls.assert_called_once()

    @patch("torch.distributed.fsdp.FullyShardedDataParallel")
    def test_fsdp_wraps_model(self, mock_fsdp_cls):
        mock_model = MagicMock()
        mock_fsdp_cls.return_value = MagicMock()
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)
        wrap_model_distributed(mock_model, strategy="fsdp", ctx=ctx)
        mock_fsdp_cls.assert_called_once()

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            wrap_model_distributed(MagicMock(), strategy="invalid", ctx=DistributedContext())


class TestWrapModelFSDP:
    @patch("torch.distributed.fsdp.FullyShardedDataParallel")
    def test_fsdp_with_full_shard(self, mock_fsdp):
        from trainlib.config.schema import FSDPConfig

        mock_fsdp.return_value = MagicMock()
        model = MagicMock()
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)
        fsdp_config = FSDPConfig(sharding_strategy="full_shard")

        wrap_model_distributed(model, strategy="fsdp", ctx=ctx, fsdp_config=fsdp_config)

        mock_fsdp.assert_called_once()
        call_kwargs = mock_fsdp.call_args[1]
        assert "sharding_strategy" in call_kwargs

    @patch("torch.distributed.fsdp.FullyShardedDataParallel")
    def test_fsdp_with_cpu_offload(self, mock_fsdp):
        from trainlib.config.schema import FSDPConfig

        mock_fsdp.return_value = MagicMock()
        model = MagicMock()
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)
        fsdp_config = FSDPConfig(cpu_offload=True)

        wrap_model_distributed(model, strategy="fsdp", ctx=ctx, fsdp_config=fsdp_config)

        call_kwargs = mock_fsdp.call_args[1]
        assert "cpu_offload" in call_kwargs

    @patch("torch.distributed.fsdp.FullyShardedDataParallel")
    def test_fsdp_with_shard_grad_op(self, mock_fsdp):
        from trainlib.config.schema import FSDPConfig

        mock_fsdp.return_value = MagicMock()
        model = MagicMock()
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)
        fsdp_config = FSDPConfig(sharding_strategy="shard_grad_op")

        wrap_model_distributed(model, strategy="fsdp", ctx=ctx, fsdp_config=fsdp_config)
        mock_fsdp.assert_called_once()

    @patch("torch.distributed.fsdp.FullyShardedDataParallel")
    def test_fsdp_with_mixed_precision(self, mock_fsdp):
        from trainlib.config.schema import FSDPConfig

        mock_fsdp.return_value = MagicMock()
        model = MagicMock()
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)
        fsdp_config = FSDPConfig(mixed_precision=True)

        wrap_model_distributed(
            model, strategy="fsdp", ctx=ctx, fsdp_config=fsdp_config, mixed_precision="bf16"
        )

        call_kwargs = mock_fsdp.call_args[1]
        assert "mixed_precision" in call_kwargs

    @patch("torch.distributed.fsdp.FullyShardedDataParallel")
    def test_fsdp_without_config_uses_defaults(self, mock_fsdp):
        mock_fsdp.return_value = MagicMock()
        model = MagicMock()
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)

        wrap_model_distributed(model, strategy="fsdp", ctx=ctx)
        mock_fsdp.assert_called_once_with(model)


class TestWrapModelDeepSpeed:
    def test_deepspeed_initializes_engine(self):
        from trainlib.config.schema import DeepSpeedConfig

        model = MagicMock()
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)
        ds_config = DeepSpeedConfig(zero_stage=2)

        mock_engine = MagicMock()
        with patch.dict("sys.modules", {"deepspeed": MagicMock()}):
            import sys

            mock_ds = sys.modules["deepspeed"]
            mock_ds.initialize.return_value = (mock_engine, None, None, None)

            result = wrap_model_distributed(
                model, strategy="deepspeed", ctx=ctx, deepspeed_config=ds_config
            )

        assert result == mock_engine

    def test_deepspeed_without_config_returns_model(self):
        model = MagicMock()
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)

        result = wrap_model_distributed(model, strategy="deepspeed", ctx=ctx)
        assert result is model


class TestWrapModelDDP:
    @patch("torch.nn.parallel.DistributedDataParallel")
    def test_ddp_uses_find_unused_parameters_false(self, mock_ddp):
        mock_ddp.return_value = MagicMock()
        model = MagicMock()
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)

        wrap_model_distributed(model, strategy="ddp", ctx=ctx)

        call_kwargs = mock_ddp.call_args[1]
        assert call_kwargs["find_unused_parameters"] is False
