import pytest
from unittest.mock import patch, MagicMock
from trainlib.trainer.distributed import (
    wrap_model_distributed,
    get_strategy,
    DistributedContext,
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


class TestWrapModelDistributed:
    def test_none_strategy_returns_model(self):
        mock_model = MagicMock()
        ctx = DistributedContext()
        result = wrap_model_distributed(mock_model, strategy="none", ctx=ctx)
        assert result is mock_model

    @patch("trainlib.trainer.distributed.DistributedDataParallel")
    def test_ddp_wraps_model(self, mock_ddp_cls):
        mock_model = MagicMock()
        mock_ddp_cls.return_value = MagicMock()
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)
        result = wrap_model_distributed(mock_model, strategy="ddp", ctx=ctx)
        mock_ddp_cls.assert_called_once()

    @patch("trainlib.trainer.distributed.FullyShardedDataParallel")
    def test_fsdp_wraps_model(self, mock_fsdp_cls):
        mock_model = MagicMock()
        mock_fsdp_cls.return_value = MagicMock()
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)
        result = wrap_model_distributed(mock_model, strategy="fsdp", ctx=ctx)
        mock_fsdp_cls.assert_called_once()

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            wrap_model_distributed(MagicMock(), strategy="invalid", ctx=DistributedContext())
