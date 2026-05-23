from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DistributedContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def device(self) -> torch.device:
        if self.is_distributed:
            return torch.device(f"cuda:{self.local_rank}")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def get_strategy(strategy: str, world_size: int = 1) -> str:
    if strategy == "auto":
        return "fsdp" if world_size > 1 else "none"
    return strategy


def init_distributed() -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        return DistributedContext()

    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    return DistributedContext(rank=rank, world_size=world_size, local_rank=local_rank)


def cleanup_distributed(ctx: DistributedContext) -> None:
    if not ctx.is_distributed:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_distributed(
    model: Any,
    *,
    strategy: str,
    ctx: DistributedContext,
    **kwargs: Any,
) -> Any:
    if strategy == "none":
        return model

    if strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        return DistributedDataParallel(
            model,
            device_ids=[ctx.local_rank] if ctx.local_rank >= 0 else None,
        )

    if strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel

        return FullyShardedDataParallel(model, **kwargs)

    if strategy == "deepspeed":
        return model  # DeepSpeed init handled separately

    raise ValueError(f"Unknown strategy: '{strategy}'. Valid options: none, ddp, fsdp, deepspeed.")
