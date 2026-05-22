from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import FullyShardedDataParallel


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


def get_strategy(strategy: str, world_size: int = 1) -> str:
    if strategy == "auto":
        return "fsdp" if world_size > 1 else "none"
    return strategy


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
        return DistributedDataParallel(
            model,
            device_ids=[ctx.local_rank] if ctx.local_rank >= 0 else None,
        )

    if strategy == "fsdp":
        return FullyShardedDataParallel(model, **kwargs)

    if strategy == "deepspeed":
        return model  # DeepSpeed init handled separately

    raise ValueError(
        f"Unknown strategy: '{strategy}'. "
        f"Valid options: none, ddp, fsdp, deepspeed."
    )
