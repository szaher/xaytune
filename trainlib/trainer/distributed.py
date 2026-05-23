from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from trainlib.trainer.device import get_device


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
        return get_device(self.local_rank)


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
    fsdp_config: Any | None = None,
    deepspeed_config: Any | None = None,
    mixed_precision: str = "bf16",
    **kwargs: Any,
) -> Any:
    if strategy == "none":
        return model

    if strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        return DistributedDataParallel(
            model,
            device_ids=[ctx.local_rank] if ctx.local_rank >= 0 else None,
            find_unused_parameters=False,
        )

    if strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel

        fsdp_kwargs: dict[str, Any] = {}

        if fsdp_config is not None:
            from torch.distributed.fsdp import CPUOffload, ShardingStrategy

            strategy_map = {
                "full_shard": ShardingStrategy.FULL_SHARD,
                "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
                "no_shard": ShardingStrategy.NO_SHARD,
            }
            fsdp_kwargs["sharding_strategy"] = strategy_map[fsdp_config.sharding_strategy]

            if fsdp_config.cpu_offload:
                fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)

            if fsdp_config.backward_prefetch is not None:
                from torch.distributed.fsdp import BackwardPrefetch

                prefetch_map = {
                    "backward_pre": BackwardPrefetch.BACKWARD_PRE,
                    "backward_post": BackwardPrefetch.BACKWARD_POST,
                }
                fsdp_kwargs["backward_prefetch"] = prefetch_map[fsdp_config.backward_prefetch]

            if fsdp_config.mixed_precision:
                from torch.distributed.fsdp import MixedPrecision as FSDPMixedPrecision

                dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
                mp_dtype = dtype_map.get(mixed_precision)
                if mp_dtype is not None:
                    fsdp_kwargs["mixed_precision"] = FSDPMixedPrecision(
                        param_dtype=mp_dtype,
                        reduce_dtype=mp_dtype,
                        buffer_dtype=mp_dtype,
                    )

        fsdp_kwargs.update(kwargs)
        return FullyShardedDataParallel(model, **fsdp_kwargs)

    if strategy == "deepspeed":
        if deepspeed_config is not None:
            import deepspeed as ds

            config_dict: dict[str, Any] = {}
            if deepspeed_config.config_file is not None:
                import json

                with open(deepspeed_config.config_file) as f:
                    config_dict = json.load(f)
            else:
                config_dict = {
                    "zero_optimization": {"stage": deepspeed_config.zero_stage},
                    "train_batch_size": "auto",
                    "train_micro_batch_size_per_gpu": "auto",
                }

            engine, _, _, _ = ds.initialize(model=model, config=config_dict)
            return engine
        return model

    raise ValueError(f"Unknown strategy: '{strategy}'. Valid options: none, ddp, fsdp, deepspeed.")
