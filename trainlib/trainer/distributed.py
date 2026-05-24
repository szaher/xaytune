from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from trainlib.trainer.device import get_device


@dataclass
class DistributedContext:
    """Process-level distributed training state (rank, world size, device)."""

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
    """Resolve ``"auto"`` strategy to ``"fsdp"`` (multi-GPU) or ``"none"`` (single)."""
    if strategy == "auto":
        return "fsdp" if world_size > 1 else "none"
    return strategy


def init_distributed() -> DistributedContext:
    """Initialize distributed training from environment variables (``RANK``, ``WORLD_SIZE``)."""
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
    """Destroy the process group if distributed training is active."""
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
    """Wrap a model with the chosen distributed strategy (DDP, FSDP, or DeepSpeed)."""
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

            if getattr(fsdp_config, "auto_wrap_min_params", 0) > 0:
                from functools import partial

                from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

                fsdp_kwargs["auto_wrap_policy"] = partial(
                    size_based_auto_wrap_policy,
                    min_num_params=fsdp_config.auto_wrap_min_params,
                )

            fsdp_kwargs["forward_prefetch"] = getattr(
                fsdp_config, "forward_prefetch", False,
            )
            fsdp_kwargs["sync_module_states"] = getattr(
                fsdp_config, "sync_module_states", True,
            )
            fsdp_kwargs["limit_all_gathers"] = getattr(
                fsdp_config, "limit_all_gathers", True,
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
                zero_opt: dict[str, Any] = {
                    "stage": deepspeed_config.zero_stage,
                    "overlap_comm": getattr(deepspeed_config, "overlap_comm", True),
                    "contiguous_gradients": getattr(
                        deepspeed_config, "contiguous_gradients", True,
                    ),
                    "reduce_bucket_size": getattr(
                        deepspeed_config, "reduce_bucket_size", 500_000_000,
                    ),
                }

                if getattr(deepspeed_config, "offload_optimizer", False):
                    zero_opt["offload_optimizer"] = {"device": "cpu", "pin_memory": True}

                if getattr(deepspeed_config, "offload_param", False):
                    zero_opt["offload_param"] = {"device": "cpu", "pin_memory": True}

                if deepspeed_config.zero_stage == 3:
                    zero_opt["stage3_prefetch_bucket_size"] = getattr(
                        deepspeed_config, "stage3_prefetch_bucket_size", 50_000_000,
                    )
                    zero_opt["stage3_param_persistence_threshold"] = getattr(
                        deepspeed_config, "stage3_param_persistence_threshold", 100_000,
                    )

                config_dict = {
                    "zero_optimization": zero_opt,
                    "train_batch_size": "auto",
                    "train_micro_batch_size_per_gpu": "auto",
                }

            engine, _, _, _ = ds.initialize(model=model, config=config_dict)
            return engine
        return model

    raise ValueError(f"Unknown strategy: '{strategy}'. Valid options: none, ddp, fsdp, deepspeed.")
