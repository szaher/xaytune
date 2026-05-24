# Trainer

The trainer module contains the training loop, checkpointing, scheduling, distributed strategies, and the LR finder.

---

## Trainer

::: xaytune.trainer.loop.Trainer

## Checkpointing

::: xaytune.trainer.checkpointing.save_checkpoint

::: xaytune.trainer.checkpointing.load_checkpoint

::: xaytune.trainer.checkpointing.find_latest_checkpoint

::: xaytune.trainer.async_checkpoint.AsyncCheckpointSaver

## Scheduling

::: xaytune.trainer.scheduler.create_scheduler

::: xaytune.trainer.scheduler.resolve_warmup_steps

## LR Finder

::: xaytune.trainer.lr_finder.lr_find

::: xaytune.trainer.lr_finder.LRFinderResult

## Distributed Training

::: xaytune.trainer.distributed.DistributedContext

::: xaytune.trainer.distributed.get_strategy

::: xaytune.trainer.distributed.wrap_model_distributed

::: xaytune.trainer.distributed.init_distributed

::: xaytune.trainer.distributed.cleanup_distributed

## Device Utilities

::: xaytune.trainer.device.get_device

::: xaytune.trainer.device.get_device_type

::: xaytune.trainer.device.seed_all
