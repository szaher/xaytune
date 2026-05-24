# Trainer

The trainer module contains the training loop, checkpointing, scheduling, distributed strategies, and the LR finder.

---

## Trainer

::: trainlib.trainer.loop.Trainer

## Checkpointing

::: trainlib.trainer.checkpointing.save_checkpoint

::: trainlib.trainer.checkpointing.load_checkpoint

::: trainlib.trainer.checkpointing.find_latest_checkpoint

::: trainlib.trainer.async_checkpoint.AsyncCheckpointSaver

## Scheduling

::: trainlib.trainer.scheduler.create_scheduler

::: trainlib.trainer.scheduler.resolve_warmup_steps

## LR Finder

::: trainlib.trainer.lr_finder.lr_find

::: trainlib.trainer.lr_finder.LRFinderResult

## Distributed Training

::: trainlib.trainer.distributed.DistributedContext

::: trainlib.trainer.distributed.get_strategy

::: trainlib.trainer.distributed.wrap_model_distributed

::: trainlib.trainer.distributed.init_distributed

::: trainlib.trainer.distributed.cleanup_distributed

## Device Utilities

::: trainlib.trainer.device.get_device

::: trainlib.trainer.device.get_device_type

::: trainlib.trainer.device.seed_all
