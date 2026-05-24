from xaytune.config.schema import TrainerConfig
from xaytune.trainer.async_checkpoint import AsyncCheckpointSaver
from xaytune.trainer.callbacks import CallbackManager, TrainState
from xaytune.trainer.checkpointing import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from xaytune.trainer.device import get_device, get_device_type, seed_all
from xaytune.trainer.distributed import (
    DistributedContext,
    get_strategy,
    wrap_model_distributed,
)
from xaytune.trainer.early_stopping import register_early_stopping_callbacks
from xaytune.trainer.eval_callback import register_eval_callbacks
from xaytune.trainer.loop import Trainer
from xaytune.trainer.lr_finder import LRFinderResult, lr_find
from xaytune.trainer.scheduler import create_scheduler, resolve_warmup_steps

_global_callback_manager = CallbackManager()

on = _global_callback_manager.on

__all__ = [
    "AsyncCheckpointSaver",
    "CallbackManager",
    "create_scheduler",
    "DistributedContext",
    "find_latest_checkpoint",
    "get_device",
    "get_device_type",
    "get_strategy",
    "load_checkpoint",
    "lr_find",
    "LRFinderResult",
    "on",
    "register_early_stopping_callbacks",
    "register_eval_callbacks",
    "resolve_warmup_steps",
    "save_checkpoint",
    "seed_all",
    "TrainState",
    "Trainer",
    "TrainerConfig",
    "wrap_model_distributed",
]
