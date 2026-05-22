from trainlib.config.schema import TrainerConfig
from trainlib.trainer.callbacks import CallbackManager, TrainState
from trainlib.trainer.checkpointing import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from trainlib.trainer.distributed import (
    DistributedContext,
    get_strategy,
    wrap_model_distributed,
)
from trainlib.trainer.loop import Trainer

_global_callback_manager = CallbackManager()

on = _global_callback_manager.on

__all__ = [
    "CallbackManager",
    "DistributedContext",
    "find_latest_checkpoint",
    "get_strategy",
    "load_checkpoint",
    "on",
    "save_checkpoint",
    "TrainState",
    "Trainer",
    "TrainerConfig",
    "wrap_model_distributed",
]
