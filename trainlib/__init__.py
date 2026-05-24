"""trainlib — An opinionated LLM training and fine-tuning library."""

__version__ = "0.6.0"

from trainlib.eval import evaluate
from trainlib.plugins import discover_plugins
from trainlib.recipes.align import align
from trainlib.recipes.finetune import finetune
from trainlib.recipes.pretrain import pretrain
from trainlib.studio.jobs import JobManager
from trainlib.trainer.lr_finder import lr_find

discover_plugins()

__all__ = [
    "__version__",
    "align",
    "evaluate",
    "finetune",
    "JobManager",
    "lr_find",
    "pretrain",
]
