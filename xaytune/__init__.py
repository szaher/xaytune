"""xaytune — An opinionated LLM training and fine-tuning library."""

__version__ = "0.6.0"

from xaytune.eval import evaluate
from xaytune.pipeline import run_pipeline as pipeline
from xaytune.plugins import discover_plugins
from xaytune.recipes.align import align
from xaytune.recipes.finetune import finetune
from xaytune.recipes.pretrain import pretrain
from xaytune.studio.jobs import JobManager
from xaytune.trainer.lr_finder import lr_find

discover_plugins()

__all__ = [
    "__version__",
    "align",
    "evaluate",
    "finetune",
    "JobManager",
    "lr_find",
    "pipeline",
    "pretrain",
]
