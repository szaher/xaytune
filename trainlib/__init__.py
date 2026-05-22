"""trainlib — An opinionated LLM training and fine-tuning library."""

__version__ = "0.1.0"

from trainlib.recipes.finetune import finetune
from trainlib.recipes.pretrain import pretrain

__all__ = ["__version__", "finetune", "pretrain"]
