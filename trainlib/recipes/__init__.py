from trainlib.utils.registry import Registry

recipe_registry = Registry("recipe")

from trainlib.recipes.finetune import finetune
from trainlib.recipes.pretrain import pretrain

recipe_registry.register("finetune")(finetune)
recipe_registry.register("pretrain")(pretrain)

__all__ = ["finetune", "pretrain", "recipe_registry"]
