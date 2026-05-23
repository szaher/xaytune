from trainlib.recipes.align import align
from trainlib.recipes.finetune import finetune
from trainlib.recipes.pretrain import pretrain
from trainlib.utils.registry import Registry

recipe_registry = Registry("recipe")

recipe_registry.register("finetune")(finetune)
recipe_registry.register("pretrain")(pretrain)
recipe_registry.register("align")(align)

__all__ = ["align", "finetune", "pretrain", "recipe_registry"]
