from xaytune.recipes.align import align
from xaytune.recipes.finetune import finetune
from xaytune.recipes.pretrain import pretrain
from xaytune.utils.registry import Registry

recipe_registry = Registry("recipe")

recipe_registry.register("finetune")(finetune)
recipe_registry.register("pretrain")(pretrain)
recipe_registry.register("align")(align)

__all__ = ["align", "finetune", "pretrain", "recipe_registry"]
