from trainlib.models.loader import ModelResult, load_model
from trainlib.models.peft import apply_lora, get_target_modules
from trainlib.models.registry import model_registry

register_model = model_registry.register

__all__ = ["ModelResult", "load_model", "apply_lora", "get_target_modules", "model_registry", "register_model"]
