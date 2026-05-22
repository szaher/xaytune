from trainlib.models.loader import ModelResult, load_model
from trainlib.models.registry import model_registry

register_model = model_registry.register

__all__ = ["ModelResult", "load_model", "model_registry", "register_model"]
