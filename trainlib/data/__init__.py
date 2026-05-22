from trainlib.data.loader import load_dataset
from trainlib.data.registry import format_registry
import trainlib.data.formats  # register built-in formats

register_format = format_registry.register

__all__ = ["load_dataset", "format_registry", "register_format"]
