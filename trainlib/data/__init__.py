import trainlib.data.formats  # register built-in formats  # noqa: F401
from trainlib.data.formats import apply_chat_template
from trainlib.data.loader import load_dataset
from trainlib.data.packing import pack_sequences
from trainlib.data.preferences import load_preference_dataset
from trainlib.data.registry import format_registry
from trainlib.data.tokenizer import (
    StreamingTokenizedDataset,
    collate_preference,
    collate_tokenized,
    tokenize_dataset,
    tokenize_preference_dataset,
    tokenize_sample,
)
from trainlib.data.validation import DataValidationError, validate_dataset_sample

register_format = format_registry.register

__all__ = [
    "apply_chat_template",
    "collate_preference",
    "collate_tokenized",
    "load_dataset",
    "load_preference_dataset",
    "StreamingTokenizedDataset",
    "tokenize_sample",
    "format_registry",
    "register_format",
    "pack_sequences",
    "tokenize_dataset",
    "tokenize_preference_dataset",
    "DataValidationError",
    "validate_dataset_sample",
]
