from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import IterableDataset

IGNORE_INDEX = -100


def tokenize_dataset(
    data: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int = 0,
) -> list[dict[str, list[int]]]:
    """Tokenize formatted samples into input_ids/labels/attention_mask dicts.

    If samples already contain ``"input_ids"``, returns them unchanged.
    Empty texts and empty encodings are filtered out.

    Args:
        data: Formatted samples, each with a ``"text"`` key.
        tokenizer: A HuggingFace tokenizer.
        max_seq_length: Maximum sequence length (0 = use tokenizer default).

    Returns:
        List of dicts with ``input_ids``, ``labels``, and ``attention_mask``
        (all ``list[int]``).
    """
    if not data:
        return []

    if "input_ids" in data[0]:
        return data

    max_length = (
        max_seq_length if max_seq_length > 0 else getattr(tokenizer, "model_max_length", 1024)
    )

    tokenized = []
    for sample in data:
        text = sample.get("text", "")
        if not text:
            continue

        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )

        input_ids = encoded["input_ids"]
        if not input_ids:
            continue

        tokenized.append(
            {
                "input_ids": input_ids,
                "labels": list(input_ids),
                "attention_mask": encoded["attention_mask"],
            }
        )

    return tokenized


def tokenize_preference_dataset(
    data: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int = 0,
) -> list[dict[str, list[int]]]:
    """Tokenize preference pairs into chosen/rejected input_ids and masks.

    Concatenates ``prompt + chosen`` and ``prompt + rejected`` before
    tokenizing.  If samples already contain ``"chosen_input_ids"``, returns
    them unchanged.  Pairs with empty chosen or rejected text are skipped.

    Args:
        data: Preference samples with ``prompt``, ``chosen``, ``rejected``.
        tokenizer: A HuggingFace tokenizer.
        max_seq_length: Maximum sequence length (0 = use tokenizer default).

    Returns:
        List of dicts with ``chosen_input_ids``, ``chosen_attention_mask``,
        ``rejected_input_ids``, and ``rejected_attention_mask``.
    """
    if not data:
        return []

    if "chosen_input_ids" in data[0]:
        return data

    max_length = (
        max_seq_length if max_seq_length > 0 else getattr(tokenizer, "model_max_length", 1024)
    )

    tokenized = []
    for sample in data:
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")
        if not chosen or not rejected:
            continue

        chosen_text = f"{prompt}{chosen}" if prompt else chosen
        rejected_text = f"{prompt}{rejected}" if prompt else rejected

        chosen_enc = tokenizer(
            chosen_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )
        rejected_enc = tokenizer(
            rejected_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )

        if not chosen_enc["input_ids"] or not rejected_enc["input_ids"]:
            continue

        tokenized.append(
            {
                "chosen_input_ids": chosen_enc["input_ids"],
                "chosen_attention_mask": chosen_enc["attention_mask"],
                "rejected_input_ids": rejected_enc["input_ids"],
                "rejected_attention_mask": rejected_enc["attention_mask"],
            }
        )

    return tokenized


def collate_preference(
    batch: list[dict[str, Any]],
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Collate tokenized preference pairs into padded tensors.

    Pads chosen and rejected sequences independently to their respective
    max lengths within the batch.

    Args:
        batch: List of tokenized preference dicts.
        pad_token_id: Token id for input padding (masks use 0).

    Returns:
        Dict with ``chosen_input_ids``, ``chosen_attention_mask``,
        ``rejected_input_ids``, and ``rejected_attention_mask`` tensors.
    """
    result: dict[str, torch.Tensor] = {}

    for prefix in ("chosen", "rejected"):
        ids_key = f"{prefix}_input_ids"
        mask_key = f"{prefix}_attention_mask"

        max_len = max(len(_to_list(sample[ids_key])) for sample in batch)

        all_ids = []
        all_mask = []
        for sample in batch:
            ids = _to_list(sample[ids_key])
            pad_len = max_len - len(ids)
            all_ids.append(ids + [pad_token_id] * pad_len)
            mask = _to_list(sample.get(mask_key, [1] * len(ids)))
            all_mask.append(mask + [0] * pad_len)

        result[ids_key] = torch.tensor(all_ids, dtype=torch.long)
        result[mask_key] = torch.tensor(all_mask, dtype=torch.long)

    return result


def _to_list(val: Any) -> list[int]:
    if isinstance(val, torch.Tensor):
        return val.tolist()
    return list(val)


def collate_tokenized(
    batch: list[dict[str, Any]],
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Collate tokenized samples into padded tensors for model input.

    Pads all sequences to the longest in the batch.  Labels are padded
    with ``-100`` (cross-entropy ignore index).

    Args:
        batch: List of tokenized dicts with ``input_ids`` keys.
        pad_token_id: Token id for input padding (masks use 0).

    Returns:
        Dict with ``input_ids``, ``labels``, and ``attention_mask`` tensors.
    """
    max_len = max(len(sample["input_ids"]) for sample in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for sample in batch:
        ids = _to_list(sample["input_ids"])
        seq_len = len(ids)
        pad_len = max_len - seq_len

        input_ids.append(ids + [pad_token_id] * pad_len)
        lab = _to_list(sample.get("labels", sample["input_ids"]))
        labels.append(lab + [IGNORE_INDEX] * pad_len)
        mask = _to_list(sample.get("attention_mask", [1] * seq_len))
        attention_mask.append(mask + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def tokenize_sample(
    sample: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int = 0,
) -> dict[str, list[int]] | None:
    """Tokenize a single formatted sample.

    Args:
        sample: A dict with a ``"text"`` key.
        tokenizer: A HuggingFace tokenizer.
        max_seq_length: Maximum sequence length (0 = use tokenizer default).

    Returns:
        Dict with ``input_ids``, ``labels``, ``attention_mask``, or
        ``None`` if the text is empty or produces no tokens.
    """
    if "input_ids" in sample:
        return sample

    text = sample.get("text", "")
    if not text:
        return None

    max_length = (
        max_seq_length if max_seq_length > 0 else getattr(tokenizer, "model_max_length", 1024)
    )

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )

    input_ids = encoded["input_ids"]
    if not input_ids:
        return None

    return {
        "input_ids": input_ids,
        "labels": list(input_ids),
        "attention_mask": encoded["attention_mask"],
    }


class StreamingTokenizedDataset(IterableDataset):
    """Wraps a HuggingFace IterableDataset with on-the-fly tokenization.

    Args:
        dataset: A HuggingFace ``IterableDataset`` yielding formatted samples.
        tokenizer: A HuggingFace tokenizer.
        max_seq_length: Maximum sequence length (0 = use tokenizer default).
    """

    def __init__(
        self,
        dataset: Any,
        tokenizer: Any,
        max_seq_length: int = 0,
    ) -> None:
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        for sample in self._dataset:
            tokenized = tokenize_sample(
                sample,
                self._tokenizer,
                self._max_seq_length,
            )
            if tokenized is not None:
                yield tokenized
