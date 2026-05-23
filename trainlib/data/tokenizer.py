from __future__ import annotations

from typing import Any

import torch

IGNORE_INDEX = -100


def tokenize_dataset(
    data: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int = 0,
) -> list[dict[str, list[int]]]:
    if not data:
        return []

    if "input_ids" in data[0]:
        return data

    max_length = max_seq_length if max_seq_length > 0 else getattr(
        tokenizer, "model_max_length", 1024
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

        tokenized.append({
            "input_ids": input_ids,
            "labels": list(input_ids),
            "attention_mask": encoded["attention_mask"],
        })

    return tokenized


def _to_list(val: Any) -> list[int]:
    if isinstance(val, torch.Tensor):
        return val.tolist()
    return list(val)


def collate_tokenized(
    batch: list[dict[str, Any]],
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
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
