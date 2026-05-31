from __future__ import annotations

from typing import Any

from xaytune.data.agent_formats import AgentMessage

IGNORE_INDEX = -100


def tokenize_agent_dataset(
    data: list[list[AgentMessage]],
    tokenizer: Any,
    max_seq_length: int = 0,
) -> list[dict[str, list[int]]]:
    if not data:
        return []

    max_length = (
        max_seq_length
        if max_seq_length > 0
        else getattr(tokenizer, "model_max_length", 1024)
    )

    tokenized = []
    for messages in data:
        all_ids: list[int] = []
        all_labels: list[int] = []

        for msg in messages:
            encoded = tokenizer(
                msg.content,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            msg_ids = encoded["input_ids"]

            all_ids.extend(msg_ids)
            if msg.trainable:
                all_labels.extend(msg_ids)
            else:
                all_labels.extend([IGNORE_INDEX] * len(msg_ids))

        if max_seq_length > 0 and len(all_ids) > max_length:
            all_ids = all_ids[:max_length]
            all_labels = all_labels[:max_length]

        if not all_ids:
            continue

        tokenized.append({
            "input_ids": all_ids,
            "labels": all_labels,
            "attention_mask": [1] * len(all_ids),
        })

    return tokenized
