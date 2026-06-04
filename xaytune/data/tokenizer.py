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

        labels = list(input_ids)

        prompt_text = sample.get("prompt_text")
        if prompt_text:
            prompt_enc = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=True,
            )
            prompt_len = len(prompt_enc["input_ids"])
            for i in range(min(prompt_len, len(labels))):
                labels[i] = IGNORE_INDEX

        tokenized.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": encoded["attention_mask"],
            }
        )

    return tokenized


def tokenize_multiturn(
    data: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int = 0,
) -> list[dict[str, list[int]]]:
    """Tokenize multi-turn conversations with per-turn label masking.

    Each turn is tokenized independently. Assistant turns are trainable
    (labels = token IDs), all other roles are masked (labels = -100).

    Args:
        data: Samples with a ``"turns"`` key containing a list of
            ``{"role": "user"|"assistant"|"system", "content": "..."}``.
        tokenizer: A HuggingFace tokenizer.
        max_seq_length: Maximum sequence length (0 = use tokenizer default).

    Returns:
        List of dicts with ``input_ids``, ``labels``, and ``attention_mask``.
    """
    if not data:
        return []

    if "input_ids" in data[0]:
        return data

    max_length = (
        max_seq_length if max_seq_length > 0 else getattr(tokenizer, "model_max_length", 1024)
    )

    use_chat_template = bool(
        data[0].get("_use_chat_template")
        and hasattr(tokenizer, "apply_chat_template")
    )

    tokenized = []
    for sample in data:
        turns = sample.get("turns", [])
        if not turns:
            continue

        all_ids: list[int] = []
        all_labels: list[int] = []

        if use_chat_template:
            messages = [{"role": t["role"], "content": t["content"]} for t in turns]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            full_enc = tokenizer(
                full_text, truncation=True, max_length=max_length,
                padding=False, return_attention_mask=False,
            )
            all_ids = full_enc["input_ids"]
            all_labels = list(all_ids)

            cursor = 0
            for turn in turns:
                prompt_msgs = messages[: cursor + 1]
                prefix = tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=False,
                )
                prefix_enc = tokenizer(
                    prefix, truncation=True, max_length=max_length,
                    padding=False, return_attention_mask=False,
                )
                turn_end = len(prefix_enc["input_ids"])

                if cursor == 0:
                    prev_end = 0
                else:
                    prev_msgs = messages[:cursor]
                    prev_text = tokenizer.apply_chat_template(
                        prev_msgs, tokenize=False, add_generation_prompt=False,
                    )
                    prev_enc = tokenizer(
                        prev_text, truncation=True, max_length=max_length,
                        padding=False, return_attention_mask=False,
                    )
                    prev_end = len(prev_enc["input_ids"])

                if turn["role"] != "assistant":
                    for j in range(prev_end, min(turn_end, len(all_labels))):
                        all_labels[j] = IGNORE_INDEX

                cursor += 1
        else:
            for i, turn in enumerate(turns):
                role = turn.get("role", "user")
                content = turn.get("content", "")

                text = f"### {role.capitalize()}:\n{content}"
                if i < len(turns) - 1:
                    text += "\n\n"

                enc = tokenizer(
                    text,
                    truncation=False,
                    padding=False,
                    add_special_tokens=(i == 0),
                    return_attention_mask=False,
                )
                turn_ids = enc["input_ids"]

                all_ids.extend(turn_ids)
                if role == "assistant":
                    all_labels.extend(turn_ids)
                else:
                    all_labels.extend([IGNORE_INDEX] * len(turn_ids))

            all_ids = all_ids[:max_length]
            all_labels = all_labels[:max_length]

        if not all_ids:
            continue

        tokenized.append(
            {
                "input_ids": all_ids,
                "labels": all_labels,
                "attention_mask": [1] * len(all_ids),
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

        prompt_len = 0
        if prompt:
            prompt_enc = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=True,
            )
            prompt_len = len(prompt_enc["input_ids"])

        tokenized.append(
            {
                "chosen_input_ids": chosen_enc["input_ids"],
                "chosen_attention_mask": chosen_enc["attention_mask"],
                "rejected_input_ids": rejected_enc["input_ids"],
                "rejected_attention_mask": rejected_enc["attention_mask"],
                "prompt_length": prompt_len,
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

    if "prompt_length" in batch[0]:
        result["prompt_length"] = torch.tensor(
            [sample.get("prompt_length", 0) for sample in batch], dtype=torch.long
        )

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


def tokenize_prompt_dataset(
    data: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int = 0,
) -> list[dict[str, list[int]]]:
    """Tokenize prompts only for online RL generation.

    Each sample needs a ``"prompt"`` field. Returns prompt token IDs and
    masks — completions are generated online during training.

    Args:
        data: Samples with a ``"prompt"`` key.
        tokenizer: A HuggingFace tokenizer.
        max_seq_length: Maximum prompt length (0 = use tokenizer default).

    Returns:
        List of dicts with ``prompt_input_ids`` and ``prompt_attention_mask``.
    """
    if not data:
        return []

    if "prompt_input_ids" in data[0]:
        return data

    max_length = (
        max_seq_length if max_seq_length > 0 else getattr(tokenizer, "model_max_length", 1024)
    )

    tokenized = []
    for sample in data:
        prompt = sample.get("prompt", "")
        if not prompt:
            continue

        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )

        if not encoded["input_ids"]:
            continue

        tokenized.append(
            {
                "prompt_input_ids": encoded["input_ids"],
                "prompt_attention_mask": encoded["attention_mask"],
            }
        )

    return tokenized


def collate_prompt(
    batch: list[dict[str, Any]],
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Collate tokenized prompts into padded tensors.

    Args:
        batch: List of dicts with ``prompt_input_ids``.
        pad_token_id: Token id for padding.

    Returns:
        Dict with ``prompt_input_ids`` and ``prompt_attention_mask`` tensors.
    """
    max_len = max(len(_to_list(sample["prompt_input_ids"])) for sample in batch)

    all_ids = []
    all_mask = []
    for sample in batch:
        ids = _to_list(sample["prompt_input_ids"])
        pad_len = max_len - len(ids)
        all_ids.append(ids + [pad_token_id] * pad_len)
        mask = _to_list(sample.get("prompt_attention_mask", [1] * len(ids)))
        all_mask.append(mask + [0] * pad_len)

    return {
        "prompt_input_ids": torch.tensor(all_ids, dtype=torch.long),
        "prompt_attention_mask": torch.tensor(all_mask, dtype=torch.long),
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

    labels = list(input_ids)

    prompt_text = sample.get("prompt_text")
    if prompt_text:
        prompt_enc = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=True,
        )
        prompt_len = len(prompt_enc["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = IGNORE_INDEX

    return {
        "input_ids": input_ids,
        "labels": labels,
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
            if "turns" in sample:
                batch = tokenize_multiturn(
                    [sample], self._tokenizer, self._max_seq_length
                )
                if batch:
                    yield batch[0]
            else:
                tokenized = tokenize_sample(
                    sample,
                    self._tokenizer,
                    self._max_seq_length,
                )
                if tokenized is not None:
                    yield tokenized
