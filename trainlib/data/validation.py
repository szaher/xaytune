from __future__ import annotations

from typing import Any

import torch


class DataValidationError(ValueError):
    """Raised when a data batch fails validation checks."""


def validate_batch(
    batch: dict[str, Any],
    *,
    max_seq_length: int = 0,
) -> list[str]:
    """Check a single batch dict for common data issues.

    Returns a list of human-readable issue strings (empty = valid).
    """
    issues: list[str] = []

    if not isinstance(batch, dict):
        issues.append(f"Batch must be a dict, got {type(batch).__name__}")
        return issues

    is_preference = "chosen_input_ids" in batch
    if "input_ids" not in batch and not is_preference:
        issues.append("Batch missing required field: 'input_ids'")

    if "input_ids" in batch:
        ids = batch["input_ids"]
        if not isinstance(ids, torch.Tensor):
            issues.append(f"'input_ids' should be a Tensor, got {type(ids).__name__}")
        elif ids.dtype not in (torch.long, torch.int, torch.int32):
            issues.append(f"'input_ids' dtype should be integer, got {ids.dtype}")

        if max_seq_length > 0 and isinstance(ids, torch.Tensor) and ids.ndim >= 1:
            seq_len = ids.shape[-1]
            if seq_len > max_seq_length:
                issues.append(
                    f"Sequence length {seq_len} exceeds max_seq_length {max_seq_length}"
                )

    if "labels" in batch and "input_ids" in batch:
        ids = batch["input_ids"]
        labels = batch["labels"]
        if isinstance(ids, torch.Tensor) and isinstance(labels, torch.Tensor):
            if ids.shape != labels.shape:
                issues.append(
                    f"'labels' shape {labels.shape} doesn't match 'input_ids' shape {ids.shape}"
                )

    if "attention_mask" in batch and "input_ids" in batch:
        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        if isinstance(ids, torch.Tensor) and isinstance(mask, torch.Tensor):
            if ids.shape != mask.shape:
                issues.append(
                    f"'attention_mask' shape {mask.shape} doesn't match "
                    f"'input_ids' shape {ids.shape}"
                )

    return issues


def validate_dataset_sample(
    dataloader: Any,
    *,
    max_seq_length: int = 0,
) -> None:
    """Draw one batch from a dataloader and validate it.

    Raises:
        DataValidationError: If the dataset is empty or the batch has issues.
    """
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        raise DataValidationError("Dataset is empty")

    issues = validate_batch(batch, max_seq_length=max_seq_length)
    if issues:
        raise DataValidationError(
            "Data validation failed:\n" + "\n".join(f"  - {i}" for i in issues)
        )
