from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from trainlib.data.validation import (
    DataValidationError,
    validate_batch,
    validate_dataset_sample,
)


class TestValidateBatch:
    def test_valid_batch_no_issues(self):
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "labels": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        assert validate_batch(batch) == []

    def test_missing_input_ids_and_text(self):
        issues = validate_batch({"labels": torch.tensor([1])})
        assert any("input_ids" in i for i in issues)

    def test_text_field_accepted(self):
        issues = validate_batch({"text": "hello"})
        assert issues == []

    def test_non_dict_batch(self):
        issues = validate_batch("not a dict")
        assert any("dict" in i for i in issues)

    def test_wrong_dtype(self):
        batch = {"input_ids": torch.tensor([1.0, 2.0])}
        issues = validate_batch(batch)
        assert any("integer" in i for i in issues)

    def test_seq_length_exceeds_max(self):
        batch = {"input_ids": torch.randint(0, 100, (1, 512))}
        issues = validate_batch(batch, max_seq_length=256)
        assert any("512" in i and "256" in i for i in issues)

    def test_seq_length_within_max(self):
        batch = {"input_ids": torch.randint(0, 100, (1, 128))}
        issues = validate_batch(batch, max_seq_length=256)
        assert issues == []

    def test_labels_shape_mismatch(self):
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "labels": torch.tensor([[1, 2]]),
        }
        issues = validate_batch(batch)
        assert any("labels" in i and "shape" in i for i in issues)

    def test_attention_mask_shape_mismatch(self):
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        issues = validate_batch(batch)
        assert any("attention_mask" in i and "shape" in i for i in issues)


class TestValidateDatasetSample:
    def test_valid_dataset_passes(self):
        data = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
            }
        ]
        dl = DataLoader(data, batch_size=1)
        validate_dataset_sample(dl)

    def test_empty_dataset_raises(self):
        dl = DataLoader([], batch_size=1)
        with pytest.raises(DataValidationError, match="empty"):
            validate_dataset_sample(dl)

    def test_invalid_data_raises(self):
        data = [{"labels": torch.tensor([1, 2])}]
        dl = DataLoader(data, batch_size=1)
        with pytest.raises(DataValidationError, match="input_ids"):
            validate_dataset_sample(dl)
