import math
from unittest.mock import MagicMock, patch

import pytest
import torch

from xaytune.eval.evaluate import evaluate


def _mock_model(loss_value: float, vocab: int = 8) -> MagicMock:
    """A stand-in model returning a realistic ``(loss, logits)`` output.

    ``logits`` must be a real tensor shaped to the batch: ``evaluate`` argmaxes
    it and masks the result against ``labels``, so a bare ``MagicMock`` yields a
    non-indexable stand-in instead of exercising the metric path.
    """

    def _forward(**batch):
        labels = torch.as_tensor(batch["labels"])
        output = MagicMock()
        output.loss = MagicMock()
        output.loss.item.return_value = loss_value
        output.logits = torch.zeros(*labels.shape, vocab)
        return output

    model = MagicMock(side_effect=_forward)
    model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
    return model


def _batch(seq_len: int = 2) -> dict:
    return {
        "input_ids": torch.arange(seq_len).unsqueeze(0),
        "labels": torch.arange(seq_len).unsqueeze(0),
    }


class TestEvaluate:
    @patch("xaytune.models.load_model")
    def test_evaluate_with_model_path(self, mock_load_model):
        mock_model = _mock_model(0.5)
        mock_result = MagicMock()
        mock_result.model = mock_model
        mock_result.tokenizer = MagicMock()
        mock_load_model.return_value = mock_result

        results = evaluate(
            model="output/my-model",
            dataset=[_batch()],
            metrics=["loss"],
        )

        assert "loss" in results
        mock_load_model.assert_called_once()

    def test_evaluate_with_model_object(self):
        results = evaluate(
            model=_mock_model(0.3),
            dataset=[_batch(seq_len=1)],
            metrics=["loss"],
        )

        assert results["loss"] == 0.3

    def test_evaluate_multiple_metrics(self):
        results = evaluate(
            model=_mock_model(1.0),
            dataset=[_batch(), _batch()],
            metrics=["loss", "perplexity"],
        )

        assert results["loss"] == 1.0
        assert results["perplexity"] == pytest.approx(math.exp(1.0), rel=1e-5)

    def test_evaluate_default_metrics(self):
        results = evaluate(model=_mock_model(0.5), dataset=[_batch(seq_len=1)])

        assert "loss" in results
        assert "perplexity" in results

    def test_evaluate_returns_dict(self):
        results = evaluate(
            model=_mock_model(0.5),
            dataset=[_batch(seq_len=1)],
            metrics=["loss"],
        )

        assert isinstance(results, dict)

    def test_evaluate_empty_dataset(self):
        results = evaluate(model=_mock_model(0.0), dataset=[], metrics=["loss"])
        assert results["loss"] == 0.0

    def test_evaluate_accepts_list_labels(self):
        """Plain-list batch values must not crash the metric path.

        ``evaluate`` explicitly passes non-tensor batch values through when
        moving the batch to the device; masking them used to raise
        ``IndexError``/``AttributeError`` a few lines later.
        """
        results = evaluate(
            model=_mock_model(0.5),
            dataset=[{"input_ids": [[0, 1]], "labels": [[0, 1]]}],
            metrics=["loss"],
        )

        assert results["loss"] == 0.5

    def test_evaluate_masks_ignore_index(self):
        """Positions labelled -100 are excluded from prediction metrics."""
        batch = {
            "input_ids": torch.tensor([[0, 1]]),
            "labels": torch.tensor([[-100, 1]]),
        }
        results = evaluate(
            model=_mock_model(0.5),
            dataset=[batch],
            metrics=["token_accuracy"],
        )

        # Only the unmasked position is scored; logits are all zeros so the
        # argmax is 0, which does not match the label 1.
        assert results["token_accuracy"] == 0.0
