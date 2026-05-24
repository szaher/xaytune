from unittest.mock import MagicMock, patch

from xaytune.eval.evaluate import evaluate


class TestEvaluate:
    @patch("xaytune.models.load_model")
    def test_evaluate_with_model_path(self, mock_load_model):
        mock_result = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_result.model = mock_model
        mock_result.tokenizer = mock_tokenizer
        mock_load_model.return_value = mock_result

        mock_model.return_value = MagicMock()
        mock_model.return_value.loss = MagicMock()
        mock_model.return_value.loss.item.return_value = 0.5

        results = evaluate(
            model="output/my-model",
            dataset=[{"input_ids": [1, 2], "labels": [1, 2]}],
            metrics=["loss"],
        )

        assert "loss" in results
        mock_load_model.assert_called_once()

    def test_evaluate_with_model_object(self):
        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.loss = MagicMock()
        mock_model.return_value.loss.item.return_value = 0.3

        results = evaluate(
            model=mock_model,
            dataset=[{"input_ids": [1], "labels": [1]}],
            metrics=["loss"],
        )

        assert "loss" in results

    def test_evaluate_multiple_metrics(self):
        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.loss = MagicMock()
        mock_model.return_value.loss.item.return_value = 1.0

        results = evaluate(
            model=mock_model,
            dataset=[
                {"input_ids": [1, 2], "labels": [1, 2]},
                {"input_ids": [3, 4], "labels": [3, 4]},
            ],
            metrics=["loss", "perplexity"],
        )

        assert "loss" in results
        assert "perplexity" in results
        assert results["loss"] == 1.0

    def test_evaluate_default_metrics(self):
        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.loss = MagicMock()
        mock_model.return_value.loss.item.return_value = 0.5

        results = evaluate(
            model=mock_model,
            dataset=[{"input_ids": [1], "labels": [1]}],
        )

        assert "loss" in results
        assert "perplexity" in results

    def test_evaluate_returns_dict(self):
        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.loss = MagicMock()
        mock_model.return_value.loss.item.return_value = 0.5

        results = evaluate(
            model=mock_model,
            dataset=[{"input_ids": [1], "labels": [1]}],
            metrics=["loss"],
        )

        assert isinstance(results, dict)

    def test_evaluate_empty_dataset(self):
        mock_model = MagicMock()
        results = evaluate(model=mock_model, dataset=[], metrics=["loss"])
        assert results["loss"] == 0.0
