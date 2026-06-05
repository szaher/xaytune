import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from xaytune.cli import main


class TestCliEvalDefaults:
    @patch("xaytune.eval.benchmarks.benchmark_evaluate")
    def test_eval_benchmarks_does_not_crash(self, mock_bench):
        mock_bench.return_value = {"mmlu": {"acc,none": 0.65}}
        result = main(["eval", "--model", "test-model", "--benchmarks", "mmlu"])
        assert result == 0
        mock_bench.assert_called_once_with(
            model="test-model",
            benchmarks=["mmlu"],
            num_fewshot=None,
        )

    @patch("xaytune.eval.benchmarks.benchmark_evaluate")
    def test_eval_multiple_benchmarks(self, mock_bench):
        mock_bench.return_value = {"mmlu": {"acc": 0.6}, "gsm8k": {"acc": 0.4}}
        result = main(["eval", "--model", "m", "--benchmarks", "mmlu,gsm8k"])
        assert result == 0
        call_kwargs = mock_bench.call_args
        assert sorted(call_kwargs[1]["benchmarks"]) == ["gsm8k", "mmlu"]

    def test_eval_no_benchmarks_no_dataset_returns_error(self, capsys):
        result = main(["eval", "--model", "test-model"])
        assert result == 1
        captured = capsys.readouterr()
        assert "benchmarks" in captured.err.lower() or "dataset" in captured.err.lower()


class TestCliEvalWithDataset:
    @patch("xaytune.eval.evaluate")
    def test_eval_dataset_default_metrics(self, mock_evaluate):
        mock_evaluate.return_value = {"loss": 0.5, "perplexity": 1.65}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"text": "hello"}) + "\n")
            path = f.name

        result = main(["eval", "--model", "test-model", "--dataset", path])
        assert result == 0
        call_kwargs = mock_evaluate.call_args[1]
        assert call_kwargs["metrics"] == ["loss", "perplexity"]

    @patch("xaytune.eval.evaluate")
    def test_eval_dataset_custom_metrics(self, mock_evaluate):
        mock_evaluate.return_value = {"loss": 0.5, "token_accuracy": 0.9}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"text": "hello"}) + "\n")
            path = f.name

        result = main([
            "eval", "--model", "test-model",
            "--dataset", path,
            "--metrics", "loss,token_accuracy",
        ])
        assert result == 0
        call_kwargs = mock_evaluate.call_args[1]
        assert call_kwargs["metrics"] == ["loss", "token_accuracy"]

    def test_eval_missing_dataset_returns_error(self, capsys):
        result = main([
            "eval", "--model", "test-model",
            "--dataset", "/nonexistent/path/data.jsonl",
        ])
        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower() or "error" in captured.err.lower()

    @patch("xaytune.eval.evaluate")
    def test_eval_dataset_without_metrics_flag_uses_defaults(self, mock_evaluate):
        mock_evaluate.return_value = {"loss": 0.3, "perplexity": 1.35}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"input": "x"}) + "\n")
            path = f.name

        result = main(["eval", "--model", "m", "--dataset", path])
        assert result == 0
        call_kwargs = mock_evaluate.call_args[1]
        assert "loss" in call_kwargs["metrics"]
        assert "perplexity" in call_kwargs["metrics"]
