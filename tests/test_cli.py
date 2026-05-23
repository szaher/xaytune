import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "test_config" / "fixtures"


class TestCLI:
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "trainlib.cli", *args],
            capture_output=True,
            text=True,
        )

    def test_no_args_shows_help(self):
        result = self._run()
        assert result.returncode != 0 or "usage" in result.stdout.lower() or "usage" in result.stderr.lower()

    def test_train_requires_config(self):
        result = self._run("train")
        assert result.returncode != 0

    def test_train_with_config(self):
        result = self._run("train", "--config", str(FIXTURES / "full_config.yaml"), "--dry-run")
        assert result.returncode == 0
        assert "my-model" in result.stdout

    def test_train_with_overrides(self):
        result = self._run(
            "train",
            "--config", str(FIXTURES / "full_config.yaml"),
            "--dry-run",
            "--override", "model.name=overridden",
        )
        assert result.returncode == 0
        assert "overridden" in result.stdout

    def test_train_invalid_config(self):
        result = self._run("train", "--config", "nonexistent.yaml")
        assert result.returncode != 0

    def test_version(self):
        result = self._run("--version")
        assert "0.1.0" in result.stdout


from unittest.mock import patch, MagicMock
from trainlib.cli import main


class TestTrainCommand:
    @patch("trainlib.cli.load_config")
    @patch("trainlib.cli.validate_config")
    @patch("trainlib.cli.recipe_registry")
    def test_train_runs_recipe(self, mock_registry, mock_validate, mock_load):
        mock_config = MagicMock()
        mock_config.recipe = "finetune"
        mock_config.model_dump_json.return_value = "{}"
        mock_load.return_value = mock_config
        mock_recipe = MagicMock()
        mock_registry.get.return_value = mock_recipe

        result = main(["train", "--config", "test.yaml"])

        mock_registry.get.assert_called_once_with("finetune")
        mock_recipe.assert_called_once_with(config=mock_config)
        assert result == 0


class TestListCommand:
    def test_list_recipes(self, capsys):
        result = main(["list", "recipes"])
        captured = capsys.readouterr()
        assert result == 0
        assert "finetune" in captured.out
        assert "pretrain" in captured.out
        assert "align" in captured.out

    def test_list_formats(self, capsys):
        result = main(["list", "formats"])
        captured = capsys.readouterr()
        assert result == 0
        assert "alpaca" in captured.out

    def test_list_metrics(self, capsys):
        result = main(["list", "metrics"])
        captured = capsys.readouterr()
        assert result == 0
        assert "loss" in captured.out

    def test_list_no_type(self, capsys):
        result = main(["list"])
        captured = capsys.readouterr()
        assert result == 0
        assert "recipes" in captured.out.lower() or "Recipes" in captured.out

    def test_list_unknown_type(self, capsys):
        result = main(["list", "unknown"])
        captured = capsys.readouterr()
        assert result == 1


class TestEvalCommand:
    def test_eval_parser_exists(self):
        from trainlib.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["eval", "--model", "test-model", "--benchmarks", "mmlu"])
        assert args.command == "eval"
        assert args.model == "test-model"
        assert args.benchmarks == "mmlu"

    def test_eval_with_metrics(self):
        from trainlib.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "eval", "--model", "test-model",
            "--metrics", "loss,perplexity",
            "--dataset", "data/eval.jsonl",
        ])
        assert args.metrics == "loss,perplexity"
        assert args.dataset == "data/eval.jsonl"

    def test_eval_benchmarks_calls_benchmark_evaluate(self):
        from unittest.mock import patch, MagicMock
        from trainlib.cli import main

        with patch("trainlib.eval.benchmarks.benchmark_evaluate") as mock_bench:
            mock_bench.return_value = {"mmlu": {"acc,none": 0.65}}
            result = main(["eval", "--model", "test-model", "--benchmarks", "mmlu,gsm8k"])

        assert result == 0
        mock_bench.assert_called_once_with(
            model="test-model",
            benchmarks=["mmlu", "gsm8k"],
            num_fewshot=None,
        )

    def test_eval_num_fewshot(self):
        from unittest.mock import patch
        from trainlib.cli import main

        with patch("trainlib.eval.benchmarks.benchmark_evaluate") as mock_bench:
            mock_bench.return_value = {}
            main(["eval", "--model", "m", "--benchmarks", "mmlu", "--num-fewshot", "5"])

        call_kwargs = mock_bench.call_args[1]
        assert call_kwargs["num_fewshot"] == 5

    def test_eval_requires_model(self):
        from trainlib.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["eval"])
        assert exc_info.value.code != 0

    def test_eval_prints_results(self, capsys):
        from unittest.mock import patch
        from trainlib.cli import main

        with patch("trainlib.eval.benchmarks.benchmark_evaluate") as mock_bench:
            mock_bench.return_value = {"mmlu": {"acc,none": 0.65}}
            main(["eval", "--model", "m", "--benchmarks", "mmlu"])

        captured = capsys.readouterr()
        assert "mmlu" in captured.out
