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
