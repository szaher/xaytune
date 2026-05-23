import subprocess
import sys

import pytest

import trainlib


class TestPackageMetadata:
    def test_version_is_string(self):
        assert isinstance(trainlib.__version__, "0.1.0".__class__)

    def test_version_matches_pyproject(self):
        assert trainlib.__version__ == "0.1.0"

    def test_all_exports(self):
        expected = {"__version__", "align", "evaluate", "finetune", "pretrain"}
        assert set(trainlib.__all__) == expected


class TestPythonModule:
    def test_python_m_trainlib_runs(self):
        result = subprocess.run(
            [sys.executable, "-m", "trainlib", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_python_m_trainlib_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "trainlib", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "train" in result.stdout


class TestPyTyped:
    def test_py_typed_exists(self):
        import importlib.resources as resources
        ref = resources.files("trainlib") / "py.typed"
        assert ref.is_file()


class TestImports:
    def test_top_level_imports(self):
        from trainlib import finetune, pretrain, align, evaluate
        assert callable(finetune)
        assert callable(pretrain)
        assert callable(align)
        assert callable(evaluate)

    def test_submodule_imports(self):
        from trainlib.models import load_model, register_model
        from trainlib.data import load_dataset, register_format
        from trainlib.trainer import Trainer, on
        from trainlib.eval import evaluate, register_metric
        from trainlib.export import merge, save, push_to_hub
        from trainlib.logging import setup_logging
        from trainlib.recipes import recipe_registry

    def test_align_losses_importable(self):
        from trainlib.recipes.align import (
            dpo_loss, grpo_loss, orpo_loss, simpo_loss,
            ppo_clip_loss, reinforce_loss,
        )


class TestEntryPoint:
    def test_cli_entry_point_defined(self):
        from importlib.metadata import entry_points
        eps = entry_points()
        console_scripts = eps.select(group="console_scripts")
        names = [ep.name for ep in console_scripts]
        assert "trainlib" in names

    def test_cli_entry_point_resolves(self):
        from importlib.metadata import entry_points
        eps = entry_points()
        console_scripts = eps.select(group="console_scripts")
        trainlib_ep = [ep for ep in console_scripts if ep.name == "trainlib"][0]
        fn = trainlib_ep.load()
        assert callable(fn)
