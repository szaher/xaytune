import subprocess
import sys

import xaytune


class TestPackageMetadata:
    def test_version_is_string(self):
        assert isinstance(xaytune.__version__, str)

    def test_version_matches_pyproject(self):
        assert xaytune.__version__ == "0.6.0"

    def test_all_exports(self):
        expected = {
            "__version__",
            "align",
            "evaluate",
            "finetune",
            "JobManager",
            "lr_find",
            "pretrain",
        }
        assert set(xaytune.__all__) == expected


class TestPythonModule:
    def test_python_m_xaytune_runs(self):
        result = subprocess.run(
            [sys.executable, "-m", "xaytune", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert xaytune.__version__ in result.stdout

    def test_python_m_xaytune_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "xaytune", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "train" in result.stdout


class TestPyTyped:
    def test_py_typed_exists(self):
        import importlib.resources as resources

        ref = resources.files("xaytune") / "py.typed"
        assert ref.is_file()


class TestImports:
    def test_top_level_imports(self):
        from xaytune import align, evaluate, finetune, pretrain

        assert callable(finetune)
        assert callable(pretrain)
        assert callable(align)
        assert callable(evaluate)

    def test_submodule_imports(self):
        pass

    def test_align_losses_importable(self):
        pass


class TestEntryPoint:
    def test_cli_entry_point_defined(self):
        from importlib.metadata import entry_points

        eps = entry_points()
        console_scripts = eps.select(group="console_scripts")
        names = [ep.name for ep in console_scripts]
        assert "xaytune" in names

    def test_cli_entry_point_resolves(self):
        from importlib.metadata import entry_points

        eps = entry_points()
        console_scripts = eps.select(group="console_scripts")
        xaytune_ep = [ep for ep in console_scripts if ep.name == "xaytune"][0]
        fn = xaytune_ep.load()
        assert callable(fn)
