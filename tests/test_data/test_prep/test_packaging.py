import tomllib
from pathlib import Path


class TestPackaging:
    def test_data_prep_extra_exists(self):
        pyproject = Path("pyproject.toml")
        config = tomllib.loads(pyproject.read_text())
        extras = config["project"]["optional-dependencies"]
        assert "data-prep" in extras
        assert any("datasketch" in dep for dep in extras["data-prep"])
        assert any("langdetect" in dep for dep in extras["data-prep"])

    def test_synth_extra_exists(self):
        pyproject = Path("pyproject.toml")
        config = tomllib.loads(pyproject.read_text())
        extras = config["project"]["optional-dependencies"]
        assert "synth" in extras
        assert any("openai" in dep for dep in extras["synth"])

    def test_data_all_includes_both(self):
        pyproject = Path("pyproject.toml")
        config = tomllib.loads(pyproject.read_text())
        extras = config["project"]["optional-dependencies"]
        assert "data-all" in extras
        data_all = extras["data-all"]
        assert any("data-prep" in dep for dep in data_all)
        assert any("synth" in dep for dep in data_all)

    def test_all_includes_data(self):
        pyproject = Path("pyproject.toml")
        config = tomllib.loads(pyproject.read_text())
        extras = config["project"]["optional-dependencies"]
        all_deps = extras["all"]
        assert any("data-prep" in dep or "data-all" in dep for dep in all_deps)
