from pathlib import Path

import pytest

from trainlib.config.parser import apply_overrides, load_config, merge_dicts

FIXTURES = Path(__file__).parent / "fixtures"


class TestMergeDicts:
    def test_shallow_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = merge_dicts(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge(self):
        base = {"model": {"name": "a", "quantization": "4bit"}}
        override = {"model": {"name": "b"}}
        result = merge_dicts(base, override)
        assert result == {"model": {"name": "b", "quantization": "4bit"}}

    def test_override_does_not_mutate_base(self):
        base = {"model": {"name": "a"}}
        override = {"model": {"name": "b"}}
        merge_dicts(base, override)
        assert base["model"]["name"] == "a"


class TestApplyOverrides:
    def test_dot_notation(self):
        data = {"model": {"name": "a"}, "trainer": {"batch_size": 4}}
        overrides = ["model.name=b", "trainer.batch_size=8"]
        result = apply_overrides(data, overrides)
        assert result["model"]["name"] == "b"
        assert result["trainer"]["batch_size"] == 8

    def test_nested_creation(self):
        data = {}
        overrides = ["model.name=my-model"]
        result = apply_overrides(data, overrides)
        assert result["model"]["name"] == "my-model"

    def test_boolean_parsing(self):
        data = {}
        overrides = ["model.trust_remote_code=true"]
        result = apply_overrides(data, overrides)
        assert result["model"]["trust_remote_code"] is True

    def test_numeric_parsing(self):
        data = {}
        overrides = ["trainer.learning_rate=1e-5", "trainer.batch_size=16"]
        result = apply_overrides(data, overrides)
        assert result["trainer"]["learning_rate"] == 1e-5
        assert result["trainer"]["batch_size"] == 16


class TestLoadConfig:
    def test_load_full_config(self):
        cfg = load_config(str(FIXTURES / "full_config.yaml"))
        assert cfg.recipe == "finetune"
        assert cfg.method == "lora"
        assert cfg.model.name == "my-model"
        assert cfg.model.quantization == "4bit"
        assert cfg.data.path == "data.jsonl"
        assert cfg.lora.rank == 32
        assert cfg.trainer.batch_size == 8

    def test_load_with_inheritance(self):
        cfg = load_config(str(FIXTURES / "child_config.yaml"))
        assert cfg.model.name == "meta-llama/Llama-3.1-8B"
        assert cfg.model.quantization == "4bit"  # inherited from base
        assert cfg.lora.rank == 16  # inherited from base
        assert cfg.trainer.num_epochs == 5  # overridden in child
        assert cfg.trainer.learning_rate == 2e-4  # inherited from base

    def test_load_with_cli_overrides(self):
        cfg = load_config(
            str(FIXTURES / "full_config.yaml"),
            overrides=["model.name=different-model", "trainer.num_epochs=10"],
        )
        assert cfg.model.name == "different-model"
        assert cfg.trainer.num_epochs == 10
        assert cfg.lora.rank == 32  # unchanged

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_resolved_config_is_serializable(self):
        cfg = load_config(str(FIXTURES / "full_config.yaml"))
        d = cfg.model_dump()
        assert isinstance(d, dict)
        assert d["model"]["name"] == "my-model"
