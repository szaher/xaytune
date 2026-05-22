from pathlib import Path

import pytest

from trainlib.config import load_config, validate_config, get_defaults_dir
from trainlib.config.schema import TrainConfig


class TestDefaults:
    def test_defaults_dir_exists(self):
        d = get_defaults_dir()
        assert d.is_dir()

    def test_lora_default_exists(self):
        d = get_defaults_dir()
        assert (d / "lora.yaml").exists()

    def test_qlora_default_exists(self):
        d = get_defaults_dir()
        assert (d / "qlora.yaml").exists()

    def test_full_finetune_default_exists(self):
        d = get_defaults_dir()
        assert (d / "full_finetune.yaml").exists()

    def test_pretrain_default_exists(self):
        d = get_defaults_dir()
        assert (d / "pretrain.yaml").exists()


class TestConfigPublicAPI:
    def test_load_config_importable(self):
        from trainlib.config import load_config
        assert callable(load_config)

    def test_validate_config_importable(self):
        from trainlib.config import validate_config
        assert callable(validate_config)

    def test_schema_classes_importable(self):
        from trainlib.config import TrainConfig, ModelConfig, DataConfig
        assert TrainConfig is not None
