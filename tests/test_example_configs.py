from pathlib import Path
import yaml
import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "configs" / "examples"


class TestExampleConfigs:
    def test_examples_dir_exists(self):
        assert EXAMPLES_DIR.is_dir()

    def test_lora_finetune_exists(self):
        assert (EXAMPLES_DIR / "lora_finetune.yaml").is_file()

    def test_qlora_finetune_exists(self):
        assert (EXAMPLES_DIR / "qlora_finetune.yaml").is_file()

    def test_full_finetune_exists(self):
        assert (EXAMPLES_DIR / "full_finetune.yaml").is_file()

    def test_pretrain_exists(self):
        assert (EXAMPLES_DIR / "pretrain.yaml").is_file()

    def test_dpo_align_exists(self):
        assert (EXAMPLES_DIR / "dpo_align.yaml").is_file()

    def test_all_examples_are_valid_yaml(self):
        for f in EXAMPLES_DIR.glob("*.yaml"):
            data = yaml.safe_load(f.read_text())
            assert isinstance(data, dict), f"{f.name} is not a valid YAML mapping"
            assert "model" in data, f"{f.name} missing 'model' key"
            assert "data" in data, f"{f.name} missing 'data' key"
