from pathlib import Path

import yaml

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

    def test_grpo_align_exists(self):
        assert (EXAMPLES_DIR / "grpo_align.yaml").is_file()

    def test_orpo_align_exists(self):
        assert (EXAMPLES_DIR / "orpo_align.yaml").is_file()

    def test_simpo_align_exists(self):
        assert (EXAMPLES_DIR / "simpo_align.yaml").is_file()

    def test_ppo_align_exists(self):
        assert (EXAMPLES_DIR / "ppo_align.yaml").is_file()

    def test_reinforce_align_exists(self):
        assert (EXAMPLES_DIR / "reinforce_align.yaml").is_file()

    def test_all_training_examples_are_valid_yaml(self):
        """Every training config must declare a model and a dataset.

        ``pipeline.yaml`` is a ``PipelineConfig``, not a ``TrainConfig`` — it
        declares stages instead, so it is validated separately below.
        """
        for f in EXAMPLES_DIR.glob("*.yaml"):
            data = yaml.safe_load(f.read_text())
            assert isinstance(data, dict), f"{f.name} is not a valid YAML mapping"
            if "stages" in data:
                continue
            assert "model" in data, f"{f.name} missing 'model' key"
            assert "data" in data, f"{f.name} missing 'data' key"

    def test_pipeline_example_is_valid_yaml(self):
        data = yaml.safe_load((EXAMPLES_DIR / "pipeline.yaml").read_text())
        assert isinstance(data, dict)
        assert "name" in data
        assert isinstance(data.get("stages"), list) and data["stages"]
        for stage in data["stages"]:
            assert "name" in stage, f"pipeline stage missing 'name': {stage}"

    def test_all_alignment_examples_have_method(self):
        alignment_configs = [
            "dpo_align.yaml",
            "grpo_align.yaml",
            "orpo_align.yaml",
            "simpo_align.yaml",
            "ppo_align.yaml",
            "reinforce_align.yaml",
        ]
        for name in alignment_configs:
            data = yaml.safe_load((EXAMPLES_DIR / name).read_text())
            assert data.get("recipe") == "align", f"{name} missing recipe: align"
            assert "method" in data, f"{name} missing 'method' key"

    def test_covers_all_alignment_methods(self):
        expected_methods = {"dpo", "grpo", "orpo", "simpo", "ppo", "reinforce"}
        found = set()
        for f in EXAMPLES_DIR.glob("*_align.yaml"):
            data = yaml.safe_load(f.read_text())
            if data.get("method"):
                found.add(data["method"])
        assert found == expected_methods

    def test_covers_all_finetune_methods(self):
        expected = {"lora_finetune.yaml", "qlora_finetune.yaml", "full_finetune.yaml"}
        found = {f.name for f in EXAMPLES_DIR.glob("*finetune.yaml")}
        assert found == expected

    def test_every_example_is_a_recognised_kind(self):
        """Guards against a stray or half-finished file landing in examples/.

        A hardcoded file count was previously asserted here; it went stale as
        soon as new examples were added, without catching anything a shipped
        example could actually get wrong.
        """
        for f in EXAMPLES_DIR.glob("*.yaml"):
            data = yaml.safe_load(f.read_text())
            is_pipeline = "stages" in data
            is_training = "model" in data and "data" in data
            assert is_pipeline or is_training, (
                f"{f.name} is neither a training config (model + data) "
                f"nor a pipeline config (stages)"
            )
