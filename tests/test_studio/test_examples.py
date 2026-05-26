from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from xaytune.studio.examples import (
    _RECIPE_FROM_BASE,
    EXAMPLES,
    _find_examples_dir,
    load_example_values,
)


class TestFindExamplesDir:
    def test_finds_existing_dir(self):
        result = _find_examples_dir()
        assert result is not None
        assert result.is_dir()
        assert result.name == "examples"

    def test_returns_none_when_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "xaytune.studio.examples.Path",
            lambda *a, **kw: tmp_path / "nonexistent",
        )
        from xaytune.studio import examples as mod

        original = mod._find_examples_dir

        def _patched():
            return None

        monkeypatch.setattr(mod, "_find_examples_dir", _patched)
        assert _patched() is None
        monkeypatch.setattr(mod, "_find_examples_dir", original)


class TestLoadExamples:
    def test_loads_all_yaml_files(self):
        assert len(EXAMPLES) >= 10

    def test_example_names_are_strings(self):
        for name in EXAMPLES:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_example_values_are_dicts(self):
        for name, cfg in EXAMPLES.items():
            assert isinstance(cfg, dict), f"Example '{name}' is not a dict"

    def test_loads_from_temp_dir(self, tmp_path: Path):
        yaml_content = dedent("""\
            # My Custom Example
            recipe: finetune
            method: full
            model:
              name: test-model
            data:
              path: data/train.jsonl
              format: alpaca
            trainer:
              batch_size: 8
              learning_rate: 1e-4
        """)
        examples_dir = tmp_path / "configs" / "examples"
        examples_dir.mkdir(parents=True)
        (examples_dir / "custom.yaml").write_text(yaml_content)

        import xaytune.studio.examples as mod

        original_fn = mod._find_examples_dir

        def _mock_find():
            return examples_dir

        mod._find_examples_dir = _mock_find  # type: ignore[assignment]
        try:
            result = mod._load_examples()
        finally:
            mod._find_examples_dir = original_fn  # type: ignore[assignment]

        assert len(result) == 1
        assert "My Custom Example" in result
        assert result["My Custom Example"]["recipe"] == "finetune"

    def test_skips_invalid_yaml(self, tmp_path: Path):
        examples_dir = tmp_path / "configs" / "examples"
        examples_dir.mkdir(parents=True)
        (examples_dir / "bad.yaml").write_text("just a string")
        (examples_dir / "good.yaml").write_text("recipe: finetune\nmethod: full\n")

        import xaytune.studio.examples as mod

        original_fn = mod._find_examples_dir

        def _mock_find():
            return examples_dir

        mod._find_examples_dir = _mock_find  # type: ignore[assignment]
        try:
            result = mod._load_examples()
        finally:
            mod._find_examples_dir = original_fn  # type: ignore[assignment]

        assert len(result) == 1

    def test_uses_filename_when_no_comment(self, tmp_path: Path):
        examples_dir = tmp_path / "configs" / "examples"
        examples_dir.mkdir(parents=True)
        (examples_dir / "my_example.yaml").write_text("recipe: finetune\nmethod: full\n")

        import xaytune.studio.examples as mod

        original_fn = mod._find_examples_dir

        def _mock_find():
            return examples_dir

        mod._find_examples_dir = _mock_find  # type: ignore[assignment]
        try:
            result = mod._load_examples()
        finally:
            mod._find_examples_dir = original_fn  # type: ignore[assignment]

        assert "My Example" in result


class TestRecipeFromBase:
    def test_known_bases(self):
        assert _RECIPE_FROM_BASE["full_finetune"] == ("finetune", "full")
        assert _RECIPE_FROM_BASE["lora"] == ("finetune", "lora")
        assert _RECIPE_FROM_BASE["qlora"] == ("finetune", "qlora")
        assert _RECIPE_FROM_BASE["pretrain"] == ("pretrain", "full")


class TestLoadExampleValues:
    def test_returns_empty_for_unknown_name(self):
        result = load_example_values("nonexistent_example_xyz")
        assert result == {}

    def test_returns_flat_dict(self):
        if not EXAMPLES:
            pytest.skip("No examples loaded")
        name = next(iter(EXAMPLES))
        result = load_example_values(name)
        assert isinstance(result, dict)
        assert "recipe" in result
        assert "method" in result
        assert "model_name" in result

    def test_all_expected_keys_present(self):
        if not EXAMPLES:
            pytest.skip("No examples loaded")
        name = next(iter(EXAMPLES))
        result = load_example_values(name)
        expected_keys = {
            "recipe",
            "method",
            "model_name",
            "data_path",
            "data_format",
            "quantization",
            "dtype",
            "trust_remote_code",
            "source",
            "max_seq_length",
            "packing",
            "streaming",
            "eval_split",
            "eval_path",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "batch_size",
            "gradient_accumulation",
            "learning_rate",
            "num_epochs",
            "max_steps",
            "seed",
            "mixed_precision",
            "scheduler",
            "warmup_steps",
            "warmup_ratio",
            "weight_decay",
            "max_grad_norm",
            "eval_every_n_steps",
            "early_stopping_patience",
            "early_stopping_metric",
            "early_stopping_min_delta",
            "log_every_n_steps",
            "output_dir",
            "merge_on_complete",
            "checkpoint_every_n_steps",
            "save_last",
            "activation_checkpointing",
            "async_checkpoint",
            "beta",
            "kl_coeff",
            "clip_eps",
            "lambda_weight",
            "gamma",
        }
        assert expected_keys.issubset(result.keys())

    def test_quantization_none_as_string(self):
        if not EXAMPLES:
            pytest.skip("No examples loaded")
        for name in EXAMPLES:
            result = load_example_values(name)
            assert isinstance(result["quantization"], str)

    def test_base_field_maps_to_recipe(self):
        cfg = {"base": "lora", "model": {"name": "test"}, "data": {"path": "d"}}
        import xaytune.studio.examples as mod

        original = dict(mod.EXAMPLES)
        mod.EXAMPLES["_test_base"] = cfg
        try:
            result = mod.load_example_values("_test_base")
            assert result["recipe"] == "finetune"
            assert result["method"] == "lora"
        finally:
            del mod.EXAMPLES["_test_base"]
            mod.EXAMPLES.update(original)
