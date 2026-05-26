from __future__ import annotations

from xaytune.studio.codegen import generate_code


class TestGenerateCode:
    def test_finetune_minimal(self):
        code = generate_code(
            recipe="finetune",
            method="full",
            model_name="meta-llama/Llama-3-8B",
            data_path="data/train.jsonl",
            data_format="alpaca",
        )
        assert "xaytune.finetune(" in code
        assert "model='meta-llama/Llama-3-8B'" in code
        assert "dataset='data/train.jsonl'" in code
        assert "method=" not in code
        assert "format=" not in code
        assert "num_epochs=" not in code

    def test_finetune_non_default_method(self):
        code = generate_code(
            recipe="finetune",
            method="lora",
            model_name="test-model",
            data_path="data.jsonl",
            data_format="alpaca",
        )
        assert "method='lora'" in code

    def test_finetune_non_default_epochs(self):
        code = generate_code(
            recipe="finetune",
            method="full",
            model_name="test-model",
            data_path="data.jsonl",
            data_format="alpaca",
            num_epochs=1,
        )
        assert "num_epochs=1" in code

    def test_align_defaults(self):
        code = generate_code(
            recipe="align",
            method="dpo",
            model_name="test-model",
            data_path="prefs.jsonl",
            data_format="preference",
            num_epochs=1,
            learning_rate=5e-6,
        )
        assert "xaytune.align(" in code
        assert "model='test-model'" in code
        assert "method=" not in code
        assert "format=" not in code
        assert "num_epochs=" not in code
        assert "learning_rate=" not in code

    def test_align_grpo(self):
        code = generate_code(
            recipe="align",
            method="grpo",
            model_name="test-model",
            data_path="prompts.jsonl",
            data_format="preference",
            num_epochs=1,
            learning_rate=5e-6,
        )
        assert "method='grpo'" in code

    def test_pretrain(self):
        code = generate_code(
            recipe="pretrain",
            method="full",
            model_name="test-model",
            data_path="corpus/",
            data_format="text",
        )
        assert "xaytune.pretrain(" in code
        assert "format=" not in code

    def test_quantization_included(self):
        code = generate_code(
            recipe="finetune",
            method="qlora",
            model_name="test-model",
            data_path="data.jsonl",
            data_format="alpaca",
            quantization="4bit",
        )
        assert "quantization='4bit'" in code

    def test_quantization_none_excluded(self):
        code = generate_code(
            recipe="finetune",
            method="full",
            model_name="test-model",
            data_path="data.jsonl",
            data_format="alpaca",
            quantization=None,
        )
        assert "quantization" not in code

    def test_quantization_none_string_excluded(self):
        code = generate_code(
            recipe="finetune",
            method="full",
            model_name="test-model",
            data_path="data.jsonl",
            data_format="alpaca",
            quantization="None",
        )
        assert "quantization" not in code

    def test_non_default_trainer_fields(self):
        code = generate_code(
            recipe="finetune",
            method="full",
            model_name="test-model",
            data_path="data.jsonl",
            data_format="alpaca",
            max_steps=500,
            seed=123,
        )
        assert "max_steps=500" in code
        assert "seed=123" in code

    def test_default_trainer_fields_excluded(self):
        code = generate_code(
            recipe="finetune",
            method="full",
            model_name="test-model",
            data_path="data.jsonl",
            data_format="alpaca",
            seed=42,
            max_steps=-1,
        )
        assert "seed=" not in code
        assert "max_steps=" not in code

    def test_lora_params_for_lora_method(self):
        code = generate_code(
            recipe="finetune",
            method="lora",
            model_name="test-model",
            data_path="data.jsonl",
            data_format="alpaca",
            lora_rank=32,
        )
        assert "lora_rank=32" in code

    def test_lora_params_excluded_for_full(self):
        code = generate_code(
            recipe="finetune",
            method="full",
            model_name="test-model",
            data_path="data.jsonl",
            data_format="alpaca",
            lora_rank=32,
        )
        assert "lora_rank" not in code

    def test_output_starts_with_import(self):
        code = generate_code(
            recipe="finetune",
            method="full",
            model_name="test",
            data_path="data",
            data_format="alpaca",
        )
        assert code.startswith("import xaytune\n")

    def test_output_ends_with_print(self):
        code = generate_code(
            recipe="finetune",
            method="full",
            model_name="test",
            data_path="data",
            data_format="alpaca",
        )
        assert "print(" in code
        assert "Final loss" in code

    def test_eval_path_included(self):
        code = generate_code(
            recipe="finetune",
            method="full",
            model_name="test",
            data_path="data",
            data_format="alpaca",
            eval_path="eval.jsonl",
        )
        assert "eval_path='eval.jsonl'" in code

    def test_eval_path_empty_excluded(self):
        code = generate_code(
            recipe="finetune",
            method="full",
            model_name="test",
            data_path="data",
            data_format="alpaca",
            eval_path="",
        )
        assert "eval_path" not in code

    def test_non_default_data_fields(self):
        code = generate_code(
            recipe="finetune",
            method="full",
            model_name="test",
            data_path="data",
            data_format="alpaca",
            max_seq_length=4096,
            packing=False,
        )
        assert "max_seq_length=4096" in code
        assert "packing=False" in code
