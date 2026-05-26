# xaytune Models & Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement model loading (HuggingFace + custom), PEFT wrapping (LoRA/QLoRA), and the data pipeline (format registry, built-in formats, packing, preference datasets).

**Architecture:** Layer 1 building blocks. Models and data are independent modules that recipes will compose. Both use the Registry pattern from Plan 1. Tests use small/mock models to avoid GPU requirements.

**Tech Stack:** PyTorch, HuggingFace Transformers, PEFT, bitsandbytes, HF datasets, pytest

---

## Plan Sequence

This is **Plan 2 of 6** — depends on Plan 1 (Foundation) being complete.

---

### Task 1: Model Registry & load_model Interface

**Files:**
- Create: `xaytune/models/registry.py`
- Create: `xaytune/models/loader.py`
- Modify: `xaytune/models/__init__.py`
- Create: `tests/test_models/__init__.py`
- Create: `tests/test_models/test_loader.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_models/__init__.py` (empty).

Create `tests/test_models/test_loader.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from xaytune.models import load_model, register_model
from xaytune.models.registry import model_registry


class TestModelRegistry:
    def test_register_custom_model(self):
        @register_model("test-model")
        class TestModel:
            pass

        assert model_registry.has("test-model")
        assert model_registry.get("test-model") is TestModel

    def test_list_registered_models(self):
        registered = model_registry.list()
        assert isinstance(registered, list)


class TestLoadModel:
    @patch("xaytune.models.loader.AutoModelForCausalLM")
    @patch("xaytune.models.loader.AutoTokenizer")
    def test_load_from_hub(self, mock_tokenizer_cls, mock_model_cls):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        result = load_model("some-org/some-model")

        mock_model_cls.from_pretrained.assert_called_once()
        mock_tokenizer_cls.from_pretrained.assert_called_once()
        assert result.model is mock_model
        assert result.tokenizer is mock_tokenizer

    @patch("xaytune.models.loader.AutoModelForCausalLM")
    @patch("xaytune.models.loader.AutoTokenizer")
    def test_load_with_dtype(self, mock_tokenizer_cls, mock_model_cls):
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()

        load_model("some-model", dtype="float16")

        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert "torch_dtype" in call_kwargs

    @patch("xaytune.models.loader.AutoModelForCausalLM")
    @patch("xaytune.models.loader.AutoTokenizer")
    def test_load_with_trust_remote_code(self, mock_tokenizer_cls, mock_model_cls):
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()

        load_model("some-model", trust_remote_code=True)

        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert call_kwargs.get("trust_remote_code") is True

    def test_load_model_result_has_config(self):
        """ModelResult stores the config used to load."""
        from xaytune.models.loader import ModelResult
        result = ModelResult(model=MagicMock(), tokenizer=MagicMock(), name="test")
        assert result.name == "test"

    @patch("xaytune.models.loader.AutoModelForCausalLM")
    @patch("xaytune.models.loader.AutoTokenizer")
    def test_load_with_quantization_4bit(self, mock_tokenizer_cls, mock_model_cls):
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()

        load_model("some-model", quantization="4bit")

        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert "quantization_config" in call_kwargs

    @patch("xaytune.models.loader.AutoModelForCausalLM")
    @patch("xaytune.models.loader.AutoTokenizer")
    def test_load_with_quantization_8bit(self, mock_tokenizer_cls, mock_model_cls):
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()

        load_model("some-model", quantization="8bit")

        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert "quantization_config" in call_kwargs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models/test_loader.py -v`

Expected: FAIL — ImportError

- [ ] **Step 3: Implement ModelResult dataclass and loader**

Create `xaytune/models/loader.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class ModelResult:
    model: Any
    tokenizer: Any
    name: str
    quantization: str | None = None
    peft_applied: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


_DTYPE_MAP = {
    "auto": "auto",
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _get_quantization_config(quantization: str) -> BitsAndBytesConfig:
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported quantization: {quantization}. Use '4bit' or '8bit'.")


def load_model(
    name_or_path: str,
    *,
    quantization: str | None = None,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    device_map: str = "auto",
) -> ModelResult:
    torch_dtype = _DTYPE_MAP.get(dtype, "auto")

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
    }

    if quantization:
        model_kwargs["quantization_config"] = _get_quantization_config(quantization)

    model = AutoModelForCausalLM.from_pretrained(name_or_path, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return ModelResult(
        model=model,
        tokenizer=tokenizer,
        name=name_or_path,
        quantization=quantization,
    )
```

- [ ] **Step 4: Implement model registry**

Create `xaytune/models/registry.py`:

```python
from xaytune.utils.registry import Registry

model_registry = Registry("model")
```

- [ ] **Step 5: Wire up models __init__.py**

Update `xaytune/models/__init__.py`:

```python
from xaytune.models.loader import ModelResult, load_model
from xaytune.models.registry import model_registry

register_model = model_registry.register

__all__ = ["ModelResult", "load_model", "model_registry", "register_model"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_models/test_loader.py -v`

Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add xaytune/models/ tests/test_models/
git commit -m "feat: add model loading with quantization and custom model registry"
```

---

### Task 2: PEFT Wrapping (LoRA/QLoRA)

**Files:**
- Create: `xaytune/models/peft.py`
- Create: `tests/test_models/test_peft.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_models/test_peft.py`:

```python
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from xaytune.models.peft import apply_lora, get_target_modules
from xaytune.models.loader import ModelResult


class TestGetTargetModules:
    def test_auto_returns_default(self):
        mock_model = MagicMock()
        mock_model.config.model_type = "llama"
        modules = get_target_modules("auto", mock_model)
        assert isinstance(modules, list)
        assert len(modules) > 0

    def test_explicit_modules(self):
        mock_model = MagicMock()
        modules = get_target_modules(["q_proj", "v_proj"], mock_model)
        assert modules == ["q_proj", "v_proj"]

    def test_unknown_model_type_returns_common(self):
        mock_model = MagicMock()
        mock_model.config.model_type = "totally_unknown_model_xyz"
        modules = get_target_modules("auto", mock_model)
        assert isinstance(modules, list)
        assert len(modules) > 0


class TestApplyLora:
    @patch("xaytune.models.peft.get_peft_model")
    @patch("xaytune.models.peft.LoraConfig")
    def test_apply_lora_basic(self, mock_lora_config_cls, mock_get_peft):
        mock_model = MagicMock()
        mock_model.config.model_type = "llama"
        mock_peft_model = MagicMock()
        mock_get_peft.return_value = mock_peft_model

        model_result = ModelResult(
            model=mock_model, tokenizer=MagicMock(), name="test"
        )

        result = apply_lora(model_result, rank=16, alpha=32, dropout=0.05)

        mock_lora_config_cls.assert_called_once()
        mock_get_peft.assert_called_once()
        assert result.model is mock_peft_model
        assert result.peft_applied is True

    @patch("xaytune.models.peft.get_peft_model")
    @patch("xaytune.models.peft.LoraConfig")
    def test_apply_lora_custom_target_modules(self, mock_lora_config_cls, mock_get_peft):
        mock_model = MagicMock()
        mock_model.config.model_type = "llama"
        mock_get_peft.return_value = MagicMock()

        model_result = ModelResult(
            model=mock_model, tokenizer=MagicMock(), name="test"
        )

        apply_lora(model_result, rank=8, alpha=16, target_modules=["q_proj", "k_proj"])

        call_kwargs = mock_lora_config_cls.call_args[1]
        assert call_kwargs["target_modules"] == ["q_proj", "k_proj"]

    @patch("xaytune.models.peft.get_peft_model")
    @patch("xaytune.models.peft.LoraConfig")
    def test_apply_lora_sets_rank_and_alpha(self, mock_lora_config_cls, mock_get_peft):
        mock_model = MagicMock()
        mock_model.config.model_type = "llama"
        mock_get_peft.return_value = MagicMock()

        model_result = ModelResult(
            model=mock_model, tokenizer=MagicMock(), name="test"
        )

        apply_lora(model_result, rank=64, alpha=128, dropout=0.1)

        call_kwargs = mock_lora_config_cls.call_args[1]
        assert call_kwargs["r"] == 64
        assert call_kwargs["lora_alpha"] == 128
        assert call_kwargs["lora_dropout"] == 0.1

    @patch("xaytune.models.peft.get_peft_model")
    @patch("xaytune.models.peft.LoraConfig")
    def test_apply_lora_preserves_tokenizer(self, mock_lora_config_cls, mock_get_peft):
        mock_model = MagicMock()
        mock_model.config.model_type = "llama"
        mock_tokenizer = MagicMock()
        mock_get_peft.return_value = MagicMock()

        model_result = ModelResult(
            model=mock_model, tokenizer=mock_tokenizer, name="test"
        )

        result = apply_lora(model_result, rank=16, alpha=32)
        assert result.tokenizer is mock_tokenizer
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models/test_peft.py -v`

Expected: FAIL — ImportError

- [ ] **Step 3: Implement PEFT wrapping**

Create `xaytune/models/peft.py`:

```python
from __future__ import annotations

from typing import Union

from peft import LoraConfig, get_peft_model, TaskType

from xaytune.models.loader import ModelResult

_AUTO_TARGET_MODULES: dict[str, list[str]] = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
}

_DEFAULT_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def get_target_modules(
    target_modules: Union[str, list[str]], model: object
) -> list[str]:
    if isinstance(target_modules, list):
        return target_modules

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type and model_type in _AUTO_TARGET_MODULES:
        return _AUTO_TARGET_MODULES[model_type]

    return _DEFAULT_MODULES


def apply_lora(
    model_result: ModelResult,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Union[str, list[str]] = "auto",
) -> ModelResult:
    resolved_modules = get_target_modules(target_modules, model_result.model)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=resolved_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    peft_model = get_peft_model(model_result.model, lora_config)

    return ModelResult(
        model=peft_model,
        tokenizer=model_result.tokenizer,
        name=model_result.name,
        quantization=model_result.quantization,
        peft_applied=True,
        metadata={**model_result.metadata, "lora_rank": rank, "lora_alpha": alpha},
    )
```

- [ ] **Step 4: Export from models __init__**

Update `xaytune/models/__init__.py` to add:

```python
from xaytune.models.peft import apply_lora, get_target_modules
```

And add `"apply_lora"` and `"get_target_modules"` to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_models/test_peft.py -v`

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add xaytune/models/ tests/test_models/
git commit -m "feat: add LoRA/QLoRA PEFT wrapping with auto target module detection"
```

---

### Task 3: Data Format Registry & load_dataset Interface

**Files:**
- Create: `xaytune/data/registry.py`
- Create: `xaytune/data/loader.py`
- Modify: `xaytune/data/__init__.py`
- Create: `tests/test_data/__init__.py`
- Create: `tests/test_data/test_loader.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data/__init__.py` (empty).

Create `tests/test_data/test_loader.py`:

```python
import json
import tempfile
from pathlib import Path

import pytest

from xaytune.data import load_dataset, register_format
from xaytune.data.registry import format_registry


class TestFormatRegistry:
    def test_register_custom_format(self):
        @register_format("test-custom-fmt")
        def parse(sample):
            return {"text": sample["content"]}

        assert format_registry.has("test-custom-fmt")

    def test_list_formats(self):
        formats = format_registry.list()
        assert isinstance(formats, list)


class TestLoadDataset:
    def _write_jsonl(self, data: list[dict], path: Path):
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    def test_load_jsonl_with_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = [
                {"instruction": "Say hi", "input": "", "output": "Hello!"},
                {"instruction": "Count", "input": "1,2", "output": "3"},
            ]
            path = Path(tmpdir) / "data.jsonl"
            self._write_jsonl(data, path)

            ds = load_dataset(str(path), format="alpaca")
            assert len(ds) == 2
            assert "text" in ds[0] or "instruction" in ds[0]

    def test_load_with_custom_format(self):
        @register_format("my-test-fmt")
        def parse(sample):
            return {"text": f"Q: {sample['q']}\nA: {sample['a']}"}

        with tempfile.TemporaryDirectory() as tmpdir:
            data = [{"q": "Hello?", "a": "Hi!"}]
            path = Path(tmpdir) / "data.jsonl"
            self._write_jsonl(data, path)

            ds = load_dataset(str(path), format="my-test-fmt")
            assert len(ds) == 1
            assert "Q: Hello?" in ds[0]["text"]

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent.jsonl", format="alpaca")

    def test_load_unknown_format_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            path.write_text('{"a": 1}\n')
            with pytest.raises(KeyError, match="not found"):
                load_dataset(str(path), format="nonexistent_format_xyz")

    def test_load_with_eval_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = [
                {"instruction": f"Task {i}", "input": "", "output": f"Result {i}"}
                for i in range(20)
            ]
            path = Path(tmpdir) / "data.jsonl"
            self._write_jsonl(data, path)

            train_ds, eval_ds = load_dataset(
                str(path), format="alpaca", eval_split=0.2
            )
            assert len(train_ds) + len(eval_ds) == 20
            assert len(eval_ds) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_loader.py -v`

Expected: FAIL — ImportError

- [ ] **Step 3: Implement data registry**

Create `xaytune/data/registry.py`:

```python
from xaytune.utils.registry import Registry

format_registry = Registry("format")
```

- [ ] **Step 4: Implement data loader**

Create `xaytune/data/loader.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Union

from xaytune.data.registry import format_registry


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _split_dataset(
    data: list[dict], eval_split: float
) -> tuple[list[dict], list[dict]]:
    split_idx = len(data) - int(len(data) * eval_split)
    return data[:split_idx], data[split_idx:]


def load_dataset(
    path: str,
    *,
    format: str,
    eval_split: float = 0.0,
    **kwargs: Any,
) -> Union[list[dict], tuple[list[dict], list[dict]]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    format_fn = format_registry.get(format)
    raw_data = _load_jsonl(path)
    processed = [format_fn(sample) for sample in raw_data]

    if eval_split > 0:
        return _split_dataset(processed, eval_split)

    return processed
```

- [ ] **Step 5: Wire up data __init__.py**

Update `xaytune/data/__init__.py`:

```python
from xaytune.data.loader import load_dataset
from xaytune.data.registry import format_registry

register_format = format_registry.register

__all__ = ["load_dataset", "format_registry", "register_format"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_data/test_loader.py -v`

Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add xaytune/data/ tests/test_data/
git commit -m "feat: add data loading with format registry and eval split support"
```

---

### Task 4: Built-in Data Formats

**Files:**
- Create: `xaytune/data/formats.py`
- Create: `tests/test_data/test_formats.py`

Implements the built-in format parsers: alpaca, sharegpt, chat (OpenAI), and text.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data/test_formats.py`:

```python
import pytest

from xaytune.data.formats import format_alpaca, format_sharegpt, format_chat, format_text
from xaytune.data.registry import format_registry


class TestAlpacaFormat:
    def test_with_input(self):
        sample = {"instruction": "Translate", "input": "Hello", "output": "Hola"}
        result = format_alpaca(sample)
        assert "Translate" in result["text"]
        assert "Hello" in result["text"]
        assert "Hola" in result["text"]

    def test_without_input(self):
        sample = {"instruction": "Say hi", "input": "", "output": "Hello!"}
        result = format_alpaca(sample)
        assert "Say hi" in result["text"]
        assert "Hello!" in result["text"]

    def test_registered(self):
        assert format_registry.has("alpaca")


class TestShareGPTFormat:
    def test_single_turn(self):
        sample = {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello!"},
            ]
        }
        result = format_sharegpt(sample)
        assert "Hi" in result["text"]
        assert "Hello!" in result["text"]

    def test_multi_turn(self):
        sample = {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello!"},
                {"from": "human", "value": "How are you?"},
                {"from": "gpt", "value": "Fine, thanks!"},
            ]
        }
        result = format_sharegpt(sample)
        assert "How are you?" in result["text"]
        assert "Fine, thanks!" in result["text"]

    def test_registered(self):
        assert format_registry.has("sharegpt")


class TestChatFormat:
    def test_openai_format(self):
        sample = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        result = format_chat(sample)
        assert "You are helpful." in result["text"]
        assert "Hi" in result["text"]
        assert "Hello!" in result["text"]

    def test_registered(self):
        assert format_registry.has("chat")


class TestTextFormat:
    def test_text_field(self):
        sample = {"text": "Hello world"}
        result = format_text(sample)
        assert result["text"] == "Hello world"

    def test_content_field(self):
        sample = {"content": "Hello world"}
        result = format_text(sample)
        assert result["text"] == "Hello world"

    def test_registered(self):
        assert format_registry.has("text")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_formats.py -v`

Expected: FAIL — ImportError

- [ ] **Step 3: Implement built-in formats**

Create `xaytune/data/formats.py`:

```python
from __future__ import annotations

from typing import Any

from xaytune.data.registry import format_registry


@format_registry.register("alpaca")
def format_alpaca(sample: dict[str, Any]) -> dict[str, str]:
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")

    if input_text:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    return {"text": text}


@format_registry.register("sharegpt")
def format_sharegpt(sample: dict[str, Any]) -> dict[str, str]:
    conversations = sample.get("conversations", [])
    parts = []
    for turn in conversations:
        role = turn.get("from", turn.get("role", ""))
        value = turn.get("value", turn.get("content", ""))
        if role in ("human", "user"):
            parts.append(f"### User:\n{value}")
        elif role in ("gpt", "assistant"):
            parts.append(f"### Assistant:\n{value}")
        elif role == "system":
            parts.append(f"### System:\n{value}")
    return {"text": "\n\n".join(parts)}


@format_registry.register("chat")
def format_chat(sample: dict[str, Any]) -> dict[str, str]:
    messages = sample.get("messages", [])
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        parts.append(f"### {role.capitalize()}:\n{content}")
    return {"text": "\n\n".join(parts)}


@format_registry.register("text")
def format_text(sample: dict[str, Any]) -> dict[str, str]:
    text = sample.get("text", sample.get("content", ""))
    return {"text": text}
```

- [ ] **Step 4: Import formats in data __init__ to trigger registration**

Update `xaytune/data/__init__.py` to add:

```python
import xaytune.data.formats  # register built-in formats
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_data/test_formats.py -v`

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add xaytune/data/ tests/test_data/
git commit -m "feat: add built-in data formats (alpaca, sharegpt, chat, text)"
```

---

### Task 5: Preference Dataset

**Files:**
- Create: `xaytune/data/preferences.py`
- Create: `tests/test_data/test_preferences.py`

Preference datasets for alignment training: (prompt, chosen, rejected) triples.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data/test_preferences.py`:

```python
import json
import tempfile
from pathlib import Path

import pytest

from xaytune.data.preferences import load_preference_dataset
from xaytune.data.registry import format_registry


class TestPreferenceDataset:
    def _write_jsonl(self, data: list[dict], path: Path):
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    def test_load_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = [
                {
                    "prompt": "Say something nice",
                    "chosen": "You look great today!",
                    "rejected": "Whatever.",
                },
                {
                    "prompt": "Explain gravity",
                    "chosen": "Gravity is a fundamental force...",
                    "rejected": "Stuff falls down.",
                },
            ]
            path = Path(tmpdir) / "prefs.jsonl"
            self._write_jsonl(data, path)

            ds = load_preference_dataset(str(path))
            assert len(ds) == 2
            assert ds[0]["prompt"] == "Say something nice"
            assert ds[0]["chosen"] == "You look great today!"
            assert ds[0]["rejected"] == "Whatever."

    def test_load_with_eval_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = [
                {"prompt": f"P{i}", "chosen": f"C{i}", "rejected": f"R{i}"}
                for i in range(10)
            ]
            path = Path(tmpdir) / "prefs.jsonl"
            self._write_jsonl(data, path)

            train, val = load_preference_dataset(str(path), eval_split=0.2)
            assert len(train) + len(val) == 10
            assert len(val) == 2

    def test_validates_required_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = [{"prompt": "Hi", "chosen": "Hello"}]  # missing rejected
            path = Path(tmpdir) / "prefs.jsonl"
            self._write_jsonl(data, path)

            with pytest.raises(ValueError, match="rejected"):
                load_preference_dataset(str(path))

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_preference_dataset("nonexistent.jsonl")

    def test_preference_format_registered(self):
        assert format_registry.has("preference")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_preferences.py -v`

Expected: FAIL — ImportError

- [ ] **Step 3: Implement preference dataset**

Create `xaytune/data/preferences.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from xaytune.data.registry import format_registry

_REQUIRED_FIELDS = {"prompt", "chosen", "rejected"}


@format_registry.register("preference")
def format_preference(sample: dict[str, Any]) -> dict[str, str]:
    return {
        "prompt": sample["prompt"],
        "chosen": sample["chosen"],
        "rejected": sample["rejected"],
    }


def load_preference_dataset(
    path: str,
    *,
    eval_split: float = 0.0,
) -> Union[list[dict], tuple[list[dict], list[dict]]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Preference dataset not found: {path}")

    items = []
    with open(file_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            missing = _REQUIRED_FIELDS - set(sample.keys())
            if missing:
                raise ValueError(
                    f"Row {i}: missing required fields: {', '.join(sorted(missing))}. "
                    f"Preference data must have: prompt, chosen, rejected."
                )
            items.append(format_preference(sample))

    if eval_split > 0:
        split_idx = len(items) - int(len(items) * eval_split)
        return items[:split_idx], items[split_idx:]

    return items
```

- [ ] **Step 4: Import in data __init__**

Update `xaytune/data/__init__.py` to add:

```python
from xaytune.data.preferences import load_preference_dataset
```

And add `"load_preference_dataset"` to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_data/test_preferences.py -v`

Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add xaytune/data/ tests/test_data/
git commit -m "feat: add preference dataset loading for alignment training"
```

---

### Task 6: Sequence Packing

**Files:**
- Create: `xaytune/data/packing.py`
- Create: `tests/test_data/test_packing.py`

Packs multiple short sequences into a single sequence up to max_seq_length to maximize GPU utilization.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data/test_packing.py`:

```python
import pytest
from xaytune.data.packing import pack_sequences


class TestPackSequences:
    def test_basic_packing(self):
        sequences = [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [4, 5]},
            {"input_ids": [6, 7, 8, 9]},
        ]
        packed = pack_sequences(sequences, max_seq_length=6, pad_token_id=0)
        # [1,2,3,4,5,0] and [6,7,8,9,0,0]
        assert len(packed) == 2
        assert len(packed[0]["input_ids"]) == 6
        assert len(packed[1]["input_ids"]) == 6

    def test_sequences_exceeding_max_are_truncated(self):
        sequences = [{"input_ids": [1, 2, 3, 4, 5, 6, 7, 8]}]
        packed = pack_sequences(sequences, max_seq_length=4, pad_token_id=0)
        assert len(packed) == 1
        assert len(packed[0]["input_ids"]) == 4

    def test_empty_input(self):
        packed = pack_sequences([], max_seq_length=10, pad_token_id=0)
        assert packed == []

    def test_attention_mask_generated(self):
        sequences = [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [4, 5]},
        ]
        packed = pack_sequences(sequences, max_seq_length=8, pad_token_id=0)
        assert len(packed) == 1
        mask = packed[0]["attention_mask"]
        assert sum(mask) == 5  # 3 + 2 real tokens
        assert len(mask) == 8

    def test_single_long_sequence(self):
        sequences = [{"input_ids": list(range(100))}]
        packed = pack_sequences(sequences, max_seq_length=50, pad_token_id=0)
        assert len(packed) == 1
        assert len(packed[0]["input_ids"]) == 50

    def test_labels_mask_padding(self):
        sequences = [
            {"input_ids": [1, 2, 3]},
        ]
        packed = pack_sequences(sequences, max_seq_length=6, pad_token_id=0)
        labels = packed[0]["labels"]
        assert labels[:3] == [1, 2, 3]
        assert labels[3:] == [-100, -100, -100]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_packing.py -v`

Expected: FAIL — ImportError

- [ ] **Step 3: Implement sequence packing**

Create `xaytune/data/packing.py`:

```python
from __future__ import annotations

from typing import Any

IGNORE_INDEX = -100


def pack_sequences(
    sequences: list[dict[str, list[int]]],
    *,
    max_seq_length: int,
    pad_token_id: int,
) -> list[dict[str, list[int]]]:
    if not sequences:
        return []

    packed: list[dict[str, list[int]]] = []
    current_ids: list[int] = []
    current_labels: list[int] = []

    for seq in sequences:
        ids = seq["input_ids"][:max_seq_length]

        if len(current_ids) + len(ids) > max_seq_length:
            if current_ids:
                packed.append(_pad_and_finalize(
                    current_ids, current_labels, max_seq_length, pad_token_id
                ))
            current_ids = ids[:]
            current_labels = ids[:]
        else:
            current_ids.extend(ids)
            current_labels.extend(ids)

    if current_ids:
        packed.append(_pad_and_finalize(
            current_ids, current_labels, max_seq_length, pad_token_id
        ))

    return packed


def _pad_and_finalize(
    input_ids: list[int],
    labels: list[int],
    max_seq_length: int,
    pad_token_id: int,
) -> dict[str, list[int]]:
    seq_len = len(input_ids)
    pad_len = max_seq_length - seq_len

    attention_mask = [1] * seq_len + [0] * pad_len
    padded_ids = input_ids + [pad_token_id] * pad_len
    padded_labels = labels + [IGNORE_INDEX] * pad_len

    return {
        "input_ids": padded_ids,
        "attention_mask": attention_mask,
        "labels": padded_labels,
    }
```

- [ ] **Step 4: Export from data __init__**

Update `xaytune/data/__init__.py` to add:

```python
from xaytune.data.packing import pack_sequences
```

And add `"pack_sequences"` to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_data/test_packing.py -v`

Expected: All 6 tests PASS

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -v`

Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add xaytune/data/ tests/test_data/
git commit -m "feat: add sequence packing for efficient GPU utilization"
```

---

## Self-Review

**Spec coverage:** This plan covers spec sections 2 (Models & PEFT) and 3 (Data Pipeline) completely. Model loading, quantization, LoRA/QLoRA wrapping, custom model registration, data loading, all built-in formats (alpaca, sharegpt, chat, text), preference datasets, sequence packing, and custom format registration via decorators.

**Placeholder scan:** No TBDs or TODOs. All code is complete.

**Type consistency:** `ModelResult`, `load_model`, `register_model`, `apply_lora`, `get_target_modules`, `load_dataset`, `register_format`, `load_preference_dataset`, `pack_sequences` — names are consistent across all tasks and match the spec.
