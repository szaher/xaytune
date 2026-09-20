import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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
            train_ds, eval_ds = load_dataset(str(path), format="alpaca", eval_split=0.2)
            assert len(train_ds) + len(eval_ds) == 20
            assert len(eval_ds) == 4


class TestAutoChatTemplate:
    def _write_jsonl(self, data: list[dict], path: Path):
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    def test_chat_format_defers_template_to_tokenizer(self):
        """Chat data is loaded as structured turns, not pre-rendered text.

        The chat template is applied later, in ``tokenize_multiturn``,
        so that per-turn label masking (assistant turns trainable, everything
        else masked) is still possible. Loading used to render a ``text`` field
        eagerly, which threw that structure away.
        """
        tokenizer = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            data = [{"messages": [{"role": "user", "content": "hi"}]}]
            path = Path(tmpdir) / "data.jsonl"
            self._write_jsonl(data, path)
            ds = load_dataset(str(path), format="chat", tokenizer=tokenizer)

            assert ds[0]["turns"] == [{"role": "user", "content": "hi"}]
            assert ds[0]["_use_chat_template"] is True
            # Rendering is deferred, so the template is not applied at load time.
            tokenizer.apply_chat_template.assert_not_called()

    def test_sharegpt_format_defers_template_to_tokenizer(self):
        tokenizer = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            data = [{"conversations": [{"from": "human", "value": "hello"}]}]
            path = Path(tmpdir) / "data.jsonl"
            self._write_jsonl(data, path)
            ds = load_dataset(str(path), format="sharegpt", tokenizer=tokenizer)

            # ShareGPT "human"/"gpt" roles are normalised to "user"/"assistant".
            assert ds[0]["turns"] == [{"role": "user", "content": "hello"}]
            assert ds[0]["_use_chat_template"] is True
            tokenizer.apply_chat_template.assert_not_called()

    def test_flagged_dataset_applies_template_at_tokenization(self):
        """The deferred flag is what makes the tokenizer use the template."""
        from xaytune.data.tokenizer import tokenize_multiturn

        tokenizer = MagicMock()
        tokenizer.model_max_length = 512
        tokenizer.apply_chat_template.return_value = "<s>user: hi</s>"
        tokenizer.side_effect = lambda text, **kw: {
            "input_ids": list(range(1, len(text.split()) + 1)),
            "attention_mask": [1] * len(text.split()),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            data = [{"messages": [{"role": "user", "content": "hi"}]}]
            path = Path(tmpdir) / "data.jsonl"
            self._write_jsonl(data, path)
            ds = load_dataset(str(path), format="chat", tokenizer=tokenizer)

            tokenize_multiturn(list(ds), tokenizer)
            assert tokenizer.apply_chat_template.called

    def test_alpaca_format_ignores_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "should not be used"

        with tempfile.TemporaryDirectory() as tmpdir:
            data = [{"instruction": "Say hi", "input": "", "output": "Hello!"}]
            path = Path(tmpdir) / "data.jsonl"
            self._write_jsonl(data, path)
            ds = load_dataset(str(path), format="alpaca", tokenizer=tokenizer)
            assert "Instruction" in ds[0]["text"]
            tokenizer.apply_chat_template.assert_not_called()

    def test_no_tokenizer_omits_chat_template_flag(self):
        """Without a tokenizer there is no template to defer to."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = [{"messages": [{"role": "user", "content": "hi"}]}]
            path = Path(tmpdir) / "data.jsonl"
            self._write_jsonl(data, path)
            ds = load_dataset(str(path), format="chat")

            assert ds[0]["turns"] == [{"role": "user", "content": "hi"}]
            assert "_use_chat_template" not in ds[0]


class TestLoadDatasetHuggingFace:
    @patch("datasets.load_dataset")
    def test_hf_source_loads_from_hub(self, mock_hf_load):
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(
            return_value=iter(
                [
                    {"text": "hello"},
                    {"text": "world"},
                ]
            )
        )
        mock_ds.__len__ = MagicMock(return_value=2)
        mock_hf_load.return_value = mock_ds

        result = load_dataset("org/dataset", format="text", source="huggingface")

        mock_hf_load.assert_called_once_with("org/dataset", split="train")
        assert len(result) == 2

    @patch("datasets.load_dataset")
    def test_hf_source_with_eval_split(self, mock_hf_load):
        train_ds = MagicMock()
        train_ds.__iter__ = MagicMock(return_value=iter([{"text": f"t{i}"} for i in range(8)]))
        train_ds.__len__ = MagicMock(return_value=8)

        eval_ds = MagicMock()
        eval_ds.__iter__ = MagicMock(return_value=iter([{"text": f"e{i}"} for i in range(2)]))
        eval_ds.__len__ = MagicMock(return_value=2)

        mock_raw = MagicMock()
        mock_raw.train_test_split.return_value = {"train": train_ds, "test": eval_ds}
        mock_hf_load.return_value = mock_raw

        train, val = load_dataset(
            "org/dataset",
            format="text",
            source="huggingface",
            eval_split=0.2,
        )

        mock_raw.train_test_split.assert_called_once_with(test_size=0.2)
        assert len(train) == 8
        assert len(val) == 2

    @patch("datasets.load_dataset")
    def test_hf_streaming_returns_iterable(self, mock_hf_load):
        mock_ds = MagicMock()
        mock_ds.map.return_value = mock_ds
        mock_hf_load.return_value = mock_ds

        load_dataset(
            "org/dataset",
            format="text",
            source="huggingface",
            streaming=True,
        )

        mock_hf_load.assert_called_once_with(
            "org/dataset",
            split="train",
            streaming=True,
        )
        mock_ds.map.assert_called_once()
