import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trainlib.data import load_dataset, register_format
from trainlib.data.registry import format_registry


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

    def test_chat_format_uses_tokenizer_template(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<s>user: hi</s>"

        with tempfile.TemporaryDirectory() as tmpdir:
            data = [{"messages": [{"role": "user", "content": "hi"}]}]
            path = Path(tmpdir) / "data.jsonl"
            self._write_jsonl(data, path)
            ds = load_dataset(str(path), format="chat", tokenizer=tokenizer)
            assert ds[0]["text"] == "<s>user: hi</s>"
            tokenizer.apply_chat_template.assert_called_once()

    def test_sharegpt_format_uses_tokenizer_template(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<s>user: hello</s>"

        with tempfile.TemporaryDirectory() as tmpdir:
            data = [{"conversations": [{"from": "human", "value": "hello"}]}]
            path = Path(tmpdir) / "data.jsonl"
            self._write_jsonl(data, path)
            ds = load_dataset(str(path), format="sharegpt", tokenizer=tokenizer)
            assert ds[0]["text"] == "<s>user: hello</s>"
            tokenizer.apply_chat_template.assert_called_once()

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

    def test_no_tokenizer_uses_default_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = [{"messages": [{"role": "user", "content": "hi"}]}]
            path = Path(tmpdir) / "data.jsonl"
            self._write_jsonl(data, path)
            ds = load_dataset(str(path), format="chat")
            assert "User" in ds[0]["text"]


class TestLoadDatasetHuggingFace:
    @patch("datasets.load_dataset")
    def test_hf_source_loads_from_hub(self, mock_hf_load):
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter([
            {"text": "hello"},
            {"text": "world"},
        ]))
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
            "org/dataset", format="text", source="huggingface", eval_split=0.2,
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
            "org/dataset", format="text", source="huggingface", streaming=True,
        )

        mock_hf_load.assert_called_once_with(
            "org/dataset", split="train", streaming=True,
        )
        mock_ds.map.assert_called_once()
