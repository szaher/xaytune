import json
import tempfile
from pathlib import Path
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
