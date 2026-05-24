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
            data = [{"prompt": f"P{i}", "chosen": f"C{i}", "rejected": f"R{i}"} for i in range(10)]
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
