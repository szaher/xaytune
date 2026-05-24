from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from xaytune.studio.data_preview import (
    _extract_text,
    compute_tokenization_stats,
    preview_dataset,
)


def _write_jsonl(path: Path, samples: list[dict]) -> None:
    with path.open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


class TestPreviewDataset:
    def test_loads_samples(self, tmp_path):
        p = tmp_path / "data.jsonl"
        data = [{"text": f"sample {i}"} for i in range(10)]
        _write_jsonl(p, data)

        result = preview_dataset(str(p), num_samples=3)
        assert len(result) == 3
        assert result[0]["text"] == "sample 0"

    def test_returns_empty_for_missing_file(self):
        result = preview_dataset("/nonexistent/path.jsonl")
        assert result == []

    def test_returns_empty_for_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        result = preview_dataset(str(p))
        assert result == []

    def test_handles_invalid_json_lines(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"text": "good"}\nnot json\n{"text": "also good"}\n')
        result = preview_dataset(str(p))
        assert len(result) == 2

    def test_respects_num_samples(self, tmp_path):
        p = tmp_path / "data.jsonl"
        data = [{"text": f"s{i}"} for i in range(20)]
        _write_jsonl(p, data)
        result = preview_dataset(str(p), num_samples=5)
        assert len(result) == 5


class TestExtractText:
    def test_text_field(self):
        assert _extract_text({"text": "hello"}, "completion") == "hello"

    def test_prompt_chosen(self):
        text = _extract_text({"prompt": "Q?", "chosen": "A"}, "preference")
        assert "Q?" in text
        assert "A" in text

    def test_instruction_output(self):
        text = _extract_text({"instruction": "Do X", "input": "Y", "output": "Z"}, "alpaca")
        assert "Do X" in text
        assert "Y" in text
        assert "Z" in text

    def test_empty_sample(self):
        assert _extract_text({}, "alpaca") == ""


class TestComputeTokenizationStats:
    def test_computes_stats(self, tmp_path):
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text: list(range(len(text.split())))

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        p = tmp_path / "data.jsonl"
        data = [{"text": " ".join(["word"] * (i + 1))} for i in range(10)]
        _write_jsonl(p, data)

        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            stats = compute_tokenization_stats(str(p), tokenizer_name="test-tok")
            assert stats["count"] == 10
            assert stats["min"] >= 0
            assert stats["max"] >= stats["min"]
            assert "avg" in stats
            assert "p50" in stats
            assert "p90" in stats
            assert "p99" in stats
            assert len(stats["histogram"]) > 0

    def test_returns_empty_for_bad_tokenizer(self, tmp_path):
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [{"text": "hello"}])
        stats = compute_tokenization_stats(str(p), tokenizer_name="nonexistent/model")
        assert stats == {}

    def test_returns_empty_for_missing_file(self):
        stats = compute_tokenization_stats("/nonexistent.jsonl", tokenizer_name="t")
        assert stats == {}
