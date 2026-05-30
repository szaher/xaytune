import tempfile
import json
from pathlib import Path

from xaytune.data.prep.dedup import deduplicate


def _write_jsonl(path, samples):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


class TestExactDedup:
    def test_removes_exact_duplicates(self):
        samples = [
            {"text": "hello world"},
            {"text": "hello world"},
            {"text": "goodbye world"},
        ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            _write_jsonl(path, samples)
            result = deduplicate(str(path), method="exact", field="text")
            assert len(result.dataset) == 2
            texts = {s["text"] for s in result.dataset}
            assert texts == {"hello world", "goodbye world"}

    def test_no_duplicates(self):
        samples = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            _write_jsonl(path, samples)
            result = deduplicate(str(path), method="exact", field="text")
            assert len(result.dataset) == 3

    def test_all_duplicates(self):
        samples = [{"text": "same"}] * 5
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            _write_jsonl(path, samples)
            result = deduplicate(str(path), method="exact", field="text")
            assert len(result.dataset) == 1

    def test_report_counts(self):
        samples = [{"text": "a"}, {"text": "a"}, {"text": "b"}]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            _write_jsonl(path, samples)
            result = deduplicate(str(path), method="exact", field="text")
            assert result.report.input_rows == 3
            assert result.report.output_rows == 2
            assert result.report.steps[0].details["exact_dupes"] == 1


class TestFieldAutoDetect:
    def test_detects_text_field(self):
        samples = [{"text": "hello", "id": 1}, {"text": "hello", "id": 2}]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            _write_jsonl(path, samples)
            result = deduplicate(str(path), method="exact")
            assert len(result.dataset) == 1

    def test_detects_output_field(self):
        samples = [{"instruction": "hi", "output": "hello"}, {"instruction": "yo", "output": "hello"}]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            _write_jsonl(path, samples)
            result = deduplicate(str(path), method="exact")
            assert len(result.dataset) == 1

    def test_falls_back_to_first_string_field(self):
        samples = [{"count": 1, "content": "same"}, {"count": 2, "content": "same"}]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            _write_jsonl(path, samples)
            result = deduplicate(str(path), method="exact")
            assert len(result.dataset) == 1


class TestAcceptsList:
    def test_deduplicate_from_list(self):
        samples = [{"text": "a"}, {"text": "a"}, {"text": "b"}]
        result = deduplicate(samples, method="exact", field="text")
        assert len(result.dataset) == 2


class TestMinHashDedup:
    def test_removes_near_duplicates(self):
        samples = [
            {"text": "The quick brown fox jumps over the lazy dog"},
            {"text": "The quick brown fox jumps over the lazy cat"},
            {"text": "Completely different sentence about programming"},
        ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            _write_jsonl(path, samples)
            result = deduplicate(str(path), method="minhash", field="text", threshold=0.5)
            assert len(result.dataset) == 2

    def test_high_threshold_keeps_similar(self):
        samples = [
            {"text": "The quick brown fox jumps over the lazy dog"},
            {"text": "The quick brown fox jumps over the lazy cat"},
        ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            _write_jsonl(path, samples)
            result = deduplicate(str(path), method="minhash", field="text", threshold=0.99)
            assert len(result.dataset) == 2


class TestBothMethod:
    def test_removes_exact_and_near(self):
        samples = [
            {"text": "hello world"},
            {"text": "hello world"},
            {"text": "hello worl"},
        ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            _write_jsonl(path, samples)
            result = deduplicate(str(path), method="both", field="text", threshold=0.5)
            assert len(result.dataset) == 1
            assert result.report.steps[0].details["exact_dupes"] >= 1
