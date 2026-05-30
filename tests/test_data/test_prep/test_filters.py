import json
import tempfile
from pathlib import Path

from xaytune.data.prep.filters import filter_dataset, filter_registry


def _write_jsonl(path, samples):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


class TestLengthFilter:
    def test_min_chars(self):
        samples = [{"text": "hi"}, {"text": "hello world this is long enough"}]
        result = filter_dataset(
            samples, filters=[{"type": "length", "min_chars": 10}], field="text"
        )
        assert len(result.dataset) == 1
        assert result.dataset[0]["text"].startswith("hello")

    def test_max_chars(self):
        samples = [{"text": "short"}, {"text": "x" * 200}]
        result = filter_dataset(
            samples, filters=[{"type": "length", "max_chars": 100}], field="text"
        )
        assert len(result.dataset) == 1
        assert result.dataset[0]["text"] == "short"

    def test_min_and_max(self):
        samples = [{"text": "ab"}, {"text": "hello world"}, {"text": "x" * 500}]
        result = filter_dataset(
            samples, filters=[{"type": "length", "min_chars": 5, "max_chars": 100}], field="text"
        )
        assert len(result.dataset) == 1


class TestRegexFilter:
    def test_drop_pattern(self):
        samples = [
            {"text": "check out https://example.com for details"},
            {"text": "no urls here"},
        ]
        result = filter_dataset(
            samples, filters=[{"type": "regex", "drop_pattern": r"https?://\S+"}], field="text"
        )
        assert len(result.dataset) == 1
        assert result.dataset[0]["text"] == "no urls here"

    def test_keep_pattern(self):
        samples = [
            {"text": "Python is great"},
            {"text": "Java is fine"},
        ]
        result = filter_dataset(
            samples, filters=[{"type": "regex", "keep_pattern": r"Python"}], field="text"
        )
        assert len(result.dataset) == 1


class TestDecontaminateFilter:
    def test_removes_overlapping(self):
        train = [
            {"text": "the quick brown fox jumps over the lazy dog"},
            {"text": "completely unique training example"},
        ]
        reference = [
            {"text": "the quick brown fox jumps over the lazy dog"},
        ]
        with tempfile.TemporaryDirectory() as d:
            ref_path = Path(d) / "test.jsonl"
            _write_jsonl(ref_path, reference)
            result = filter_dataset(
                train,
                filters=[{"type": "decontaminate", "reference": str(ref_path), "ngram_size": 5}],
                field="text",
            )
            assert len(result.dataset) == 1
            assert result.dataset[0]["text"] == "completely unique training example"


class TestMultipleFilters:
    def test_chain(self):
        samples = [
            {"text": "hi"},
            {"text": "hello world this is a good example"},
            {"text": "visit https://spam.com for more"},
        ]
        result = filter_dataset(
            samples,
            filters=[
                {"type": "length", "min_chars": 10},
                {"type": "regex", "drop_pattern": r"https?://\S+"},
            ],
            field="text",
        )
        assert len(result.dataset) == 1
        assert result.dataset[0]["text"] == "hello world this is a good example"

    def test_report_per_filter(self):
        samples = [{"text": "hi"}, {"text": "hello world"}, {"text": "x" * 500}]
        result = filter_dataset(
            samples,
            filters=[
                {"type": "length", "min_chars": 5, "max_chars": 100},
            ],
            field="text",
        )
        assert result.report.steps[0].name == "filter:length"
        assert result.report.steps[0].rows_removed == 2


class TestFromFile:
    def test_filter_from_file_path(self):
        samples = [{"text": "hi"}, {"text": "hello world long text"}]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            _write_jsonl(path, samples)
            result = filter_dataset(
                str(path), filters=[{"type": "length", "min_chars": 10}], field="text"
            )
            assert len(result.dataset) == 1


class TestCustomFilter:
    def test_register_and_use(self):
        @filter_registry.register("no-exclamation")
        def no_exclamation(sample, field):
            return "!" not in sample[field]

        samples = [{"text": "hello!"}, {"text": "hello"}]
        result = filter_dataset(samples, filters=[{"type": "no-exclamation"}], field="text")
        assert len(result.dataset) == 1
        assert result.dataset[0]["text"] == "hello"
