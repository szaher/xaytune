import json
import tempfile
from pathlib import Path

from xaytune.data.prep import (
    PrepResult,
    convert,
    deduplicate,
    filter_dataset,
    pipeline,
)


def _write_jsonl(path, samples):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


class TestEndToEnd:
    def test_convert_filter_dedup_pipeline(self):
        """Full pipeline: convert CSV-like → alpaca, filter short, dedup."""
        samples = [
            {
                "question": "What is AI?",
                "answer": "Artificial Intelligence is the simulation of human intelligence.",
            },
            {
                "question": "What is AI?",
                "answer": "Artificial Intelligence is the simulation of human intelligence.",
            },
            {"question": "Hi", "answer": "Hey"},
            {
                "question": "Explain ML",
                "answer": "Machine learning is a subset of AI that learns from data.",
            },
        ]
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "data.jsonl"
            dst = Path(d) / "ready.jsonl"
            _write_jsonl(src, samples)

            result = pipeline(
                input=str(src),
                output=str(dst),
                steps=[
                    {
                        "convert": {
                            "source_format": "alpaca",
                            "target_format": "alpaca",
                            "field_map": {"question": "instruction", "answer": "output"},
                        }
                    },
                    {"filter": {"min_chars": 20}},
                    {"deduplicate": {"method": "exact"}},
                ],
            )

            assert isinstance(result, PrepResult)
            assert len(result.dataset) == 2
            assert len(result.report.steps) == 3
            assert result.report.input_rows == 4
            assert result.report.output_rows == 2
            assert dst.exists()

    def test_standalone_operations_compose(self):
        """Use individual operations manually and chain results."""
        samples = [
            {"instruction": "Q1", "input": "", "output": "Short"},
            {
                "instruction": "Q2",
                "input": "",
                "output": "A much longer and more detailed answer here",
            },
            {
                "instruction": "Q3",
                "input": "",
                "output": "A much longer and more detailed answer here",
            },
            {
                "instruction": "Q4",
                "input": "",
                "output": "Another substantial answer with good content",
            },
        ]

        filtered = filter_dataset(
            samples,
            filters=[{"type": "length", "min_chars": 20}],
            field="output",
        )
        assert len(filtered.dataset) == 3

        deduped = deduplicate(filtered.dataset, method="exact", field="output")
        assert len(deduped.dataset) == 2

        converted = convert(deduped.dataset, source_format="alpaca", target_format="sharegpt")
        assert len(converted.dataset) == 2
        assert all("conversations" in s for s in converted.dataset)

    def test_imports_from_package(self):
        """Verify all public API is importable from the package."""
        from xaytune.data.prep import (
            convert,
            deduplicate,
            filter_dataset,
            generate,
            pipeline,
            register_filter,
        )

        assert callable(deduplicate)
        assert callable(filter_dataset)
        assert callable(convert)
        assert callable(generate)
        assert callable(pipeline)
        assert callable(register_filter)
