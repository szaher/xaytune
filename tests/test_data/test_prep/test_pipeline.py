import json
import tempfile
from pathlib import Path

import yaml

from xaytune.data.prep.pipeline import pipeline


def _write_jsonl(path, samples):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


class TestPipelineBasic:
    def test_filter_then_dedup(self):
        samples = [
            {"text": "hi"},
            {"text": "hello world this is a real sentence"},
            {"text": "hello world this is a real sentence"},
            {"text": "another good long sentence here"},
        ]
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "data.jsonl"
            dst = Path(d) / "out.jsonl"
            _write_jsonl(src, samples)
            result = pipeline(
                input=str(src),
                output=str(dst),
                steps=[
                    {"filter": {"min_chars": 10}},
                    {"deduplicate": {"method": "exact"}},
                ],
            )
            assert len(result.dataset) == 2
            assert dst.exists()

    def test_report_has_all_steps(self):
        samples = [
            {"text": "hi"},
            {"text": "hello world long enough"},
            {"text": "hello world long enough"},
        ]
        result = pipeline(
            input=samples,
            steps=[
                {"filter": {"min_chars": 10}},
                {"deduplicate": {"method": "exact"}},
            ],
        )
        assert len(result.report.steps) == 2
        assert result.report.input_rows == 3
        assert result.report.output_rows == 1


class TestPipelineConvert:
    def test_convert_then_filter(self):
        samples = [
            {"instruction": "Q1", "input": "", "output": "short"},
            {"instruction": "Q2", "input": "", "output": "This is a much longer answer that passes the filter"},
        ]
        result = pipeline(
            input=samples,
            steps=[
                {"convert": {"source_format": "alpaca", "target_format": "sharegpt"}},
                {"filter": {"min_chars": 20}},
            ],
        )
        assert len(result.dataset) == 1
        assert "conversations" in result.dataset[0]


class TestPipelineFromYAML:
    def test_yaml_config(self):
        samples = [
            {"text": "short"},
            {"text": "this is long enough to pass the length filter"},
            {"text": "this is long enough to pass the length filter"},
        ]
        config = {
            "steps": [
                {"filter": {"min_chars": 10}},
                {"deduplicate": {"method": "exact"}},
            ],
        }
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "data.jsonl"
            cfg = Path(d) / "prep.yaml"
            dst = Path(d) / "out.jsonl"
            _write_jsonl(src, samples)
            config["input"] = str(src)
            config["output"] = str(dst)
            cfg.write_text(yaml.dump(config))
            result = pipeline(config=str(cfg))
            assert len(result.dataset) == 1
            assert dst.exists()


class TestPipelineEmpty:
    def test_no_steps(self):
        samples = [{"text": "hello"}]
        result = pipeline(input=samples, steps=[])
        assert len(result.dataset) == 1

    def test_empty_input(self):
        result = pipeline(input=[], steps=[{"filter": {"min_chars": 10}}])
        assert len(result.dataset) == 0
