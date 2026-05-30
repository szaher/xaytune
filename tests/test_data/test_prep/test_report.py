import json
import tempfile
from pathlib import Path

from xaytune.data.prep.report import PrepReport, PrepResult, StepReport


class TestStepReport:
    def test_create(self):
        step = StepReport(
            name="deduplicate",
            input_rows=100,
            output_rows=85,
            details={"exact_dupes": 10, "near_dupes": 5},
        )
        assert step.name == "deduplicate"
        assert step.input_rows == 100
        assert step.output_rows == 85
        assert step.details["exact_dupes"] == 10

    def test_rows_removed(self):
        step = StepReport(name="filter", input_rows=100, output_rows=70, details={})
        assert step.rows_removed == 30


class TestPrepReport:
    def test_single_step(self):
        report = PrepReport(
            input_rows=100,
            output_rows=85,
            steps=[
                StepReport(name="deduplicate", input_rows=100, output_rows=85, details={}),
            ],
        )
        assert report.input_rows == 100
        assert report.output_rows == 85

    def test_summary_contains_counts(self):
        report = PrepReport(
            input_rows=100,
            output_rows=85,
            steps=[
                StepReport(name="deduplicate", input_rows=100, output_rows=85, details={}),
            ],
        )
        text = report.summary()
        assert "100" in text
        assert "85" in text

    def test_multi_step(self):
        report = PrepReport(
            input_rows=1000,
            output_rows=500,
            steps=[
                StepReport(name="filter", input_rows=1000, output_rows=700, details={}),
                StepReport(name="deduplicate", input_rows=700, output_rows=500, details={}),
            ],
        )
        assert len(report.steps) == 2
        assert report.output_rows == 500


class TestPrepResult:
    def test_create(self):
        data = [{"text": "hello"}, {"text": "world"}]
        report = PrepReport(input_rows=3, output_rows=2, steps=[])
        result = PrepResult(dataset=data, report=report)
        assert len(result.dataset) == 2

    def test_save_jsonl(self):
        data = [{"text": "hello"}, {"text": "world"}]
        report = PrepReport(input_rows=2, output_rows=2, steps=[])
        result = PrepResult(dataset=data, report=report)

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.jsonl"
            result.save(str(path))
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 2
            assert json.loads(lines[0]) == {"text": "hello"}

    def test_save_json(self):
        data = [{"text": "a"}]
        report = PrepReport(input_rows=1, output_rows=1, steps=[])
        result = PrepResult(dataset=data, report=report)

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.json"
            result.save(str(path), format="json")
            loaded = json.loads(path.read_text())
            assert loaded == [{"text": "a"}]
