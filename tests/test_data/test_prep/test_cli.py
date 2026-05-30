import json
import tempfile
from pathlib import Path

from xaytune.cli import main


def _write_jsonl(path, samples):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


class TestDataDeduplicateCLI:
    def test_deduplicate(self):
        samples = [{"text": "hello"}, {"text": "hello"}, {"text": "world"}]
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "data.jsonl"
            dst = Path(d) / "out.jsonl"
            _write_jsonl(src, samples)
            ret = main(["data", "deduplicate", str(src), "-o", str(dst), "--method", "exact"])
            assert ret == 0
            assert dst.exists()
            lines = dst.read_text().strip().split("\n")
            assert len(lines) == 2


class TestDataFilterCLI:
    def test_filter_min_chars(self):
        samples = [{"text": "hi"}, {"text": "hello world long text"}]
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "data.jsonl"
            dst = Path(d) / "out.jsonl"
            _write_jsonl(src, samples)
            ret = main(["data", "filter", str(src), "-o", str(dst), "--min-chars", "10"])
            assert ret == 0
            lines = dst.read_text().strip().split("\n")
            assert len(lines) == 1


class TestDataConvertCLI:
    def test_convert(self):
        samples = [{"instruction": "Hi", "input": "", "output": "Hello"}]
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "data.jsonl"
            dst = Path(d) / "out.jsonl"
            _write_jsonl(src, samples)
            ret = main(
                [
                    "data",
                    "convert",
                    str(src),
                    "-o",
                    str(dst),
                    "--from",
                    "alpaca",
                    "--to",
                    "sharegpt",
                ]
            )
            assert ret == 0
            lines = dst.read_text().strip().split("\n")
            loaded = json.loads(lines[0])
            assert "conversations" in loaded


class TestDataPipelineCLI:
    def test_pipeline_from_yaml(self):
        import yaml

        samples = [
            {"text": "hi"},
            {"text": "hello world this is long"},
            {"text": "hello world this is long"},
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
            ret = main(["data", "pipeline", str(cfg)])
            assert ret == 0
            assert dst.exists()


class TestDataNoSubcommand:
    def test_no_subcommand_prints_help(self):
        ret = main(["data"])
        assert ret == 1
