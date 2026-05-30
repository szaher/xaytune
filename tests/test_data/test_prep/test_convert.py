import csv
import json
import tempfile
from pathlib import Path

from xaytune.data.prep.convert import convert


def _write_jsonl(path, samples):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


class TestAlpacaToShareGPT:
    def test_basic(self):
        samples = [{"instruction": "Say hello", "input": "", "output": "Hello!"}]
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src.jsonl"
            dst = Path(d) / "dst.jsonl"
            _write_jsonl(src, samples)
            result = convert(str(src), output=str(dst), source_format="alpaca", target_format="sharegpt")
            assert len(result.dataset) == 1
            item = result.dataset[0]
            assert "conversations" in item
            convs = item["conversations"]
            assert convs[0]["from"] == "human"
            assert convs[0]["value"] == "Say hello"
            assert convs[1]["from"] == "gpt"
            assert convs[1]["value"] == "Hello!"

    def test_with_input(self):
        samples = [{"instruction": "Translate", "input": "Hello", "output": "Hola"}]
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src.jsonl"
            dst = Path(d) / "dst.jsonl"
            _write_jsonl(src, samples)
            result = convert(str(src), output=str(dst), source_format="alpaca", target_format="sharegpt")
            user_msg = result.dataset[0]["conversations"][0]["value"]
            assert "Translate" in user_msg
            assert "Hello" in user_msg


class TestShareGPTToAlpaca:
    def test_basic(self):
        samples = [{"conversations": [
            {"from": "human", "value": "What is 2+2?"},
            {"from": "gpt", "value": "4"},
        ]}]
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src.jsonl"
            dst = Path(d) / "dst.jsonl"
            _write_jsonl(src, samples)
            result = convert(str(src), output=str(dst), source_format="sharegpt", target_format="alpaca")
            item = result.dataset[0]
            assert item["instruction"] == "What is 2+2?"
            assert item["output"] == "4"


class TestChatToAlpaca:
    def test_basic(self):
        samples = [{"messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]}]
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src.jsonl"
            dst = Path(d) / "dst.jsonl"
            _write_jsonl(src, samples)
            result = convert(str(src), output=str(dst), source_format="chat", target_format="alpaca")
            item = result.dataset[0]
            assert item["instruction"] == "Hi"
            assert item["output"] == "Hello!"


class TestCSVConversion:
    def test_csv_to_alpaca(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "data.csv"
            dst = Path(d) / "out.jsonl"
            with open(src, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["question", "answer"])
                writer.writeheader()
                writer.writerow({"question": "What is AI?", "answer": "Artificial Intelligence"})
            result = convert(
                str(src), output=str(dst),
                source_format="csv", target_format="alpaca",
                field_map={"question": "instruction", "answer": "output"},
            )
            assert len(result.dataset) == 1
            assert result.dataset[0]["instruction"] == "What is AI?"


class TestOutputFile:
    def test_saves_output(self):
        samples = [{"instruction": "Hi", "input": "", "output": "Hello"}]
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src.jsonl"
            dst = Path(d) / "dst.jsonl"
            _write_jsonl(src, samples)
            convert(str(src), output=str(dst), source_format="alpaca", target_format="sharegpt")
            assert dst.exists()
            lines = dst.read_text().strip().split("\n")
            assert len(lines) == 1


class TestFromList:
    def test_convert_from_list(self):
        samples = [{"instruction": "Hi", "input": "", "output": "Hello"}]
        result = convert(samples, source_format="alpaca", target_format="sharegpt")
        assert len(result.dataset) == 1
        assert "conversations" in result.dataset[0]
