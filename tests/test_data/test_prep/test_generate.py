import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xaytune.data.prep.generate import generate


def _write_jsonl(path, samples):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


def _mock_openai_response(content: str):
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


class TestAugment:
    @patch("xaytune.data.prep.generate._call_llm_sync")
    def test_augment_from_seeds(self, mock_call):
        mock_call.return_value = json.dumps(
            {
                "instruction": "Explain gravity",
                "output": "Gravity is a force...",
            }
        )
        seeds = [{"instruction": "What is physics?", "output": "The study of matter and energy."}]
        result = generate(
            mode="augment",
            seed=seeds,
            n=2,
            format="alpaca",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        assert len(result.dataset) == 2
        assert mock_call.call_count == 2

    @patch("xaytune.data.prep.generate._call_llm_sync")
    def test_augment_from_file(self, mock_call):
        mock_call.return_value = json.dumps(
            {
                "instruction": "New question",
                "output": "New answer",
            }
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "seeds.jsonl"
            _write_jsonl(path, [{"instruction": "Q1", "output": "A1"}])
            result = generate(
                mode="augment",
                seed=str(path),
                n=1,
                format="alpaca",
                model="test",
                api_key="test-key",
            )
            assert len(result.dataset) == 1


class TestDistill:
    @patch("xaytune.data.prep.generate._call_llm_sync")
    def test_distill_from_topic(self, mock_call):
        mock_call.return_value = json.dumps(
            {
                "instruction": "What is K8s?",
                "output": "Kubernetes is a container orchestration platform.",
            }
        )
        result = generate(
            mode="distill",
            topic="Kubernetes",
            n=3,
            format="alpaca",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        assert len(result.dataset) == 3

    def test_distill_requires_topic(self):
        with pytest.raises(ValueError, match="topic"):
            generate(mode="distill", n=1, format="alpaca", model="test", api_key="k")


class TestEvolve:
    @patch("xaytune.data.prep.generate._call_llm_sync")
    def test_evolve_increases_complexity(self, mock_call):
        mock_call.return_value = json.dumps(
            {
                "instruction": "Compare and contrast gravity in Newtonian and Einsteinian physics",
                "output": "In Newtonian physics...",
            }
        )
        seeds = [{"instruction": "What is gravity?", "output": "A force of attraction."}]
        result = generate(
            mode="evolve",
            seed=seeds,
            rounds=2,
            format="alpaca",
            model="test",
            api_key="test-key",
        )
        assert len(result.dataset) == 1
        assert mock_call.call_count == 2


class TestPostFilter:
    @patch("xaytune.data.prep.generate._call_llm_sync")
    def test_post_filter_drops_short(self, mock_call):
        mock_call.side_effect = [
            json.dumps({"instruction": "Q", "output": "A"}),
            json.dumps(
                {"instruction": "Long question here", "output": "Long detailed answer here"}
            ),
        ]
        result = generate(
            mode="augment",
            seed=[{"instruction": "X", "output": "Y"}],
            n=2,
            format="alpaca",
            model="test",
            api_key="k",
            post_filter=[{"type": "length", "min_chars": 20}],
        )
        assert len(result.dataset) == 1


class TestAPIKeyResolution:
    def test_missing_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                generate(mode="distill", topic="test", n=1, format="alpaca", model="test")

    @patch("xaytune.data.prep.generate._call_llm_sync")
    def test_env_var_key(self, mock_call):
        mock_call.return_value = json.dumps({"instruction": "Q", "output": "A"})
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            result = generate(
                mode="distill",
                topic="test",
                n=1,
                format="alpaca",
                model="test",
            )
            assert len(result.dataset) == 1


class TestReport:
    @patch("xaytune.data.prep.generate._call_llm_sync")
    def test_report_counts(self, mock_call):
        mock_call.return_value = json.dumps({"instruction": "Q", "output": "A"})
        result = generate(
            mode="distill",
            topic="test",
            n=3,
            format="alpaca",
            model="test",
            api_key="k",
        )
        assert result.report.input_rows == 0
        assert result.report.output_rows == 3
