from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestBenchmarkEvaluate:
    def test_calls_lm_eval_simple_evaluate(self):
        # Clean slate
        sys.modules.pop("trainlib.eval.benchmarks", None)

        mock_lm_eval = MagicMock()
        mock_lm_eval.simple_evaluate.return_value = {
            "results": {
                "mmlu": {"acc,none": 0.65, "acc_stderr,none": 0.01},
            }
        }
        sys.modules["lm_eval"] = mock_lm_eval

        try:
            from trainlib.eval.benchmarks import benchmark_evaluate

            results = benchmark_evaluate(
                model="test-model",
                benchmarks=["mmlu"],
            )

            mock_lm_eval.simple_evaluate.assert_called_once()
            assert "mmlu" in results
        finally:
            sys.modules.pop("lm_eval", None)
            sys.modules.pop("trainlib.eval.benchmarks", None)

    def test_multiple_benchmarks(self):
        # Clean slate
        sys.modules.pop("trainlib.eval.benchmarks", None)

        mock_lm_eval = MagicMock()
        mock_lm_eval.simple_evaluate.return_value = {
            "results": {
                "mmlu": {"acc,none": 0.65},
                "hellaswag": {"acc_norm,none": 0.72},
            }
        }
        sys.modules["lm_eval"] = mock_lm_eval

        try:
            from trainlib.eval.benchmarks import benchmark_evaluate

            results = benchmark_evaluate(
                model="test-model",
                benchmarks=["mmlu", "hellaswag"],
            )

            assert "mmlu" in results
            assert "hellaswag" in results
        finally:
            sys.modules.pop("lm_eval", None)
            sys.modules.pop("trainlib.eval.benchmarks", None)

    def test_raises_when_lm_eval_not_installed(self):
        sys.modules.pop("lm_eval", None)
        sys.modules.pop("trainlib.eval.benchmarks", None)
        sys.modules.pop("trainlib.eval", None)

        with patch.dict(sys.modules, {"lm_eval": None}):
            with pytest.raises(ImportError, match="lm-eval"):
                from trainlib.eval import benchmarks

                benchmarks.benchmark_evaluate(model="x", benchmarks=["mmlu"])

    def test_returns_dict_of_benchmark_results(self):
        # Clean slate
        sys.modules.pop("trainlib.eval.benchmarks", None)

        mock_lm_eval = MagicMock()
        mock_lm_eval.simple_evaluate.return_value = {
            "results": {
                "gsm8k": {"exact_match,none": 0.45},
            }
        }
        sys.modules["lm_eval"] = mock_lm_eval

        try:
            from trainlib.eval.benchmarks import benchmark_evaluate

            results = benchmark_evaluate(
                model="test-model",
                benchmarks=["gsm8k"],
            )

            assert isinstance(results, dict)
            assert isinstance(results["gsm8k"], dict)
        finally:
            sys.modules.pop("lm_eval", None)
            sys.modules.pop("trainlib.eval.benchmarks", None)

    def test_num_fewshot_passed_through(self):
        # Clean slate
        sys.modules.pop("trainlib.eval.benchmarks", None)

        mock_lm_eval = MagicMock()
        mock_lm_eval.simple_evaluate.return_value = {"results": {}}
        sys.modules["lm_eval"] = mock_lm_eval

        try:
            from trainlib.eval.benchmarks import benchmark_evaluate

            benchmark_evaluate(
                model="test-model",
                benchmarks=["mmlu"],
                num_fewshot=5,
            )

            call_kwargs = mock_lm_eval.simple_evaluate.call_args[1]
            assert call_kwargs["num_fewshot"] == 5
        finally:
            sys.modules.pop("lm_eval", None)
            sys.modules.pop("trainlib.eval.benchmarks", None)


class TestBenchmarkImport:
    def test_importable_from_eval(self):
        # Clean slate
        sys.modules.pop("trainlib.eval.benchmarks", None)
        sys.modules.pop("trainlib.eval", None)

        mock_lm_eval = MagicMock()
        sys.modules["lm_eval"] = mock_lm_eval
        try:
            from trainlib.eval import benchmark_evaluate

            assert callable(benchmark_evaluate)
        finally:
            sys.modules.pop("lm_eval", None)
            sys.modules.pop("trainlib.eval.benchmarks", None)
            sys.modules.pop("trainlib.eval", None)
