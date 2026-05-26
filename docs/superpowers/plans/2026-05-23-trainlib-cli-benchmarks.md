# CLI Completion & Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the remaining CLI subcommands (`eval`, `export merge/gguf/push`), GGUF export function, lm-eval-harness benchmark integration, and a `compare` command — completing xaytune's user-facing interface.

**Architecture:** The CLI (`xaytune/cli.py`) already has `train` and `list` commands. This plan adds `eval` and `export` subcommands that call existing building blocks (`xaytune.eval.evaluate`, `xaytune.export.merge/push_to_hub`) plus two new modules: `xaytune/export/gguf.py` for GGUF conversion and `xaytune/eval/benchmarks.py` for lm-eval-harness integration. The `compare` command calls `evaluate` on two models and prints a side-by-side table.

**Tech Stack:** hatchling (build), pytest, lm-eval-harness (optional dep via `eval` extra)

---

## File Structure

```
xaytune/
├── cli.py                    # (modify) — add eval, export, compare subparsers
├── eval/
│   ├── __init__.py           # (modify) — re-export benchmark_evaluate
│   └── benchmarks.py         # (create) — lm-eval-harness wrapper
├── export/
│   ├── __init__.py           # (modify) — re-export to_gguf
│   └── gguf.py               # (create) — GGUF conversion wrapper
└── tests/
    ├── test_eval/
    │   └── test_benchmarks.py # (create) — benchmark integration tests
    ├── test_export/
    │   └── test_gguf.py       # (create) — GGUF export tests
    └── test_cli.py            # (modify) — add eval/export/compare CLI tests
```

---

### Task 1: GGUF Export

**Files:**
- Create: `xaytune/export/gguf.py`
- Modify: `xaytune/export/__init__.py`
- Create: `tests/test_export/test_gguf.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_export/test_gguf.py`:

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xaytune.export.gguf import to_gguf


class TestToGguf:
    def test_calls_llama_cpp_convert(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output = tmp_path / "output.gguf"

        with patch("xaytune.export.gguf.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            to_gguf(str(model_dir), output=str(output))

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert "convert" in " ".join(str(a) for a in call_args[0][0])

    def test_default_quantization_is_q4_k_m(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output = tmp_path / "output.gguf"

        with patch("xaytune.export.gguf.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            to_gguf(str(model_dir), output=str(output))

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "Q4_K_M" in cmd

    def test_custom_quantization(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output = tmp_path / "output.gguf"

        with patch("xaytune.export.gguf.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            to_gguf(str(model_dir), output=str(output), quantization="Q8_0")

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "Q8_0" in cmd

    def test_raises_on_missing_model_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            to_gguf(str(tmp_path / "nonexistent"), output="out.gguf")

    def test_raises_on_conversion_failure(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with patch("xaytune.export.gguf.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="conversion failed")
            with pytest.raises(RuntimeError, match="GGUF conversion failed"):
                to_gguf(str(model_dir), output="out.gguf")

    def test_output_dir_created(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output = tmp_path / "nested" / "dir" / "output.gguf"

        with patch("xaytune.export.gguf.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            to_gguf(str(model_dir), output=str(output))

        assert output.parent.exists()


class TestGgufImport:
    def test_importable_from_export(self):
        from xaytune.export import to_gguf as fn
        assert callable(fn)
```

- [ ] **Step 2: Run tests to verify failures**

Run: `.venv/bin/python -m pytest tests/test_export/test_gguf.py -v`
Expected: FAIL — `xaytune.export.gguf` does not exist

- [ ] **Step 3: Implement to_gguf**

Create `xaytune/export/gguf.py`:

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def to_gguf(
    model_path: str,
    *,
    output: str,
    quantization: str = "Q4_K_M",
) -> None:
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "llama_cpp.convert",
        str(model_dir),
        "--outfile", str(output_path),
        "--outtype", quantization,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed: {result.stderr}")
```

- [ ] **Step 4: Update export __init__.py**

Add to `xaytune/export/__init__.py`:

```python
from xaytune.export.gguf import to_gguf
```

And add `"to_gguf"` to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_export/test_gguf.py -v`
Expected: All tests PASS

- [ ] **Step 6: Run full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: All tests PASS (344 + 7 new)

- [ ] **Step 7: Commit**

```bash
git add xaytune/export/gguf.py xaytune/export/__init__.py tests/test_export/test_gguf.py
git commit -m "feat: add GGUF export via llama.cpp conversion"
```

---

### Task 2: Benchmark Integration (lm-eval-harness)

**Files:**
- Create: `xaytune/eval/benchmarks.py`
- Modify: `xaytune/eval/__init__.py`
- Create: `tests/test_eval/test_benchmarks.py`

lm-eval-harness is an optional dependency (already in `[eval]` extra). This module wraps it with a simple `benchmark_evaluate(model, benchmarks)` function.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval/test_benchmarks.py`:

```python
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestBenchmarkEvaluate:
    def test_calls_lm_eval_simple_evaluate(self):
        mock_lm_eval = MagicMock()
        mock_lm_eval.simple_evaluate.return_value = {
            "results": {
                "mmlu": {"acc,none": 0.65, "acc_stderr,none": 0.01},
            }
        }
        sys.modules["lm_eval"] = mock_lm_eval

        try:
            from xaytune.eval.benchmarks import benchmark_evaluate

            results = benchmark_evaluate(
                model="test-model",
                benchmarks=["mmlu"],
            )

            mock_lm_eval.simple_evaluate.assert_called_once()
            assert "mmlu" in results
        finally:
            sys.modules.pop("lm_eval", None)
            sys.modules.pop("xaytune.eval.benchmarks", None)

    def test_multiple_benchmarks(self):
        mock_lm_eval = MagicMock()
        mock_lm_eval.simple_evaluate.return_value = {
            "results": {
                "mmlu": {"acc,none": 0.65},
                "hellaswag": {"acc_norm,none": 0.72},
            }
        }
        sys.modules["lm_eval"] = mock_lm_eval

        try:
            from xaytune.eval.benchmarks import benchmark_evaluate

            results = benchmark_evaluate(
                model="test-model",
                benchmarks=["mmlu", "hellaswag"],
            )

            assert "mmlu" in results
            assert "hellaswag" in results
        finally:
            sys.modules.pop("lm_eval", None)
            sys.modules.pop("xaytune.eval.benchmarks", None)

    def test_raises_when_lm_eval_not_installed(self):
        sys.modules.pop("lm_eval", None)
        sys.modules.pop("xaytune.eval.benchmarks", None)

        with patch.dict(sys.modules, {"lm_eval": None}):
            with pytest.raises(ImportError, match="lm-eval"):
                from xaytune.eval import benchmarks
                benchmarks.benchmark_evaluate(model="x", benchmarks=["mmlu"])

    def test_returns_dict_of_benchmark_results(self):
        mock_lm_eval = MagicMock()
        mock_lm_eval.simple_evaluate.return_value = {
            "results": {
                "gsm8k": {"exact_match,none": 0.45},
            }
        }
        sys.modules["lm_eval"] = mock_lm_eval

        try:
            from xaytune.eval.benchmarks import benchmark_evaluate

            results = benchmark_evaluate(
                model="test-model",
                benchmarks=["gsm8k"],
            )

            assert isinstance(results, dict)
            assert isinstance(results["gsm8k"], dict)
        finally:
            sys.modules.pop("lm_eval", None)
            sys.modules.pop("xaytune.eval.benchmarks", None)

    def test_num_fewshot_passed_through(self):
        mock_lm_eval = MagicMock()
        mock_lm_eval.simple_evaluate.return_value = {"results": {}}
        sys.modules["lm_eval"] = mock_lm_eval

        try:
            from xaytune.eval.benchmarks import benchmark_evaluate

            benchmark_evaluate(
                model="test-model",
                benchmarks=["mmlu"],
                num_fewshot=5,
            )

            call_kwargs = mock_lm_eval.simple_evaluate.call_args[1]
            assert call_kwargs["num_fewshot"] == 5
        finally:
            sys.modules.pop("lm_eval", None)
            sys.modules.pop("xaytune.eval.benchmarks", None)


class TestBenchmarkImport:
    def test_importable_from_eval(self):
        mock_lm_eval = MagicMock()
        sys.modules["lm_eval"] = mock_lm_eval
        try:
            sys.modules.pop("xaytune.eval.benchmarks", None)
            from xaytune.eval import benchmark_evaluate
            assert callable(benchmark_evaluate)
        finally:
            sys.modules.pop("lm_eval", None)
            sys.modules.pop("xaytune.eval.benchmarks", None)
```

- [ ] **Step 2: Run tests to verify failures**

Run: `.venv/bin/python -m pytest tests/test_eval/test_benchmarks.py -v`
Expected: FAIL — `xaytune.eval.benchmarks` does not exist

- [ ] **Step 3: Implement benchmarks.py**

Create `xaytune/eval/benchmarks.py`:

```python
from __future__ import annotations

from typing import Any

try:
    import lm_eval
except ImportError:
    lm_eval = None


def benchmark_evaluate(
    *,
    model: str,
    benchmarks: list[str],
    num_fewshot: int | None = None,
) -> dict[str, dict[str, Any]]:
    if lm_eval is None:
        raise ImportError(
            "lm-eval is required for benchmark evaluation. "
            "Install it with: pip install xaytune[eval]"
        )

    kwargs: dict[str, Any] = {
        "model": "hf",
        "model_args": f"pretrained={model}",
        "tasks": benchmarks,
    }
    if num_fewshot is not None:
        kwargs["num_fewshot"] = num_fewshot

    raw = lm_eval.simple_evaluate(**kwargs)

    return raw.get("results", {})
```

- [ ] **Step 4: Update eval __init__.py**

Add to `xaytune/eval/__init__.py`:

```python
from xaytune.eval.benchmarks import benchmark_evaluate
```

And add `"benchmark_evaluate"` to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval/test_benchmarks.py -v`
Expected: All tests PASS

- [ ] **Step 6: Run full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add xaytune/eval/benchmarks.py xaytune/eval/__init__.py tests/test_eval/test_benchmarks.py
git commit -m "feat: add lm-eval-harness benchmark integration"
```

---

### Task 3: eval CLI Subcommand

**Files:**
- Modify: `xaytune/cli.py`
- Modify: `tests/test_cli.py`

Add `xaytune eval --model X --benchmarks mmlu,gsm8k` and `xaytune eval --model X --metrics loss,perplexity --dataset eval.jsonl`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py` (add new class after existing test classes):

```python
class TestEvalCommand:
    def test_eval_parser_exists(self):
        from xaytune.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["eval", "--model", "test-model", "--benchmarks", "mmlu"])
        assert args.command == "eval"
        assert args.model == "test-model"
        assert args.benchmarks == "mmlu"

    def test_eval_with_metrics(self):
        from xaytune.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "eval", "--model", "test-model",
            "--metrics", "loss,perplexity",
            "--dataset", "data/eval.jsonl",
        ])
        assert args.metrics == "loss,perplexity"
        assert args.dataset == "data/eval.jsonl"

    def test_eval_benchmarks_calls_benchmark_evaluate(self):
        from unittest.mock import patch, MagicMock
        from xaytune.cli import main

        with patch("xaytune.cli.benchmark_evaluate") as mock_bench:
            mock_bench.return_value = {"mmlu": {"acc,none": 0.65}}
            result = main(["eval", "--model", "test-model", "--benchmarks", "mmlu,gsm8k"])

        assert result == 0
        mock_bench.assert_called_once_with(
            model="test-model",
            benchmarks=["mmlu", "gsm8k"],
            num_fewshot=None,
        )

    def test_eval_num_fewshot(self):
        from unittest.mock import patch
        from xaytune.cli import main

        with patch("xaytune.cli.benchmark_evaluate") as mock_bench:
            mock_bench.return_value = {}
            main(["eval", "--model", "m", "--benchmarks", "mmlu", "--num-fewshot", "5"])

        call_kwargs = mock_bench.call_args[1]
        assert call_kwargs["num_fewshot"] == 5

    def test_eval_requires_model(self):
        from xaytune.cli import main
        result = main(["eval"])
        assert result != 0

    def test_eval_prints_results(self, capsys):
        from unittest.mock import patch
        from xaytune.cli import main

        with patch("xaytune.cli.benchmark_evaluate") as mock_bench:
            mock_bench.return_value = {"mmlu": {"acc,none": 0.65}}
            main(["eval", "--model", "m", "--benchmarks", "mmlu"])

        captured = capsys.readouterr()
        assert "mmlu" in captured.out
```

- [ ] **Step 2: Run tests to verify failures**

Run: `.venv/bin/python -m pytest tests/test_cli.py::TestEvalCommand -v`
Expected: FAIL — eval subparser does not exist

- [ ] **Step 3: Add eval subcommand to cli.py**

In `xaytune/cli.py`, add to `_build_parser()` after the `list_parser` block:

```python
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_parser.add_argument("--model", required=True, help="Model path or HF Hub name")
    eval_parser.add_argument("--benchmarks", default=None, help="Comma-separated benchmarks (e.g., mmlu,gsm8k)")
    eval_parser.add_argument("--metrics", default=None, help="Comma-separated metrics (e.g., loss,perplexity)")
    eval_parser.add_argument("--dataset", default=None, help="Path to evaluation dataset")
    eval_parser.add_argument("--num-fewshot", type=int, default=None, help="Number of few-shot examples")
```

Add to `main()` after the `list` dispatch:

```python
    if args.command == "eval":
        return _handle_eval(args)
```

Add handler function:

```python
def _handle_eval(args: argparse.Namespace) -> int:
    if args.benchmarks:
        from xaytune.eval.benchmarks import benchmark_evaluate

        benchmarks = [b.strip() for b in args.benchmarks.split(",")]
        results = benchmark_evaluate(
            model=args.model,
            benchmarks=benchmarks,
            num_fewshot=args.num_fewshot,
        )

        for task_name, task_results in results.items():
            print(f"\n{task_name}:")
            for metric_name, value in task_results.items():
                print(f"  {metric_name}: {value}")

        return 0

    if args.dataset:
        import json
        from pathlib import Path
        from xaytune.eval import evaluate

        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"Error: Dataset not found: {args.dataset}", file=sys.stderr)
            return 1

        data = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
        metrics = [m.strip() for m in args.metrics.split(",")] if args.metrics else None

        results = evaluate(model=args.model, dataset=data, metrics=metrics)

        for metric_name, value in results.items():
            print(f"{metric_name}: {value:.4f}")

        return 0

    print("Error: Provide --benchmarks or --dataset", file=sys.stderr)
    return 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cli.py::TestEvalCommand -v`
Expected: All tests PASS

- [ ] **Step 5: Run full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add xaytune/cli.py tests/test_cli.py
git commit -m "feat: add eval CLI subcommand with benchmark and dataset modes"
```

---

### Task 4: export CLI Subcommand

**Files:**
- Modify: `xaytune/cli.py`
- Modify: `tests/test_cli.py`

Add `xaytune export merge/gguf/push` sub-subcommands.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
class TestExportCommand:
    def test_export_merge_parser(self):
        from xaytune.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["export", "merge", "--checkpoint", "ckpt/", "--output", "out/"])
        assert args.command == "export"
        assert args.export_command == "merge"
        assert args.checkpoint == "ckpt/"
        assert args.output == "out/"

    def test_export_gguf_parser(self):
        from xaytune.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["export", "gguf", "--model", "model/", "--output", "out.gguf"])
        assert args.command == "export"
        assert args.export_command == "gguf"
        assert args.model == "model/"

    def test_export_gguf_default_quant(self):
        from xaytune.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["export", "gguf", "--model", "m/", "--output", "o.gguf"])
        assert args.quant == "Q4_K_M"

    def test_export_push_parser(self):
        from xaytune.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["export", "push", "--model", "model/", "--repo", "user/repo"])
        assert args.command == "export"
        assert args.export_command == "push"
        assert args.repo == "user/repo"

    def test_export_merge_calls_merge(self):
        from unittest.mock import patch
        from xaytune.cli import main

        with patch("xaytune.cli._export_merge") as mock_merge:
            mock_merge.return_value = 0
            result = main(["export", "merge", "--checkpoint", "ckpt/", "--output", "out/"])

        assert result == 0

    def test_export_gguf_calls_to_gguf(self):
        from unittest.mock import patch
        from xaytune.cli import main

        with patch("xaytune.export.gguf.to_gguf") as mock_gguf:
            result = main(["export", "gguf", "--model", "m/", "--output", "o.gguf", "--quant", "Q8_0"])

        assert result == 0
        mock_gguf.assert_called_once_with("m/", output="o.gguf", quantization="Q8_0")

    def test_export_push_calls_push_to_hub(self):
        from unittest.mock import patch
        from xaytune.cli import main

        with patch("xaytune.export.hub.push_to_hub") as mock_push:
            result = main(["export", "push", "--model", "m/", "--repo", "user/repo"])

        assert result == 0
        mock_push.assert_called_once_with("m/", repo="user/repo")

    def test_export_no_subcommand_shows_help(self, capsys):
        from xaytune.cli import main
        result = main(["export"])
        assert result == 1
```

- [ ] **Step 2: Run tests to verify failures**

Run: `.venv/bin/python -m pytest tests/test_cli.py::TestExportCommand -v`
Expected: FAIL — export subparser does not exist

- [ ] **Step 3: Add export subcommand to cli.py**

In `xaytune/cli.py`, add to `_build_parser()` after the eval parser block:

```python
    export_parser = subparsers.add_parser("export", help="Export and convert models")
    export_subparsers = export_parser.add_subparsers(dest="export_command", help="Export commands")

    merge_parser = export_subparsers.add_parser("merge", help="Merge LoRA adapters into base model")
    merge_parser.add_argument("--checkpoint", required=True, help="Path to LoRA checkpoint")
    merge_parser.add_argument("--output", required=True, help="Output directory for merged model")

    gguf_parser = export_subparsers.add_parser("gguf", help="Convert model to GGUF format")
    gguf_parser.add_argument("--model", required=True, help="Path to model directory")
    gguf_parser.add_argument("--output", required=True, help="Output GGUF file path")
    gguf_parser.add_argument("--quant", default="Q4_K_M", help="Quantization type (default: Q4_K_M)")

    push_parser = export_subparsers.add_parser("push", help="Push model to Hugging Face Hub")
    push_parser.add_argument("--model", required=True, help="Path to model directory")
    push_parser.add_argument("--repo", required=True, help="HF Hub repo (e.g., username/model-name)")
```

Add to `main()` after the eval dispatch:

```python
    if args.command == "export":
        return _handle_export(args)
```

Add handler functions:

```python
def _handle_export(args: argparse.Namespace) -> int:
    if args.export_command is None:
        print("Error: Specify an export command: merge, gguf, push", file=sys.stderr)
        return 1

    if args.export_command == "merge":
        return _export_merge(args)

    if args.export_command == "gguf":
        return _export_gguf(args)

    if args.export_command == "push":
        return _export_push(args)

    return 1


def _export_merge(args: argparse.Namespace) -> int:
    from xaytune.export import merge

    try:
        merge(args.checkpoint, save_to=args.output)
        print(f"Merged model saved to {args.output}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _export_gguf(args: argparse.Namespace) -> int:
    from xaytune.export.gguf import to_gguf

    try:
        to_gguf(args.model, output=args.output, quantization=args.quant)
        print(f"GGUF model saved to {args.output}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _export_push(args: argparse.Namespace) -> int:
    from xaytune.export.hub import push_to_hub

    try:
        push_to_hub(args.model, repo=args.repo)
        print(f"Model pushed to {args.repo}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cli.py::TestExportCommand -v`
Expected: All tests PASS

- [ ] **Step 5: Run full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add xaytune/cli.py tests/test_cli.py
git commit -m "feat: add export CLI subcommand with merge, gguf, and push"
```

---

### Task 5: compare CLI Subcommand

**Files:**
- Modify: `xaytune/cli.py`
- Modify: `tests/test_cli.py`

Add `xaytune compare model-a/ model-b/ --benchmarks mmlu,gsm8k` for side-by-side evaluation.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
class TestCompareCommand:
    def test_compare_parser(self):
        from xaytune.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["compare", "model-a/", "model-b/", "--benchmarks", "mmlu,gsm8k"])
        assert args.command == "compare"
        assert args.models == ["model-a/", "model-b/"]
        assert args.benchmarks == "mmlu,gsm8k"

    def test_compare_calls_benchmark_evaluate_twice(self):
        from unittest.mock import patch, call
        from xaytune.cli import main

        results_a = {"mmlu": {"acc,none": 0.65}}
        results_b = {"mmlu": {"acc,none": 0.70}}

        with patch("xaytune.cli.benchmark_evaluate", side_effect=[results_a, results_b]) as mock_bench:
            result = main(["compare", "model-a/", "model-b/", "--benchmarks", "mmlu"])

        assert result == 0
        assert mock_bench.call_count == 2

    def test_compare_prints_table(self, capsys):
        from unittest.mock import patch
        from xaytune.cli import main

        results_a = {"mmlu": {"acc,none": 0.65}}
        results_b = {"mmlu": {"acc,none": 0.70}}

        with patch("xaytune.cli.benchmark_evaluate", side_effect=[results_a, results_b]):
            main(["compare", "model-a/", "model-b/", "--benchmarks", "mmlu"])

        captured = capsys.readouterr()
        assert "model-a/" in captured.out
        assert "model-b/" in captured.out
        assert "mmlu" in captured.out

    def test_compare_requires_two_models(self):
        from xaytune.cli import main
        result = main(["compare", "model-a/", "--benchmarks", "mmlu"])
        assert result != 0
```

- [ ] **Step 2: Run tests to verify failures**

Run: `.venv/bin/python -m pytest tests/test_cli.py::TestCompareCommand -v`
Expected: FAIL — compare subparser does not exist

- [ ] **Step 3: Add compare subcommand to cli.py**

In `xaytune/cli.py`, add to `_build_parser()` after the export parser block:

```python
    compare_parser = subparsers.add_parser("compare", help="Compare two models side-by-side")
    compare_parser.add_argument("models", nargs="+", help="Model paths to compare (exactly 2)")
    compare_parser.add_argument("--benchmarks", required=True, help="Comma-separated benchmarks")
    compare_parser.add_argument("--num-fewshot", type=int, default=None, help="Number of few-shot examples")
```

Add to `main()` after the export dispatch:

```python
    if args.command == "compare":
        return _handle_compare(args)
```

Add handler function:

```python
def _handle_compare(args: argparse.Namespace) -> int:
    if len(args.models) != 2:
        print("Error: compare requires exactly 2 models", file=sys.stderr)
        return 1

    from xaytune.eval.benchmarks import benchmark_evaluate

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    model_a, model_b = args.models

    results_a = benchmark_evaluate(
        model=model_a, benchmarks=benchmarks, num_fewshot=args.num_fewshot,
    )
    results_b = benchmark_evaluate(
        model=model_b, benchmarks=benchmarks, num_fewshot=args.num_fewshot,
    )

    all_tasks = sorted(set(results_a) | set(results_b))
    header = f"{'Benchmark':<20} {'Metric':<25} {model_a:<15} {model_b:<15}"
    print(header)
    print("-" * len(header))

    for task in all_tasks:
        metrics_a = results_a.get(task, {})
        metrics_b = results_b.get(task, {})
        all_metrics = sorted(set(metrics_a) | set(metrics_b))
        for metric in all_metrics:
            val_a = metrics_a.get(metric, "N/A")
            val_b = metrics_b.get(metric, "N/A")
            if isinstance(val_a, float):
                val_a = f"{val_a:.4f}"
            if isinstance(val_b, float):
                val_b = f"{val_b:.4f}"
            print(f"{task:<20} {metric:<25} {val_a:<15} {val_b:<15}")

    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cli.py::TestCompareCommand -v`
Expected: All tests PASS

- [ ] **Step 5: Run full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add xaytune/cli.py tests/test_cli.py
git commit -m "feat: add compare CLI subcommand for side-by-side model evaluation"
```
