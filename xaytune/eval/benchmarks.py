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
    """Run lm-eval-harness benchmarks against a HuggingFace model.

    Args:
        model: HuggingFace model name or local path.
        benchmarks: Benchmark task names (e.g. ``["mmlu", "gsm8k"]``).
        num_fewshot: Number of few-shot examples. ``None`` uses each
            benchmark's default.

    Returns:
        Dict mapping benchmark names to their result dicts.

    Raises:
        ImportError: If ``lm-eval`` is not installed.
    """
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

    return raw.get("results", {})  # type: ignore[no-any-return]
