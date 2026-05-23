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
            "Install it with: pip install trainlib[eval]"
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
