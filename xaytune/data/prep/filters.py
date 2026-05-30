from __future____ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from xaytune.data.prep.report import PrepReport, PrepResult, StepReport
from xaytune.utils.registry import Registry

filter_registry = Registry("filter")

FilterFn = Callable[[dict[str, Any], str], bool]

_FIELD_PRIORITY = ("text", "output", "response")


def _load_input(source: str | list[dict]) -> list[dict[str, Any]]:
    if isinstance(source, list):
        return source
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _detect_field(sample: dict[str, Any]) -> str:
    for name in _FIELD_PRIORITY:
        if name in sample and isinstance(sample[name], str):
            return name
    for name, value in sample.items():
        if isinstance(value, str):
            return name
    raise ValueError("No string field found for filtering. Specify field= explicitly.")


def _build_length_filter(min_chars: int = 0, max_chars: int = -1, **_: Any) -> FilterFn:
    def fn(sample: dict, field: str) -> bool:
        length = len(sample[field])
        if length < min_chars:
            return False
        if max_chars > 0 and length > max_chars:
            return False
        return True
    return fn


def _build_regex_filter(
    drop_pattern: str | None = None, keep_pattern: str | None = None, **_: Any
) -> FilterFn:
    drop_re = re.compile(drop_pattern) if drop_pattern else None
    keep_re = re.compile(keep_pattern) if keep_pattern else None

    def fn(sample: dict, field: str) -> bool:
        text = sample[field]
        if drop_re and drop_re.search(text):
            return False
        if keep_re and not keep_re.search(text):
            return False
        return True
    return fn


def _build_language_filter(keep: list[str], **_: Any) -> FilterFn:
    try:
        from langdetect import detect
    except ImportError:
        raise ImportError(
            "Language filtering requires langdetect. "
            "Install with: pip install xaytune[data-prep]"
        )

    def fn(sample: dict, field: str) -> bool:
        try:
            lang = detect(sample[field])
            return lang in keep
        except Exception:
            return False
    return fn


def _build_decontaminate_filter(
    reference: str, ngram_size: int = 13, field: str = "text", **_: Any
) -> FilterFn:
    ref_data = _load_input(reference)
    ref_ngrams: set[str] = set()
    for item in ref_data:
        text = item.get(field, "")
        words = text.split()
        for i in range(len(words) - ngram_size + 1):
            ref_ngrams.add(" ".join(words[i : i + ngram_size]))

    def fn(sample: dict, f: str) -> bool:
        text = sample[f]
        words = text.split()
        for i in range(len(words) - ngram_size + 1):
            if " ".join(words[i : i + ngram_size]) in ref_ngrams:
                return False
        return True
    return fn


_BUILTIN_BUILDERS: dict[str, Callable[..., FilterFn]] = {
    "length": _build_length_filter,
    "regex": _build_regex_filter,
    "language": _build_language_filter,
    "decontaminate": _build_decontaminate_filter,
}


def _resolve_filter(spec: dict[str, Any], field: str) -> tuple[str, FilterFn]:
    filter_type = spec["type"]
    params = {k: v for k, v in spec.items() if k != "type"}
    params.setdefault("field", field)

    if filter_type in _BUILTIN_BUILDERS:
        return filter_type, _BUILTIN_BUILDERS[filter_type](**params)

    if filter_registry.has(filter_type):
        return filter_type, filter_registry.get(filter_type)

    raise ValueError(
        f"Unknown filter type: '{filter_type}'. "
        f"Built-in: {', '.join(_BUILTIN_BUILDERS)}. "
        f"Registered: {', '.join(filter_registry.list())}"
    )


def filter_dataset(
    source: str | list[dict],
    *,
    filters: list[dict[str, Any]],
    field: str | None = None,
) -> PrepResult:
    samples = _load_input(source)
    input_rows = len(samples)

    if not samples:
        return PrepResult(
            dataset=[],
            report=PrepReport(input_rows=0, output_rows=0, steps=[]),
        )

    if field is None:
        field = _detect_field(samples[0])

    steps: list[StepReport] = []
    for spec in filters:
        before = len(samples)
        filter_name, filter_fn = _resolve_filter(spec, field)
        samples = [s for s in samples if filter_fn(s, field)]
        steps.append(StepReport(
            name=f"filter:{filter_name}",
            input_rows=before,
            output_rows=len(samples),
            details=spec,
        ))

    return PrepResult(
        dataset=samples,
        report=PrepReport(
            input_rows=input_rows,
            output_rows=len(samples),
            steps=steps,
        ),
    )
