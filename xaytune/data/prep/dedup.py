from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from xaytune.data.prep.report import PrepReport, PrepResult, StepReport

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
    raise ValueError("No string field found for deduplication. Specify field= explicitly.")


def _exact_dedup(samples: list[dict], field: str) -> tuple[list[dict], int]:
    seen: set[str] = set()
    result = []
    dupes = 0
    for sample in samples:
        h = hashlib.sha256(sample[field].encode()).hexdigest()
        if h in seen:
            dupes += 1
        else:
            seen.add(h)
            result.append(sample)
    return result, dupes


def _minhash_dedup(
    samples: list[dict], field: str, threshold: float, num_perm: int, ngram: int
) -> tuple[list[dict], int]:
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        raise ImportError(
            "Near-duplicate dedup requires datasketch. "
            "Install with: pip install xaytune[data-prep]"
        )

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: list[MinHash] = []

    for i, sample in enumerate(samples):
        text = sample[field]
        m = MinHash(num_perm=num_perm)
        for j in range(len(text) - ngram + 1):
            m.update(text[j : j + ngram].encode("utf-8"))
        minhashes.append(m)
        try:
            lsh.insert(str(i), m)
        except ValueError:
            pass

    keep = set()
    removed = 0
    for i, m in enumerate(minhashes):
        if i in keep or i in {-1}:
            continue
        matches = lsh.query(m)
        match_indices = sorted(int(x) for x in matches)
        if match_indices:
            keep.add(match_indices[0])
            removed += len(match_indices) - 1

    result = [samples[i] for i in sorted(keep)]
    return result, removed


def deduplicate(
    source: str | list[dict],
    *,
    method: Literal["exact", "minhash", "both"] = "both",
    threshold: float = 0.85,
    field: str | None = None,
    num_perm: int = 128,
    ngram: int = 5,
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

    exact_dupes = 0
    near_dupes = 0

    if method in ("exact", "both"):
        samples, exact_dupes = _exact_dedup(samples, field)

    if method in ("minhash", "both"):
        samples, near_dupes = _minhash_dedup(samples, field, threshold, num_perm, ngram)

    step = StepReport(
        name="deduplicate",
        input_rows=input_rows,
        output_rows=len(samples),
        details={
            "method": method,
            "exact_dupes": exact_dupes,
            "near_dupes": near_dupes,
        },
    )

    return PrepResult(
        dataset=samples,
        report=PrepReport(
            input_rows=input_rows,
            output_rows=len(samples),
            steps=[step],
        ),
    )
