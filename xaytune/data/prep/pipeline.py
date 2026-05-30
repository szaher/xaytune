from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from xaytune.data.prep.convert import convert
from xaytune.data.prep.dedup import deduplicate
from xaytune.data.prep.filters import filter_dataset
from xaytune.data.prep.report import PrepReport, PrepResult, StepReport


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


def _run_step(
    samples: list[dict], step_name: str, step_params: dict[str, Any]
) -> tuple[list[dict], StepReport]:
    before = len(samples)

    if step_name == "filter":
        result = filter_dataset(samples, filters=[_as_filter_spec(step_params)], field=step_params.get("field"))
        return result.dataset, result.report.steps[0] if result.report.steps else StepReport(
            name="filter", input_rows=before, output_rows=len(result.dataset), details=step_params
        )

    if step_name == "deduplicate":
        result = deduplicate(samples, **step_params)
        return result.dataset, result.report.steps[0] if result.report.steps else StepReport(
            name="deduplicate", input_rows=before, output_rows=len(result.dataset), details=step_params
        )

    if step_name == "convert":
        result = convert(samples, **step_params)
        return result.dataset, result.report.steps[0] if result.report.steps else StepReport(
            name="convert", input_rows=before, output_rows=len(result.dataset), details=step_params
        )

    raise ValueError(f"Unknown pipeline step: '{step_name}'. Supported: filter, deduplicate, convert")


def _as_filter_spec(params: dict[str, Any]) -> dict[str, Any]:
    if "type" in params:
        return params

    spec: dict[str, Any] = {}
    if "min_chars" in params or "max_chars" in params:
        spec = {"type": "length"}
        if "min_chars" in params:
            spec["min_chars"] = params["min_chars"]
        if "max_chars" in params:
            spec["max_chars"] = params["max_chars"]
        return spec

    if "language" in params:
        return {"type": "language", "keep": [params["language"]] if isinstance(params["language"], str) else params["language"]}

    if "drop_regex" in params:
        return {"type": "regex", "drop_pattern": params["drop_regex"]}

    return {"type": "length", **params}


def pipeline(
    *,
    input: str | list[dict] | None = None,
    output: str | None = None,
    steps: list[dict[str, Any]] | None = None,
    config: str | None = None,
) -> PrepResult:
    if config is not None:
        cfg_path = Path(config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {config}")
        cfg = yaml.safe_load(cfg_path.read_text())
        input = cfg.get("input", input)
        output = cfg.get("output", output)
        steps = cfg.get("steps", steps)

    if input is None:
        raise ValueError("pipeline() requires input= or a config file with 'input'.")
    if steps is None:
        steps = []

    samples = _load_input(input)
    input_rows = len(samples)
    all_steps: list[StepReport] = []

    for step_dict in steps:
        for step_name, step_params in step_dict.items():
            if step_params is None:
                step_params = {}
            samples, step_report = _run_step(samples, step_name, step_params)
            all_steps.append(step_report)

    result = PrepResult(
        dataset=samples,
        report=PrepReport(
            input_rows=input_rows,
            output_rows=len(samples),
            steps=all_steps,
        ),
    )

    if output:
        result.save(output)

    return result
