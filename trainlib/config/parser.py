from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from trainlib.config.schema import TrainConfig


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_value(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null" or value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(data)
    for override in overrides:
        key, _, value = override.partition("=")
        parts = key.split(".")
        target = result
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = _parse_value(value)
    return result


def _resolve_inheritance(data: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    base_path = data.pop("base", None)
    if base_path is None:
        return data

    full_base_path = config_dir / base_path
    if not full_base_path.exists():
        raise FileNotFoundError(f"Base config not found: {full_base_path}")

    with open(full_base_path) as f:
        base_data = yaml.safe_load(f)

    base_data = _resolve_inheritance(base_data, full_base_path.parent)
    return merge_dicts(base_data, data)


def load_config(
    path: str,
    overrides: list[str] | None = None,
) -> TrainConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    data = _resolve_inheritance(data, config_path.parent)

    if overrides:
        data = apply_overrides(data, overrides)

    return TrainConfig(**data)
