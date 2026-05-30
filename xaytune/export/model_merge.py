from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class MergeResult:
    output_path: str
    method: str
    models: list[str]
    params: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        model_list = ", ".join(self.models)
        lines = [
            f"Merge method: {self.method}",
            f"Models: {model_list}",
            f"Output: {self.output_path}",
        ]
        for k, v in self.params.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def _linear_merge(
    state_dicts: list[dict[str, Tensor]], weights: list[float]
) -> dict[str, Tensor]:
    merged: dict[str, Tensor] = {}
    keys = state_dicts[0].keys()
    for key in keys:
        tensors = [sd[key] for sd in state_dicts]
        merged[key] = sum(w * t for w, t in zip(weights, tensors))
    return merged
