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


def _slerp_tensor(a: Tensor, b: Tensor, t: float) -> Tensor:
    if a.dim() <= 1 and a.numel() <= 1:
        return (1 - t) * a + t * b

    a_flat = a.flatten().float()
    b_flat = b.flatten().float()

    a_norm = torch.nn.functional.normalize(a_flat, dim=0)
    b_norm = torch.nn.functional.normalize(b_flat, dim=0)

    dot = torch.clamp(torch.dot(a_norm, b_norm), -1.0, 1.0)
    theta = torch.acos(dot)

    if theta.abs() < 1e-6:
        return ((1 - t) * a + t * b).to(a.dtype)

    sin_theta = torch.sin(theta)
    w_a = torch.sin((1 - t) * theta) / sin_theta
    w_b = torch.sin(t * theta) / sin_theta

    return (w_a * a.float() + w_b * b.float()).to(a.dtype)


def _slerp_merge(
    sd_a: dict[str, Tensor], sd_b: dict[str, Tensor], t: float
) -> dict[str, Tensor]:
    merged: dict[str, Tensor] = {}
    for key in sd_a.keys():
        if not isinstance(sd_a[key], Tensor):
            merged[key] = sd_a[key]
            continue
        merged[key] = _slerp_tensor(sd_a[key], sd_b[key], t)
    return merged


def _ties_merge(
    state_dicts: list[dict[str, Tensor]],
    base_sd: dict[str, Tensor],
    density: float,
    weight: float,
) -> dict[str, Tensor]:
    merged: dict[str, Tensor] = {}

    for key in base_sd.keys():
        base_tensor = base_sd[key].float()

        task_vectors = [(sd[key].float() - base_tensor) for sd in state_dicts]

        trimmed = []
        for tv in task_vectors:
            flat = tv.flatten()
            k = max(1, int(density * flat.numel()))
            threshold = flat.abs().topk(k).values[-1]
            mask = flat.abs() >= threshold
            trimmed_flat = flat * mask.float()
            trimmed.append(trimmed_flat.reshape(tv.shape))

        stacked = torch.stack(trimmed)
        sign_magnitude = stacked.sum(dim=0)
        elected_sign = torch.sign(sign_magnitude)

        aligned = []
        for tv in trimmed:
            mask = torch.sign(tv) == elected_sign
            aligned.append(tv * mask.float())

        stacked_aligned = torch.stack(aligned)
        counts = (stacked_aligned != 0).float().sum(dim=0).clamp(min=1)
        avg_task_vector = stacked_aligned.sum(dim=0) / counts

        merged[key] = (base_tensor + weight * avg_task_vector).to(base_sd[key].dtype)

    return merged
