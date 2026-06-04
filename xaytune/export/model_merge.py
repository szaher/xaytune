from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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


def _linear_merge(state_dicts: list[dict[str, Tensor]], weights: list[float]) -> dict[str, Tensor]:
    merged: dict[str, Tensor] = {}
    keys = state_dicts[0].keys()
    for key in keys:
        tensors = [sd[key] for sd in state_dicts]
        acc = weights[0] * tensors[0]
        for w, t in zip(weights[1:], tensors[1:]):
            acc = acc + w * t
        merged[key] = acc
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


def _slerp_merge(sd_a: dict[str, Tensor], sd_b: dict[str, Tensor], t: float) -> dict[str, Tensor]:
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


def _dare_merge(
    state_dicts: list[dict[str, Tensor]],
    base_sd: dict[str, Tensor],
    density: float,
    weight: float,
    seed: int,
) -> dict[str, Tensor]:
    merged: dict[str, Tensor] = {}
    gen = torch.Generator()
    gen.manual_seed(seed)

    for key in base_sd.keys():
        base_tensor = base_sd[key].float()

        task_vectors = [(sd[key].float() - base_tensor) for sd in state_dicts]

        rescaled = []
        for tv in task_vectors:
            mask = torch.bernoulli(torch.full_like(tv, density), generator=gen).bool()
            dropped = tv * mask.float()
            if density > 0:
                dropped = dropped / density
            rescaled.append(dropped)

        avg_task_vector = torch.stack(rescaled).mean(dim=0)
        merged[key] = (base_tensor + weight * avg_task_vector).to(base_sd[key].dtype)

    return merged


def _load_state_dict(path: str) -> dict[str, Tensor]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    return model.state_dict()


def _load_tokenizer(path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def _save_merged(state_dict: dict[str, Tensor], tokenizer: Any, output: str, model_paths: list[str] | None = None) -> None:
    from transformers import AutoConfig

    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, str(path / "pytorch_model.bin"))
    tokenizer.save_pretrained(output)
    if model_paths:
        config = AutoConfig.from_pretrained(model_paths[0])
        config.save_pretrained(output)


def model_merge(
    models: list[str],
    method: Literal["linear", "slerp", "ties", "dare"],
    output: str,
    *,
    weights: list[float] | None = None,
    t: float = 0.5,
    base_model: str | None = None,
    density: float = 0.5,
    weight: float = 1.0,
    seed: int = 42,
) -> MergeResult:
    if method == "slerp" and len(models) != 2:
        raise ValueError(f"SLERP requires exactly 2 models, got {len(models)}")

    if method in ("ties", "dare") and base_model is None:
        raise ValueError(f"{method.upper()} requires base_model= argument")

    if weights is not None and len(weights) != len(models):
        raise ValueError(f"weights length ({len(weights)}) must match models count ({len(models)})")

    if not 0.0 <= density <= 1.0:
        raise ValueError(f"density must be between 0.0 and 1.0, got {density}")

    if not 0.0 <= t <= 1.0:
        raise ValueError(f"t must be between 0.0 and 1.0, got {t}")

    state_dicts = [_load_state_dict(m) for m in models]

    ref_keys = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], 1):
        if set(sd.keys()) != ref_keys:
            raise ValueError(
                f"Model {models[i]} has different state_dict keys than {models[0]}. "
                f"All models must have the same architecture."
            )

    tokenizer = _load_tokenizer(models[0])

    params: dict[str, Any] = {"method": method}

    if method == "linear":
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        merged_sd = _linear_merge(state_dicts, weights)
        params["weights"] = weights
    elif method == "slerp":
        merged_sd = _slerp_merge(state_dicts[0], state_dicts[1], t)
        params["t"] = t
    elif method == "ties":
        assert base_model is not None
        base_sd = _load_state_dict(base_model)
        merged_sd = _ties_merge(state_dicts, base_sd, density, weight)
        params.update({"density": density, "weight": weight, "base_model": base_model})
    elif method == "dare":
        assert base_model is not None
        base_sd = _load_state_dict(base_model)
        merged_sd = _dare_merge(state_dicts, base_sd, density, weight, seed)
        params.update(
            {"density": density, "weight": weight, "seed": seed, "base_model": base_model}
        )
    else:
        raise ValueError(f"Unknown merge method: {method}")

    _save_merged(merged_sd, tokenizer, output, model_paths=models)

    return MergeResult(
        output_path=output,
        method=method,
        models=models,
        params=params,
    )
