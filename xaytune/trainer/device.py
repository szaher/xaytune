from __future__ import annotations

import random

import torch


def get_device_type() -> str:
    """Detect the best available device type (``"cuda"``, ``"mps"``, or ``"cpu"``)."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device(local_rank: int = 0, *, device_type: str | None = None) -> torch.device:
    """Return a :class:`torch.device` for the given rank and device type."""
    dt = device_type or get_device_type()
    if dt == "cuda":
        return torch.device(f"cuda:{local_rank}")
    return torch.device(dt)


def seed_all(seed: int) -> None:
    """Seed Python, PyTorch CPU, CUDA, MPS, and NumPy random generators."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    try:
        import numpy

        numpy.random.seed(seed)
    except ImportError:
        pass


def supports_amp(device_type: str) -> bool:
    """Return whether *device_type* supports ``torch.amp.autocast``."""
    if device_type == "cuda":
        return True
    if device_type == "mps":
        return True
    return False


def supports_grad_scaler(device_type: str, dtype: torch.dtype | None) -> bool:
    """Return whether GradScaler is needed (CUDA + fp16 only)."""
    return device_type == "cuda" and dtype == torch.float16


def detect_device_type_from_model(model: torch.nn.Module) -> str:
    """Infer the device type from the model's first parameter."""
    try:
        first_param = next(iter(model.parameters()))
        if first_param.is_cuda:
            return "cuda"
        if hasattr(first_param, "is_mps") and first_param.is_mps:
            return "mps"
    except (StopIteration, TypeError):
        pass
    return "cpu"
