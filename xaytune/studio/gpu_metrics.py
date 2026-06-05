from __future__ import annotations


def get_gpu_metrics() -> dict[str, float]:
    """Return current GPU memory and utilization metrics, or empty dict on CPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}

        util = get_gpu_utilization()
        return {
            "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
            "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
            "gpu_memory_peak_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
            "gpu_utilization": util if util is not None else 0.0,
        }
    except Exception:
        return {}


def get_gpu_utilization() -> float | None:
    """Return current GPU utilization percentage (0-100), or None on CPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        return float(torch.cuda.utilization())
    except Exception:
        return None
