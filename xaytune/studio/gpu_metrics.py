from __future__ import annotations


def get_gpu_metrics() -> dict[str, float]:
    """Return current GPU memory metrics, or empty dict on CPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}

        return {
            "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
            "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
            "gpu_memory_peak_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        }
    except Exception:
        return {}
