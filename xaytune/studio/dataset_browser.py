from __future__ import annotations

from typing import Any


def search_datasets(
    query: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search HuggingFace Hub for datasets matching a query.

    Returns a list of dicts with ``dataset_id``, ``downloads``, ``likes``,
    and ``tags``.  Returns an empty list on network errors or if
    ``huggingface_hub`` is not installed.
    """
    if not query or not query.strip():
        return []
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        datasets = api.list_datasets(
            search=query,
            limit=limit,
            sort="downloads",
        )
        results = []
        for d in datasets:
            tags = getattr(d, "tags", []) or []
            results.append(
                {
                    "dataset_id": d.id,
                    "downloads": getattr(d, "downloads", 0) or 0,
                    "likes": getattr(d, "likes", 0) or 0,
                    "tags": ", ".join(tags[:5]),
                }
            )
        return results
    except Exception:
        return []


def preview_hf_dataset(
    dataset_id: str,
    split: str = "train",
    num_samples: int = 5,
) -> list[dict[str, Any]]:
    """Load first N samples from a HuggingFace dataset using streaming."""
    if not dataset_id or not dataset_id.strip():
        return []
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_id, split=split, streaming=True)
        samples = []
        for i, row in enumerate(ds):
            if i >= num_samples:
                break
            samples.append(dict(row))
        return samples
    except Exception:
        return []


def get_dataset_info(dataset_id: str) -> dict[str, Any]:
    """Get metadata about a HuggingFace dataset."""
    if not dataset_id or not dataset_id.strip():
        return {}
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.dataset_info(dataset_id)
        description = getattr(info, "description", "") or ""
        return {
            "id": info.id,
            "description": description[:500],
            "downloads": getattr(info, "downloads", 0) or 0,
            "tags": getattr(info, "tags", []) or [],
        }
    except Exception:
        return {}
