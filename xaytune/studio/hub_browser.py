from __future__ import annotations

from typing import Any


def search_models(
    query: str,
    limit: int = 20,
    filter_task: str = "text-generation",
) -> list[dict[str, Any]]:
    """Search HuggingFace Hub for models matching a query.

    Returns a list of dicts with ``model_id``, ``downloads``, and ``likes``.
    Returns an empty list on network errors or if ``huggingface_hub`` is not
    installed.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        models = api.list_models(
            search=query,
            limit=limit,
            pipeline_tag=filter_task if filter_task else None,
            sort="downloads",
            direction=-1,
        )
        results = []
        for m in models:
            results.append(
                {
                    "model_id": m.id,
                    "downloads": getattr(m, "downloads", 0) or 0,
                    "likes": getattr(m, "likes", 0) or 0,
                }
            )
        return results
    except Exception:
        return []
