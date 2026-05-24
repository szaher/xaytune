from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from xaytune.studio.hub_browser import search_models


def _mock_model(model_id: str, downloads: int, likes: int) -> MagicMock:
    m = MagicMock()
    m.id = model_id
    m.downloads = downloads
    m.likes = likes
    return m


def _setup_mock_hub(models: list[MagicMock] | None = None) -> MagicMock:
    mock_api = MagicMock()
    mock_api.list_models.return_value = models or []
    mock_module = MagicMock()
    mock_module.HfApi.return_value = mock_api
    return mock_module, mock_api


class TestSearchModels:
    def test_returns_results(self):
        mock_module, mock_api = _setup_mock_hub(
            [
                _mock_model("org/model-a", 1000, 50),
                _mock_model("org/model-b", 500, 30),
            ]
        )
        with patch.dict(sys.modules, {"huggingface_hub": mock_module}):
            results = search_models("model")
            assert len(results) == 2
            assert results[0]["model_id"] == "org/model-a"
            assert results[0]["downloads"] == 1000
            assert results[0]["likes"] == 50

    def test_respects_limit(self):
        mock_module, mock_api = _setup_mock_hub()
        with patch.dict(sys.modules, {"huggingface_hub": mock_module}):
            search_models("test", limit=5)
            call_kwargs = mock_api.list_models.call_args[1]
            assert call_kwargs["limit"] == 5

    def test_passes_filter_task(self):
        mock_module, mock_api = _setup_mock_hub()
        with patch.dict(sys.modules, {"huggingface_hub": mock_module}):
            search_models("test", filter_task="text-generation")
            call_kwargs = mock_api.list_models.call_args[1]
            assert call_kwargs["pipeline_tag"] == "text-generation"

    def test_empty_filter_task(self):
        mock_module, mock_api = _setup_mock_hub()
        with patch.dict(sys.modules, {"huggingface_hub": mock_module}):
            search_models("test", filter_task="")
            call_kwargs = mock_api.list_models.call_args[1]
            assert call_kwargs["pipeline_tag"] is None

    def test_returns_empty_on_error(self):
        mock_module = MagicMock()
        mock_module.HfApi.side_effect = Exception("no network")
        with patch.dict(sys.modules, {"huggingface_hub": mock_module}):
            results = search_models("test")
            assert results == []

    def test_handles_missing_attributes(self):
        m = MagicMock(spec=[])
        m.id = "org/model"
        mock_module, mock_api = _setup_mock_hub([m])
        with patch.dict(sys.modules, {"huggingface_hub": mock_module}):
            results = search_models("test")
            assert len(results) == 1
            assert results[0]["downloads"] == 0
            assert results[0]["likes"] == 0
