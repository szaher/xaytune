from unittest.mock import MagicMock, patch

from xaytune.studio.dataset_browser import get_dataset_info, preview_hf_dataset, search_datasets


class TestSearchDatasets:
    @patch("huggingface_hub.HfApi.list_datasets")
    def test_returns_list(self, mock_list):
        mock_ds = MagicMock()
        mock_ds.id = "tatsu-lab/alpaca"
        mock_ds.downloads = 50000
        mock_ds.likes = 200
        mock_ds.tags = ["task_categories:text-generation", "language:en"]
        mock_list.return_value = [mock_ds]

        results = search_datasets("alpaca")
        assert len(results) == 1
        assert results[0]["dataset_id"] == "tatsu-lab/alpaca"
        assert results[0]["downloads"] == 50000
        assert results[0]["likes"] == 200
        assert "task_categories:text-generation" in results[0]["tags"]

    def test_empty_query_returns_empty(self):
        assert search_datasets("") == []
        assert search_datasets("  ") == []

    @patch("huggingface_hub.HfApi.list_datasets")
    def test_network_error_returns_empty(self, mock_list):
        mock_list.side_effect = ConnectionError("timeout")
        assert search_datasets("alpaca") == []


class TestPreviewHFDataset:
    @patch("datasets.load_dataset")
    def test_returns_samples(self, mock_load):
        mock_ds = [
            {"instruction": "Add 2+2", "output": "4"},
            {"instruction": "Say hi", "output": "Hello"},
        ]
        mock_load.return_value = iter(mock_ds)

        samples = preview_hf_dataset("test/dataset", num_samples=5)
        assert len(samples) == 2
        assert samples[0]["instruction"] == "Add 2+2"

    @patch("datasets.load_dataset")
    def test_error_returns_empty(self, mock_load):
        mock_load.side_effect = Exception("not found")
        assert preview_hf_dataset("nonexistent/ds") == []

    def test_empty_id_returns_empty(self):
        assert preview_hf_dataset("") == []


class TestGetDatasetInfo:
    @patch("huggingface_hub.HfApi.dataset_info")
    def test_returns_info(self, mock_info_fn):
        mock_info = MagicMock()
        mock_info.id = "tatsu-lab/alpaca"
        mock_info.description = "A dataset for instruction following."
        mock_info.downloads = 50000
        mock_info.tags = ["en", "text-generation"]
        mock_info_fn.return_value = mock_info

        info = get_dataset_info("tatsu-lab/alpaca")
        assert info["id"] == "tatsu-lab/alpaca"
        assert info["downloads"] == 50000
        assert "en" in info["tags"]

    def test_empty_id_returns_empty(self):
        assert get_dataset_info("") == {}

    @patch("huggingface_hub.HfApi.dataset_info")
    def test_error_returns_empty(self, mock_info_fn):
        mock_info_fn.side_effect = Exception("not found")
        assert get_dataset_info("bad/id") == {}
