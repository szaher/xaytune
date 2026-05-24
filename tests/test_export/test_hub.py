from unittest.mock import MagicMock, patch

import pytest

from xaytune.export.hub import push_to_hub


class TestPushToHub:
    @patch("xaytune.models.load_model")
    def test_push_to_hub_with_path(self, mock_load_model):
        mock_result = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_result.model = mock_model
        mock_result.tokenizer = mock_tokenizer
        mock_load_model.return_value = mock_result

        push_to_hub("output/my-model", repo="user/my-model")

        mock_model.push_to_hub.assert_called_once_with("user/my-model")
        mock_tokenizer.push_to_hub.assert_called_once_with("user/my-model")

    def test_push_to_hub_with_objects(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        push_to_hub(mock_model, repo="user/my-model", tokenizer=mock_tokenizer)

        mock_model.push_to_hub.assert_called_once_with("user/my-model")
        mock_tokenizer.push_to_hub.assert_called_once_with("user/my-model")

    def test_push_requires_repo(self):
        with pytest.raises(ValueError, match="repo"):
            push_to_hub("output/model")
