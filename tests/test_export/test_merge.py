import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trainlib.export.merge import merge, save


class TestMerge:
    @patch("trainlib.models.load_model")
    def test_merge_loads_and_merges(self, mock_load_model):
        mock_result = MagicMock()
        mock_model = MagicMock()
        mock_model.merge_and_unload.return_value = MagicMock()
        mock_result.model = mock_model
        mock_result.tokenizer = MagicMock()
        mock_result.peft_applied = True
        mock_load_model.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            merge("checkpoint/lora", save_to=tmpdir)

        mock_model.merge_and_unload.assert_called_once()

    @patch("trainlib.models.load_model")
    def test_merge_saves_model_and_tokenizer(self, mock_load_model):
        mock_result = MagicMock()
        mock_merged = MagicMock()
        mock_result.model = MagicMock()
        mock_result.model.merge_and_unload.return_value = mock_merged
        mock_result.tokenizer = MagicMock()
        mock_result.peft_applied = True
        mock_load_model.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            merge("checkpoint/lora", save_to=tmpdir)

            mock_merged.save_pretrained.assert_called_once_with(tmpdir)
            mock_result.tokenizer.save_pretrained.assert_called_once_with(tmpdir)

    @patch("trainlib.models.load_model")
    def test_merge_non_peft_raises(self, mock_load_model):
        mock_result = MagicMock()
        mock_result.peft_applied = False
        mock_load_model.return_value = mock_result

        with pytest.raises(ValueError, match="not a PEFT model"):
            merge("checkpoint/full", save_to="output/")


class TestSave:
    def test_save_creates_directory(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "saved-model"
            save(mock_model, mock_tokenizer, output_dir=str(output_dir))

            assert output_dir.exists()

    def test_save_calls_save_pretrained(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            save(mock_model, mock_tokenizer, output_dir=tmpdir)

            mock_model.save_pretrained.assert_called_once_with(tmpdir)
            mock_tokenizer.save_pretrained.assert_called_once_with(tmpdir)

    def test_save_with_metadata(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            save(
                mock_model,
                mock_tokenizer,
                output_dir=tmpdir,
                metadata={"recipe": "finetune", "method": "lora"},
            )

            meta_path = Path(tmpdir) / "trainlib_metadata.json"
            assert meta_path.exists()
            meta = json.loads(meta_path.read_text())
            assert meta["recipe"] == "finetune"
