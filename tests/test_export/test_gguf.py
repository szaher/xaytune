from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from xaytune.export.gguf import to_gguf


class TestToGguf:
    def test_calls_llama_cpp_convert(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output = tmp_path / "output.gguf"

        with patch("xaytune.export.gguf.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            to_gguf(str(model_dir), output=str(output))

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert "convert" in " ".join(str(a) for a in call_args[0][0])

    def test_default_quantization_is_q4_k_m(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output = tmp_path / "output.gguf"

        with patch("xaytune.export.gguf.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            to_gguf(str(model_dir), output=str(output))

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "Q4_K_M" in cmd

    def test_custom_quantization(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output = tmp_path / "output.gguf"

        with patch("xaytune.export.gguf.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            to_gguf(str(model_dir), output=str(output), quantization="Q8_0")

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "Q8_0" in cmd

    def test_raises_on_missing_model_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            to_gguf(str(tmp_path / "nonexistent"), output="out.gguf")

    def test_raises_on_conversion_failure(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with patch("xaytune.export.gguf.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="conversion failed")
            with pytest.raises(RuntimeError, match="GGUF conversion failed"):
                to_gguf(str(model_dir), output="out.gguf")

    def test_output_dir_created(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output = tmp_path / "nested" / "dir" / "output.gguf"

        with patch("xaytune.export.gguf.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            to_gguf(str(model_dir), output=str(output))

        assert output.parent.exists()


class TestGgufImport:
    def test_importable_from_export(self):
        from xaytune.export import to_gguf as fn

        assert callable(fn)
