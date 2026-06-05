"""Tests for QLoRA prepare_model_for_kbit_training (BUG-035 / TASK-028).

Verifies that apply_lora() calls prepare_model_for_kbit_training()
for quantized models and skips it for non-quantized models.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch


class TestQLoRAPreparation:
    def _make_model_result(self, quantization=None):
        from xaytune.models.loader import ModelResult

        model = MagicMock()
        model.config = MagicMock()
        model.config.model_type = "llama"
        model.parameters.return_value = iter([torch.nn.Parameter(torch.randn(2, 2))])
        tokenizer = MagicMock()
        return ModelResult(
            model=model,
            tokenizer=tokenizer,
            name="test-model",
            quantization=quantization,
        )

    @patch("xaytune.models.peft.get_peft_model")
    @patch("xaytune.models.peft.prepare_model_for_kbit_training")
    def test_4bit_calls_prepare(self, mock_prepare, mock_get_peft):
        mock_prepare.return_value = MagicMock()
        mock_get_peft.return_value = MagicMock()

        from xaytune.models.peft import apply_lora

        model_result = self._make_model_result(quantization="4bit")
        apply_lora(model_result)

        mock_prepare.assert_called_once()

    @patch("xaytune.models.peft.get_peft_model")
    @patch("xaytune.models.peft.prepare_model_for_kbit_training")
    def test_8bit_calls_prepare(self, mock_prepare, mock_get_peft):
        mock_prepare.return_value = MagicMock()
        mock_get_peft.return_value = MagicMock()

        from xaytune.models.peft import apply_lora

        model_result = self._make_model_result(quantization="8bit")
        apply_lora(model_result)

        mock_prepare.assert_called_once()

    @patch("xaytune.models.peft.get_peft_model")
    @patch("xaytune.models.peft.prepare_model_for_kbit_training")
    def test_no_quantization_skips_prepare(self, mock_prepare, mock_get_peft):
        mock_get_peft.return_value = MagicMock()

        from xaytune.models.peft import apply_lora

        model_result = self._make_model_result(quantization=None)
        apply_lora(model_result)

        mock_prepare.assert_not_called()

    @patch("xaytune.models.peft.get_peft_model")
    @patch("xaytune.models.peft.prepare_model_for_kbit_training")
    def test_returns_model_result_with_peft_applied(self, mock_prepare, mock_get_peft):
        mock_prepare.return_value = MagicMock()
        peft_model = MagicMock()
        mock_get_peft.return_value = peft_model

        from xaytune.models.peft import apply_lora

        model_result = self._make_model_result(quantization="4bit")
        result = apply_lora(model_result)

        assert result.peft_applied is True
        assert result.model is peft_model
