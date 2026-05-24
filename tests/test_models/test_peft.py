from unittest.mock import MagicMock, patch

from xaytune.models.loader import ModelResult
from xaytune.models.peft import apply_lora, get_target_modules


class TestGetTargetModules:
    def test_auto_returns_default(self):
        mock_model = MagicMock()
        mock_model.config.model_type = "llama"
        modules = get_target_modules("auto", mock_model)
        assert isinstance(modules, list)
        assert len(modules) > 0

    def test_explicit_modules(self):
        mock_model = MagicMock()
        modules = get_target_modules(["q_proj", "v_proj"], mock_model)
        assert modules == ["q_proj", "v_proj"]

    def test_unknown_model_type_returns_common(self):
        mock_model = MagicMock()
        mock_model.config.model_type = "totally_unknown_model_xyz"
        modules = get_target_modules("auto", mock_model)
        assert isinstance(modules, list)
        assert len(modules) > 0


class TestApplyLora:
    @patch("xaytune.models.peft.get_peft_model")
    @patch("xaytune.models.peft.LoraConfig")
    def test_apply_lora_basic(self, mock_lora_config_cls, mock_get_peft):
        mock_model = MagicMock()
        mock_model.config.model_type = "llama"
        mock_peft_model = MagicMock()
        mock_get_peft.return_value = mock_peft_model
        model_result = ModelResult(model=mock_model, tokenizer=MagicMock(), name="test")
        result = apply_lora(model_result, rank=16, alpha=32, dropout=0.05)
        mock_lora_config_cls.assert_called_once()
        mock_get_peft.assert_called_once()
        assert result.model is mock_peft_model
        assert result.peft_applied is True

    @patch("xaytune.models.peft.get_peft_model")
    @patch("xaytune.models.peft.LoraConfig")
    def test_apply_lora_custom_target_modules(self, mock_lora_config_cls, mock_get_peft):
        mock_model = MagicMock()
        mock_model.config.model_type = "llama"
        mock_get_peft.return_value = MagicMock()
        model_result = ModelResult(model=mock_model, tokenizer=MagicMock(), name="test")
        apply_lora(model_result, rank=8, alpha=16, target_modules=["q_proj", "k_proj"])
        call_kwargs = mock_lora_config_cls.call_args[1]
        assert call_kwargs["target_modules"] == ["q_proj", "k_proj"]

    @patch("xaytune.models.peft.get_peft_model")
    @patch("xaytune.models.peft.LoraConfig")
    def test_apply_lora_sets_rank_and_alpha(self, mock_lora_config_cls, mock_get_peft):
        mock_model = MagicMock()
        mock_model.config.model_type = "llama"
        mock_get_peft.return_value = MagicMock()
        model_result = ModelResult(model=mock_model, tokenizer=MagicMock(), name="test")
        apply_lora(model_result, rank=64, alpha=128, dropout=0.1)
        call_kwargs = mock_lora_config_cls.call_args[1]
        assert call_kwargs["r"] == 64
        assert call_kwargs["lora_alpha"] == 128
        assert call_kwargs["lora_dropout"] == 0.1

    @patch("xaytune.models.peft.get_peft_model")
    @patch("xaytune.models.peft.LoraConfig")
    def test_apply_lora_preserves_tokenizer(self, mock_lora_config_cls, mock_get_peft):
        mock_model = MagicMock()
        mock_model.config.model_type = "llama"
        mock_tokenizer = MagicMock()
        mock_get_peft.return_value = MagicMock()
        model_result = ModelResult(model=mock_model, tokenizer=mock_tokenizer, name="test")
        result = apply_lora(model_result, rank=16, alpha=32)
        assert result.tokenizer is mock_tokenizer
