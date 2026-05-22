import pytest
from unittest.mock import patch, MagicMock
from trainlib.models import load_model, register_model
from trainlib.models.registry import model_registry


class TestModelRegistry:
    def test_register_custom_model(self):
        @register_model("test-model")
        class TestModel:
            pass
        assert model_registry.has("test-model")
        assert model_registry.get("test-model") is TestModel

    def test_list_registered_models(self):
        registered = model_registry.list()
        assert isinstance(registered, list)


class TestLoadModel:
    @patch("trainlib.models.loader.AutoModelForCausalLM")
    @patch("trainlib.models.loader.AutoTokenizer")
    def test_load_from_hub(self, mock_tokenizer_cls, mock_model_cls):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        result = load_model("some-org/some-model")
        mock_model_cls.from_pretrained.assert_called_once()
        mock_tokenizer_cls.from_pretrained.assert_called_once()
        assert result.model is mock_model
        assert result.tokenizer is mock_tokenizer

    @patch("trainlib.models.loader.AutoModelForCausalLM")
    @patch("trainlib.models.loader.AutoTokenizer")
    def test_load_with_dtype(self, mock_tokenizer_cls, mock_model_cls):
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        load_model("some-model", dtype="float16")
        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert "torch_dtype" in call_kwargs

    @patch("trainlib.models.loader.AutoModelForCausalLM")
    @patch("trainlib.models.loader.AutoTokenizer")
    def test_load_with_trust_remote_code(self, mock_tokenizer_cls, mock_model_cls):
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        load_model("some-model", trust_remote_code=True)
        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert call_kwargs.get("trust_remote_code") is True

    def test_load_model_result_has_config(self):
        from trainlib.models.loader import ModelResult
        result = ModelResult(model=MagicMock(), tokenizer=MagicMock(), name="test")
        assert result.name == "test"

    @patch("trainlib.models.loader.AutoModelForCausalLM")
    @patch("trainlib.models.loader.AutoTokenizer")
    def test_load_with_quantization_4bit(self, mock_tokenizer_cls, mock_model_cls):
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        load_model("some-model", quantization="4bit")
        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert "quantization_config" in call_kwargs

    @patch("trainlib.models.loader.AutoModelForCausalLM")
    @patch("trainlib.models.loader.AutoTokenizer")
    def test_load_with_quantization_8bit(self, mock_tokenizer_cls, mock_model_cls):
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        load_model("some-model", quantization="8bit")
        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert "quantization_config" in call_kwargs
