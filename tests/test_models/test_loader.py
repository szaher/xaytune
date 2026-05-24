from unittest.mock import MagicMock, patch

from xaytune.models import load_model, register_model
from xaytune.models.registry import model_registry


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
    @patch("xaytune.models.loader.AutoModelForCausalLM")
    @patch("xaytune.models.loader.AutoTokenizer")
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

    @patch("xaytune.models.loader.AutoModelForCausalLM")
    @patch("xaytune.models.loader.AutoTokenizer")
    def test_load_with_dtype(self, mock_tokenizer_cls, mock_model_cls):
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        load_model("some-model", dtype="float16")
        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert "torch_dtype" in call_kwargs

    @patch("xaytune.models.loader.AutoModelForCausalLM")
    @patch("xaytune.models.loader.AutoTokenizer")
    def test_load_with_trust_remote_code(self, mock_tokenizer_cls, mock_model_cls):
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        load_model("some-model", trust_remote_code=True)
        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert call_kwargs.get("trust_remote_code") is True

    def test_load_model_result_has_config(self):
        from xaytune.models.loader import ModelResult

        result = ModelResult(model=MagicMock(), tokenizer=MagicMock(), name="test")
        assert result.name == "test"

    @patch("xaytune.models.loader.AutoModelForCausalLM")
    @patch("xaytune.models.loader.AutoTokenizer")
    def test_load_with_quantization_4bit(self, mock_tokenizer_cls, mock_model_cls):
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        load_model("some-model", quantization="4bit")
        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert "quantization_config" in call_kwargs

    @patch("xaytune.models.loader.AutoModelForCausalLM")
    @patch("xaytune.models.loader.AutoTokenizer")
    def test_load_with_quantization_8bit(self, mock_tokenizer_cls, mock_model_cls):
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        load_model("some-model", quantization="8bit")
        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert "quantization_config" in call_kwargs


class TestRegistryModelLoading:
    def test_registry_loader_called(self):
        from xaytune.models.loader import ModelResult

        mock_result = ModelResult(
            model=MagicMock(),
            tokenizer=MagicMock(),
            name="my-arch",
        )

        @register_model("test-registry-arch", override=True)
        def my_loader(name_or_path, **kwargs):
            return mock_result

        result = load_model("test-registry-arch")
        assert result is mock_result

    @patch("xaytune.models.loader.AutoModelForCausalLM")
    @patch("xaytune.models.loader.AutoTokenizer")
    def test_unregistered_falls_through_to_hf(self, mock_tok, mock_model):
        mock_model.from_pretrained.return_value = MagicMock()
        mock_tok.from_pretrained.return_value = MagicMock()
        result = load_model("not-in-registry-xyz")
        mock_model.from_pretrained.assert_called_once()
        assert result.name == "not-in-registry-xyz"

    def test_registry_loader_receives_kwargs(self):
        from xaytune.models.loader import ModelResult

        received = {}

        @register_model("test-kwargs-arch", override=True)
        def my_loader(name_or_path, **kwargs):
            received.update(kwargs)
            return ModelResult(
                model=MagicMock(),
                tokenizer=MagicMock(),
                name=name_or_path,
            )

        load_model("test-kwargs-arch", dtype="bf16", quantization="4bit")
        assert received["dtype"] == "bf16"
        assert received["quantization"] == "4bit"
