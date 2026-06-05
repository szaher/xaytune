import warnings
from unittest.mock import MagicMock

import pytest

from xaytune.logging.base import LoggingBackend, LoggingManager
from xaytune.logging.mlflow import _flatten_dict


class FailingBackend(LoggingBackend):
    def log_scalar(self, key: str, value: float, step: int) -> None:
        raise RuntimeError("Backend crashed")

    def log_config(self, config: dict) -> None:
        raise RuntimeError("Backend crashed")

    def close(self) -> None:
        pass


class TestLoggingExceptionIsolation:
    def test_log_scalar_backend_failure_does_not_crash(self):
        manager = LoggingManager()
        manager.add_backend(FailingBackend())

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            manager.log_scalar("loss", 0.5, 1)

        assert len(w) >= 1
        assert "failed" in str(w[0].message).lower()

    def test_log_config_backend_failure_does_not_crash(self):
        manager = LoggingManager()
        manager.add_backend(FailingBackend())

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            manager.log_config({"lr": 0.001})

        assert len(w) >= 1
        assert "failed" in str(w[0].message).lower()

    def test_failing_backend_does_not_block_others(self):
        manager = LoggingManager()
        failing = FailingBackend()
        healthy = MagicMock(spec=LoggingBackend)
        manager.add_backend(failing)
        manager.add_backend(healthy)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            manager.log_scalar("loss", 0.5, 1)

        healthy.log_scalar.assert_called_once_with("loss", 0.5, 1)

    def test_warning_includes_backend_name(self):
        manager = LoggingManager()
        manager.add_backend(FailingBackend())

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            manager.log_scalar("loss", 0.5, 1)

        assert "FailingBackend" in str(w[0].message)


class TestMlflowFlatten:
    def test_flat_dict_unchanged(self):
        result = _flatten_dict({"a": 1, "b": "x"})
        assert result == {"a": "1", "b": "x"}

    def test_nested_dict_flattened(self):
        result = _flatten_dict({"model": {"name": "llama"}})
        assert result == {"model.name": "llama"}

    def test_deeply_nested(self):
        result = _flatten_dict({"a": {"b": {"c": 42}}})
        assert result == {"a.b.c": "42"}

    def test_mixed_nesting(self):
        result = _flatten_dict({"lr": 0.001, "model": {"name": "x", "dtype": "fp16"}})
        assert result == {"lr": "0.001", "model.name": "x", "model.dtype": "fp16"}

    def test_empty_dict(self):
        result = _flatten_dict({})
        assert result == {}

    def test_custom_prefix(self):
        result = _flatten_dict({"a": 1}, prefix="config")
        assert result == {"config.a": "1"}


class TestImportGuards:
    def test_peft_import_error_message(self):
        from xaytune.models.peft import apply_lora
        from xaytune.models.loader import ModelResult

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([MagicMock()])
        mock_tokenizer = MagicMock()
        model_result = ModelResult(
            model=mock_model,
            tokenizer=mock_tokenizer,
            name="test",
            quantization=None,
            peft_applied=False,
            metadata={},
        )

        # If peft is not installed, apply_lora should raise ImportError with install hint
        # If peft IS installed, this test still verifies the guard exists in the source
        import xaytune.models.peft as peft_module

        if peft_module.get_peft_model is None:
            with pytest.raises(ImportError, match="pip install"):
                apply_lora(model_result)
        else:
            # peft is installed; verify the guard code path exists by checking source
            import inspect

            source = inspect.getsource(apply_lora)
            assert "pip install" in source

    def test_mlflow_import_guard(self):
        import xaytune.logging.mlflow as mlflow_mod

        if mlflow_mod.mlflow is None:
            with pytest.raises(ImportError, match="pip install"):
                from xaytune.logging.mlflow import MLflowBackend
                MLflowBackend()
        else:
            import inspect

            source = inspect.getsource(mlflow_mod.MLflowBackend.__init__)
            assert "pip install" in source
