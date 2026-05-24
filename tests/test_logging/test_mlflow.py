from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Mock mlflow before any xaytune imports
mock_mlflow = MagicMock()
mock_mlflow.__spec__ = MagicMock()
sys.modules["mlflow"] = mock_mlflow


class TestMLflowBackend:
    def test_init_starts_run(self):
        import mlflow

        from xaytune.logging.mlflow import MLflowBackend

        mlflow.reset_mock()
        MLflowBackend(run_name="test-run")
        mlflow.start_run.assert_called_once_with(run_name="test-run")

    def test_log_scalar(self):
        import mlflow

        from xaytune.logging.mlflow import MLflowBackend

        mlflow.reset_mock()
        backend = MLflowBackend()
        backend.log_scalar("loss", 0.5, 10)

        mlflow.log_metric.assert_called_once_with("loss", 0.5, step=10)

    def test_log_config(self):
        import mlflow

        from xaytune.logging.mlflow import MLflowBackend

        mlflow.reset_mock()
        backend = MLflowBackend()
        backend.log_config({"lr": 0.001, "epochs": 3})

        mlflow.log_params.assert_called_once_with({"lr": 0.001, "epochs": 3})

    def test_close_ends_run(self):
        import mlflow

        from xaytune.logging.mlflow import MLflowBackend

        mlflow.reset_mock()
        backend = MLflowBackend()
        backend.close()

        mlflow.end_run.assert_called_once()

    def test_is_logging_backend(self):
        from xaytune.logging.base import LoggingBackend
        from xaytune.logging.mlflow import MLflowBackend

        backend = MLflowBackend()
        assert isinstance(backend, LoggingBackend)

    def test_default_run_name(self):
        import mlflow

        from xaytune.logging.mlflow import MLflowBackend

        mlflow.reset_mock()
        MLflowBackend()
        mlflow.start_run.assert_called_once_with(run_name=None)
