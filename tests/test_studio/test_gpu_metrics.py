from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from xaytune.studio.gpu_metrics import get_gpu_metrics


class TestGetGpuMetrics:
    def test_returns_empty_on_cpu(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = get_gpu_metrics()
            assert result == {}

    def test_returns_metrics_when_cuda_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 100
        mock_torch.cuda.memory_reserved.return_value = 1024 * 1024 * 200
        mock_torch.cuda.max_memory_allocated.return_value = 1024 * 1024 * 150
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = get_gpu_metrics()
            assert result["gpu_memory_allocated_mb"] == 100.0
            assert result["gpu_memory_reserved_mb"] == 200.0
            assert result["gpu_memory_peak_mb"] == 150.0

    def test_returns_empty_on_exception(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("boom")
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = get_gpu_metrics()
            assert result == {}
