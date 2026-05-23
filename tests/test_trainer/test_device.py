from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from trainlib.trainer.device import (
    detect_device_type_from_model,
    get_device,
    get_device_type,
    seed_all,
    supports_amp,
    supports_grad_scaler,
)


class TestGetDeviceType:
    @patch("trainlib.trainer.device.torch")
    def test_cuda_preferred(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        assert get_device_type() == "cuda"

    @patch("trainlib.trainer.device.torch")
    def test_mps_fallback(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        assert get_device_type() == "mps"

    @patch("trainlib.trainer.device.torch")
    def test_cpu_fallback(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        assert get_device_type() == "cpu"


class TestGetDevice:
    def test_explicit_cuda(self):
        dev = get_device(2, device_type="cuda")
        assert dev == torch.device("cuda:2")

    def test_explicit_mps(self):
        dev = get_device(0, device_type="mps")
        assert dev == torch.device("mps")

    def test_explicit_cpu(self):
        dev = get_device(0, device_type="cpu")
        assert dev == torch.device("cpu")


class TestSeedAll:
    @patch("trainlib.trainer.device.torch")
    @patch("trainlib.trainer.device.random")
    def test_seeds_random_and_torch(self, mock_random, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        seed_all(42)

        mock_random.seed.assert_called_once_with(42)
        mock_torch.manual_seed.assert_called_once_with(42)

    @patch("trainlib.trainer.device.torch")
    @patch("trainlib.trainer.device.random")
    def test_seeds_cuda_when_available(self, mock_random, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = False

        seed_all(42)

        mock_torch.cuda.manual_seed_all.assert_called_once_with(42)

    @patch("trainlib.trainer.device.torch")
    @patch("trainlib.trainer.device.random")
    def test_seeds_mps_when_available(self, mock_random, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        seed_all(42)

        mock_torch.mps.manual_seed.assert_called_once_with(42)


class TestSupportsAmp:
    def test_cuda_supports_amp(self):
        assert supports_amp("cuda") is True

    def test_mps_supports_amp(self):
        assert supports_amp("mps") is True

    def test_cpu_no_amp(self):
        assert supports_amp("cpu") is False


class TestSupportsGradScaler:
    def test_cuda_fp16(self):
        assert supports_grad_scaler("cuda", torch.float16) is True

    def test_cuda_bf16(self):
        assert supports_grad_scaler("cuda", torch.bfloat16) is False

    def test_mps_fp16(self):
        assert supports_grad_scaler("mps", torch.float16) is False

    def test_cpu(self):
        assert supports_grad_scaler("cpu", torch.float16) is False


class TestDetectDeviceType:
    def test_cuda_model(self):
        model = MagicMock()
        param = MagicMock()
        param.is_cuda = True
        model.parameters.return_value = iter([param])
        assert detect_device_type_from_model(model) == "cuda"

    def test_mps_model(self):
        model = MagicMock()
        param = MagicMock()
        param.is_cuda = False
        param.is_mps = True
        model.parameters.return_value = iter([param])
        assert detect_device_type_from_model(model) == "mps"

    def test_cpu_model(self):
        model = MagicMock()
        param = MagicMock()
        param.is_cuda = False
        param.is_mps = False
        model.parameters.return_value = iter([param])
        assert detect_device_type_from_model(model) == "cpu"

    def test_empty_model(self):
        model = MagicMock()
        model.parameters.return_value = iter([])
        assert detect_device_type_from_model(model) == "cpu"
