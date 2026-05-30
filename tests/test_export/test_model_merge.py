import torch
import pytest

from xaytune.export.model_merge import MergeResult, _linear_merge


class TestMergeResult:
    def test_create(self):
        result = MergeResult(
            output_path="/tmp/merged",
            method="linear",
            models=["model-a", "model-b"],
            params={"weights": [0.5, 0.5]},
        )
        assert result.output_path == "/tmp/merged"
        assert result.method == "linear"
        assert len(result.models) == 2

    def test_summary(self):
        result = MergeResult(
            output_path="/tmp/merged",
            method="slerp",
            models=["model-a", "model-b"],
            params={"t": 0.6},
        )
        text = result.summary()
        assert "slerp" in text
        assert "model-a" in text


class TestLinearMerge:
    def test_two_models_equal_weights(self):
        sd_a = {"layer.weight": torch.tensor([1.0, 2.0, 3.0])}
        sd_b = {"layer.weight": torch.tensor([3.0, 4.0, 5.0])}
        merged = _linear_merge([sd_a, sd_b], weights=[0.5, 0.5])
        expected = torch.tensor([2.0, 3.0, 4.0])
        assert torch.allclose(merged["layer.weight"], expected)

    def test_two_models_unequal_weights(self):
        sd_a = {"w": torch.tensor([10.0, 0.0])}
        sd_b = {"w": torch.tensor([0.0, 10.0])}
        merged = _linear_merge([sd_a, sd_b], weights=[0.7, 0.3])
        expected = torch.tensor([7.0, 3.0])
        assert torch.allclose(merged["w"], expected)

    def test_three_models(self):
        sd_a = {"w": torch.tensor([1.0])}
        sd_b = {"w": torch.tensor([2.0])}
        sd_c = {"w": torch.tensor([3.0])}
        merged = _linear_merge([sd_a, sd_b, sd_c], weights=[1 / 3, 1 / 3, 1 / 3])
        assert torch.allclose(merged["w"], torch.tensor([2.0]), atol=1e-6)

    def test_multiple_keys(self):
        sd_a = {"a": torch.tensor([1.0]), "b": torch.tensor([10.0])}
        sd_b = {"a": torch.tensor([3.0]), "b": torch.tensor([20.0])}
        merged = _linear_merge([sd_a, sd_b], weights=[0.5, 0.5])
        assert torch.allclose(merged["a"], torch.tensor([2.0]))
        assert torch.allclose(merged["b"], torch.tensor([15.0]))

    def test_2d_tensors(self):
        sd_a = {"w": torch.zeros(3, 4)}
        sd_b = {"w": torch.ones(3, 4)}
        merged = _linear_merge([sd_a, sd_b], weights=[0.5, 0.5])
        assert torch.allclose(merged["w"], torch.full((3, 4), 0.5))
