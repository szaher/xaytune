import torch
import pytest

from xaytune.export.model_merge import MergeResult, _linear_merge, _slerp_merge, _slerp_tensor, _ties_merge


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


class TestSlerpTensor:
    def test_t_zero_returns_a(self):
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        result = _slerp_tensor(a, b, t=0.0)
        assert torch.allclose(result, a, atol=1e-6)

    def test_t_one_returns_b(self):
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        result = _slerp_tensor(a, b, t=1.0)
        assert torch.allclose(result, b, atol=1e-6)

    def test_t_half_is_midpoint(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        result = _slerp_tensor(a, b, t=0.5)
        assert torch.allclose(result[0], result[1], atol=1e-6)
        assert result.norm() > 0

    def test_parallel_vectors_fallback(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([2.0, 4.0, 6.0])
        result = _slerp_tensor(a, b, t=0.5)
        expected = 0.5 * a + 0.5 * b
        assert torch.allclose(result, expected, atol=1e-6)

    def test_1d_bias_fallback(self):
        a = torch.tensor([1.0])
        b = torch.tensor([3.0])
        result = _slerp_tensor(a, b, t=0.5)
        assert torch.allclose(result, torch.tensor([2.0]), atol=1e-6)


class TestSlerpMerge:
    def test_basic(self):
        sd_a = {"w": torch.tensor([1.0, 0.0, 0.0]), "b": torch.tensor([0.0])}
        sd_b = {"w": torch.tensor([0.0, 1.0, 0.0]), "b": torch.tensor([2.0])}
        merged = _slerp_merge(sd_a, sd_b, t=0.0)
        assert torch.allclose(merged["w"], sd_a["w"], atol=1e-6)

    def test_t_one(self):
        sd_a = {"w": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0])}
        sd_b = {"w": torch.tensor([0.0, 1.0]), "b": torch.tensor([2.0])}
        merged = _slerp_merge(sd_a, sd_b, t=1.0)
        assert torch.allclose(merged["w"], sd_b["w"], atol=1e-6)
        assert torch.allclose(merged["b"], sd_b["b"], atol=1e-6)

    def test_preserves_all_keys(self):
        sd_a = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0]), "c": torch.tensor([3.0])}
        sd_b = {"a": torch.tensor([4.0]), "b": torch.tensor([5.0]), "c": torch.tensor([6.0])}
        merged = _slerp_merge(sd_a, sd_b, t=0.5)
        assert set(merged.keys()) == {"a", "b", "c"}


class TestTiesMerge:
    def test_basic_with_known_values(self):
        base = {"w": torch.tensor([0.0, 0.0, 0.0, 0.0])}
        sd_a = {"w": torch.tensor([1.0, -0.1, 0.5, -2.0])}
        sd_b = {"w": torch.tensor([0.8, 0.1, -0.5, -1.5])}
        merged = _ties_merge([sd_a, sd_b], base, density=1.0, weight=1.0)
        assert "w" in merged
        assert merged["w"].shape == torch.Size([4])

    def test_density_trims_values(self):
        base = {"w": torch.zeros(10)}
        sd_a = {"w": torch.randn(10)}
        sd_b = {"w": torch.randn(10)}
        merged_full = _ties_merge([sd_a, sd_b], base, density=1.0, weight=1.0)
        merged_sparse = _ties_merge([sd_a, sd_b], base, density=0.3, weight=1.0)
        assert not torch.allclose(merged_full["w"], merged_sparse["w"])

    def test_weight_scales_result(self):
        base = {"w": torch.tensor([0.0, 0.0])}
        sd_a = {"w": torch.tensor([2.0, 2.0])}
        sd_b = {"w": torch.tensor([2.0, 2.0])}
        merged_1 = _ties_merge([sd_a, sd_b], base, density=1.0, weight=1.0)
        merged_2 = _ties_merge([sd_a, sd_b], base, density=1.0, weight=2.0)
        ratio = merged_2["w"] / merged_1["w"]
        assert torch.allclose(ratio, torch.tensor([2.0, 2.0]), atol=1e-6)

    def test_result_includes_base(self):
        base = {"w": torch.tensor([10.0, 20.0])}
        sd_a = {"w": torch.tensor([11.0, 21.0])}
        merged = _ties_merge([sd_a], base, density=1.0, weight=1.0)
        assert merged["w"][0] > 10.0

    def test_sign_election(self):
        base = {"w": torch.tensor([0.0, 0.0])}
        sd_a = {"w": torch.tensor([1.0, -1.0])}
        sd_b = {"w": torch.tensor([2.0, -2.0])}
        sd_c = {"w": torch.tensor([-0.5, 0.5])}
        merged = _ties_merge([sd_a, sd_b, sd_c], base, density=1.0, weight=1.0)
        assert merged["w"][0] > 0
        assert merged["w"][1] < 0
