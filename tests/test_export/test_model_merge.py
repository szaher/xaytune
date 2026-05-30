import torch
import pytest
from unittest.mock import patch, MagicMock
from importlib import import_module

import xaytune.export.model_merge as mm_module
from xaytune.export.model_merge import MergeResult, _linear_merge, _slerp_merge, _slerp_tensor, _ties_merge, _dare_merge, model_merge

# Ensure we have the actual module, not the function
mm_module = import_module('xaytune.export.model_merge')


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


class TestDareMerge:
    def test_basic(self):
        base = {"w": torch.tensor([0.0, 0.0, 0.0, 0.0])}
        sd_a = {"w": torch.tensor([1.0, 2.0, 3.0, 4.0])}
        sd_b = {"w": torch.tensor([2.0, 3.0, 4.0, 5.0])}
        merged = _dare_merge([sd_a, sd_b], base, density=1.0, weight=1.0, seed=42)
        assert "w" in merged
        assert merged["w"].shape == torch.Size([4])

    def test_density_drops_elements(self):
        torch.manual_seed(0)
        base = {"w": torch.zeros(100)}
        sd_a = {"w": torch.ones(100)}
        merged = _dare_merge([sd_a], base, density=0.5, weight=1.0, seed=42)
        task_vec = merged["w"] - base["w"]
        nonzero = (task_vec.abs() > 1e-6).sum().item()
        assert 30 < nonzero < 70

    def test_rescaling_preserves_magnitude(self):
        base = {"w": torch.zeros(1000)}
        sd_a = {"w": torch.ones(1000)}
        merged_full = _dare_merge([sd_a], base, density=1.0, weight=1.0, seed=42)
        merged_half = _dare_merge([sd_a], base, density=0.5, weight=1.0, seed=42)
        mean_full = (merged_full["w"] - base["w"]).mean()
        mean_half = (merged_half["w"] - base["w"]).mean()
        assert abs(mean_full.item() - mean_half.item()) < 0.15

    def test_deterministic_with_seed(self):
        base = {"w": torch.zeros(50)}
        sd_a = {"w": torch.randn(50)}
        merged_1 = _dare_merge([sd_a], base, density=0.5, weight=1.0, seed=123)
        merged_2 = _dare_merge([sd_a], base, density=0.5, weight=1.0, seed=123)
        assert torch.allclose(merged_1["w"], merged_2["w"])

    def test_weight_scales_result(self):
        base = {"w": torch.tensor([0.0, 0.0])}
        sd_a = {"w": torch.tensor([1.0, 1.0])}
        merged_1 = _dare_merge([sd_a], base, density=1.0, weight=1.0, seed=42)
        merged_2 = _dare_merge([sd_a], base, density=1.0, weight=2.0, seed=42)
        diff_1 = merged_1["w"] - base["w"]
        diff_2 = merged_2["w"] - base["w"]
        assert torch.allclose(diff_2, 2 * diff_1, atol=1e-6)

    def test_result_includes_base(self):
        base = {"w": torch.tensor([10.0, 20.0])}
        sd_a = {"w": torch.tensor([11.0, 21.0])}
        merged = _dare_merge([sd_a], base, density=1.0, weight=1.0, seed=42)
        assert torch.allclose(merged["w"], sd_a["w"], atol=1e-6)


class TestValidation:
    def test_slerp_requires_two_models(self):
        with pytest.raises(ValueError, match="exactly 2"):
            model_merge(models=["a", "b", "c"], method="slerp", output="/tmp/out")

    def test_ties_requires_base_model(self):
        with pytest.raises(ValueError, match="base_model"):
            model_merge(models=["a", "b"], method="ties", output="/tmp/out")

    def test_dare_requires_base_model(self):
        with pytest.raises(ValueError, match="base_model"):
            model_merge(models=["a", "b"], method="dare", output="/tmp/out")

    def test_weights_length_mismatch(self):
        with pytest.raises(ValueError, match="weights"):
            model_merge(models=["a", "b"], method="linear", output="/tmp/out", weights=[0.5])

    def test_density_out_of_range(self):
        with pytest.raises(ValueError, match="density"):
            model_merge(models=["a", "b"], method="ties", output="/tmp/out", base_model="base", density=1.5)

    def test_t_out_of_range(self):
        with pytest.raises(ValueError, match="t"):
            model_merge(models=["a", "b"], method="slerp", output="/tmp/out", t=2.0)

    def test_mismatched_keys(self):
        sd_a = {"layer1.w": torch.tensor([1.0])}
        sd_b = {"layer2.w": torch.tensor([1.0])}
        with patch.object(mm_module, "_load_state_dict", side_effect=[sd_a, sd_b]):
            with patch.object(mm_module, "_load_tokenizer", return_value=MagicMock()):
                with pytest.raises(ValueError, match="keys"):
                    model_merge(models=["a", "b"], method="linear", output="/tmp/out")


class TestModelMergeDispatch:
    def _make_sd(self):
        return {"layer.weight": torch.randn(4, 4), "layer.bias": torch.randn(4)}

    @patch.object(mm_module, "_save_merged")
    @patch.object(mm_module, "_load_tokenizer")
    @patch.object(mm_module, "_load_state_dict")
    def test_linear_dispatch(self, mock_load, mock_tok, mock_save):
        mock_load.side_effect = [self._make_sd(), self._make_sd()]
        mock_tok.return_value = MagicMock()
        result = model_merge(models=["a", "b"], method="linear", output="/tmp/out")
        assert result.method == "linear"
        mock_save.assert_called_once()

    @patch.object(mm_module, "_save_merged")
    @patch.object(mm_module, "_load_tokenizer")
    @patch.object(mm_module, "_load_state_dict")
    def test_slerp_dispatch(self, mock_load, mock_tok, mock_save):
        mock_load.side_effect = [self._make_sd(), self._make_sd()]
        mock_tok.return_value = MagicMock()
        result = model_merge(models=["a", "b"], method="slerp", output="/tmp/out", t=0.3)
        assert result.method == "slerp"
        assert result.params["t"] == 0.3

    @patch.object(mm_module, "_save_merged")
    @patch.object(mm_module, "_load_tokenizer")
    @patch.object(mm_module, "_load_state_dict")
    def test_ties_dispatch(self, mock_load, mock_tok, mock_save):
        base_sd = self._make_sd()
        mock_load.side_effect = [self._make_sd(), self._make_sd(), base_sd]
        mock_tok.return_value = MagicMock()
        result = model_merge(models=["a", "b"], method="ties", output="/tmp/out", base_model="base", density=0.7)
        assert result.method == "ties"
        assert result.params["density"] == 0.7

    @patch.object(mm_module, "_save_merged")
    @patch.object(mm_module, "_load_tokenizer")
    @patch.object(mm_module, "_load_state_dict")
    def test_dare_dispatch(self, mock_load, mock_tok, mock_save):
        base_sd = self._make_sd()
        mock_load.side_effect = [self._make_sd(), self._make_sd(), base_sd]
        mock_tok.return_value = MagicMock()
        result = model_merge(models=["a", "b"], method="dare", output="/tmp/out", base_model="base")
        assert result.method == "dare"
