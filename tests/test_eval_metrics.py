import math

from xaytune.eval.metrics import compute_loss, compute_perplexity, compute_token_accuracy


class TestComputeLoss:
    def test_mean_of_values(self):
        assert compute_loss([1.0, 2.0, 3.0]) == 2.0

    def test_single_value(self):
        assert compute_loss([5.0]) == 5.0

    def test_empty_returns_zero(self):
        assert compute_loss([]) == 0.0

    def test_identical_values(self):
        assert compute_loss([0.5, 0.5, 0.5]) == 0.5


class TestComputePerplexity:
    def test_exp_of_mean_loss(self):
        result = compute_perplexity([1.0])
        assert abs(result - math.e) < 0.01

    def test_multiple_losses(self):
        result = compute_perplexity([1.0, 2.0, 3.0])
        assert abs(result - math.exp(2.0)) < 0.01

    def test_empty_returns_zero(self):
        assert compute_perplexity([]) == 0.0

    def test_zero_loss(self):
        result = compute_perplexity([0.0])
        assert abs(result - 1.0) < 0.01


class TestComputeTokenAccuracy:
    def test_perfect_match(self):
        assert compute_token_accuracy([1, 2, 3], [1, 2, 3]) == 1.0

    def test_no_match(self):
        assert compute_token_accuracy([1, 2, 3], [4, 5, 6]) == 0.0

    def test_partial_match(self):
        result = compute_token_accuracy([1, 2, 3], [1, 5, 3])
        assert abs(result - 2 / 3) < 0.01

    def test_empty_returns_zero(self):
        assert compute_token_accuracy([], []) == 0.0

    def test_single_element_match(self):
        assert compute_token_accuracy([42], [42]) == 1.0

    def test_single_element_mismatch(self):
        assert compute_token_accuracy([42], [99]) == 0.0

    def test_large_ids(self):
        preds = [30000, 30001, 30002]
        refs = [30000, 30001, 30002]
        assert compute_token_accuracy(preds, refs) == 1.0
