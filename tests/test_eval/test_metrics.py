import pytest
import torch
from trainlib.eval.metrics import metric_registry, register_metric


class TestMetricRegistry:
    def test_register_and_get(self):
        @register_metric("test-metric")
        def my_metric(predictions, references) -> float:
            return 1.0

        assert metric_registry.has("test-metric")
        fn = metric_registry.get("test-metric")
        assert fn([], []) == 1.0

    def test_register_returns_original(self):
        @register_metric("identity-metric")
        def my_fn(predictions, references) -> float:
            return 0.5

        assert my_fn([], []) == 0.5

    def test_unknown_metric_raises(self):
        with pytest.raises(KeyError, match="not found"):
            metric_registry.get("nonexistent-metric")

    def test_list_metrics(self):
        metrics = metric_registry.list()
        assert "loss" in metrics
        assert "perplexity" in metrics
        assert "token_accuracy" in metrics


class TestBuiltinMetrics:
    def test_loss_metric(self):
        compute_loss = metric_registry.get("loss")
        losses = [0.5, 0.3, 0.4]
        result = compute_loss(losses)
        assert abs(result - 0.4) < 1e-5

    def test_loss_metric_empty(self):
        compute_loss = metric_registry.get("loss")
        result = compute_loss([])
        assert result == 0.0

    def test_perplexity_metric(self):
        compute_ppl = metric_registry.get("perplexity")
        losses = [1.0, 2.0, 3.0]
        result = compute_ppl(losses)
        import math
        assert abs(result - math.exp(2.0)) < 0.01

    def test_perplexity_empty(self):
        compute_ppl = metric_registry.get("perplexity")
        result = compute_ppl([])
        assert result == 0.0

    def test_token_accuracy(self):
        compute_acc = metric_registry.get("token_accuracy")
        predictions = [1, 2, 3, 4, 5]
        references = [1, 2, 0, 4, 0]
        result = compute_acc(predictions, references)
        assert abs(result - 0.6) < 1e-5

    def test_token_accuracy_perfect(self):
        compute_acc = metric_registry.get("token_accuracy")
        predictions = [1, 2, 3]
        references = [1, 2, 3]
        result = compute_acc(predictions, references)
        assert result == 1.0

    def test_token_accuracy_empty(self):
        compute_acc = metric_registry.get("token_accuracy")
        result = compute_acc([], [])
        assert result == 0.0
