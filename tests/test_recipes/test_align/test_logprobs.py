import torch

from xaytune.recipes.align.logprobs import (
    get_model_logps,
    get_per_token_logps,
    get_sequence_logps,
)


class TestGetPerTokenLogps:
    def test_shape(self):
        logits = torch.randn(2, 5, 100)
        labels = torch.randint(0, 100, (2, 5))
        result = get_per_token_logps(logits, labels)
        assert result.shape == (2, 4)

    def test_values_are_log_probs(self):
        logits = torch.randn(1, 3, 10)
        labels = torch.randint(0, 10, (1, 3))
        result = get_per_token_logps(logits, labels)
        assert (result <= 0).all()


class TestGetSequenceLogps:
    def test_sums_per_token(self):
        logits = torch.randn(2, 5, 100)
        labels = torch.randint(0, 100, (2, 5))
        result = get_sequence_logps(logits, labels)
        assert result.shape == (2,)

    def test_mask_zeros_padding(self):
        logits = torch.randn(1, 5, 100)
        labels = torch.randint(0, 100, (1, 5))
        mask_all = torch.ones(1, 5)
        mask_partial = torch.tensor([[1, 1, 1, 0, 0]])

        full = get_sequence_logps(logits, labels, mask_all)
        partial = get_sequence_logps(logits, labels, mask_partial)
        assert partial.item() >= full.item()

    def test_no_mask_same_as_all_ones(self):
        logits = torch.randn(1, 4, 50)
        labels = torch.randint(0, 50, (1, 4))
        no_mask = get_sequence_logps(logits, labels, mask=None)
        ones_mask = get_sequence_logps(logits, labels, mask=torch.ones(1, 4))
        assert torch.allclose(no_mask, ones_mask)


class TestGetModelLogps:
    def test_with_mock_model(self):
        class FakeModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None):
                logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 100)
                return type("Out", (), {"logits": logits})()

        model = FakeModel()
        input_ids = torch.randint(0, 100, (2, 5))
        result = get_model_logps(model, input_ids)
        assert result.shape == (2,)
        assert (result <= 0).all()

    def test_uses_labels_when_provided(self):
        class FakeModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None):
                logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 100)
                return type("Out", (), {"logits": logits})()

        model = FakeModel()
        input_ids = torch.randint(0, 100, (1, 5))
        labels = torch.randint(0, 100, (1, 5))
        result = get_model_logps(model, input_ids, labels=labels)
        assert result.shape == (1,)
