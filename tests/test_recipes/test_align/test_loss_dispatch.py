import torch

from trainlib.recipes.align.loss_dispatch import (
    create_alignment_loss_fn,
    is_alignment_method,
)


class TestIsAlignmentMethod:
    def test_alignment_methods(self):
        for method in ("dpo", "grpo", "ppo", "orpo", "simpo", "reinforce"):
            assert is_alignment_method(method)

    def test_non_alignment_methods(self):
        for method in ("full", "lora", "qlora"):
            assert not is_alignment_method(method)


class _FakeOutput:
    def __init__(self, batch_size, seq_len, vocab_size=100):
        self.logits = torch.randn(batch_size, seq_len, vocab_size)
        self.loss = torch.tensor(1.0, requires_grad=True)


class _FakeModel(torch.nn.Module):
    def __init__(self, vocab_size=100):
        super().__init__()
        self._vocab_size = vocab_size
        self._dummy = torch.nn.Linear(1, 1)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        bs, seq_len = input_ids.shape
        return _FakeOutput(bs, seq_len, self._vocab_size)


def _make_dpo_batch():
    return {
        "chosen_input_ids": torch.randint(0, 100, (1, 5)),
        "chosen_attention_mask": torch.ones(1, 5),
        "rejected_input_ids": torch.randint(0, 100, (1, 5)),
        "rejected_attention_mask": torch.ones(1, 5),
    }


def _make_grpo_batch():
    return {
        "input_ids": torch.randint(0, 100, (1, 5)),
        "attention_mask": torch.ones(1, 5),
        "advantages": torch.tensor([1.0]),
    }


class TestCreateAlignmentLossFn:
    def test_dpo_creates_loss(self):
        model = _FakeModel()
        ref_model = _FakeModel()
        loss_fn = create_alignment_loss_fn(
            method="dpo", ref_model=ref_model
        )
        loss = loss_fn(model, _make_dpo_batch(), None)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_grpo_creates_loss(self):
        model = _FakeModel()
        ref_model = _FakeModel()
        loss_fn = create_alignment_loss_fn(
            method="grpo", ref_model=ref_model
        )
        loss = loss_fn(model, _make_grpo_batch(), None)
        assert isinstance(loss, torch.Tensor)

    def test_orpo_creates_loss(self):
        model = _FakeModel()
        loss_fn = create_alignment_loss_fn(method="orpo")
        batch = _make_dpo_batch()
        outputs = _FakeOutput(1, 5)
        loss = loss_fn(model, batch, outputs)
        assert isinstance(loss, torch.Tensor)

    def test_simpo_creates_loss(self):
        model = _FakeModel()
        loss_fn = create_alignment_loss_fn(method="simpo")
        loss = loss_fn(model, _make_dpo_batch(), None)
        assert isinstance(loss, torch.Tensor)

    def test_ppo_creates_loss(self):
        model = _FakeModel()
        loss_fn = create_alignment_loss_fn(method="ppo")
        batch = {
            "input_ids": torch.randint(0, 100, (1, 5)),
            "attention_mask": torch.ones(1, 5),
            "old_logprobs": torch.tensor([-2.0]),
            "advantages": torch.tensor([1.0]),
        }
        loss = loss_fn(model, batch, None)
        assert isinstance(loss, torch.Tensor)

    def test_reinforce_creates_loss(self):
        model = _FakeModel()
        loss_fn = create_alignment_loss_fn(method="reinforce")
        loss = loss_fn(model, _make_grpo_batch(), None)
        assert isinstance(loss, torch.Tensor)

    def test_unknown_method_falls_back_to_model_loss(self):
        loss_fn = create_alignment_loss_fn(method="unknown")
        outputs = _FakeOutput(1, 5)
        loss = loss_fn(None, {}, outputs)
        assert loss is outputs.loss

    def test_sft_batch_falls_back_to_model_loss(self):
        loss_fn = create_alignment_loss_fn(method="dpo", ref_model=_FakeModel())
        outputs = _FakeOutput(1, 5)
        sft_batch = {"input_ids": torch.randint(0, 100, (1, 5))}
        loss = loss_fn(None, sft_batch, outputs)
        assert loss is outputs.loss
