import torch
import torch.nn as nn

from trainlib.trainer.lr_finder import LRFinderResult, lr_find


def _make_model():
    return nn.Linear(10, 1)


def _make_dataloader(num_batches=20):
    return [
        {"input": torch.randn(4, 10), "target": torch.randn(4, 1)}
        for _ in range(num_batches)
    ]


class _WrapperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        self.criterion = nn.MSELoss()

    def forward(self, input, target):
        pred = self.linear(input)
        loss = self.criterion(pred, target)
        return type("Out", (), {"loss": loss})()


class TestLRFinderResult:
    def test_to_dict(self):
        result = LRFinderResult(
            lrs=[1e-7, 1e-6], losses=[1.0, 0.9], suggested_lr=1e-6
        )
        d = result.to_dict()
        assert d["lrs"] == [1e-7, 1e-6]
        assert d["losses"] == [1.0, 0.9]
        assert d["suggested_lr"] == 1e-6

    def test_to_dict_none_suggested(self):
        result = LRFinderResult(lrs=[], losses=[], suggested_lr=None)
        assert result.to_dict()["suggested_lr"] is None


class TestLRFind:
    def test_returns_lr_finder_result(self):
        model = _WrapperModel()
        dl = _make_dataloader()
        result = lr_find(model, dl, num_iterations=10)
        assert isinstance(result, LRFinderResult)
        assert isinstance(result.lrs, list)
        assert isinstance(result.losses, list)

    def test_lrs_increase_exponentially(self):
        model = _WrapperModel()
        dl = _make_dataloader()
        result = lr_find(model, dl, num_iterations=10)
        for i in range(1, len(result.lrs)):
            assert result.lrs[i] > result.lrs[i - 1]

    def test_losses_recorded(self):
        model = _WrapperModel()
        dl = _make_dataloader()
        result = lr_find(model, dl, num_iterations=10)
        assert len(result.losses) == len(result.lrs)
        assert len(result.losses) > 0

    def test_model_state_restored(self):
        model = _WrapperModel()
        dl = _make_dataloader()
        params_before = {
            k: v.clone() for k, v in model.state_dict().items()
        }
        lr_find(model, dl, num_iterations=10)
        params_after = model.state_dict()
        for key in params_before:
            assert torch.equal(params_before[key], params_after[key])

    def test_suggested_lr_within_range(self):
        model = _WrapperModel()
        dl = _make_dataloader()
        result = lr_find(
            model, dl, start_lr=1e-6, end_lr=0.1, num_iterations=20
        )
        if result.suggested_lr is not None:
            assert 1e-6 <= result.suggested_lr <= 0.1

    def test_early_stopping_on_divergence(self):
        model = _WrapperModel()
        dl = _make_dataloader(50)
        result = lr_find(
            model,
            dl,
            start_lr=1e-7,
            end_lr=100.0,
            num_iterations=200,
            divergence_threshold=2.0,
        )
        assert len(result.lrs) < 200

    def test_custom_loss_fn(self):
        model = _WrapperModel()
        dl = _make_dataloader()
        calls = []

        def custom_loss(m, batch, outputs):
            calls.append(1)
            return outputs.loss

        result = lr_find(model, dl, num_iterations=5, loss_fn=custom_loss)
        assert len(calls) == len(result.lrs)

    def test_empty_dataloader_raises(self):
        model = _WrapperModel()
        try:
            lr_find(model, [], num_iterations=10)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "empty" in str(e).lower()

    def test_single_iteration(self):
        model = _WrapperModel()
        dl = _make_dataloader()
        result = lr_find(model, dl, num_iterations=1)
        assert len(result.lrs) >= 1
        assert result.suggested_lr is not None
