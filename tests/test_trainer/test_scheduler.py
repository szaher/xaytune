from __future__ import annotations

import pytest
import torch

from xaytune.trainer.scheduler import create_scheduler, resolve_warmup_steps


def _make_optimizer(lr: float = 1.0) -> torch.optim.Optimizer:
    param = torch.randn(2, 2, requires_grad=True)
    return torch.optim.SGD([param], lr=lr)


class TestResolveWarmupSteps:
    def test_explicit_steps_takes_precedence(self):
        assert resolve_warmup_steps(warmup_steps=20, warmup_ratio=0.0, total_steps=100) == 20

    def test_ratio_converts_to_steps(self):
        assert resolve_warmup_steps(warmup_steps=0, warmup_ratio=0.1, total_steps=100) == 10

    def test_both_zero_returns_zero(self):
        assert resolve_warmup_steps(warmup_steps=0, warmup_ratio=0.0, total_steps=100) == 0

    def test_steps_preferred_over_ratio(self):
        assert resolve_warmup_steps(warmup_steps=5, warmup_ratio=0.5, total_steps=100) == 5


class TestCreateScheduler:
    def test_cosine_with_warmup(self):
        opt = _make_optimizer(lr=1.0)
        sched = create_scheduler(opt, "cosine", total_steps=10, warmup_steps=2)

        lrs = []
        for _ in range(10):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()

        assert lrs[0] < 0.01
        assert lrs[2] > 0.9
        assert lrs[-1] < lrs[2]

    def test_cosine_no_warmup(self):
        opt = _make_optimizer(lr=1.0)
        sched = create_scheduler(opt, "cosine", total_steps=10, warmup_steps=0)

        lrs = []
        for _ in range(10):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()

        assert lrs[0] == pytest.approx(1.0)
        assert lrs[-1] < lrs[0]

    def test_linear_with_warmup(self):
        opt = _make_optimizer(lr=1.0)
        sched = create_scheduler(opt, "linear", total_steps=10, warmup_steps=2)

        lrs = []
        for _ in range(10):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()

        assert lrs[0] < 0.01
        assert lrs[2] > 0.9
        assert lrs[-1] < lrs[2]
        for i in range(3, len(lrs)):
            assert lrs[i] <= lrs[i - 1]

    def test_linear_no_warmup(self):
        opt = _make_optimizer(lr=1.0)
        sched = create_scheduler(opt, "linear", total_steps=10, warmup_steps=0)

        lrs = []
        for _ in range(10):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()

        assert lrs[0] == pytest.approx(1.0)
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1]

    def test_constant(self):
        opt = _make_optimizer(lr=0.5)
        sched = create_scheduler(opt, "constant", total_steps=10, warmup_steps=0)

        for _ in range(10):
            assert opt.param_groups[0]["lr"] == pytest.approx(0.5)
            opt.step()
            sched.step()

    def test_constant_ignores_warmup_steps(self):
        opt = _make_optimizer(lr=0.5)
        create_scheduler(opt, "constant", total_steps=10, warmup_steps=5)

        assert opt.param_groups[0]["lr"] == pytest.approx(0.5)

    def test_constant_with_warmup(self):
        opt = _make_optimizer(lr=1.0)
        sched = create_scheduler(opt, "constant_with_warmup", total_steps=10, warmup_steps=4)

        lrs = []
        for _ in range(10):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()

        assert lrs[0] < 0.01
        assert lrs[3] < 1.0
        assert lrs[4] == pytest.approx(1.0)
        assert lrs[-1] == pytest.approx(1.0)

    def test_invalid_type_raises(self):
        opt = _make_optimizer()
        with pytest.raises(ValueError, match="Unknown scheduler type"):
            create_scheduler(opt, "invalid", total_steps=10, warmup_steps=0)
