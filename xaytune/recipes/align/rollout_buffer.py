"""Rollout buffer for PPO multi-epoch training over collected trajectories."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch


@dataclass
class Rollout:
    """A batch of collected PPO trajectories.

    All tensors have the same first dimension (number of rollout samples).
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    old_logprobs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    prompt_lengths: torch.Tensor

    def __len__(self) -> int:
        return self.input_ids.size(0)


class RolloutBuffer:
    """Stores a single rollout and yields shuffled mini-batches for PPO training.

    Usage::

        buffer = RolloutBuffer()
        buffer.store(rollout)
        for epoch in range(ppo_epochs):
            for batch in buffer.iterate(mini_batch_size=8):
                loss = train_step(batch)
        buffer.clear()
    """

    def __init__(self) -> None:
        self._rollout: Rollout | None = None

    @property
    def size(self) -> int:
        return len(self._rollout) if self._rollout is not None else 0

    def store(self, rollout: Rollout) -> None:
        self._rollout = rollout

    def iterate(
        self,
        mini_batch_size: int,
        shuffle: bool = True,
    ) -> list[dict[str, torch.Tensor]]:
        if self._rollout is None:
            return []

        n = len(self._rollout)
        if shuffle:
            indices = torch.randperm(n)
        else:
            indices = torch.arange(n)

        field_names = [f.name for f in fields(Rollout)]
        batches: list[dict[str, torch.Tensor]] = []

        for start in range(0, n, mini_batch_size):
            batch_idx = indices[start : start + mini_batch_size]
            batch: dict[str, Any] = {}
            for name in field_names:
                tensor = getattr(self._rollout, name)
                batch[name] = tensor[batch_idx]
            batches.append(batch)

        return batches

    def clear(self) -> None:
        self._rollout = None
