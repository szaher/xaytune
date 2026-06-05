"""Value head for PPO — projects LM hidden states to scalar values."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """Linear projection from the last hidden state to a scalar value estimate.

    Extracts the hidden state of the last non-padding token per sequence
    and projects it to a single value via a linear layer.

    Args:
        hidden_size: Dimension of the LM's hidden states.
        dropout: Dropout probability before the linear layer.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.linear.bias)
        nn.init.normal_(self.linear.weight, std=0.01)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sequence values from hidden states.

        Args:
            hidden_states: ``(batch, seq_len, hidden_size)`` from the LM.
            attention_mask: ``(batch, seq_len)`` with 1s for real tokens.

        Returns:
            Values tensor of shape ``(batch,)``.
        """
        seq_lens = attention_mask.sum(dim=1).long() - 1
        seq_lens = seq_lens.clamp(min=0)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_idx, seq_lens]
        return self.linear(self.dropout(last_hidden)).squeeze(-1)


def get_values(
    model: Any,
    value_head: ValueHead,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Forward pass through model + value head to get per-sequence values."""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    hidden = outputs.hidden_states[-1]
    return value_head(hidden, attention_mask)
