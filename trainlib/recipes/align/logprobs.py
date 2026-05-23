from __future__ import annotations

import torch
import torch.nn.functional as F


def get_per_token_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target = labels[:, 1:]
    return torch.gather(log_probs, dim=2, index=target.unsqueeze(2)).squeeze(2)


def get_sequence_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    per_token = get_per_token_logps(logits, labels)
    if mask is not None:
        per_token = per_token * mask[:, 1:]
    return per_token.sum(dim=-1)


def get_model_logps(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    if labels is None:
        labels = input_ids
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return get_sequence_logps(outputs.logits, labels, attention_mask)
