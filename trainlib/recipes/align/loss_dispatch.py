from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from trainlib.recipes.align.dpo import dpo_loss
from trainlib.recipes.align.grpo import grpo_loss
from trainlib.recipes.align.logprobs import get_sequence_logps
from trainlib.recipes.align.orpo import orpo_loss
from trainlib.recipes.align.ppo import ppo_clip_loss, reinforce_loss
from trainlib.recipes.align.simpo import simpo_loss

ALIGNMENT_METHODS = {"dpo", "grpo", "ppo", "orpo", "simpo", "reinforce"}

_PAIR_METHODS = {"dpo", "orpo", "simpo"}
_RL_METHODS = {"grpo", "ppo", "reinforce"}


def _has_alignment_fields(method: str, batch: dict[str, Any]) -> bool:
    if method in _PAIR_METHODS:
        return "chosen_input_ids" in batch and "rejected_input_ids" in batch
    if method in _RL_METHODS:
        return "advantages" in batch
    return False


def is_alignment_method(method: str) -> bool:
    """Return whether *method* is a known alignment method."""
    return method in ALIGNMENT_METHODS


def create_alignment_loss_fn(
    *,
    method: str,
    ref_model: Any | None = None,
    beta: float = 0.1,
    kl_coeff: float = 0.04,
    lambda_weight: float = 1.0,
    gamma: float = 0.5,
    clip_eps: float = 0.2,
) -> Callable[..., torch.Tensor]:
    """Create a loss function for the given alignment method.

    Returns a callable ``(model, batch, outputs) -> loss`` that handles
    forward passes on chosen/rejected pairs and reference model inference.
    """
    def loss_fn(
        model: Any,
        batch: dict[str, Any],
        outputs: Any,
    ) -> torch.Tensor:
        if not _has_alignment_fields(method, batch):
            return outputs.loss if hasattr(outputs, "loss") else outputs

        if method == "dpo":
            return _dpo_step(model, batch, ref_model, beta=beta)
        elif method == "grpo":
            return _grpo_step(
                model, batch, ref_model, kl_coeff=kl_coeff
            )
        elif method == "orpo":
            return _orpo_step(
                model, batch, outputs, lambda_weight=lambda_weight
            )
        elif method == "simpo":
            return _simpo_step(model, batch, beta=beta, gamma=gamma)
        elif method == "ppo":
            return _ppo_step(model, batch, clip_eps=clip_eps)
        elif method == "reinforce":
            return _reinforce_step(model, batch)
        else:
            raise ValueError(f"Unknown alignment method: {method}")

    return loss_fn


def _dpo_step(
    model: Any,
    batch: dict[str, Any],
    ref_model: Any,
    *,
    beta: float,
) -> torch.Tensor:
    chosen_ids = batch["chosen_input_ids"]
    chosen_mask = batch.get("chosen_attention_mask")
    rejected_ids = batch["rejected_input_ids"]
    rejected_mask = batch.get("rejected_attention_mask")

    chosen_out = model(input_ids=chosen_ids, attention_mask=chosen_mask)
    rejected_out = model(input_ids=rejected_ids, attention_mask=rejected_mask)

    policy_chosen_logps = get_sequence_logps(
        chosen_out.logits, chosen_ids, chosen_mask
    )
    policy_rejected_logps = get_sequence_logps(
        rejected_out.logits, rejected_ids, rejected_mask
    )

    with torch.no_grad():
        ref_chosen_out = ref_model(
            input_ids=chosen_ids, attention_mask=chosen_mask
        )
        ref_rejected_out = ref_model(
            input_ids=rejected_ids, attention_mask=rejected_mask
        )
        ref_chosen_logps = get_sequence_logps(
            ref_chosen_out.logits, chosen_ids, chosen_mask
        )
        ref_rejected_logps = get_sequence_logps(
            ref_rejected_out.logits, rejected_ids, rejected_mask
        )

    return dpo_loss(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        ref_chosen_logps=ref_chosen_logps,
        ref_rejected_logps=ref_rejected_logps,
        beta=beta,
    )


def _grpo_step(
    model: Any,
    batch: dict[str, Any],
    ref_model: Any,
    *,
    kl_coeff: float,
) -> torch.Tensor:
    input_ids = batch["input_ids"]
    mask = batch.get("attention_mask")
    advantages = batch["advantages"]

    outputs = model(input_ids=input_ids, attention_mask=mask)
    logprobs = get_sequence_logps(outputs.logits, input_ids, mask)

    with torch.no_grad():
        ref_outputs = ref_model(input_ids=input_ids, attention_mask=mask)
        ref_logprobs = get_sequence_logps(
            ref_outputs.logits, input_ids, mask
        )

    return grpo_loss(
        logprobs=logprobs,
        ref_logprobs=ref_logprobs,
        advantages=advantages,
        kl_coeff=kl_coeff,
    )


def _orpo_step(
    model: Any,
    batch: dict[str, Any],
    outputs: Any,
    *,
    lambda_weight: float,
) -> torch.Tensor:
    chosen_ids = batch["chosen_input_ids"]
    chosen_mask = batch.get("chosen_attention_mask")
    rejected_ids = batch["rejected_input_ids"]
    rejected_mask = batch.get("rejected_attention_mask")

    sft_loss = outputs.loss if hasattr(outputs, "loss") else outputs

    chosen_out = model(input_ids=chosen_ids, attention_mask=chosen_mask)
    rejected_out = model(input_ids=rejected_ids, attention_mask=rejected_mask)

    policy_chosen_logps = get_sequence_logps(
        chosen_out.logits, chosen_ids, chosen_mask
    )
    policy_rejected_logps = get_sequence_logps(
        rejected_out.logits, rejected_ids, rejected_mask
    )

    return orpo_loss(
        sft_loss=sft_loss,
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        lambda_weight=lambda_weight,
    )


def _simpo_step(
    model: Any,
    batch: dict[str, Any],
    *,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    chosen_ids = batch["chosen_input_ids"]
    chosen_mask = batch.get("chosen_attention_mask")
    rejected_ids = batch["rejected_input_ids"]
    rejected_mask = batch.get("rejected_attention_mask")

    chosen_out = model(input_ids=chosen_ids, attention_mask=chosen_mask)
    rejected_out = model(input_ids=rejected_ids, attention_mask=rejected_mask)

    policy_chosen_logps = get_sequence_logps(
        chosen_out.logits, chosen_ids, chosen_mask
    )
    policy_rejected_logps = get_sequence_logps(
        rejected_out.logits, rejected_ids, rejected_mask
    )

    chosen_lengths = chosen_mask.sum(dim=-1) if chosen_mask is not None else torch.tensor(
        [chosen_ids.shape[-1]], device=chosen_ids.device
    )
    rejected_lengths = rejected_mask.sum(dim=-1) if rejected_mask is not None else torch.tensor(
        [rejected_ids.shape[-1]], device=rejected_ids.device
    )

    return simpo_loss(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        chosen_lengths=chosen_lengths,
        rejected_lengths=rejected_lengths,
        beta=beta,
        gamma=gamma,
    )


def _ppo_step(
    model: Any,
    batch: dict[str, Any],
    *,
    clip_eps: float,
) -> torch.Tensor:
    input_ids = batch["input_ids"]
    mask = batch.get("attention_mask")
    old_logprobs = batch["old_logprobs"]
    advantages = batch["advantages"]

    outputs = model(input_ids=input_ids, attention_mask=mask)
    logprobs = get_sequence_logps(outputs.logits, input_ids, mask)

    return ppo_clip_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        clip_eps=clip_eps,
    )


def _reinforce_step(
    model: Any,
    batch: dict[str, Any],
) -> torch.Tensor:
    input_ids = batch["input_ids"]
    mask = batch.get("attention_mask")
    advantages = batch["advantages"]

    outputs = model(input_ids=input_ids, attention_mask=mask)
    logprobs = get_sequence_logps(outputs.logits, input_ids, mask)

    return reinforce_loss(logprobs=logprobs, advantages=advantages)
