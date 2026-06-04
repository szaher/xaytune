from __future__ import annotations

from typing import Any

import torch

from xaytune.config.schema import GenerationConfig
from xaytune.recipes.align.generation import generate_completions
from xaytune.recipes.align.logprobs import get_sequence_logps
from xaytune.recipes.align.reward_scoring import (
    compute_advantages_from_rewards,
    score_completions,
)


class OnlineRLStep:
    """Online RL training step that generates, scores, and trains in one call.

    Conforms to the ``loss_fn(model, batch, outputs) -> Tensor`` interface
    expected by :class:`~xaytune.trainer.loop.Trainer`.  Falls back to the
    pre-computed path when ``batch["advantages"]`` already exists.
    """

    def __init__(
        self,
        ref_model: Any,
        tokenizer: Any,
        method: str,
        generation_config: GenerationConfig,
        reward_name: str = "default",
        reward_kwargs: dict[str, Any] | None = None,
        kl_coeff: float = 0.04,
        clip_eps: float = 0.2,
    ) -> None:
        self._ref_model = ref_model
        self._tokenizer = tokenizer
        self._method = method
        self._gen_config = generation_config
        self._reward_name = reward_name
        self._reward_kwargs = reward_kwargs or {}
        self._kl_coeff = kl_coeff
        self._clip_eps = clip_eps

    def __call__(
        self,
        model: Any,
        batch: dict[str, Any],
        outputs: Any,
    ) -> torch.Tensor:
        if "advantages" in batch:
            from xaytune.recipes.align.loss_dispatch import create_alignment_loss_fn

            offline_fn = create_alignment_loss_fn(
                method=self._method,
                ref_model=self._ref_model,
                kl_coeff=self._kl_coeff,
                clip_eps=self._clip_eps,
            )
            return offline_fn(model, batch, outputs)

        prompt_ids = batch["prompt_input_ids"]
        prompt_mask = batch["prompt_attention_mask"]

        gen_result = generate_completions(
            model=model,
            tokenizer=self._tokenizer,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            config=self._gen_config,
        )

        rewards = score_completions(
            prompts=self._decode_prompts(prompt_ids, prompt_mask),
            responses=gen_result.response_texts,
            reward_name=self._reward_name,
            reward_kwargs=self._reward_kwargs,
        )
        rewards = rewards.to(gen_result.response_ids.device)

        advantages = compute_advantages_from_rewards(
            rewards,
            group_size=self._gen_config.group_size,
        ).to(gen_result.response_ids.device)

        full_ids = gen_result.response_ids
        full_mask = gen_result.attention_mask

        model_out = model(input_ids=full_ids, attention_mask=full_mask)
        logprobs = get_sequence_logps(model_out.logits, full_ids, full_mask)

        if self._method == "grpo":
            return self._grpo_online(logprobs, full_ids, full_mask, advantages)
        elif self._method == "ppo":
            return self._ppo_online(logprobs, full_ids, full_mask, advantages)
        elif self._method == "reinforce":
            from xaytune.recipes.align.ppo import reinforce_loss

            return reinforce_loss(logprobs=logprobs, advantages=advantages)
        else:
            raise ValueError(f"Online RL not supported for method: {self._method}")

    def _grpo_online(
        self,
        logprobs: torch.Tensor,
        full_ids: torch.Tensor,
        full_mask: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        from xaytune.recipes.align.grpo import grpo_loss

        ref_logprobs = None
        if self._ref_model is not None and self._kl_coeff > 0:
            with torch.no_grad():
                ref_out = self._ref_model(input_ids=full_ids, attention_mask=full_mask)
                ref_logprobs = get_sequence_logps(ref_out.logits, full_ids, full_mask)

        return grpo_loss(
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            kl_coeff=self._kl_coeff,
        )

    def _ppo_online(
        self,
        logprobs: torch.Tensor,
        full_ids: torch.Tensor,
        full_mask: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        from xaytune.recipes.align.ppo import ppo_clip_loss

        with torch.no_grad():
            old_out = self._ref_model(input_ids=full_ids, attention_mask=full_mask)
            old_logprobs = get_sequence_logps(old_out.logits, full_ids, full_mask)

        return ppo_clip_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            clip_eps=self._clip_eps,
        )

    def _decode_prompts(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> list[str]:
        prompts = []
        for i in range(prompt_ids.shape[0]):
            length = int(prompt_mask[i].sum().item())
            ids = prompt_ids[i, :length]
            prompts.append(self._tokenizer.decode(ids, skip_special_tokens=True))
        if self._gen_config.group_size > 1:
            expanded = []
            for p in prompts:
                expanded.extend([p] * self._gen_config.group_size)
            return expanded
        return prompts
