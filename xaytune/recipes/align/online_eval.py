from __future__ import annotations

import logging
from typing import Any

import torch

from xaytune.config.schema import GenerationConfig
from xaytune.recipes.align.reward_scoring import score_completions
from xaytune.trainer.callbacks import CallbackManager, TrainState

logger = logging.getLogger(__name__)


class OnlineEvalCallback:
    """Periodically generates completions from eval prompts and scores them.

    Injects ``online_eval/mean_reward`` and ``online_eval/std_reward`` into
    ``state.metrics`` so they appear in training logs and Studio charts.
    """

    def __init__(
        self,
        eval_prompts: list[str],
        tokenizer: Any,
        generation_config: GenerationConfig,
        reward_name: str = "default",
        reward_kwargs: dict[str, Any] | None = None,
        every_n_steps: int = 100,
    ) -> None:
        self._eval_prompts = eval_prompts
        self._tokenizer = tokenizer
        self._gen_config = generation_config
        self._reward_name = reward_name
        self._reward_kwargs = reward_kwargs or {}
        self._every_n_steps = every_n_steps

    def register(self, callback_manager: CallbackManager) -> None:
        @callback_manager.on("step_end")
        def _eval_step(state: TrainState) -> None:
            if state.global_step == 0:
                return
            if state.global_step % self._every_n_steps != 0:
                return
            metrics = self.evaluate(state)
            state.metrics.update(metrics)

    def evaluate(self, state: TrainState) -> dict[str, float]:
        """Generate completions for eval prompts and compute reward stats."""
        if not self._eval_prompts:
            return {}

        prompt_ids_list = []
        mask_list = []
        for prompt in self._eval_prompts:
            encoded = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            prompt_ids_list.append(encoded["input_ids"].squeeze(0))
            mask_list.append(encoded["attention_mask"].squeeze(0))

        max_len = max(ids.shape[0] for ids in prompt_ids_list)
        pad_id = self._tokenizer.pad_token_id or 0
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(prompt_ids_list, mask_list):
            pad_len = max_len - ids.shape[0]
            padded_ids.append(torch.cat([torch.full((pad_len,), pad_id, dtype=ids.dtype), ids]))
            padded_masks.append(torch.cat([torch.zeros(pad_len, dtype=mask.dtype), mask]))

        prompt_ids = torch.stack(padded_ids)
        prompt_mask = torch.stack(padded_masks)

        try:
            from xaytune.recipes.align.generation import generate_completions as _gen

            gen_result = _gen(
                model=state.metrics.get("_model"),
                tokenizer=self._tokenizer,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                config=self._gen_config,
            )

            rewards = score_completions(
                prompts=self._eval_prompts,
                responses=gen_result.response_texts,
                reward_name=self._reward_name,
                reward_kwargs=self._reward_kwargs,
            )

            mean_reward = float(rewards.mean().item())
            std_reward = float(rewards.std().item()) if rewards.numel() > 1 else 0.0

            logger.info(
                "Online eval step %d: mean_reward=%.4f std_reward=%.4f",
                state.global_step,
                mean_reward,
                std_reward,
            )

            return {
                "online_eval/mean_reward": mean_reward,
                "online_eval/std_reward": std_reward,
            }
        except Exception:
            logger.warning("Online eval failed at step %d", state.global_step, exc_info=True)
            return {}
