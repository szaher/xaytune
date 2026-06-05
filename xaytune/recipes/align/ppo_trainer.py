"""Full PPO trainer with value model, rollout buffer, and multi-epoch optimization.

Implements the complete PPO algorithm for LLM alignment:
1. Collect rollouts by generating responses and scoring rewards
2. Compute sequence-level values and advantages
3. Train for K epochs over the rollout buffer with clipped policy + value loss

Usage::

    trainer = PPOTrainer(model, ref_model, value_head, tokenizer, config, cb)
    state = trainer.train(prompt_dataloader, generation_config, reward_name)
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from xaytune.recipes.align.generation import generate_completions
from xaytune.recipes.align.logprobs import get_sequence_logps
from xaytune.recipes.align.ppo import ppo_clip_loss, ppo_value_loss
from xaytune.recipes.align.reward_scoring import score_completions
from xaytune.recipes.align.rollout_buffer import Rollout, RolloutBuffer
from xaytune.recipes.align.value_head import ValueHead
from xaytune.trainer.callbacks import CallbackManager, TrainState

logger = logging.getLogger(__name__)


def _extract_prompts(
    batch: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract prompt IDs and mask from whatever batch format the dataloader provides."""
    if "prompt_input_ids" in batch:
        return batch["prompt_input_ids"].to(device), batch["prompt_attention_mask"].to(device)
    if "input_ids" in batch:
        return batch["input_ids"].to(device), batch.get(
            "attention_mask", torch.ones_like(batch["input_ids"])
        ).to(device)
    if "chosen_input_ids" in batch:
        return batch["chosen_input_ids"].to(device), batch.get(
            "chosen_attention_mask", torch.ones_like(batch["chosen_input_ids"])
        ).to(device)
    raise KeyError(
        f"PPO batch must contain 'prompt_input_ids', 'input_ids', or 'chosen_input_ids'. "
        f"Got keys: {list(batch.keys())}"
    )


class PPOTrainer:
    """PPO training loop for LLM alignment.

    Manages the collect→train cycle: generates responses with the current
    policy, scores them with a reward function, estimates values, computes
    advantages, and optimises the policy for multiple epochs over the
    collected rollout.

    Args:
        model: The language model being trained.
        ref_model: Frozen reference model for KL computation (or ``None``).
        value_head: :class:`ValueHead` module for value estimation.
        tokenizer: HuggingFace tokenizer for decoding generated tokens.
        config: Full :class:`TrainConfig` (uses ``ppo`` and ``trainer`` fields).
        callback_manager: For ``train_start``, ``step_end``, ``train_end`` events.
    """

    def __init__(
        self,
        model: Any,
        ref_model: Any | None,
        value_head: ValueHead,
        tokenizer: Any,
        config: Any,
        callback_manager: CallbackManager | None = None,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.value_head = value_head
        self.tokenizer = tokenizer
        self.ppo_config = config.ppo
        self.trainer_config = config.trainer
        self.cb = callback_manager or CallbackManager()
        self.buffer = RolloutBuffer()

    def train(
        self,
        prompt_dataloader: Any,
        generation_config: Any,
        reward_name: str = "default",
        reward_kwargs: dict[str, Any] | None = None,
    ) -> TrainState:
        """Run the full PPO training loop.

        Args:
            prompt_dataloader: DataLoader yielding prompt batches
                (dicts with ``prompt_input_ids`` and ``prompt_attention_mask``).
            generation_config: :class:`GenerationConfig` for response sampling.
            reward_name: Registered reward function name.
            reward_kwargs: Extra kwargs for the reward function.

        Returns:
            Final :class:`TrainState` with PPO-specific metrics.
        """
        device = next(self.model.parameters()).device

        optimizer = torch.optim.AdamW(
            [
                {"params": self.model.parameters(), "lr": self.trainer_config.learning_rate},
                {"params": self.value_head.parameters(), "lr": self.trainer_config.learning_rate},
            ],
            weight_decay=self.trainer_config.weight_decay,
        )

        self.value_head.to(device)

        num_iterations = self.trainer_config.max_steps
        if num_iterations <= 0:
            try:
                num_iterations = len(prompt_dataloader) * self.trainer_config.num_epochs
            except TypeError:
                num_iterations = 1000

        state = TrainState(
            num_epochs=self.trainer_config.num_epochs,
            max_steps=num_iterations,
        )

        self.cb.fire("train_start", state)
        prompt_iter = iter(prompt_dataloader)

        for iteration in range(num_iterations):
            try:
                prompt_batch = next(prompt_iter)
            except StopIteration:
                prompt_iter = iter(prompt_dataloader)
                prompt_batch = next(prompt_iter)

            prompt_ids, prompt_mask = _extract_prompts(prompt_batch, device)

            rollout = self._collect_rollout(
                prompt_ids,
                prompt_mask,
                generation_config,
                reward_name,
                reward_kwargs or {},
            )
            self.buffer.store(rollout)

            total_policy_loss = 0.0
            total_value_loss = 0.0
            n_steps = 0

            self.model.train()
            self.value_head.train()

            for _ppo_epoch in range(self.ppo_config.ppo_epochs):
                for mini_batch in self.buffer.iterate(
                    self.ppo_config.mini_batch_size, shuffle=True
                ):
                    p_loss, v_loss = self._training_step(mini_batch, optimizer)
                    total_policy_loss += p_loss
                    total_value_loss += v_loss
                    n_steps += 1

            self.buffer.clear()

            avg_policy = total_policy_loss / max(n_steps, 1)
            avg_value = total_value_loss / max(n_steps, 1)

            state.global_step = iteration + 1
            state.metrics["loss"] = avg_policy + self.ppo_config.value_coeff * avg_value
            state.metrics["policy_loss"] = avg_policy
            state.metrics["value_loss"] = avg_value
            state.metrics["mean_reward"] = rollout.rewards.mean().item()
            state.metrics["mean_advantage"] = rollout.advantages.mean().item()
            state.metrics["mean_value"] = rollout.values.mean().item()

            self.cb.fire("step_end", state)

            if state.should_stop:
                break

            logger.info(
                f"PPO iter {iteration + 1}/{num_iterations} — "
                f"reward={rollout.rewards.mean():.3f} "
                f"policy_loss={avg_policy:.4f} "
                f"value_loss={avg_value:.4f}"
            )

        self.cb.fire("train_end", state)
        return state

    @torch.no_grad()
    def _collect_rollout(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        generation_config: Any,
        reward_name: str,
        reward_kwargs: dict[str, Any],
    ) -> Rollout:
        """Generate responses, score rewards, compute values and advantages."""
        self.model.eval()
        self.value_head.eval()

        gen_result = generate_completions(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            config=generation_config,
        )

        prompts_text = []
        for i in range(prompt_ids.shape[0]):
            length = int(prompt_mask[i].sum().item())
            ids = prompt_ids[i, :length]
            prompts_text.append(
                self.tokenizer.decode(ids, skip_special_tokens=True)
            )
        if generation_config.group_size > 1:
            prompts_text = [
                p
                for p in prompts_text
                for _ in range(generation_config.group_size)
            ]

        rewards = score_completions(
            prompts=prompts_text,
            responses=gen_result.response_texts,
            reward_name=reward_name,
            reward_kwargs=reward_kwargs,
        ).to(gen_result.response_ids.device)

        full_ids = gen_result.response_ids
        full_mask = gen_result.attention_mask
        prompt_lengths = gen_result.prompt_lengths

        outputs = self.model(
            input_ids=full_ids,
            attention_mask=full_mask,
            output_hidden_states=True,
        )
        old_logprobs = get_sequence_logps(
            outputs.logits, full_ids, full_mask, prompt_length=prompt_lengths
        )
        values = self.value_head(outputs.hidden_states[-1], full_mask)

        advantages = rewards - values
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
        returns = rewards.clone()

        return Rollout(
            input_ids=full_ids,
            attention_mask=full_mask,
            old_logprobs=old_logprobs,
            rewards=rewards,
            values=values,
            advantages=advantages,
            returns=returns,
            prompt_lengths=prompt_lengths,
        )

    def _training_step(
        self,
        batch: dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, float]:
        """Single PPO optimization step on a mini-batch."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
        new_logprobs = get_sequence_logps(
            outputs.logits,
            batch["input_ids"],
            batch["attention_mask"],
            prompt_length=batch["prompt_lengths"],
        )
        new_values = self.value_head(
            outputs.hidden_states[-1], batch["attention_mask"]
        )

        policy_loss = ppo_clip_loss(
            logprobs=new_logprobs,
            old_logprobs=batch["old_logprobs"],
            advantages=batch["advantages"],
            clip_eps=self.ppo_config.clip_eps,
        )
        value_loss = ppo_value_loss(
            values=new_values,
            returns=batch["returns"],
        )

        kl = (new_logprobs - batch["old_logprobs"]).mean()

        total_loss = (
            policy_loss
            + self.ppo_config.value_coeff * value_loss
            + self.ppo_config.kl_coeff * kl
        )

        optimizer.zero_grad()
        total_loss.backward()
        if self.ppo_config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.ppo_config.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.value_head.parameters(), self.ppo_config.max_grad_norm
            )
        optimizer.step()

        return policy_loss.item(), value_loss.item()
