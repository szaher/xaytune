# Alignment Losses

xaytune supports six alignment methods. Each has a dedicated loss function and can be selected via `create_alignment_loss_fn()`.

| Method | Function | Paper |
|--------|----------|-------|
| DPO | `dpo_loss` | Rafailov et al., 2023 |
| SimPO | `simpo_loss` | Meng et al., 2024 |
| ORPO | `orpo_loss` | Hong et al., 2024 |
| GRPO | `grpo_loss` | Shao et al., 2024 |
| PPO | `ppo_clip_loss` | Schulman et al., 2017 |
| REINFORCE | `reinforce_loss` | Williams, 1992 |

---

## Loss Dispatch

::: xaytune.recipes.align.loss_dispatch.create_alignment_loss_fn

::: xaytune.recipes.align.loss_dispatch.is_alignment_method

## DPO

::: xaytune.recipes.align.dpo.dpo_loss

## SimPO

::: xaytune.recipes.align.simpo.simpo_loss

## ORPO

::: xaytune.recipes.align.orpo.orpo_loss

## GRPO

::: xaytune.recipes.align.grpo.grpo_loss

::: xaytune.recipes.align.grpo.compute_group_advantages

## PPO / REINFORCE

::: xaytune.recipes.align.ppo.ppo_clip_loss

::: xaytune.recipes.align.ppo.ppo_value_loss

::: xaytune.recipes.align.ppo.reinforce_loss

## Log-Probabilities

::: xaytune.recipes.align.logprobs.get_per_token_logps

::: xaytune.recipes.align.logprobs.get_sequence_logps

::: xaytune.recipes.align.logprobs.get_model_logps

## Rewards

::: xaytune.recipes.align.rewards.default_reward

::: xaytune.recipes.align.rewards.length_penalty_reward

::: xaytune.recipes.align.rewards.format_check_reward

::: xaytune.recipes.align.rewards.composite_reward

## Agent Rewards

Reward functions for agent alignment with GRPO/PPO. Score agent responses based on tool usage quality, task completion, and efficiency. All rewards use `<tool_call>` tag parsing with pluggable custom parsers.

```python
# In training config:
online_rl:
  reward_name: agent_composite
  reward_kwargs:
    expected_tools: ["search", "calculator"]
    success_markers: ["Done"]
    max_steps: 5
```

::: xaytune.recipes.align.agent_rewards.tool_use_quality_reward

::: xaytune.recipes.align.agent_rewards.task_completion_reward

::: xaytune.recipes.align.agent_rewards.efficiency_reward

::: xaytune.recipes.align.agent_rewards.agent_composite_reward

::: xaytune.recipes.align.agent_rewards.parse_tool_calls

::: xaytune.recipes.align.agent_rewards.ParsedToolCall
