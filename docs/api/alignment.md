# Alignment Losses

trainlib supports six alignment methods. Each has a dedicated loss function and can be selected via `create_alignment_loss_fn()`.

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

::: trainlib.recipes.align.loss_dispatch.create_alignment_loss_fn

::: trainlib.recipes.align.loss_dispatch.is_alignment_method

## DPO

::: trainlib.recipes.align.dpo.dpo_loss

## SimPO

::: trainlib.recipes.align.simpo.simpo_loss

## ORPO

::: trainlib.recipes.align.orpo.orpo_loss

## GRPO

::: trainlib.recipes.align.grpo.grpo_loss

::: trainlib.recipes.align.grpo.compute_group_advantages

## PPO / REINFORCE

::: trainlib.recipes.align.ppo.ppo_clip_loss

::: trainlib.recipes.align.ppo.ppo_value_loss

::: trainlib.recipes.align.ppo.reinforce_loss

## Log-Probabilities

::: trainlib.recipes.align.logprobs.get_per_token_logps

::: trainlib.recipes.align.logprobs.get_sequence_logps

::: trainlib.recipes.align.logprobs.get_model_logps

## Rewards

::: trainlib.recipes.align.rewards.default_reward

::: trainlib.recipes.align.rewards.length_penalty_reward

::: trainlib.recipes.align.rewards.format_check_reward

::: trainlib.recipes.align.rewards.composite_reward
