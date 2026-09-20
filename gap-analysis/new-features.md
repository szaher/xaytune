# New Features / Enhancements Backlog

Original audit: 2026-06-03 15:00
Reconciled against the tree: 2026-09-21

> **This file still contains live work.** FEAT-002 and FEAT-005 are partial,
> and FEAT-006 and FEAT-010 are not started. Read it alongside
> `missing-features.md`, not as a historical record.

## Shipped

FEAT-001 (response-only loss masking), FEAT-003 (QLoRA k-bit preparation),
FEAT-004 (prompt-aware preference tokenization), FEAT-007 (multi-stage
pipeline, `xaytune/pipeline.py`) and FEAT-008 (real-time Studio monitoring).

## Partial

**FEAT-005 — DeepSpeed-aware training loop.** The loop half shipped:
`Trainer.train()` detects the engine, delegates `backward`/`step` to it, and
skips the GradScaler. What did not is the ownership contract underneath.
`wrap_model_distributed()` builds a DeepSpeed config with no `optimizer` and no
`scheduler` key, passes no optimizer to `ds.initialize()`, and discards the
optimizer and scheduler it returns — so neither DeepSpeed nor the trainer owns
them. Delegating to an engine that has no optimizer is not multi-GPU training.
Tracked by **BUG-036** and **TASK-029**; see `implementation-plan/backlog.md`
for the ordering problem that makes it a design change rather than a patch.

**FEAT-002 — full PPO with rollout buffer and GAE.** Most of it shipped:
`PPOTrainer`, `RolloutBuffer`, `ValueHead`, the clipped policy objective, the
value loss and multi-epoch optimization over each rollout. **GAE did not.**
`PPOTrainer._collect_rollout()` computes

```python
advantages = rewards - values
returns = rewards.clone()
```

with no λ recursion and no bootstrapping.

The reason is structural, not an omission to be patched in isolation.
`ValueHead` projects the last non-padding token's hidden state to **one scalar
per sequence**, and `score_completions()` returns **one terminal reward per
sequence**. Over a single-step episode GAE(λ) degenerates to exactly
`δ₀ = r − V(s₀)`, which is what the code computes — so the present form is not
wrong for the formulation it implements. But that formulation is a contextual
bandit, and GAE has nothing to average over in it.

**GAE requires a multi-step trajectory with a value estimate and a reward at
each timestep.** What counts as a timestep is a design choice this file should
not foreclose:

| Timestep | Shape of the change |
|---|---|
| Token — conventional for LLM PPO | `ValueHead` emits `[B, T]`, plus a per-token reward, typically a KL penalty against the reference policy |
| Turn, for multi-turn agent training | value and reward per turn |
| Environment step, for tool use or RL environments | value and reward per environment step |

Whichever is chosen, both the values and the rewards have to be carried through
`Rollout` and the buffer at that granularity, which is why this is a design
change rather than a function to drop in.

Until then the honest scope is:

| | |
|---|---|
| ✓ | rollout collection, rollout buffer, value head |
| ✓ | clipped policy objective, value loss, multiple PPO epochs |
| ✗ | GAE — blocked on per-token values and per-token rewards |

Note also that the README's "PPO — simplified clipped policy gradient" refers to
the **offline/precomputed-advantage** path and remains accurate. The two should
not be conflated:

```text
offline PPO path   -> simplified clipped PG
online PPOTrainer  -> rollout / value head / multi-epoch PPO, sequence-level
                      advantage, no GAE
```

## Not started

- **FEAT-006** — evaluation with prompt-masked metrics (Should)
- **FEAT-010** — AWQ/GPTQ quantization export (Could)
- **FEAT-009** — experiment comparison, still deliberately Won't (use native
  MLflow/W&B tooling)

Note FEAT-002, FEAT-007 and FEAT-008 were all rated Could and built anyway, so
the MoSCoW column below reflects the audit's priorities at the time rather than
what actually got built.

---

Priority method: **MoSCoW** (Must/Should/Could/Won't for this release cycle)

| ID | Feature Idea | User Problem | Value Hypothesis | Scope | Data/Schema Impact | Security/Privacy | Telemetry | Priority |
|----|-------------|-------------|-----------------|-------|-------------------|-----------------|-----------|----------|
| FEAT-001 | Response-only loss masking with configurable boundary detection | Users doing instruction tuning get suboptimal models because loss is computed on prompt tokens | Correct masking is table-stakes for any SFT framework. Without it, xaytune is objectively worse than TRL/Axolotl | MVP: alpaca + chat + sharegpt masking | Format functions must return structured output (prompt/response boundary) instead of flat text | None | Track avg response token ratio vs total tokens | Must |
| FEAT-002 | Full PPO trainer with rollout buffer and GAE | Researchers wanting real RLHF cannot use xaytune's "PPO" — it's just clipped PG | Real PPO is the gold standard for RLHF. Missing it limits xaytune to offline alignment only | Later: rollout buffer, value model, GAE, multi-epoch | Rollout buffer storage (RAM), value model checkpoints | None | Track KL divergence, reward stats, value loss | Could |
| FEAT-003 | Automatic mixed-precision QLoRA with proper preparation | Consumer GPU users (24GB cards) need reliable QLoRA. Current path skips critical preparation | QLoRA is the #1 use case for hobbyists and small teams. Must work correctly | MVP: add prepare_model_for_kbit_training call | None | None | Track dtype distribution across model layers | Must |
| FEAT-004 | Prompt-response aware preference tokenization | Alignment quality is degraded because prompt log-probs are included in preference scoring | Every competitor (TRL, Axolotl) masks prompt tokens in preference methods. This is expected behavior | MVP: track prompt_length in preference tokenization, mask in log-prob computation | Preference batches get `chosen_prompt_length` / `rejected_prompt_length` fields | None | Track response-only log-prob ratio | Must |
| FEAT-005 | DeepSpeed-aware training loop | Multi-GPU users can't benefit from ZeRO memory savings — the training loop ignores the DeepSpeed engine | DeepSpeed is the primary way to scale to larger models on multi-GPU. Broken = unusable for enterprise | MVP: detect engine type, delegate backward/step | None | None | Track DeepSpeed memory stats if available | Must |
| FEAT-006 | Evaluation with prompt-masked metrics | Users want to measure model quality on response tokens only, not prompt tokens | Standard practice in LLM eval. Without it, metrics are noisy | Later: extend evaluate() to support prompt masking | None | None | Track masked vs unmasked metric deltas | Should |
| FEAT-007 | Curriculum learning / multi-stage training pipeline | Advanced users want SFT → DPO → eval as a single pipeline rather than manual steps | Reduces operational overhead, enables automated hyperparameter search across stages | Later: pipeline config format, stage dependencies | Multi-stage config schema | None | Track per-stage metrics, stage transition events | Could |
| FEAT-008 | Real-time training monitoring in Studio | Studio users can launch training but can't see live loss curves or GPU utilization during training | Visual feedback is critical for catching bad runs early and not wasting GPU hours | Later: WebSocket/SSE streaming from trainer callbacks to Studio UI | None | None — local only | Track training events in real-time | Could |
| FEAT-009 | Wandb/MLflow experiment comparison | Users running multiple experiments have no built-in way to compare results | Experiment tracking is essential for systematic hyperparameter tuning | Later: extend Studio with experiment comparison view | None | WandB API key handling | Track experiment metadata, comparison frequency | Won't (use native tools) |
| FEAT-010 | Model quantization export (AWQ/GPTQ) | Users want to export quantized models for efficient inference, not just GGUF | AWQ and GPTQ are widely used for vLLM/TGI serving. Missing = users must use separate tools | Later: AWQ and GPTQ quantization in export pipeline | None | None | Track export format distribution | Could |

## Summary

Two separate things, kept apart because the old single table mixed them.

**Original MoSCoW priorities (2026-06-03)** — what the audit thought mattered:

| Priority | Features |
|----------|----------|
| Must | FEAT-001, 003, 004, 005 |
| Should | FEAT-006 |
| Could | FEAT-002, 007, 008, 010 |
| Won't | FEAT-009 (use WandB/MLflow native) |

**Current status (2026-09-21)** — what actually exists:

| Status | Features |
|--------|----------|
| Shipped | FEAT-001, 003, 004, 007, 008 |
| Partial | FEAT-002 (no GAE), FEAT-005 (DeepSpeed ownership) |
| Not started | FEAT-006, FEAT-010 |
| Won't | FEAT-009 |

The two diverge: FEAT-002, 007 and 008 were rated Could and were built anyway,
while FEAT-005 was rated Must and is only partly done. Priority did not predict
what got built, so read the status table for state and the priority table only
as a record of the audit's judgement at the time.

The earlier version of this summary put FEAT-002 in the Must row as "renamed/
documented", folding together its MoSCoW rating, its implementation status, and
the separate BUG-034 documentation fix (TASK-031, which is done). Those are
three different things.

The "Must" features were not optional enhancements — they were correctness fixes that happened to require new code. Of those four, three have shipped; FEAT-005 (DeepSpeed) is partial, so that path does not yet work end to end.
