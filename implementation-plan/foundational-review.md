# Foundational Review Findings

Last updated: 2026-06-03 14:00

## Source

External architectural review of xaytune. Findings independently verified against source code on 2026-06-03.

## Verified Issues

All items below are confirmed by reading the actual source. No speculative claims.

### FOUND-001: SFT prompt masking is missing (CONFIRMED — CRITICAL)
**File:** `xaytune/data/tokenizer.py:62`
**Evidence:** `"labels": list(input_ids)` — labels equal input_ids for the full sequence. No prompt masking. For instruction tuning (alpaca/chat/sharegpt), the model trains on prompt tokens too, wasting capacity and degrading instruction-following quality.
**Impact:** Every instruction-tuning user gets suboptimal results. This is the single most important correctness issue in the codebase.

### FOUND-002: Preference log-probs include prompt tokens (CONFIRMED — HIGH)
**File:** `xaytune/recipes/align/logprobs.py:24-26`
**Evidence:** `get_sequence_logps()` masks by `attention_mask` (padding only). DPO/ORPO/SimPO sum log-probs over the full sequence including the shared prompt. The prompt portion is identical for chosen and rejected, diluting the preference signal.
**Impact:** Alignment training is less effective than it should be. Larger models and longer prompts amplify the dilution.

### FOUND-003: ORPO crashes end-to-end (CONFIRMED — CRITICAL)
**File:** `xaytune/trainer/loop.py:185-193`, `xaytune/recipes/align/loss_dispatch.py:138-163`, `xaytune/recipes/align/orpo.py:22`
**Evidence:** Training loop sets `skip_forward=True` for preference batches → passes `outputs=None` to loss_fn → `_orpo_step` extracts `sft_loss = None` → `orpo_loss()` does `None + tensor` → `TypeError`. ORPO is completely broken — it has never worked end-to-end.
**Impact:** Any user calling `xaytune.align(method="orpo")` gets an immediate crash.

### FOUND-004: PPO is not a real PPO implementation (CONFIRMED — HIGH)
**File:** `xaytune/recipes/align/ppo.py`
**Evidence:** Contains only `ppo_clip_loss()` (clipped policy gradient) and `ppo_value_loss()`. Missing: rollout buffer, old policy behavior tracking, value model/baseline, multiple optimization epochs over rollout data, GAE advantage estimation, KL control, response-only token-level training, reward normalization. This is a clipped REINFORCE variant, not PPO.
**Impact:** Calling this PPO is misleading. Users expecting PPO-quality alignment get a weaker algorithm.

### FOUND-005: QLoRA missing `prepare_model_for_kbit_training()` (CONFIRMED — HIGH)
**File:** `xaytune/models/peft.py:46`
**Evidence:** `get_peft_model()` is called directly on a 4-bit model without calling `peft.prepare_model_for_kbit_training()` first. That function disables gradients on non-LoRA params, casts layernorm to float32, and enables gradient checkpointing — all critical for stable 4-bit training.
**Impact:** QLoRA training may produce garbage gradients, dtype mismatches, or silently train all parameters (not just adapters), negating the memory savings.

### FOUND-006: DeepSpeed training loop is broken (CONFIRMED — CRITICAL)
**File:** `xaytune/trainer/distributed.py:209`, `xaytune/trainer/loop.py:57-63,220-238`
**Evidence:** `ds.initialize(model=model, config=config_dict)` returns a DeepSpeed engine, but the training loop creates its own `AdamW` optimizer (line 58) and calls `loss.backward()` / `optimizer.step()` directly. DeepSpeed requires `engine.backward(loss)` / `engine.step()`. The optimizer must be passed to `ds.initialize()`, not created separately.
**Impact:** DeepSpeed training silently uses vanilla PyTorch backward/step, defeating the purpose of DeepSpeed (no ZeRO, no offloading, no overlapping).

### FOUND-007: Studio bypasses alignment loss setup (CONFIRMED — HIGH)
**File:** `xaytune/studio/app.py` → calls `setup_training()` + `trainer.train()` directly
**Evidence:** Alignment-specific loss functions (DPO, GRPO, etc.) are constructed inside `align()` (in `recipes/align/align.py:161-186`), not in `setup_training()`. Studio builds a config and calls `setup_training()` directly, then `trainer.train()` without loss_fn. Result: Studio alignment jobs use default cross-entropy loss, not the alignment loss.
**Impact:** Studio-launched DPO/GRPO/ORPO jobs do vanilla SFT instead of alignment training. Users get no error — just wrong results.

## Verdict Alignment

The external review's core claim is correct: **xaytune has a clean architecture but overclaims its feature coverage.** The implementation is too thin for the surface area advertised.

The most critical gaps are not edge cases — they affect the default training paths:
1. SFT masking (affects every instruction-tuning job)
2. ORPO crash (advertised feature that has never worked)
3. DeepSpeed broken (advertised feature that silently falls back to vanilla PyTorch)
4. QLoRA incomplete (advertised feature that produces wrong gradients)

These should be fixed or removed (with clear "experimental" labels) before any public claim of support.
