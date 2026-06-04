# Bug List

Last updated: 2026-06-03 15:00

## Status Legend

- **FIXED** — Fixed during this audit session. Code committed but not yet pushed.
- **OPEN** — Confirmed defect, not yet fixed. Has a TASK-### in `implementation-plan/backlog.md`.

---

## Critical / Blocker

| ID | Title | Severity | Status | Task |
|----|-------|----------|--------|------|
| BUG-031 | SFT prompt masking missing | Critical | OPEN | TASK-025 |
| BUG-033 | ORPO crashes end-to-end | Critical | OPEN | TASK-027 |
| BUG-036 | DeepSpeed training loop broken | Critical | OPEN | TASK-029 |
| BUG-011 | ORPO numerical instability (NaN/Inf) | Critical | OPEN | TASK-001 |
| BUG-004 | global_step counts micro-batches | Critical | FIXED | — |
| BUG-005 | Reported loss divided by gradient_accumulation | Critical | FIXED | — |
| BUG-003 | GRPO OOM — deepcopy for all alignment methods | Critical | FIXED | — |

## High

| ID | Title | Severity | Status | Task |
|----|-------|----------|--------|------|
| BUG-032 | Preference log-probs include prompt tokens | High | OPEN | TASK-026 |
| BUG-035 | QLoRA missing prepare_model_for_kbit_training | High | OPEN | TASK-028 |
| BUG-034 | PPO is not real PPO (misleading name) | High | OPEN | TASK-031 |
| BUG-037 | Studio bypasses alignment loss setup | High | OPEN | TASK-030 |
| BUG-012 | model_merge output missing config.json | High | OPEN | TASK-013 |
| BUG-013 | torch.load missing map_location | High | OPEN | TASK-004 |
| BUG-014 | reinforce excluded from config validation | High | OPEN | TASK-006 |
| BUG-022 | Optional backend imports crash without deps | High | OPEN | TASK-012 |
| BUG-006 | token_accuracy always returns 0.0 | High | FIXED | — |
| BUG-007 | evaluate() device mismatch crash | High | FIXED | — |

## Medium

| ID | Title | Severity | Status | Task |
|----|-------|----------|--------|------|
| BUG-015 | MLflow log_params crash with nested config | Medium | OPEN | TASK-010 |
| BUG-016 | Studio data format dropdown wrong choices | Medium | OPEN | TASK-018 |
| BUG-017 | CLI eval crashes without --metrics | Medium | OPEN | TASK-019 |
| BUG-018 | seed_all missing numpy | Medium | OPEN | TASK-020 |
| BUG-019 | Distributed init hardcodes NCCL | Medium | OPEN | TASK-021 |
| BUG-020 | eval_callback dummy metrics for non-loss | Medium | OPEN | TASK-003 |
| BUG-021 | LR finder no device transfer | Medium | OPEN | TASK-022 |
| BUG-023 | Logging log_scalar no exception isolation | Medium | OPEN | TASK-011 |
| BUG-024 | format_text silent empty for unknown keys | Medium | OPEN | TASK-016 |
| BUG-025 | preferences.py split without shuffle | Medium | OPEN | TASK-017 |
| BUG-026 | Checkpoint metadata non-serializable tensors | Medium | OPEN | TASK-005 |
| BUG-008 | Unknown kwargs silently ignored | Medium | FIXED | — |
| BUG-009 | _split_dataset doesn't shuffle | Medium | FIXED | — |
| BUG-010 | Streaming+eval_split silently drops eval | Medium | FIXED | — |

## Low

| ID | Title | Severity | Status | Task |
|----|-------|----------|--------|------|
| BUG-027 | SimPO zero-length sequence guard | Low | OPEN | TASK-002 |
| BUG-028 | Agent tokenizer BOS duplication | Low | OPEN | TASK-024 |
| BUG-029 | Constant scheduler ignores warmup_steps | Low | OPEN | TASK-023 |
| BUG-030 | Studio theme not applied | Low | OPEN | — |
| BUG-001 | trainlib references in example notebooks | Low | FIXED | — |
| BUG-002 | 22 documentation errors in example notebooks | Low | FIXED | — |

---

## Detailed Bug Descriptions

### BUG-031: SFT prompt masking missing
**Severity:** Critical
**User impact:** Every instruction-tuning user gets suboptimal models. Loss is computed on prompt tokens (instruction, input), wasting capacity.
**Repro:** `xaytune.finetune(model="X", dataset="alpaca.jsonl", format="alpaca")` — inspect labels tensor, observe labels == input_ids for full sequence.
**Expected:** Labels should be `-100` for prompt tokens, real IDs only for output/assistant tokens.
**Actual:** `labels = list(input_ids)` for the entire sequence.
**Root cause:** `tokenize_dataset()` in `data/tokenizer.py:62` copies input_ids to labels without masking.
**Evidence:** `data/tokenizer.py:59-64`
**Proposed fix:** Format functions return `{"prompt": "...", "response": "..."}` instead of `{"text": "..."}`. Tokenizer masks prompt portion of labels with `-100`.
**Risk:** Changes training dynamics for all users. Models trained before fix are not reproducible.
**Tests to add:** `tests/test_sft_masking.py` — verify labels mask for alpaca, chat, sharegpt, text formats.
**Owner:** Backend / Training

### BUG-033: ORPO crashes end-to-end
**Severity:** Critical
**User impact:** `xaytune.align(method="orpo")` crashes immediately. Feature is advertised but has never worked.
**Repro:** `xaytune.align(model="X", dataset="prefs.jsonl", method="orpo")`
**Expected:** ORPO training runs and produces a loss.
**Actual:** `TypeError: unsupported operand type(s) for +: 'NoneType' and 'Tensor'`
**Root cause:** `trainer/loop.py:185-193` sets `skip_forward=True` for preference batches → passes `outputs=None` → `_orpo_step` extracts `sft_loss = None` → `orpo_loss()` does `None + tensor`.
**Evidence:** `trainer/loop.py:185-193`, `recipes/align/loss_dispatch.py:138-163`, `recipes/align/orpo.py:22`
**Proposed fix:** `_orpo_step` computes its own forward pass for SFT loss when `outputs is None`.
**Risk:** Low — ORPO was never functional.
**Tests to add:** `tests/test_orpo_e2e.py` — ORPO training on toy dataset completes without crash.
**Owner:** Backend / Alignment

### BUG-036: DeepSpeed training loop broken
**Severity:** Critical
**User impact:** Users requesting `strategy="deepspeed"` get vanilla PyTorch training. No ZeRO memory savings, no offloading. GPU resources wasted.
**Repro:** Set `trainer.strategy="deepspeed"`, observe no memory reduction vs `strategy="auto"`.
**Expected:** `engine.backward(loss)` and `engine.step()` called. ZeRO active.
**Actual:** `loss.backward()` and `optimizer.step()` called on vanilla objects. DeepSpeed engine wraps model but training loop ignores it.
**Root cause:** `distributed.py:209` returns a DeepSpeed engine but `loop.py:58` creates its own `AdamW`. `loop.py:220-238` calls vanilla backward/step.
**Evidence:** `trainer/distributed.py:209`, `trainer/loop.py:57-63,220-238`
**Proposed fix:** Detect DeepSpeed engine in Trainer. Delegate backward/step/optimizer to engine. Skip GradScaler.
**Risk:** Medium — training loop complexity increases. Need engine type detection.
**Tests to add:** `tests/test_deepspeed_loop.py` — verify engine.backward() and engine.step() called.
**Owner:** Backend / Distributed

### BUG-011: ORPO numerical instability
**Severity:** Critical
**User impact:** ORPO training produces NaN/Inf when log-probs approach 0 (probability ~1.0).
**Repro:** Call `orpo_loss()` with `policy_chosen_logps=tensor(0.0)`.
**Expected:** Finite loss value.
**Actual:** `exp(0) = 1`, `1 - 1 = 0`, division by zero → Inf → NaN.
**Root cause:** `orpo.py:15-16` computes explicit odds `exp(lp) / (1 - exp(lp))`.
**Evidence:** `recipes/align/orpo.py:14-18`
**Proposed fix:** Use numerically stable formulation: `log_odds_ratio = chosen_logps - rejected_logps`, then `F.logsigmoid(log_odds_ratio)`.
**Risk:** Low — only changes behavior for edge-case inputs.
**Tests to add:** Edge-case test with logps=0.
**Owner:** Backend / Alignment

### BUG-032: Preference log-probs include prompt tokens
**Severity:** High
**User impact:** DPO/GRPO/SimPO/ORPO alignment quality degraded. Prompt log-probs are identical for chosen/rejected — they add noise, not signal.
**Repro:** Inspect `get_sequence_logps()` output — includes prompt token contributions.
**Expected:** Only response tokens contribute to sequence log-probability.
**Actual:** `mask[:, 1:]` masks only padding, not prompt tokens.
**Root cause:** `logprobs.py:24-26` uses attention_mask (padding only) as the mask.
**Evidence:** `recipes/align/logprobs.py:17-26`
**Proposed fix:** Add `prompt_length` parameter to `get_sequence_logps()`. Zero out first `prompt_length` positions.
**Risk:** Changes alignment loss values. Existing results not reproducible.
**Tests to add:** Unit test verifying prompt exclusion.
**Owner:** Backend / Alignment

### BUG-035: QLoRA missing prepare_model_for_kbit_training
**Severity:** High
**User impact:** QLoRA training may produce garbage gradients or dtype mismatches.
**Repro:** `xaytune.finetune(model="X", method="qlora")` — observe no `prepare_model_for_kbit_training` call.
**Expected:** 4-bit model prepared before LoRA application.
**Actual:** `get_peft_model()` called directly on quantized model.
**Root cause:** `peft.py:46` skips `prepare_model_for_kbit_training()`.
**Evidence:** `models/peft.py:29-54`
**Proposed fix:** Call `prepare_model_for_kbit_training(model)` when `model_result.quantization` is set.
**Risk:** Low — additive fix.
**Tests to add:** Unit test verifying preparation is called for quantized models.
**Owner:** Backend / Models

### BUG-034: PPO is not real PPO
**Severity:** High
**User impact:** Users expecting PPO-quality alignment get a simplified clipped policy gradient. Missing: rollout buffer, GAE, value model, multiple optimization epochs.
**Repro:** Read `recipes/align/ppo.py` — contains only `ppo_clip_loss()` and `ppo_value_loss()`.
**Expected:** Full PPO trainer with rollout collection and multi-epoch optimization.
**Actual:** Single clipped PG loss function.
**Root cause:** Incomplete implementation labeled as "PPO".
**Evidence:** `recipes/align/ppo.py`
**Proposed fix:** Rename to "clipped_pg" or add prominent "experimental/simplified" documentation.
**Risk:** API breaking if renamed. Prefer documentation fix.
**Tests to add:** None — documentation change only.
**Owner:** Backend / Documentation

### BUG-037: Studio bypasses alignment loss
**Severity:** High
**User impact:** Studio-launched DPO/GRPO/ORPO training jobs use default cross-entropy loss, producing wrong results with no error.
**Repro:** Launch alignment training from Studio UI.
**Expected:** Alignment loss function used.
**Actual:** `setup_training()` + `trainer.train()` called directly — alignment loss constructed only inside `align()`.
**Root cause:** Studio doesn't call `align()`, it calls `setup_training()` directly.
**Evidence:** `studio/app.py` training launch path.
**Proposed fix:** Studio calls `xaytune.align(config=config)` for align recipes.
**Risk:** Low — Studio alignment was already broken.
**Tests to add:** Test that Studio align produces correct loss_fn.
**Owner:** Backend / Studio

### BUG-012: model_merge output missing config.json
**Severity:** High
**User impact:** `model_merge()` output cannot be loaded by `from_pretrained()`.
**Evidence:** `export/model_merge.py` — `_save_merged()` saves `pytorch_model.bin` but not `config.json`.
**Proposed fix:** Load and save `AutoConfig` from first source model.
**Tests to add:** Verify output is loadable.
**Owner:** Backend / Export

### BUG-013: torch.load missing map_location
**Severity:** High
**User impact:** Checkpoint resume fails across devices (GPU checkpoint on CPU).
**Evidence:** `trainer/checkpointing.py` — 4 `torch.load()` calls without `map_location`. `trainer/loop.py` — 3 more.
**Proposed fix:** Add `map_location="cpu"` to all calls.
**Tests to add:** Cross-device checkpoint round-trip.
**Owner:** Backend / Trainer

### BUG-014: reinforce excluded from validation
**Severity:** High
**User impact:** `method="reinforce"` rejected by CLI despite being supported.
**Evidence:** `config/validation.py` — `_ALIGN_METHODS` missing `"reinforce"`.
**Proposed fix:** Add `"reinforce"` to the set.
**Tests to add:** Validation test with reinforce config.
**Owner:** Backend / Config

### BUG-022: Optional backend imports crash
**Severity:** High
**User impact:** `import xaytune.logging.wandb` crashes if `wandb` not installed. `from peft import ...` crashes at module level.
**Evidence:** `logging/tensorboard.py`, `logging/wandb.py`, `logging/mlflow.py`, `models/peft.py` — all have unconditional top-level imports.
**Proposed fix:** Guard with try/except, raise helpful message on first use.
**Tests to add:** Mock `sys.modules` to simulate missing package.
**Owner:** Backend / Imports

### BUG-015: MLflow log_params crash
**Severity:** Medium
**Evidence:** `logging/mlflow.py:18` — `mlflow.log_params(config)` receives nested dict. MLflow requires flat `dict[str, str]`.
**Proposed fix:** Flatten config with dot-notation keys.
**Owner:** Backend / Logging

### BUG-016: Studio format dropdown wrong
**Severity:** Medium
**Evidence:** `studio/app.py:~343` — lists `["alpaca", "sharegpt", "completion", "pretrain"]`. Should be `["alpaca", "sharegpt", "chat", "text", "preference"]`.
**Owner:** Backend / Studio

### BUG-017: CLI eval crashes without --metrics
**Severity:** Medium
**Evidence:** `cli.py:~387` — `args.metrics` is None, `.split()` called on None.
**Proposed fix:** Default to `"loss,perplexity"`.
**Owner:** Backend / CLI

### BUG-018: seed_all missing numpy
**Severity:** Medium
**Evidence:** `trainer/device.py` — seeds `torch` and `random` but not `numpy`.
**Owner:** Backend / Trainer

### BUG-019: Distributed init hardcodes NCCL
**Severity:** Medium
**Evidence:** `trainer/distributed.py:52` — `backend="nccl"` always. Fails on CPU.
**Proposed fix:** `"nccl" if torch.cuda.is_available() else "gloo"`.
**Owner:** Backend / Distributed

### BUG-020: eval_callback dummy metrics
**Severity:** Medium
**Evidence:** `trainer/eval_callback.py:59-64` — non-loss metrics get `compute_fn([], [])`.
**Proposed fix:** Same pattern as evaluate.py fix — collect predictions/references.
**Owner:** Backend / Eval

### BUG-021: LR finder no device transfer
**Severity:** Medium
**Evidence:** `trainer/lr_finder.py:93-125` — batch not moved to model device.
**Owner:** Backend / Trainer

### BUG-023: Logging no exception isolation
**Severity:** Medium
**Evidence:** `logging/base.py:29-33` — one backend failure kills all logging and potentially training.
**Owner:** Backend / Logging

### BUG-024: format_text silent empty
**Severity:** Medium
**Evidence:** `data/formats.py:53-57` — returns `{"text": ""}` for unknown keys without warning.
**Owner:** Backend / Data

### BUG-025: Preference split without shuffle
**Severity:** Medium
**Evidence:** `data/preferences.py:61` — splits by index without shuffling.
**Owner:** Backend / Data

### BUG-026: Checkpoint tensor serialization
**Severity:** Medium
**Evidence:** `trainer/checkpointing.py:~47` — `json.dumps(state.metrics)` may contain tensors.
**Owner:** Backend / Trainer

### BUG-027: SimPO zero-length guard
**Severity:** Low
**Evidence:** `recipes/align/simpo.py:17-18` — divides by lengths without clamp.
**Owner:** Backend / Alignment

### BUG-028: Agent tokenizer BOS duplication
**Severity:** Low
**Evidence:** `data/agent_tokenizer.py:28-33` — each message tokenized independently, producing multiple BOS.
**Owner:** Backend / Data

### BUG-029: Constant scheduler ignores warmup
**Severity:** Low
**Evidence:** `trainer/scheduler.py:50-51` — `"constant"` lambda returns 1.0, ignoring warmup_steps.
**Owner:** Backend / Trainer

### BUG-030: Studio theme not applied
**Severity:** Low
**Evidence:** `studio/server.py:25` — theme passed to `launch()` instead of `gr.Blocks()`.
**Owner:** Backend / Studio
