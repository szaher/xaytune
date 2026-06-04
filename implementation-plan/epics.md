# Epics

Last updated: 2026-06-03 14:00

---

## EPIC-0: Foundational Training Correctness

**Problem:** The core training paths produce wrong results by design. SFT trains on prompt tokens (wasting capacity). Preference alignment includes prompt log-probs (diluting signal). ORPO crashes end-to-end. QLoRA skips critical preparation. DeepSpeed integration is non-functional. Studio bypasses alignment loss setup entirely. These are not edge cases — they affect the default, advertised paths.

**Source:** External architectural review, verified against source code 2026-06-03.

**In-scope:**
- `data/tokenizer.py` — SFT label masking for instruction formats
- `recipes/align/logprobs.py` — response-only log-prob masking for preference methods
- `trainer/loop.py` — ORPO forward-pass handling (skip_forward bug)
- `recipes/align/orpo.py` — ORPO loss receiving None sft_loss
- `models/peft.py` — `prepare_model_for_kbit_training()` call
- `trainer/loop.py` + `trainer/distributed.py` — DeepSpeed backward/step integration
- `studio/app.py` — alignment loss wiring for Studio-launched jobs
- `recipes/align/ppo.py` — rename or mark experimental

**Out-of-scope:** Full PPO implementation (rollout buffer, GAE, value model). That is a new feature, not a fix.

**Key risks:**
- SFT masking changes training behavior for all users. Models trained before the fix were trained differently. Document in changelog.
- Preference log-prob masking changes DPO/SimPO/GRPO loss values. Existing alignment results are not reproducible after fix.
- DeepSpeed fix requires training loop to detect wrapped engine type and delegate backward/step. Increased code complexity.

**Exit criteria:**
- Instruction-tuning formats (alpaca, chat, sharegpt) mask prompt tokens in labels (`-100`).
- DPO/ORPO/SimPO/GRPO log-prob computation excludes prompt tokens.
- `xaytune.align(method="orpo")` completes without crash.
- QLoRA calls `prepare_model_for_kbit_training()` before `get_peft_model()`.
- DeepSpeed-wrapped models use `engine.backward()` / `engine.step()`.
- Studio alignment jobs use the correct alignment loss function.
- PPO is marked experimental or renamed to "clipped_pg".

**Success metrics:**
- SFT loss computed on response tokens only — verified by comparing labels tensor against expected mask.
- ORPO training runs to completion on a toy dataset.
- DeepSpeed ZeRO-2 training shows memory reduction vs vanilla PyTorch (proves engine is actually used).

---

## EPIC-1: Training Loop Correctness

**Problem:** The core training loop has numerical and counting bugs that produce wrong loss values, wrong step counts, and wrong training behavior with gradient accumulation. These affect every user.

**In-scope:** `trainer/loop.py` step counting, loss reporting, gradient accumulation interaction.
**Out-of-scope:** Distributed training bugs (EPIC-4), scheduler bugs (EPIC-5).

**Key risks:** Regressions in callback timing; existing checkpoints may have been saved with wrong `global_step` values.
**Exit criteria:** With `gradient_accumulation=N`, `global_step` increments once per N micro-batches, reported loss equals the raw forward-pass loss, `max_steps` stops at the correct optimizer step.
**Success metrics:** All gradient accumulation integration tests pass; loss values match PyTorch reference.

**Status:** FIXED in session. Needs test coverage.

---

## EPIC-2: Alignment Numerical Stability

**Problem:** ORPO odds computation can produce NaN/Inf. GRPO deepcopy causes OOM for reference-free methods. SimPO has no zero-length guard.

**In-scope:** `recipes/align/orpo.py`, `align.py` ref model gating, `simpo.py` defensive clamp.
**Out-of-scope:** New alignment methods, online RL pipeline completion.

**Key risks:** ORPO fix changes loss values — may affect reproducibility of existing runs.
**Exit criteria:** ORPO handles log-probs at 0 without NaN. GRPO/ORPO/SimPO/PPO/REINFORCE run without deepcopy when ref model is not needed.
**Success metrics:** All alignment loss functions return finite values for edge-case inputs.

**Status:** GRPO OOM fix applied. ORPO and SimPO pending.

---

## EPIC-3: Eval Pipeline Completeness

**Problem:** `evaluate()` and `eval_callback` cannot compute any metric beyond loss/perplexity. Device mismatch crashes GPU evaluation.

**In-scope:** `eval/evaluate.py`, `trainer/eval_callback.py`, `eval/metrics.py`.
**Out-of-scope:** Benchmark evaluation (`benchmarks.py`), agent metrics.

**Key risks:** None — additive changes.
**Exit criteria:** `token_accuracy` returns correct nonzero values. Evaluation works on GPU without manual device management.
**Success metrics:** `evaluate(metrics=["token_accuracy"])` produces values matching manual computation.

**Status:** `evaluate.py` FIXED. `eval_callback.py` still needs the same fix.

---

## EPIC-4: Checkpoint & Device Portability

**Problem:** `torch.load()` calls lack `map_location`, breaking checkpoint resume across devices. `save_checkpoint()` can fail on non-serializable tensor metrics.

**In-scope:** `trainer/checkpointing.py`, `trainer/loop.py` resume path.
**Out-of-scope:** Async checkpoint correctness.

**Key risks:** Changing `map_location` may affect loading behavior for users with existing checkpoints (shouldn't — `map_location="cpu"` is universally safe).
**Exit criteria:** Checkpoints saved on GPU load on CPU without error. Tensor metrics are serialized to Python floats.
**Success metrics:** Cross-device checkpoint round-trip test passes.

---

## EPIC-5: Config Validation & Safety

**Problem:** `validate_config()` excludes `reinforce`, isn't called from Studio/API, has no pretrain rules, and `apply_overrides()` silently accepts typos.

**In-scope:** `config/validation.py`, `config/parser.py`, `recipes/base.py`.
**Out-of-scope:** Schema-level Pydantic validators (already robust).

**Key risks:** Adding validation to `setup_training()` could break existing workflows that pass technically-invalid-but-working configs.
**Exit criteria:** `reinforce` passes validation. `apply_overrides` rejects unknown keys. `setup_training()` calls `validate_config()`.
**Success metrics:** `xaytune.align(method="reinforce")` works end-to-end.

---

## EPIC-6: Logging & Observability Robustness

**Problem:** MLflow crashes on nested config. Optional backend imports are fragile. No exception isolation between logging backends. Step-0 log spam.

**In-scope:** `logging/` module — all backends.
**Out-of-scope:** Adding new logging backends.

**Key risks:** Lazy import changes could break plugin discovery.
**Exit criteria:** MLflow, WandB, TensorBoard backends fail gracefully when their dependencies are missing or a call fails. Config logging works with nested dicts.
**Success metrics:** Training completes even when a logging backend errors transiently.

---

## EPIC-7: Export Pipeline Correctness

**Problem:** `model_merge` output lacks `config.json` (unusable). GGUF conversion uses wrong module path. Hub push silently skips tokenizer.

**In-scope:** `export/model_merge.py`, `export/gguf.py`, `export/hub.py`.
**Out-of-scope:** Adding new export formats.

**Key risks:** Changing GGUF subprocess command may break for users who patched it locally.
**Exit criteria:** Merged model output is loadable by `from_pretrained()`. GGUF conversion either works or gives a clear error. Hub push warns if tokenizer is missing.
**Success metrics:** `model_merge()` output loads without manual intervention.

---

## EPIC-8: Data Pipeline Edge Cases

**Problem:** `format_text()` silently returns empty for unknown keys. Agent tokenizer duplicates BOS. Preference data split is unshuffled.

**In-scope:** `data/formats.py`, `data/agent_tokenizer.py`, `data/preferences.py`.
**Out-of-scope:** New data formats, data prep pipeline.

**Key risks:** Warning on unknown keys could be noisy for intentionally sparse samples.
**Exit criteria:** Unknown text keys produce a warning. Preference splits are shuffled deterministically. Agent tokenization doesn't duplicate BOS.
**Success metrics:** `format_text({"body": "..."})` emits a warning, not silent empty.

---

## EPIC-9: Studio & CLI Surface Bugs

**Problem:** Studio format dropdown lists nonexistent formats. CLI eval crashes without `--metrics`. Studio theme may not apply.

**In-scope:** `studio/app.py`, `cli.py`, `studio/server.py`.
**Out-of-scope:** Studio feature additions, new CLI commands.

**Key risks:** None — isolated UI fixes.
**Exit criteria:** Studio formats match the registry. `xaytune eval` with `--dataset` but no `--metrics` uses sensible defaults.
**Success metrics:** Studio training with "chat" format works; CLI eval doesn't crash.

---

## EPIC-10: Dependency Import Safety

**Problem:** `peft` and optional logging backends crash with raw `ImportError` instead of helpful messages.

**In-scope:** `models/peft.py`, `logging/tensorboard.py`, `logging/wandb.py`, `logging/mlflow.py`.
**Out-of-scope:** All other imports.

**Key risks:** Lazy import patterns can be subtle; incorrect implementation breaks the module entirely.
**Exit criteria:** Missing optional dependency produces a clear message naming the extras group to install.
**Success metrics:** `xaytune.finetune(method="lora")` without `peft` installed gives "pip install xaytune[peft]" message.

---

## EPIC-11: Trainer Utilities

**Problem:** `seed_all()` doesn't seed numpy. Distributed init hardcodes NCCL. LR finder doesn't move batches to device. Constant scheduler ignores warmup.

**In-scope:** `trainer/device.py`, `trainer/distributed.py`, `trainer/lr_finder.py`, `trainer/scheduler.py`.
**Out-of-scope:** New trainer features.

**Key risks:** Adding numpy seed import could fail if numpy isn't installed (it's optional).
**Exit criteria:** Seeding covers torch + random + numpy (when available). Distributed backend auto-selects gloo for CPU.
**Success metrics:** `lr_find()` works on GPU without manual device handling.
