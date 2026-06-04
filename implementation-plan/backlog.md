# Backlog

Last updated: 2026-06-03 14:00

---

## Milestone: NOW

### EPIC-0: Foundational Training Correctness

---

### TASK-025: Implement SFT prompt masking for instruction formats
**Type:** Bugfix
**Source:** FOUND-001
**Epic:** EPIC-0
**Milestone:** Now
**Priority:** P0
**Severity:** Critical
**Component(s):** `data/tokenizer.py`, `data/formats.py`
**Owner role:** Backend
**Estimate:** M

**Problem / Goal**
- `tokenize_dataset()` sets `labels = list(input_ids)` for the entire sequence. For instruction-tuning formats (alpaca, chat, sharegpt), the model trains on prompt tokens too. This wastes model capacity and degrades instruction-following quality.
- Every instruction-tuning user is affected.

**Requirements**
- R1: For alpaca format, mask prompt tokens (`instruction` + `input`) with `labels=-100`. Only the `output` portion has real labels.
- R2: For chat/sharegpt format, mask all non-assistant turns with `labels=-100`. Only assistant responses have real labels.
- R3: For text format, keep current behavior (full-sequence training is correct for pretraining/CLM).
- R4: For preference format, not applicable (handled separately).

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given an alpaca sample `{"instruction": "Summarize", "input": "...", "output": "The summary is..."}`
  - When tokenized via `tokenize_dataset()`
  - Then `labels` contains `-100` for all prompt token positions and real token IDs only for the output portion
- AC2:
  - Given a chat sample `{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}`
  - When tokenized
  - Then `labels` contains `-100` for the user turn and real IDs for the assistant turn
- AC3:
  - Given a text sample `{"text": "Raw text..."}`
  - When tokenized
  - Then `labels` equals `input_ids` (unchanged behavior)

**Implementation Notes**
- The format functions (`format_alpaca`, `format_chat`, `format_sharegpt`) currently produce `{"text": "..."}` — a flattened string. To mask properly, they need to produce structured output that preserves the prompt/response boundary.
- Option A: Format functions return `{"prompt": "...", "response": "..."}` instead of `{"text": "..."}`. Tokenizer tokenizes prompt, then response, sets labels=-100 for prompt portion.
- Option B: Format functions return `{"text": "...", "response_start": <char_offset>}`. Tokenizer uses offset to find the label boundary.
- Option A is cleaner. Requires changing format functions + tokenizer.
- Files: `xaytune/data/formats.py` (format_alpaca, format_sharegpt, apply_chat_template), `xaytune/data/tokenizer.py` (tokenize_dataset).

**Testing**
- Integration tests:
  - Tokenize an alpaca sample, verify labels mask matches expected boundaries.
  - Tokenize a chat sample with 3 turns (user/assistant/user/assistant), verify only assistant turns have real labels.
  - Tokenize a text sample, verify labels equal input_ids.
- Edge cases:
  - Empty output/response.
  - Multi-turn with only user messages (should produce all -100 labels).

**Observability**
- Logs: None.
- Metrics: None.

**Dependencies**
- Blocks on: Nothing.
- Blocked-by: Nothing.
- Blocks: TASK-026 (preference masking uses similar pattern).

**Definition of Done**
- [ ] Alpaca format masks prompt tokens
- [ ] Chat/sharegpt format masks non-assistant turns
- [ ] Text format unchanged
- [ ] Integration tests with known token counts
- [ ] No regressions in existing format tests

---

### TASK-026: Mask prompt tokens from preference log-prob computation
**Type:** Bugfix
**Source:** FOUND-002
**Epic:** EPIC-0
**Milestone:** Now
**Priority:** P0
**Severity:** High
**Component(s):** `recipes/align/logprobs.py`, `data/tokenizer.py`
**Owner role:** Backend
**Estimate:** M

**Problem / Goal**
- `get_sequence_logps()` sums log-probs over the full sequence including the shared prompt. DPO/ORPO/SimPO/GRPO all include prompt log-probs in chosen/rejected scores, diluting the preference signal.
- The prompt portion is identical for chosen and rejected — it adds noise but no signal.

**Requirements**
- R1: Preference tokenization must track the prompt length for each chosen/rejected sequence.
- R2: `get_sequence_logps()` must accept a `prompt_length` or `response_mask` parameter to exclude prompt tokens.
- R3: All preference loss steps (_dpo_step, _grpo_step, _orpo_step, _simpo_step) must pass the response mask.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given a preference pair with prompt "What is 2+2?" and chosen "4" and rejected "5"
  - When DPO log-probs are computed
  - Then only the response tokens ("4" / "5") contribute to the sequence log-probability, not the prompt
- AC2:
  - Given identical prompt tokens in chosen and rejected
  - When `get_sequence_logps()` is called with response masking
  - Then the prompt portion contributes zero to the sum

**Implementation Notes**
- `tokenize_preference_dataset()` in `data/tokenizer.py` already tokenizes `prompt + chosen` and `prompt + rejected` separately. It needs to also output `chosen_prompt_length` and `rejected_prompt_length` (number of prompt tokens).
- `get_sequence_logps()` gets a new optional `prompt_length` param. When set, zero out the first `prompt_length` positions in `per_token` before summing.
- Update `collate_preference()` to pad and batch the prompt lengths.
- Update all `_*_step` functions in `loss_dispatch.py` to pass prompt lengths through.
- Files: `data/tokenizer.py`, `recipes/align/logprobs.py`, `recipes/align/loss_dispatch.py`.

**Testing**
- Unit test: synthetic logits + labels + prompt_length → verify only response tokens counted.
- Integration test: DPO step with known inputs → verify prompt doesn't affect result.

**Observability**
- Logs: None.
- Metrics: None.

**Dependencies**
- Blocks on: Nothing.
- Blocked-by: TASK-025 (uses similar masking pattern).

**Definition of Done**
- [ ] Prompt lengths tracked in preference tokenization
- [ ] Log-prob computation excludes prompt tokens
- [ ] All preference loss steps updated
- [ ] Tests verify prompt exclusion

---

### TASK-027: Fix ORPO end-to-end crash (skip_forward + None sft_loss)
**Type:** Bugfix
**Source:** FOUND-003
**Epic:** EPIC-0
**Milestone:** Now
**Priority:** P0
**Severity:** Critical
**Component(s):** `trainer/loop.py`, `recipes/align/loss_dispatch.py`
**Owner role:** Backend
**Estimate:** S

**Problem / Goal**
- Training loop sets `skip_forward=True` for all preference batches (when `chosen_input_ids` is present) and passes `outputs=None` to the loss function.
- ORPO needs the SFT loss from the forward pass. It receives `None` and crashes: `TypeError: unsupported operand type(s) for +: 'NoneType' and 'Tensor'`.
- ORPO has never worked end-to-end.

**Requirements**
- R1: ORPO must receive a valid `outputs` object containing `.loss` from a forward pass on the chosen sequence.
- R2: The `skip_forward` optimization must NOT apply to ORPO (it needs both the standard forward pass AND the preference loss).
- R3: Other preference methods (DPO, SimPO) that don't need `outputs` should still benefit from `skip_forward`.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given `xaytune.align(method="orpo", dataset="prefs.jsonl")`
  - When training runs
  - Then no crash occurs and loss is a finite float
- AC2:
  - Given `xaytune.align(method="dpo", dataset="prefs.jsonl")`
  - When training runs with preference batches
  - Then `skip_forward` is still used (no regression)

**Implementation Notes**
- Option A: In `loop.py`, make `skip_forward` conditional on the loss_fn type. ORPO loss needs `outputs`, others don't.
- Option B: In `_orpo_step`, compute the forward pass internally if `outputs is None` (call `model(**chosen_batch)` to get `sft_loss`).
- Option B is cleaner — keeps the loop generic. `_orpo_step` already calls `model()` twice for chosen/rejected, adding a third call for SFT loss is consistent.
- File: `xaytune/recipes/align/loss_dispatch.py` (`_orpo_step`).

**Testing**
- Integration test: ORPO training on a 2-sample preference dataset with a tiny model → completes without crash, loss is finite.

**Observability**
- Logs: None.
- Metrics: None.

**Dependencies**
- Blocks on: Nothing.
- Blocked-by: Nothing.

**Definition of Done**
- [ ] ORPO runs end-to-end without crash
- [ ] sft_loss is computed from the chosen forward pass
- [ ] DPO skip_forward optimization preserved
- [ ] Test added

---

### TASK-028: Add `prepare_model_for_kbit_training()` to QLoRA path
**Type:** Bugfix
**Source:** FOUND-005
**Epic:** EPIC-0
**Milestone:** Now
**Priority:** P0
**Severity:** High
**Component(s):** `models/peft.py`
**Owner role:** Backend
**Estimate:** S

**Problem / Goal**
- `apply_lora()` calls `get_peft_model()` directly on a 4-bit quantized model without calling `peft.prepare_model_for_kbit_training()` first.
- That function: (a) disables gradients on non-LoRA params, (b) casts layernorm to float32, (c) enables input gradient computation. All critical for stable 4-bit training.
- Without it, QLoRA may produce garbage gradients or dtype mismatches.

**Requirements**
- R1: When `model_result.quantization` is `"4bit"` or `"8bit"`, call `prepare_model_for_kbit_training(model)` before `get_peft_model()`.
- R2: When quantization is `None`, skip the preparation (standard LoRA path unchanged).
- R3: Guard the import — `prepare_model_for_kbit_training` is in `peft`, which is already a required import in this file.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given a model loaded with `quantization="4bit"`
  - When `apply_lora()` is called
  - Then `prepare_model_for_kbit_training()` is called before `get_peft_model()`
- AC2:
  - Given a model loaded without quantization
  - When `apply_lora()` is called
  - Then `prepare_model_for_kbit_training()` is NOT called

**Implementation Notes**
- Add `from peft import prepare_model_for_kbit_training` at the top of `peft.py`.
- Before `get_peft_model()`, check `if model_result.quantization:` and call `prepare_model_for_kbit_training(model_result.model)`.
- File: `xaytune/models/peft.py`.

**Testing**
- Unit test with mock model: verify `prepare_model_for_kbit_training` is called when quantization is set.

**Observability**
- Logs: None.
- Metrics: None.

**Dependencies**
- Blocks on: Nothing.
- Blocked-by: Nothing.

**Definition of Done**
- [ ] `prepare_model_for_kbit_training()` called for quantized models
- [ ] Skipped for non-quantized models
- [ ] Test added

---

### TASK-029: Fix DeepSpeed training loop integration
**Type:** Bugfix
**Source:** FOUND-006
**Epic:** EPIC-0
**Milestone:** Now
**Priority:** P0
**Severity:** Critical
**Component(s):** `trainer/loop.py`, `trainer/distributed.py`
**Owner role:** Backend
**Estimate:** L

**Problem / Goal**
- `ds.initialize()` returns a DeepSpeed engine, but the training loop creates its own `AdamW` optimizer and calls `loss.backward()` / `optimizer.step()` directly. DeepSpeed requires `engine.backward(loss)` / `engine.step()`.
- The optimizer must be passed to `ds.initialize()`, not created separately.
- Current state: DeepSpeed is advertised but silently falls back to vanilla PyTorch behavior (no ZeRO, no offloading).

**Requirements**
- R1: When the model is a DeepSpeed engine, use `model.backward(loss)` instead of `loss.backward()`.
- R2: When the model is a DeepSpeed engine, use `model.step()` instead of `optimizer.step()`.
- R3: The optimizer must be created by DeepSpeed (`ds.initialize(model=model, optimizer=optimizer)` or let DeepSpeed create it from config).
- R4: Gradient accumulation must be delegated to DeepSpeed (it handles it internally).
- R5: The GradScaler must NOT be used with DeepSpeed (it has its own AMP handling).

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given `trainer.strategy="deepspeed"` with ZeRO-2
  - When training runs
  - Then GPU memory usage is reduced compared to vanilla PyTorch (proves ZeRO is active)
- AC2:
  - Given a DeepSpeed-wrapped model
  - When `training_step()` runs
  - Then `engine.backward()` and `engine.step()` are called (not `loss.backward()` / `optimizer.step()`)

**Implementation Notes**
- Detect DeepSpeed engine: `hasattr(model, 'backward')` and `hasattr(model, 'module')` or `isinstance(model, deepspeed.DeepSpeedEngine)`.
- In `Trainer.train()`: if DeepSpeed, skip optimizer/scheduler creation (engine owns them). Skip GradScaler.
- In `training_step()`: if DeepSpeed, call `model.backward(loss)` and `model.step()`. Skip manual grad clipping (engine handles it).
- In `distributed.py`: pass optimizer to `ds.initialize()`.
- Files: `xaytune/trainer/loop.py`, `xaytune/trainer/distributed.py`.

**Testing**
- Integration test on GPU: DeepSpeed ZeRO-2 training completes, memory usage < vanilla PyTorch.
- Unit test: verify `engine.backward()` and `engine.step()` are called (mock engine).

**Observability**
- Logs: Log "Using DeepSpeed engine for backward/step" at training start.
- Metrics: None.

**Dependencies**
- Blocks on: Nothing.
- Blocked-by: Nothing.

**Definition of Done**
- [ ] DeepSpeed engine backward/step used
- [ ] Optimizer passed to ds.initialize
- [ ] GradScaler disabled for DeepSpeed
- [ ] Gradient accumulation delegated
- [ ] Integration test on GPU
- [ ] Unit test with mock engine

---

### TASK-030: Wire alignment loss into Studio training path
**Type:** Bugfix
**Source:** FOUND-007
**Epic:** EPIC-0
**Milestone:** Now
**Priority:** P1
**Severity:** High
**Component(s):** `studio/app.py`
**Owner role:** Backend
**Estimate:** S

**Problem / Goal**
- Studio builds a config and calls `setup_training()` + `trainer.train()` directly. Alignment loss functions are constructed inside `align()`, not in `setup_training()`.
- Studio-launched DPO/GRPO/ORPO jobs use default cross-entropy loss instead of the alignment loss.
- Users get no error — just wrong results.

**Requirements**
- R1: When Studio detects `recipe="align"`, it must use the `align()` function (or equivalent loss setup) instead of calling `trainer.train()` directly.
- R2: Alternatively, move alignment loss construction into `setup_training()` so all callers get it automatically.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given Studio config with `recipe="align", method="dpo"`
  - When training is launched from Studio
  - Then DPO loss is used (not cross-entropy)

**Implementation Notes**
- Cleanest fix: Studio should call `xaytune.align(config=config)` instead of building its own training loop.
- Alternative: move loss_fn construction from `align()` into `setup_training()` when `recipe="align"`. This benefits all callers.
- File: `xaytune/studio/app.py` (training launch function).

**Testing**
- Test that Studio with align recipe produces a loss_fn (mock or assert).

**Observability**
- Logs: None.
- Metrics: None.

**Dependencies**
- Blocks on: Nothing.
- Blocked-by: TASK-027 (ORPO must work first).

**Definition of Done**
- [ ] Studio align jobs use alignment loss
- [ ] Test added

---

### TASK-031: Rename PPO or mark experimental
**Type:** Refactor
**Source:** FOUND-004
**Epic:** EPIC-0
**Milestone:** Now
**Priority:** P1
**Severity:** High
**Component(s):** `recipes/align/ppo.py`, docs, examples
**Owner role:** Backend
**Estimate:** S

**Problem / Goal**
- The "PPO" implementation is a clipped policy gradient loss — not a full PPO trainer. It lacks rollout buffers, GAE, value models, and multiple optimization epochs. Calling it PPO is misleading.

**Requirements**
- R1: Either rename `ppo_clip_loss` to `clipped_pg_loss` and update method name, OR add prominent documentation stating it is an experimental/simplified implementation.
- R2: Update docstrings, examples, and the alignment method table.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given the method table in `04_alignment.ipynb`
  - When a user reads the PPO section
  - Then it clearly states this is a simplified clipped policy gradient, not full PPO

**Implementation Notes**
- Simplest: keep `method="ppo"` for API compatibility but update all prose to say "Clipped Policy Gradient (simplified PPO)".
- Files: `recipes/align/ppo.py` (docstrings), `examples/04_alignment.ipynb`, `docs/`.

**Testing**
- Review only — no functional change.

**Observability**
- Logs: None.

**Dependencies**
- Blocks on: Nothing.
- Blocked-by: Nothing.

**Definition of Done**
- [ ] Docstrings updated
- [ ] Example notebook updated
- [ ] No functional regression

---

### EPIC-2: Alignment Numerical Stability

---

### TASK-001: Fix ORPO numerical instability (NaN/Inf on edge-case log-probs)
**Type:** Bugfix
**Source:** BUG-011
**Epic:** EPIC-2
**Milestone:** Now
**Priority:** P0
**Severity:** Critical
**Component(s):** `recipes/align/orpo.py`
**Owner role:** Backend
**Estimate:** S

**Problem / Goal**
- `orpo_loss()` computes `exp(logps) / (1 - exp(logps))`. When log-probs approach 0 (i.e., probability ~1.0), `exp(0) = 1`, denominator = 0, producing Inf. Subsequent `torch.log()` compounds to NaN.
- Training silently produces garbage gradients or crashes.

**Requirements**
- R1: ORPO loss must return finite values for all valid log-probability inputs including 0.
- R2: Use numerically stable formulation: `log_odds_ratio = chosen_logps - rejected_logps`, then `log_sigmoid(log_odds_ratio)`.
- R3: No change to the mathematical result for normal inputs.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given log-probabilities where `chosen_logps = 0.0` (edge case)
  - When `orpo_loss()` is called
  - Then the result is a finite tensor (no NaN, no Inf)
- AC2:
  - Given normal log-probabilities (e.g., -2.0, -3.0)
  - When `orpo_loss()` is called
  - Then the result matches the original formula within 1e-6

**Implementation Notes**
- Replace explicit odds computation with: `log_odds_ratio = policy_chosen_logps - policy_rejected_logps`, then use `F.logsigmoid(log_odds_ratio)` for the OR component.
- Touch: `xaytune/recipes/align/orpo.py` (loss function), `xaytune/recipes/align/loss_dispatch.py` (`_orpo_step` if it passes odds directly).

**Testing**
- Integration tests:
  - Test `orpo_loss()` with `chosen_logps=0.0, rejected_logps=-5.0` — must return finite.
  - Test `orpo_loss()` with normal values — must match reference within tolerance.

**Observability**
- Logs: None needed.
- Metrics: None needed.

**Dependencies**
- Blocks on: Nothing.
- Blocked-by: Nothing.

**Definition of Done**
- [ ] Numerically stable formulation implemented
- [ ] Edge-case test added
- [ ] Regression test for normal inputs added
- [ ] No NaN/Inf for any valid input

---

### TASK-002: Add SimPO zero-length sequence guard
**Type:** Bugfix
**Source:** BUG-027
**Epic:** EPIC-2
**Milestone:** Now
**Priority:** P2
**Severity:** Low
**Component(s):** `recipes/align/simpo.py`
**Owner role:** Backend
**Estimate:** XS

**Problem / Goal**
- `simpo_loss()` divides by `chosen_lengths` and `rejected_lengths`. If either is 0, produces Inf.
- Defensive guard prevents silent corruption.

**Requirements**
- R1: Clamp lengths to `min=1` before division.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given `chosen_lengths` containing a 0
  - When `simpo_loss()` is called
  - Then result is finite

**Implementation Notes**
- Add `.clamp(min=1)` to both `chosen_lengths` and `rejected_lengths` before division in `simpo.py`.

**Testing**
- Unit test with zero-length input.

**Dependencies**
- Blocks on: Nothing.
- Blocked-by: Nothing.

**Definition of Done**
- [ ] Clamp added
- [ ] Test added

---

### EPIC-3: Eval Pipeline Completeness

---

### TASK-003: Fix eval_callback dummy metrics (same bug as evaluate.py)
**Type:** Bugfix
**Source:** BUG-006 / BUG-020
**Epic:** EPIC-3
**Milestone:** Now
**Priority:** P1
**Severity:** High
**Component(s):** `trainer/eval_callback.py`
**Owner role:** Backend
**Estimate:** S

**Problem / Goal**
- `eval_callback.py` has the same pattern as the (now-fixed) `evaluate.py`: non-loss/perplexity metrics get `compute_fn([], [])`, always returning 0.
- During-training eval with `token_accuracy` is broken.

**Requirements**
- R1: Collect argmax predictions and labels (masked by `-100`) during eval loop.
- R2: Pass predictions/references to non-loss metrics.
- R3: Move batches to model device before forward pass.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given `EvalConfig(metrics=["loss", "token_accuracy"])`
  - When training-time eval runs
  - Then `state.metrics["eval_token_accuracy"]` is a nonzero float

**Implementation Notes**
- Mirror the pattern applied to `eval/evaluate.py`: collect `logits.argmax(dim=-1)` and `labels` where `labels != -100`.
- File: `xaytune/trainer/eval_callback.py`.

**Testing**
- Integration test with mock model and known predictions.

**Dependencies**
- Blocks on: Nothing.
- Blocked-by: Nothing.

**Definition of Done**
- [ ] Predictions/references collected
- [ ] Device handling added
- [ ] Test added

---

### EPIC-4: Checkpoint & Device Portability

---

### TASK-004: Add `map_location` to all `torch.load()` calls
**Type:** Bugfix
**Source:** BUG-013
**Epic:** EPIC-4
**Milestone:** Now
**Priority:** P1
**Severity:** High
**Component(s):** `trainer/checkpointing.py`, `trainer/loop.py`
**Owner role:** Backend
**Estimate:** S

**Problem / Goal**
- `torch.load()` in `load_checkpoint()` and `Trainer.train()` resume path lack `map_location`. GPU checkpoints crash when loaded on CPU.

**Requirements**
- R1: All `torch.load()` calls must use `map_location="cpu"`.
- R2: After loading to CPU, tensors move to correct device during normal model/optimizer setup.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given a checkpoint saved on `cuda:0`
  - When `load_checkpoint()` is called on a CPU-only machine
  - Then checkpoint loads without error
- AC2:
  - Given a checkpoint saved on CPU
  - When loaded on a GPU machine
  - Then checkpoint loads and model is moved to GPU during setup

**Implementation Notes**
- `trainer/checkpointing.py`: 4 `torch.load()` calls at lines ~72, 78, 84, 90.
- `trainer/loop.py`: 3 `torch.load()` calls at lines ~108, 111, 114.
- Add `map_location="cpu"` to all.

**Testing**
- Unit test: save on one device, load with `map_location="cpu"`, verify no error.

**Dependencies**
- Blocks on: Nothing.
- Blocked-by: Nothing.

**Definition of Done**
- [ ] All `torch.load()` calls have `map_location="cpu"`
- [ ] Cross-device test added

---

### TASK-005: Fix checkpoint metadata serialization of tensor metrics
**Type:** Bugfix
**Source:** BUG-026
**Epic:** EPIC-4
**Milestone:** Now
**Priority:** P1
**Severity:** Medium
**Component(s):** `trainer/checkpointing.py`
**Owner role:** Backend
**Estimate:** XS

**Problem / Goal**
- `save_checkpoint()` writes `json.dumps(state.metrics)`, but metrics may contain `torch.Tensor` scalars. `TypeError` during save loses the checkpoint.

**Requirements**
- R1: Convert all metric values to Python floats before JSON serialization.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given `state.metrics = {"loss": torch.tensor(0.5)}`
  - When `save_checkpoint()` is called
  - Then `metadata.json` is written without error, containing `{"loss": 0.5}`

**Implementation Notes**
- Use `state.to_dict()` which already handles `.item()` conversion, or add inline conversion.
- File: `xaytune/trainer/checkpointing.py`, near the `json.dumps` call.

**Testing**
- Unit test with tensor-valued metrics.

**Dependencies**
- Blocks on: Nothing.
- Blocked-by: Nothing.

**Definition of Done**
- [ ] Tensor metrics converted before serialization
- [ ] Test added

---

### EPIC-5: Config Validation (Critical fix)

---

### TASK-006: Add `reinforce` to validation `_ALIGN_METHODS`
**Type:** Bugfix
**Source:** BUG-014
**Epic:** EPIC-5
**Milestone:** Now
**Priority:** P1
**Severity:** High
**Component(s):** `config/validation.py`
**Owner role:** Backend
**Estimate:** XS

**Problem / Goal**
- `_ALIGN_METHODS` in `validation.py` is `{"dpo", "grpo", "ppo", "orpo", "simpo"}`. Missing `"reinforce"`. CLI rejects valid `method: reinforce` configs.

**Requirements**
- R1: Add `"reinforce"` to `_ALIGN_METHODS`.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given config with `recipe="align", method="reinforce"`
  - When `validate_config()` is called
  - Then no validation error is raised

**Implementation Notes**
- One-line change in `xaytune/config/validation.py`.

**Testing**
- Extend existing validation tests.

**Dependencies**
- Blocks on: Nothing.
- Blocked-by: Nothing.

**Definition of Done**
- [ ] `reinforce` added to set
- [ ] Test added

---

## Milestone: NEXT

### EPIC-5: Config Validation (Full)

---

### TASK-007: Call `validate_config()` from `setup_training()`
**Type:** Missing Feature
**Source:** GAP-003
**Epic:** EPIC-5
**Milestone:** Next
**Priority:** P1
**Component(s):** `recipes/base.py`, `config/validation.py`
**Owner role:** Backend
**Estimate:** S

**Problem / Goal**
- `validate_config()` is only called from CLI. Studio and Python API users get no cross-field validation.

**Requirements**
- R1: `setup_training()` calls `validate_config(config)` before proceeding.
- R2: Validation errors produce clear messages.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given `xaytune.finetune(model="x", dataset="y", method="invalid")`
  - When called via Python API
  - Then a `ConfigValidationError` is raised with a clear message

**Implementation Notes**
- Add `from xaytune.config.validation import validate_config` and call it early in `setup_training()`.

**Testing**
- Test that invalid configs raise from the Python API.

**Dependencies**
- Blocked-by: TASK-006 (reinforce must be valid first).

**Definition of Done**
- [ ] Validation called from `setup_training()`
- [ ] Test added

---

### TASK-008: Validate `apply_overrides()` keys against schema
**Type:** Missing Feature
**Source:** GAP-005
**Epic:** EPIC-5
**Milestone:** Next
**Priority:** P1
**Component(s):** `config/parser.py`
**Owner role:** Backend
**Estimate:** S

**Problem / Goal**
- `apply_overrides()` silently creates new keys for typos (e.g., `trainer.lerning_rate=1e-4`).

**Requirements**
- R1: Reject override keys that don't exist in the `TrainConfig` schema.
- R2: Suggest closest valid key (Levenshtein distance or simple prefix match).

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given override `"trainer.lerning_rate=1e-4"`
  - When `apply_overrides()` is called
  - Then a `ValueError` is raised mentioning `"learning_rate"` as suggestion

**Implementation Notes**
- After parsing `key=value`, walk the schema to verify the key path exists.
- File: `xaytune/config/parser.py`.

**Testing**
- Test with typo'd key, valid key, nested key.

**Dependencies**
- Blocks on: Nothing.

**Definition of Done**
- [ ] Invalid keys rejected
- [ ] Suggestion provided
- [ ] Test added

---

### TASK-009: Add pretrain validation rules
**Type:** Missing Feature
**Source:** GAP-004
**Epic:** EPIC-5
**Milestone:** Next
**Priority:** P2
**Component(s):** `config/validation.py`
**Owner role:** Backend
**Estimate:** S

**Problem / Goal**
- `recipe="pretrain"` has zero validation. `method="lora"` with pretrain silently does SFT behavior.

**Requirements**
- R1: Validate pretrain method is `"full"` (only valid method).
- R2: Validate data format is `"text"`.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given `recipe="pretrain", method="lora"`
  - When `validate_config()` is called
  - Then raises error: pretrain only supports method="full"

**Implementation Notes**
- Add pretrain validation block in `validation.py`.

**Testing**
- Test valid and invalid pretrain configs.

**Dependencies**
- Blocked-by: TASK-007 (validation must be called from API).

**Definition of Done**
- [ ] Pretrain rules added
- [ ] Tests added

---

### EPIC-6: Logging Robustness

---

### TASK-010: Fix MLflow `log_params` crash with nested config
**Type:** Bugfix
**Source:** BUG-015
**Epic:** EPIC-6
**Milestone:** Next
**Priority:** P1
**Severity:** Medium
**Component(s):** `logging/mlflow.py`
**Owner role:** Backend
**Estimate:** S

**Problem / Goal**
- `mlflow.log_params(config)` receives a nested dict. MLflow requires flat `dict[str, str]`. Crashes at training start.

**Requirements**
- R1: Flatten nested config dict before calling `log_params`.
- R2: Use dot-notation keys (e.g., `model.name`, `trainer.batch_size`).

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given a `TrainConfig` with nested `model`, `data`, `trainer` sections
  - When MLflow backend `log_config()` is called
  - Then no exception is raised and params are logged with dot-notation keys

**Implementation Notes**
- Write a `_flatten_dict(d, prefix="")` helper. Apply to config before `log_params`.
- File: `xaytune/logging/mlflow.py`.

**Testing**
- Unit test with mock MLflow.

**Dependencies**
- Blocks on: Nothing.

**Definition of Done**
- [ ] Config flattened before `log_params`
- [ ] Test with nested dict

---

### TASK-011: Isolate logging backend exceptions
**Type:** Missing Feature
**Source:** BUG-023
**Epic:** EPIC-6
**Milestone:** Next
**Priority:** P1
**Component(s):** `logging/base.py`
**Owner role:** Backend
**Estimate:** S

**Problem / Goal**
- A crash in one logging backend (e.g., WandB network error) kills the training loop.

**Requirements**
- R1: Wrap each backend's `log_scalar`/`log_config` call in try/except.
- R2: Log the exception to stderr but continue training.
- R3: After 10 consecutive failures from one backend, disable it with a warning.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given WandB backend raises `ConnectionError` on `log_scalar`
  - When `log_scalar` is called
  - Then training continues and a warning is logged to stderr

**Implementation Notes**
- File: `xaytune/logging/base.py`, `log_scalar()` and `log_config()` methods.

**Testing**
- Unit test with mock backend that raises.

**Dependencies**
- Blocks on: Nothing.

**Definition of Done**
- [ ] Exception isolation implemented
- [ ] Auto-disable after threshold
- [ ] Test added

---

### TASK-012: Guard optional backend imports
**Type:** Bugfix
**Source:** BUG-022
**Epic:** EPIC-10
**Milestone:** Next
**Priority:** P1
**Severity:** High
**Component(s):** `logging/tensorboard.py`, `logging/wandb.py`, `logging/mlflow.py`, `models/peft.py`
**Owner role:** Backend
**Estimate:** S

**Problem / Goal**
- Optional dependencies (`tensorboard`, `wandb`, `mlflow`, `peft`) are imported at module top-level. Missing package = raw `ImportError`.

**Requirements**
- R1: Guard imports with try/except. Set module reference to `None` when missing.
- R2: On first use, raise `ImportError` with message: `"pip install xaytune[<extras-group>]"`.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given `peft` is not installed
  - When `xaytune.finetune(method="lora")` is called
  - Then error message says `"pip install xaytune[peft]"` or `"pip install peft"`

**Implementation Notes**
- Pattern: `try: import peft except ImportError: peft = None`. Then check at call sites.
- Files: `models/peft.py`, `logging/tensorboard.py`, `logging/wandb.py`, `logging/mlflow.py`.

**Testing**
- Test by mocking `sys.modules` to simulate missing package.

**Dependencies**
- Blocks on: Nothing.

**Definition of Done**
- [ ] All optional imports guarded
- [ ] Helpful error messages
- [ ] Test added

---

### EPIC-7: Export Pipeline

---

### TASK-013: Save `config.json` in `model_merge` output
**Type:** Bugfix
**Source:** BUG-012
**Epic:** EPIC-7
**Milestone:** Next
**Priority:** P1
**Severity:** High
**Component(s):** `export/model_merge.py`
**Owner role:** Backend
**Estimate:** S

**Problem / Goal**
- `_save_merged()` writes `pytorch_model.bin` but not `config.json`. Output is not loadable by `from_pretrained()`.

**Requirements**
- R1: Copy `config.json` from the first source model to the output directory.
- R2: If first source is a HF model name, load its config and save it.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given two local model directories
  - When `model_merge()` runs with TIES method
  - Then output directory contains both `pytorch_model.bin` and `config.json`

**Implementation Notes**
- Load `AutoConfig.from_pretrained(model_paths[0])` and call `.save_pretrained(output_dir)`.
- File: `xaytune/export/model_merge.py`.

**Testing**
- Integration test that verifies `AutoModelForCausalLM.from_pretrained(output_dir)` succeeds.

**Dependencies**
- Blocks on: Nothing.

**Definition of Done**
- [ ] `config.json` saved
- [ ] Loadability test added

---

### TASK-014: Fix GGUF conversion subprocess command
**Type:** Missing Feature
**Source:** GAP-001
**Epic:** EPIC-7
**Milestone:** Next
**Priority:** P2
**Component(s):** `export/gguf.py`
**Owner role:** Backend
**Estimate:** M

**Problem / Goal**
- Shells out to `python -m llama_cpp.convert` which doesn't exist. GGUF conversion always fails.

**Requirements**
- R1: Use correct conversion command (`convert_hf_to_gguf.py` from llama.cpp, or `llama-cpp-python` if available).
- R2: Check tool availability before running. Give clear install instructions on failure.
- R3: Support at least Q4_K_M, Q5_K_M, Q8_0 quantization presets.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given llama.cpp conversion tool is not installed
  - When `to_gguf()` is called
  - Then a clear error message explains how to install it

**Implementation Notes**
- Check for `convert_hf_to_gguf.py` on PATH or in common locations.
- File: `xaytune/export/gguf.py`.

**Testing**
- Unit test that the error message is produced when tool is missing.

**Dependencies**
- Blocks on: Nothing.

**Definition of Done**
- [ ] Correct subprocess command
- [ ] Tool availability check
- [ ] Clear error message
- [ ] Test added

---

### TASK-015: Warn when pushing model to Hub without tokenizer
**Type:** Missing Feature
**Source:** GAP-002
**Epic:** EPIC-7
**Milestone:** Next
**Priority:** P2
**Component(s):** `export/hub.py`
**Owner role:** Backend
**Estimate:** XS

**Problem / Goal**
- Pushing a model object without tokenizer silently publishes an incomplete model.

**Requirements**
- R1: Emit `warnings.warn()` when tokenizer is None and model is not a string path.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given a model object passed without tokenizer
  - When `push_to_hub()` is called
  - Then a warning is emitted about missing tokenizer

**Implementation Notes**
- File: `xaytune/export/hub.py`.

**Testing**
- Test with `warnings.catch_warnings()`.

**Dependencies**
- Blocks on: Nothing.

**Definition of Done**
- [ ] Warning added
- [ ] Test added

---

### EPIC-8: Data Pipeline

---

### TASK-016: Warn on unknown text keys in `format_text()`
**Type:** Missing Feature
**Source:** BUG-024
**Epic:** EPIC-8
**Milestone:** Next
**Priority:** P1
**Component(s):** `data/formats.py`
**Owner role:** Backend
**Estimate:** XS

**Problem / Goal**
- `format_text()` returns `{"text": ""}` silently when sample has no `"text"` or `"content"` key. Entire dataset becomes empty.

**Requirements**
- R1: When neither `"text"` nor `"content"` exists, emit a warning listing the sample's actual keys.
- R2: Warn once per unique key set (not per sample — avoid log spam).

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given a sample `{"body": "Hello"}`
  - When `format_text()` is called
  - Then a warning is emitted: `"Sample has no 'text' or 'content' key. Found keys: ['body']"`

**Implementation Notes**
- Use a module-level `_warned_keys = set()` to dedupe.
- File: `xaytune/data/formats.py`.

**Testing**
- Test with `warnings.catch_warnings()`.

**Dependencies**
- Blocks on: Nothing.

**Definition of Done**
- [ ] Warning added
- [ ] Dedupe implemented
- [ ] Test added

---

### TASK-017: Shuffle preference dataset before splitting
**Type:** Bugfix
**Source:** BUG-025 (merged with BUG-009)
**Epic:** EPIC-8
**Milestone:** Next
**Priority:** P2
**Component(s):** `data/preferences.py`
**Owner role:** Backend
**Estimate:** XS

**Problem / Goal**
- `load_preference_dataset()` splits by index without shuffling. Same bug as (fixed) `_split_dataset` in `loader.py`.

**Requirements**
- R1: Shuffle with `random.Random(42)` before splitting.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given an ordered preference dataset
  - When split with `eval_split=0.1`
  - Then eval samples are drawn from across the dataset, not just the tail

**Implementation Notes**
- Apply same pattern as `loader.py` fix.
- File: `xaytune/data/preferences.py`.

**Testing**
- Test that first/last samples are mixed after split.

**Dependencies**
- Blocks on: Nothing.

**Definition of Done**
- [ ] Shuffle added
- [ ] Test added

---

### EPIC-9: Studio & CLI

---

### TASK-018: Fix Studio data format dropdown
**Type:** Bugfix
**Source:** BUG-016
**Epic:** EPIC-9
**Milestone:** Next
**Priority:** P1
**Severity:** Medium
**Component(s):** `studio/app.py`
**Owner role:** Backend
**Estimate:** XS

**Problem / Goal**
- Dropdown lists `["alpaca", "sharegpt", "completion", "pretrain"]`. `"completion"` and `"pretrain"` are not registered formats. `"chat"`, `"text"`, `"preference"` are missing.

**Requirements**
- R1: Dropdown options must match the format registry: `["alpaca", "sharegpt", "chat", "text", "preference"]`.
- R2: Ideally, populate from `format_registry.list()` dynamically.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given a user opens Studio
  - When they click the format dropdown
  - Then they see: alpaca, sharegpt, chat, text, preference

**Implementation Notes**
- File: `xaytune/studio/app.py`, line ~343.

**Testing**
- Assert dropdown choices match registry keys.

**Dependencies**
- Blocks on: Nothing.

**Definition of Done**
- [ ] Dropdown corrected
- [ ] Test added

---

### TASK-019: Fix CLI eval crash without `--metrics`
**Type:** Bugfix
**Source:** BUG-017
**Epic:** EPIC-9
**Milestone:** Next
**Priority:** P1
**Severity:** Medium
**Component(s):** `cli.py`
**Owner role:** Backend
**Estimate:** XS

**Problem / Goal**
- `xaytune eval --model X --dataset Y` crashes with `AttributeError` because `args.metrics` is `None` and code tries to `.split()` it.

**Requirements**
- R1: Default `--metrics` to `"loss,perplexity"` when not provided.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given `xaytune eval --model X --dataset Y` with no `--metrics`
  - When the command runs
  - Then it evaluates with loss and perplexity (no crash)

**Implementation Notes**
- File: `xaytune/cli.py`, in `_handle_eval()`.

**Testing**
- CLI test with and without `--metrics`.

**Dependencies**
- Blocks on: Nothing.

**Definition of Done**
- [ ] Default metrics set
- [ ] Test added

---

## Milestone: LATER

### EPIC-11: Trainer Utilities

---

### TASK-020: Add numpy to `seed_all()`
**Type:** Bugfix
**Source:** BUG-018
**Epic:** EPIC-11
**Milestone:** Later
**Priority:** P2
**Severity:** Medium
**Component(s):** `trainer/device.py`
**Owner role:** Backend
**Estimate:** XS

**Problem / Goal**
- `seed_all()` seeds `torch` and `random` but not `numpy`. Data pipelines using numpy are non-reproducible.

**Requirements**
- R1: Seed numpy if installed. Don't crash if numpy is not available.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given numpy is installed
  - When `seed_all(42)` is called
  - Then `numpy.random.RandomState` is seeded to 42

**Implementation Notes**
- `try: import numpy; numpy.random.seed(seed) except ImportError: pass`
- File: `xaytune/trainer/device.py`.

**Testing**
- Test reproducibility of numpy random calls after seeding.

**Dependencies**
- Blocks on: Nothing.

**Definition of Done**
- [ ] Numpy seeded when available
- [ ] Test added

---

### TASK-021: Auto-select distributed backend (gloo fallback for CPU)
**Type:** Bugfix
**Source:** BUG-019
**Epic:** EPIC-11
**Milestone:** Later
**Priority:** P2
**Severity:** Medium
**Component(s):** `trainer/distributed.py`
**Owner role:** Backend
**Estimate:** XS

**Problem / Goal**
- `init_distributed()` hardcodes `backend="nccl"`. Fails on CPU or non-NVIDIA hardware.

**Requirements**
- R1: Use `"nccl"` when CUDA is available, `"gloo"` otherwise.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given `WORLD_SIZE=2` on a CPU-only machine
  - When `init_distributed()` is called
  - Then `torch.distributed.init_process_group(backend="gloo")` is used

**Implementation Notes**
- `backend = "nccl" if torch.cuda.is_available() else "gloo"`
- File: `xaytune/trainer/distributed.py`.

**Testing**
- Unit test with mocked `torch.cuda.is_available()`.

**Dependencies**
- Blocks on: Nothing.

**Definition of Done**
- [ ] Backend auto-selected
- [ ] Test added

---

### TASK-022: Fix LR finder device mismatch
**Type:** Bugfix
**Source:** BUG-021
**Epic:** EPIC-11
**Milestone:** Later
**Priority:** P2
**Severity:** Medium
**Component(s):** `trainer/lr_finder.py`
**Owner role:** Backend
**Estimate:** S

**Problem / Goal**
- `lr_find()` doesn't move batches to the model's device. Crashes on GPU models with CPU data.

**Requirements**
- R1: Detect model device and move batch tensors before forward pass.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given a model on `cuda:0` and dataloader returning CPU tensors
  - When `lr_find()` is called
  - Then it completes without device mismatch error

**Implementation Notes**
- Same pattern as `Trainer.move_batch_to_device()`.
- File: `xaytune/trainer/lr_finder.py`.

**Testing**
- Test with mock model reporting a device.

**Dependencies**
- Blocks on: Nothing.

**Definition of Done**
- [ ] Device transfer added
- [ ] Test added

---

### TASK-023: Fix constant scheduler ignoring warmup
**Type:** Bugfix
**Source:** BUG-029
**Epic:** EPIC-11
**Milestone:** Later
**Priority:** P3
**Severity:** Low
**Component(s):** `trainer/scheduler.py`
**Owner role:** Backend
**Estimate:** XS

**Problem / Goal**
- `scheduler="constant"` with `warmup_steps > 0` silently skips warmup. Only `"constant_with_warmup"` respects it.

**Requirements**
- R1: If `scheduler="constant"` and `warmup_steps > 0`, auto-upgrade to `"constant_with_warmup"` behavior.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given `scheduler="constant", warmup_steps=100`
  - When scheduler is created
  - Then the first 100 steps have linearly increasing LR

**Implementation Notes**
- File: `xaytune/trainer/scheduler.py`.

**Testing**
- Test LR schedule output for first N steps.

**Dependencies**
- Blocks on: Nothing.

**Definition of Done**
- [ ] Warmup applied
- [ ] Test added

---

### TASK-024: Fix agent tokenizer BOS duplication
**Type:** Bugfix
**Source:** BUG-028
**Epic:** EPIC-8
**Milestone:** Later
**Priority:** P3
**Severity:** Low
**Component(s):** `data/agent_tokenizer.py`
**Owner role:** Backend
**Estimate:** S

**Problem / Goal**
- `tokenize_agent_dataset()` tokenizes each message independently, producing multiple BOS tokens when concatenated.

**Requirements**
- R1: Only the first message should get BOS. Subsequent messages use `add_special_tokens=False`.

**Acceptance Criteria (Gherkin)**
- AC1:
  - Given a 3-message agent trajectory
  - When tokenized
  - Then the resulting token sequence contains exactly 1 BOS token (at position 0)

**Implementation Notes**
- File: `xaytune/data/agent_tokenizer.py`.

**Testing**
- Test with a mock tokenizer that has a known BOS token.

**Dependencies**
- Blocks on: Nothing.

**Definition of Done**
- [ ] Single BOS per sequence
- [ ] Test added
