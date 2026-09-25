# Changelog

## Unreleased

Everything below is on `main` and not yet released; `pyproject.toml` stays at
`0.6.0` until it is. The first release of the experiment control plane is
planned as `1.0.0a1`, once it can train, evaluate and decide.

### Added — the experiment control plane

- **Durable persistence** (`xaytune.storage`): SQLite, versioned migrations 001–006, revision-based optimistic concurrency, and atomic state + event + outbox transactions. A `RuntimeOperation` journal records every external effect before it is attempted, and Actions record requests such as cancellation together with their outcomes.
- **`CandidateSpec` and versioned fingerprints**: `CandidateFingerprint`, `RunHistoryFingerprint`, `ArtifactLineageFingerprint`.
- **Compile/execute boundary** (`xaytune.compilation`): `TrainerCompiler`, `TrainingExecutionSpec` and `ResolvedExecutionPlan`, with `NativeCompiler` and `TRLCompiler` for full-parameter SFT. A candidate that leaves a training-relevant value undeclared is refused, with every reason.
- **`LocalRuntime`** (`xaytune.runtimes`): workers run as separate processes, submission is idempotent through `submit_or_get`, and telemetry is versioned (v1alpha2 for training, v1alpha3 for evaluation) with a durable cursor.
- **`EmbeddedControllerHost` and `ExperimentHandle`** (`xaytune.experiment`): `submit`, `status`, `wait`, `cancel`, `events`, and `attach`, which adopts running work after a restart and never submits it twice.
- **The durable evaluation lifecycle**: `EvaluationSpec`, evaluation runs, attempts and results, evaluation cycles, and the `Evaluator` contract.
- **`NativeEvaluator`, the built-in evaluator** (`xaytune.evaluation.native`, worker `xaytune.workers.eval_native`), registered by default as `native`. It measures next-token loss, perplexity and token accuracy (token-weighted, logits at *i* against the token at *i + 1*) on a local JSONL file of held-out text, pinned by its content digest (`local_dataset()`). The worker refuses a file whose bytes changed. It declares itself `SEEDED`, records the run's seed and evaluator version on every metric, and writes a report naming the device and library versions. Everything that changes a number (format, truncation, batching, precision, metrics) is required configuration, part of the `EvaluationFingerprint`.
- **`Evaluator.supports(spec)`**: the host asks at submission, so an evaluation that cannot run exactly as declared is refused before anything trains, with every reason. A refusal from `prepare()` after training (a subject the evaluator cannot read) now fails the evaluation run with its reasons and reports the cycle stalled, instead of escaping the controller. `UnsupportedEvaluationError` carries `evaluator` and `reasons`.
- **`examples/control_plane/05_train_and_evaluate.py`**: train, evaluate the trained model, and decide against a loss `--target`.
- **Durable decisions (PR-015)**: `DecisionContext`, `DecisionProposal` and `Decision` (`xaytune.core.domain.decision`), and the `DecisionEngine` contract with the built-in `ThresholdDecisionEngine` (`xaytune.decision`). The engine is pure: it returns a proposal with no id, time or actor, which the repository adds when it records it. The input fingerprint is an explicit, versioned projection of the evidence (`decision_input_identity_v1`).
  - The engine decides from the record alone: the objective, and the evaluation results of the candidate's current cycle. It compares recorded values with the target and constraints (all six operators, applied exactly). A violated constraint gives `REJECT`, a target met gives `STOP_SUCCEEDED`, and a target missed gives `STOP_FAILED`.
  - Without a target, or with a metric missing, nothing is decided: the candidate stays `DECIDING`, with a `DecisionDeferred` event.
  - Migration 007 records each decision, with its evidence and an input fingerprint, in one commit with what it causes: `STOP_SUCCEEDED` completes the node and succeeds the experiment, setting `best_node_id`; `STOP_FAILED` rejects the node and fails the experiment; `REJECT` rejects the node and leaves the experiment `ACTIVE`, with `next_stage` `"planning"`.
  - Idempotent per cycle: a different decision for a decided cycle is refused, and a controller that restarts decides once.
  - `EmbeddedControllerHost` takes an optional `decision_engine`.
- **`examples/control_plane/`**, and control-plane *Getting started* and *Concepts* pages in the docs.
- **A release gate**: publishing a GitHub release fails unless its tag is `v` plus the `pyproject.toml` version, and `xaytune/_version.py` agrees (`scripts/check_release_version.py`). Manual TestPyPI dispatches are not gated.
- **One runtime version**: `xaytune/_version.py`, an import-free module. `xaytune.__version__` and the `xaytune_version` of every built-in plugin descriptor (`NativeCompiler`, `TRLCompiler`, `LocalRuntime`, `NativeEvaluator`) come from it instead of hardcoded literals, and a test refuses any new literal.
- **An sdist allow-list**: the source distribution contains `xaytune/`, `pyproject.toml`, `README.md`, `CHANGELOG.md` and `LICENSE`, whatever else is in the working tree. `scripts/check_sdist.py` checks the built archive in CI and before every upload, including that `LICENSE` ships and `pyproject.toml` declares `Apache-2.0`. The wheel is built from the sdist.
- **`LICENSE`**: the Apache License 2.0 text, verbatim, which the package already declared and did not ship.
- **A wheel smoke test** in CI and before every upload. The built wheel is installed into a fresh virtualenv, then must import without torch, transformers or TRL, migrate a new store, compile a candidate, round-trip the plan, and import the control-plane examples (`scripts/wheel_smoke.py`).

### Fixed — evaluation metrics (correctness)

- **`token_accuracy` scored the wrong token** in `xaytune.eval.evaluate()` and in the in-training evaluation (`eval.every_n_steps`). It compared the logits at position *i* with the label at *i*. In a causal LM those logits predict the token at *i + 1*, so a model that merely echoed its input scored perfectly: an untrained test model reported 1.0 where its next-token accuracy was 0.0. It now compares each prediction with the next label, and labels of `-100` count in neither the numerator nor the denominator (issue #36).
- **`loss` and `perplexity` depended on batching.** They averaged per-batch losses, so a batch with 3 next-token targets weighed as much as one with 300. Each batch's loss is now weighted by its target count, which gives the mean over every target in the dataset: the same examples batched differently give the same answer. `perplexity` is `exp` of that loss. A batch with no next-token target contributes nothing.
- **Reported values change.** This is a correctness fix, not an API change: the functions, their arguments and the metric names are unchanged, and the earlier values were wrong. `compute_loss` and `compute_perplexity` take an optional `weights`, one non-negative target count per loss: a mismatched or negative weight raises `ValueError`, and weights summing to zero report `0.0`, as no losses always have. Without `weights` they average plainly, exactly as before.

### Changed — positioning and dependencies

- The package description is now "Agent-native experiment control plane for model post-training and adaptation". The README and docs separate what is available today from what is planned, and label the notebooks as the legacy trainer API.
- `packaging` is a declared dependency. The TRL worker imports it to check installed releases.
- The `trl` extra pins `trl>=1.13,<1.14` and `transformers>=5.17,<5.18`, the releases the TRL worker is classified against, and `uv.lock` resolves them.

### Added

- **`xaytune.core` — control-plane domain foundation.** Typed sortable identifiers (`ExperimentId`, `RunId`, …), the `Experiment` / `ExperimentNode` / `Run` / `RunAttempt` aggregates, their separate state machines, `ExecutionOverride`, and the immutable value objects (`Actor`, `DatasetRef`, `ModelRef`, `ArtifactRef`, `RuntimeRef`, `ControllerHostRef`, `CheckpointRef`, `ResourceUsage`). The package imports without torch, transformers, peft, trl, ray or kubernetes installed, enforced by `tests/test_core/test_architecture.py`. Nothing in the existing training path uses it yet.
- **`docs/api/core.md`** documenting the new package.

### Changed

- **Top-level imports are now lazy.** `xaytune.finetune`, `.pretrain`, `.align`, `.evaluate`, `.lr_find`, `.JobManager` and `.discover_plugins` resolve on first access (PEP 562) instead of at import time, so `import xaytune` no longer pulls in torch. Every public name behaves as before; `xaytune.pipeline` is still bound eagerly because it collides with the `xaytune/pipeline.py` submodule.

### Fixed

- **`xaytune pipeline` crashed on any stage that omitted `trainer:` or `lora:`.** `_run_train_stage()` used `TrainerConfig()` and `LoraConfig()` above their import, raising `UnboundLocalError`. The `or` short-circuit meant it only fired when a stage left those sections out. `xaytune/pipeline.py` had no test coverage at all; `tests/test_pipeline.py` now covers it.
- **`import xaytune.trainer` as the first xaytune import raised `ImportError`.** `xaytune.trainer` → `xaytune.eval` → `xaytune.recipes` → `xaytune.trainer` was a circular import masked by the eager package `__init__`; `eval_callback` now resolves the metric registry lazily.
- **`apply_lora()` skipped `prepare_model_for_kbit_training` silently** if the inline import failed. The import is now hoisted alongside the other peft imports.
- **`evaluate()` raised `IndexError` on non-tensor `labels`** while explicitly passing non-tensor batch values through the device move. Labels are now coerced before masking.
- **DeepSpeed + `resume_checkpoint_dir` raised `AttributeError`.** The DeepSpeed path sets `optimizer` to `None`, but the resume branch called `optimizer.load_state_dict()` unguarded while the adjacent scaler and scheduler branches both checked. The skip is now logged rather than passing silently, since DeepSpeed restores optimizer state through its own checkpoint API.
- **CPU-only distributed training raised `AttributeError`.** `init_distributed()` called `torch.cuda.set_device()` unconditionally, so a gloo process group could not start without CUDA.
- **ORPO produced `NaN` when a sequence log-probability reached 0** (BUG-011). The odds `p / (1 - p)` diverge as `p` approaches 1, and `p == 1` is reachable: a fully-masked sequence sums to exactly 0. Probabilities are now clamped just below 1; the clamp is a no-op for realistic inputs.

### Known issues

Two tests are recorded as strict `xfail` pending a product decision, not because of a defect:

- `constant` scheduler with `warmup_steps > 0` auto-upgrades to warmup behaviour, while `test_constant_ignores_warmup_steps` asserts warmup is ignored. Deciding for the implementation makes `constant_with_warmup` redundant; deciding for the test means a requested warmup is silently dropped.
- `global_step` counts optimizer steps, while `test_gradient_accumulation_reduces_optimizer_steps` expects micro-steps. The rest of the loop agrees with the implementation, but `global_step` is user-visible in checkpoints and logs.

### Changed

- **`xaytune.core` domain records are now deeply immutable, and payloads must be canonically persistable.** `FrozenDict` freezes at construction and is backed by a `MappingProxyType`, so an instance cannot contain a mutable container however it was built. Value objects inherit `FrozenDomainModel`, whose `model_copy(update=...)` re-validates rather than assigning — Pydantic's own version skips validation, which would otherwise hand back a plain mutable dict or an unvalidated id. The four aggregates inherit `AggregateModel`, which refuses updates outright: schema validation cannot check that a transition is legal or that the revision moved, so `model_copy(update={"status": ...})` would yield a `SUCCEEDED` attempt with no `started_at` and revision 0. Aggregates change only through transition methods such as `with_status()`. Non-string mapping keys, sets, `NaN`/infinity, `bytes` and arbitrary objects are rejected with `InvalidDomainValueError` — a record that cannot round-trip cannot be fingerprinted. Pydantic's `frozen=True` blocks attribute assignment but not mutation of the containers behind it, so `snapshot.payload["optimizer"]["lr"] = 7` succeeded on a supposedly frozen scientific record, and the model kept a live reference to the caller's dict. Mappings now become `FrozenDict` and sequences become tuples at validation time, recursively; `thaw()` returns a mutable copy. Note id lists and `metadata` are now tuples and mappings rather than `list`/`dict`.
- **The state machines cover failure and cancellation.** The first tables encoded only the transitions the architecture spec drew, which left real gaps: a node could not fail while `ACTIVE`, an attempt could not fail while `STARTING`, `CHECKPOINTING` or `RECOVERING`, and nothing could be cancelled before it started. Any non-terminal state can now reach `FAILED` or `CANCELLED`, and an attempt can be `PREEMPTED` from `QUEUED` onwards. `ExperimentNodeStatus` gains `CANCELLED`, which is distinct from `REJECTED` — the latter is a judgement on merit, reachable only from `DECIDING`.

### Notes

- `mypy` is configured at `python_version = "3.12"` (was `3.10`); numpy's bundled stubs use 3.12-only syntax, and CI already runs mypy under 3.12. This means mypy no longer verifies 3.10 compatibility while `requires-python` remains `>=3.10`.
- Markdown is excluded from `ruff`; ruff 0.16 formats Markdown code blocks, which the project never opted into.

## v0.3.0

### Added

- **Tokenization pipeline** (`tokenize_dataset()`, `collate_tokenized()`) — automatic tokenization of text-format data before training. Converts `{"text": "..."}` samples to `{"input_ids", "labels", "attention_mask"}` tensors.
- **Real model integration tests** — end-to-end tests using `sshleifer/tiny-gpt2` covering forward pass, gradient flow, loss decrease, Trainer loop, and eval-during-training.
- **`@pytest.mark.slow` marker** — integration tests that download models are deselected by default.

### Changed

- `setup_training()` now auto-tokenizes text-format datasets and uses a proper `collate_fn` for padding/batching.
- `validate_batch()` now requires `input_ids` — text-only batches are rejected since tokenization is handled upstream.

## v0.2.0

### Added

- **Algorithm-specific parameters** (`method_params`) — configure DPO beta, GRPO kl_coeff, PPO clip_eps, ORPO lambda_weight, SimPO beta/gamma via config, CLI, Python API, and Studio UI.
- **Studio Simple/Advanced mode** — toggle between minimal form (recipe, model, data) and full control with all training parameters.
- **Auto chat template** — tokenizer chat templates are automatically applied for `chat` and `sharegpt` data formats when a tokenizer is available.
- **Pre-flight validation** (`preflight_check()`) — checks GPU availability, quantization CUDA requirement, data path existence, and output directory writability before training starts.
- **Dynamic method params in Studio** — selecting an alignment method (DPO, GRPO, etc.) shows its configurable hyperparameters with defaults and descriptions.

### Changed

- `align()` one-liner now accepts algorithm kwargs directly (e.g., `align(model="m", dataset="d", beta=0.2)`).
- `build_config()` accepts `method_params` dict for Studio integration.
- All alignment example configs now include `method_params` with documented defaults.
- `setup_training()` passes tokenizer to `load_dataset()` for automatic chat template application.

## v0.1.0

### Added

- Recipe-based training: `finetune`, `pretrain`, `align` recipes with registry pattern.
- Alignment methods: DPO, GRPO, PPO, ORPO, SimPO, REINFORCE.
- Fine-tuning methods: full, LoRA, QLoRA.
- Pydantic config schema with YAML parsing, dot-notation overrides, and config inheritance.
- Cross-field config validation with actionable error messages.
- CLI: `train`, `list`, `eval`, `export` (merge/gguf/push), `compare`, `lr-find`, `studio`, `launch`.
- Training Studio: Gradio web UI with Train/Monitor/History tabs, live loss plotting.
- Data pipeline: format registry (alpaca, sharegpt, chat, text), sequence packing, eval splits.
- Evaluation: metric registry (loss, perplexity), lm-eval-harness benchmark integration.
- Export: LoRA merge, GGUF conversion, HuggingFace Hub push.
- Trainer: mixed precision, gradient accumulation, gradient clipping, LR schedulers.
- Checkpointing: periodic saves, save-last, async checkpoint, resume from checkpoint.
- Early stopping with configurable patience, metric, and min delta.
- LR finder with EMA smoothing and suggested LR.
- Distributed training: DDP, FSDP, DeepSpeed via `xaytune launch`.
- Logging backends: console, WandB, MLflow, TensorBoard.
- Progress bar with Rich.
- Python API one-liners: `finetune()`, `pretrain()`, `align()`.
