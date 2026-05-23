# Changelog

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
- Distributed training: DDP, FSDP, DeepSpeed via `trainlib launch`.
- Logging backends: console, WandB, MLflow, TensorBoard.
- Progress bar with Rich.
- Python API one-liners: `finetune()`, `pretrain()`, `align()`.
