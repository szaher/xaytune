# New Features / Enhancements Backlog

Last updated: 2026-06-03 15:00

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

| Priority | Count | Focus |
|----------|-------|-------|
| Must | 5 | FEAT-001, 003, 004, 005, plus FEAT-002 renamed/documented |
| Should | 1 | FEAT-006 |
| Could | 3 | FEAT-007, 008, 010 |
| Won't | 1 | FEAT-009 (use WandB/MLflow native) |

The "Must" features are not optional enhancements — they are correctness fixes that happen to require new code. Without them, xaytune's core training paths produce wrong results.
