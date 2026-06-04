# Quality Scorecard

Last updated: 2026-06-03 15:00

Scoring: 0 = broken/absent, 1 = critically flawed, 2 = works sometimes with known bugs, 3 = functional with gaps, 4 = solid with minor issues, 5 = production-ready.

Dimensions: **R** = Reliability, **S** = Security, **P** = Performance, **M** = Maintainability, **UX** = UX Completeness, **O** = Observability.

---

## 1. SFT Fine-tuning

| R | S | P | M | UX | O | Avg |
|---|---|---|---|----|----|-----|
| 2 | 4 | 3 | 4 | 3  | 2  | 3.0 |

- **R=2**: Trains on prompt tokens due to wrong masking in `data/tokenizer.py`. Every SFT model trained with xaytune has degraded quality because the loss includes tokens the model should be copying, not learning.
- **P=3**: Packing implementation works and saves training time; no major performance regressions identified.
- **M=4**: Clean recipe pattern in `recipes/sft/` makes the SFT path easy to follow and extend.
- **UX=3**: One-liner API works for the happy path, but configuration errors are swallowed silently rather than surfaced to the user.
- **O=2**: Loss reporting was wrong (now fixed in `trainer/loop.py`), meaning users were calibrating training runs against incorrect values.

---

## 2. Pre-training

| R | S | P | M | UX | O | Avg |
|---|---|---|---|----|----|-----|
| 3 | 4 | 3 | 4 | 3  | 2  | 3.2 |

- **R=3**: Basic pre-training path works, but streaming mode has an `eval_split` bug that prevents evaluation during streaming pre-training runs.
- **M=4**: Follows the same recipe pattern as SFT; code is consistent and readable.
- **O=2**: Same loss reporting issues as SFT applied here; limited insight into training dynamics beyond loss.

---

## 3. Alignment (DPO/GRPO/ORPO/PPO)

| R | S | P | M | UX | O | Avg |
|---|---|---|---|----|----|-----|
| 1 | 4 | 2 | 3 | 2  | 1  | 2.2 |

- **R=1**: ORPO crashes at runtime (`recipes/align/orpo.py`, `trainer/loop.py`). Prompt tokens leak into log-probability calculations in `recipes/align/logprobs.py`, diluting alignment signal. GRPO had unfixed OOM. PPO in `recipes/align/ppo.py` is misleading -- it implements clipped policy gradient, not actual PPO with value function.
- **P=2**: `deepcopy` of the reference model doubled VRAM usage -- fixed for some methods but the pattern may recur in others.
- **M=3**: Each alignment method has its own recipe file, which is clean, but shared log-prob code has the masking bug affecting all methods.
- **UX=2**: The alignment method comparison table in documentation contained errors, leading users to pick methods based on wrong information.
- **O=1**: No alignment-specific metrics (reward margins, KL divergence tracking, chosen vs rejected gaps). Users have no way to tell if alignment is working.

---

## 4. Evaluation

| R | S | P | M | UX | O | Avg |
|---|---|---|---|----|----|-----|
| 2 | 4 | 3 | 3 | 3  | 2  | 2.8 |

- **R=2**: `token_accuracy` metric was broken and device mismatch caused crashes on GPU -- both now fixed. The fixes are recent and edge cases may remain.
- **M=3**: Evaluation logic is coupled to the training loop rather than being independently callable, limiting reuse.
- **O=2**: Evaluation results are logged but there is no structured output format for downstream consumption or comparison across runs.

---

## 5. Export

| R | S | P | M | UX | O | Avg |
|---|---|---|---|----|----|-----|
| 2 | 4 | 4 | 3 | 2  | 1  | 2.7 |

- **R=2**: `model_merge` in `export/model_merge.py` produces output missing `config.json`, making merged models unusable without manual intervention. GGUF conversion in `export/gguf.py` generates the wrong shell command.
- **P=4**: When export works, the merge/conversion operations are straightforward and not a bottleneck.
- **UX=2**: Users who complete training and try to export hit broken paths at the finish line -- the worst possible time to fail.
- **O=1**: No validation step confirms the exported model is loadable. Users discover failures only when trying to use the output.

---

## 6. Data Pipeline

| R | S | P | M | UX | O | Avg |
|---|---|---|---|----|----|-----|
| 3 | 4 | 3 | 4 | 3  | 1  | 3.0 |

- **R=3**: `format_text` silently returns empty strings for unrecognized formats rather than raising an error. Split logic lacked shuffling before splitting -- now fixed.
- **M=4**: Registry pattern in `data/` is clean and extensible; adding new data formats is straightforward.
- **O=1**: No statistics on dataset composition, token counts, or format distribution. Users cannot validate their data pipeline output before training.

---

## 7. Distributed Training

| R | S | P | M | UX | O | Avg |
|---|---|---|---|----|----|-----|
| 1 | 4 | 2 | 3 | 2  | 1  | 2.2 |

- **R=1**: DeepSpeed training loop in `trainer/loop.py` is broken -- it does not actually function. NCCL backend is hardcoded in `trainer/distributed.py`, preventing use on non-NVIDIA hardware.
- **P=2**: DeepSpeed does not work, so advertised multi-GPU performance gains are unachievable through that path.
- **M=3**: FSDP wrapping logic is clean and follows standard PyTorch patterns.
- **UX=2**: Users attempting multi-GPU training with DeepSpeed will waste time debugging a fundamentally broken path.
- **O=1**: No distributed-specific metrics (communication overhead, shard balance, GPU utilization per rank).

---

## 8. Config System

| R | S | P | M | UX | O | Avg |
|---|---|---|---|----|----|-----|
| 3 | 4 | 5 | 4 | 3  | 1  | 3.3 |

- **R=3**: `reinforce` is rejected as an alignment method despite being a valid option. Override typos are silently ignored rather than flagged, meaning users can run training with unintended configurations.
- **M=4**: Pydantic schema in the config system is clean and provides good type safety for the configs it does validate.
- **P=5**: Config parsing adds negligible overhead; no performance concerns.
- **O=1**: No logging of the effective config after overrides are applied. Users cannot confirm what config was actually used for a training run.

---

## 9. Training Studio

| R | S | P | M | UX | O | Avg |
|---|---|---|---|----|----|-----|
| 2 | 3 | 3 | 3 | 2  | 1  | 2.3 |

- **R=2**: Format dropdown in `studio/app.py` lists wrong options. Alignment training through the Studio bypasses the alignment-specific loss functions, producing models trained with SFT loss on preference data.
- **S=3**: No authentication, which is acceptable for a local development tool but should be documented as such.
- **UX=2**: The Studio is a demo rather than a functional tool. Users who rely on it for alignment training get silently wrong results.
- **O=1**: No training progress display or real-time metrics in the Gradio UI.

---

## 10. Logging

| R | S | P | M | UX | O | Avg |
|---|---|---|---|----|----|-----|
| 2 | 3 | 4 | 3 | 3  | 3  | 3.0 |

- **R=2**: MLflow integration crashes under certain conditions with no graceful fallback. Logger instances are not isolated, causing interference when multiple loggers are configured.
- **S=3**: Credentials for remote logging backends (MLflow, W&B) are handled via environment variables, which is standard but undocumented.
- **O=3**: When logging works, the integrations with TensorBoard, W&B, and MLflow provide reasonable visibility -- this is the one area where observability is partially adequate.

---

## 11. Checkpointing

| R | S | P | M | UX | O | Avg |
|---|---|---|---|----|----|-----|
| 2 | 4 | 3 | 4 | 3  | 2  | 3.0 |

- **R=2**: Missing `map_location` parameter means checkpoints saved on GPU cannot be loaded on CPU without manual patching. Tensor serialization issues exist for certain model configurations.
- **P=3**: Async checkpoint saving exists and works for the common case, reducing training interruption.
- **M=4**: Checkpoint logic is well-separated and follows PyTorch conventions.
- **O=2**: Checkpoint save events are logged but there is no validation that a checkpoint is loadable after saving.

---

## 12. CLI

| R | S | P | M | UX | O | Avg |
|---|---|---|---|----|----|-----|
| 2 | 4 | 4 | 4 | 3  | 2  | 3.2 |

- **R=2**: `eval` subcommand crashes when invoked without `--metrics` flag. This is the most common CLI failure path since users expect reasonable defaults.
- **M=4**: CLI is built with a clean command pattern and is easy to extend with new subcommands.
- **UX=3**: Happy-path usage works well; error messages for invalid inputs need improvement.
- **O=2**: CLI commands log to stdout but do not produce structured output for scripting.

---

## Summary Table

| Area                 | R | S | P | M | UX | O | Avg |
|----------------------|---|---|---|---|----|----|-----|
| SFT Fine-tuning      | 2 | 4 | 3 | 4 | 3  | 2  | 3.0 |
| Pre-training         | 3 | 4 | 3 | 4 | 3  | 2  | 3.2 |
| Alignment            | 1 | 4 | 2 | 3 | 2  | 1  | 2.2 |
| Evaluation           | 2 | 4 | 3 | 3 | 3  | 2  | 2.8 |
| Export               | 2 | 4 | 4 | 3 | 2  | 1  | 2.7 |
| Data Pipeline        | 3 | 4 | 3 | 4 | 3  | 1  | 3.0 |
| Distributed Training | 1 | 4 | 2 | 3 | 2  | 1  | 2.2 |
| Config System        | 3 | 4 | 5 | 4 | 3  | 1  | 3.3 |
| Training Studio      | 2 | 3 | 3 | 3 | 2  | 1  | 2.3 |
| Logging              | 2 | 3 | 4 | 3 | 3  | 3  | 3.0 |
| Checkpointing        | 2 | 4 | 3 | 4 | 3  | 2  | 3.0 |
| CLI                  | 2 | 4 | 4 | 4 | 3  | 2  | 3.2 |

**Overall weighted average: 2.8 / 5.0**

---

## Top Opportunities

The 5 lowest-scoring areas and what fixing them would unlock:

### 1. Alignment (Avg 2.2) -- Fix ORPO, log-prob masking, PPO labeling
Alignment is the highest-value differentiator for an LLM training library. Fixing ORPO to actually run, correcting log-prob masking across all methods, and either implementing real PPO or renaming the current method would make xaytune's alignment story credible. This unlocks the entire RLHF/preference-tuning market.

### 2. Distributed Training (Avg 2.2) -- Fix DeepSpeed loop, un-hardcode NCCL
Multi-GPU training is table stakes for any serious LLM training library. Fixing the DeepSpeed loop and making the backend configurable would let users actually train models at scale, which is the primary reason people reach for a training library instead of writing their own loop.

### 3. Training Studio (Avg 2.3) -- Fix alignment bypass, correct UI dropdowns
The Studio is the first thing new users try. If it works correctly, it dramatically lowers the barrier to entry. Fixing the alignment loss bypass alone would prevent users from silently getting wrong results, which is a trust-destroying experience.

### 4. Export (Avg 2.7) -- Fix model_merge config.json, fix GGUF command
Export is the last step before a user gets value from their training run. Breaking here means all prior work (data prep, training, evaluation) was wasted. These are small fixes with outsized impact on the end-to-end experience.

### 5. Evaluation (Avg 2.8) -- Harden metrics, fix device handling
Reliable evaluation is how users know if training worked. Without it, users cannot make informed decisions about hyperparameters, data quality, or when to stop training. The recent fixes are a good start but need validation across more configurations.
