# Risks and Unknowns

Last updated: 2026-06-03 15:00

---

## Top 10 Technical Risks

### 1. SFT masking wrong -- every trained model affected
**Severity: Critical** | **Status: Unfixed**

The tokenizer in `data/tokenizer.py` does not mask prompt tokens during SFT. The loss function trains on both the prompt and the completion, meaning the model learns to reproduce the prompt rather than focusing on the desired response. Every SFT model trained with xaytune has suboptimal quality as a result. This is the single highest-impact bug because SFT is the most-used feature.

### 2. ORPO never worked -- broken advertised feature
**Severity: Critical** | **Status: Unfixed**

ORPO (Odds Ratio Preference Optimization) crashes at runtime. The implementation in `recipes/align/orpo.py` does not integrate correctly with the training loop in `trainer/loop.py`. This is an advertised feature in the README and documentation. Users who choose ORPO based on documentation guidance will hit an immediate crash.

### 3. DeepSpeed loop non-functional -- wasted GPU resources
**Severity: Critical** | **Status: Unfixed**

The DeepSpeed training path in `trainer/loop.py` and `trainer/distributed.py` does not function. The loop does not properly handle DeepSpeed's engine wrapper, meaning multi-GPU training via DeepSpeed silently fails or produces incorrect results. Users allocating expensive GPU resources for distributed training get no benefit.

### 4. QLoRA unstable -- garbage gradients possible
**Severity: High** | **Status: Unfixed**

The QLoRA implementation in `models/peft.py` has stability issues that can produce garbage gradients under certain configurations. This is particularly dangerous because QLoRA is the path for users on consumer GPUs (the largest potential user base), and gradient issues manifest as silently bad model quality rather than crashes.

### 5. Preference log-probs diluted -- alignment quality degraded
**Severity: High** | **Status: Unfixed**

The log-probability calculation in `recipes/align/logprobs.py` includes prompt tokens, which dilutes the preference signal. This affects DPO, ORPO (when it doesn't crash), and other preference-based methods. The alignment training appears to work but produces weaker results than it should, making it hard for users to diagnose.

### 6. Studio alignment bypass -- wrong training results
**Severity: High** | **Status: Unfixed**

When users configure alignment training through the Gradio Studio in `studio/app.py`, the alignment-specific loss function is bypassed. The model trains with SFT loss on preference data, which is semantically wrong. Users get a model that learned to reproduce both chosen and rejected responses equally, defeating the purpose of alignment.

### 7. Gradient accumulation step counting was wrong -- existing checkpoints affected
**Severity: Medium** | **Status: FIXED**

The gradient accumulation step counter in `trainer/loop.py` was miscounting, causing optimizer steps to fire at wrong intervals. This has been fixed, but any checkpoints saved before the fix may have been trained with incorrect effective batch sizes. Users resuming from old checkpoints should be aware of this.

### 8. Loss reporting was wrong -- users calibrated on wrong values
**Severity: Medium** | **Status: FIXED**

Loss values reported during training in `trainer/loop.py` were incorrect. This has been fixed, but users who tuned hyperparameters based on reported loss values from previous runs were working with wrong data. Historical training logs from xaytune cannot be trusted for comparison.

### 9. model_merge output unusable -- broken export pipeline
**Severity: High** | **Status: Unfixed**

The model merge functionality in `export/model_merge.py` produces output directories missing `config.json`. Without this file, the merged model cannot be loaded by any standard framework (transformers, vLLM, TGI). Users who complete a training run and attempt to merge LoRA weights back into the base model get an unusable artifact.

### 10. GGUF conversion broken -- never works
**Severity: Medium** | **Status: Unfixed**

The GGUF export in `export/gguf.py` generates an incorrect shell command for the llama.cpp conversion tool. The command will fail on execution. Users wanting to run their fine-tuned models locally via llama.cpp, Ollama, or similar tools cannot use this export path.

---

## Top 10 Product Risks

### 1. Overclaiming
The README and documentation promise significantly more than the code delivers. Features like ORPO, DeepSpeed, and GGUF export are advertised as working but are broken. This creates a trust deficit when users discover the gaps, and the gap between promise and reality is large enough to damage the project's reputation.

### 2. No prompt masking
xaytune's SFT is worse than competitors (trl, axolotl, LLaMA-Factory) by default because it trains on prompt tokens. Users who benchmark xaytune against alternatives will get inferior results without understanding why, and will likely switch tools rather than investigate.

### 3. PPO misleading
The PPO implementation is actually clipped policy gradient without a value function or GAE. Users familiar with PPO from the RLHF literature expect specific behavior (critic network, advantage estimation) that is absent. This is a naming problem that could be fixed by renaming to "REINFORCE with clipping" or implementing actual PPO.

### 4. QLoRA incomplete
Users on consumer GPUs (4090, 3090, Apple Silicon) are the largest potential audience for a training library. If QLoRA produces unstable gradients, these users get bad results and have no recourse since they cannot fit full fine-tuning in memory.

### 5. DeepSpeed broken
Multi-GPU training is a core requirement for any team training models larger than 7B parameters. A broken DeepSpeed integration means these teams cannot use xaytune, eliminating a significant market segment.

### 6. Studio unusable for alignment
The Gradio Training Studio is the lowest-friction entry point for new users. If alignment training through the Studio silently produces wrong results, users will blame the alignment method rather than the tool, leading to confusion and abandonment.

### 7. No real-world validation
There is no evidence (blog posts, case studies, user reports, benchmark results) that anyone has successfully trained a model with xaytune and deployed it. Without this validation, adoption is a leap of faith.

### 8. Test coverage is structural, not behavioral
The test suite (102 files, ~16K lines) is large but relies heavily on mocking. Tests verify that functions are called with expected arguments rather than verifying that training produces correct gradients, that loss decreases, or that models improve on downstream tasks. The tests can pass while the code produces wrong results.

### 9. Documentation errors were pervasive
The alignment method comparison table, GGUF export instructions, and model merge documentation all contained errors. Even after fixes, users who read cached or previously-downloaded docs will follow wrong instructions. The trust gap means users will second-guess all documentation.

### 10. Scope too wide
xaytune v0.6.0 covers SFT, pre-training, DPO, ORPO, GRPO, PPO, model merging, GGUF export, data preparation, evaluation, distributed training, a Gradio studio, agent fine-tuning, and multi-agent conversations. For a library at this maturity level, the breadth means no individual feature gets the depth of testing and polish it needs. Competitors that do fewer things better will win on reliability.

---

## Unknowns

These are questions this audit could not answer. Each represents a potential risk that should be investigated.

- **Has anyone run a complete fine-tuning job with this library?**
  No evidence of successful end-to-end training runs was found. Without at least one validated run per recipe type, it is unknown whether the full pipeline (data loading -> training -> evaluation -> export) works end-to-end.

- **What GPU configurations are actually tested?**
  The test suite mocks CUDA. It is unknown whether xaytune works correctly on A100, H100, RTX 4090, Apple Silicon (MPS), or multi-node configurations. Each has different memory constraints, precision support, and driver requirements.

- **Does FSDP work correctly?**
  FSDP wrapping code in `trainer/distributed.py` looks clean, but it was not verified during this audit. FSDP has subtle correctness requirements (parameter ordering, gradient synchronization, checkpoint format) that can only be validated on multi-GPU hardware.

- **Are the agent fine-tuning features tested against real agent traces?**
  The agent fine-tuning and multi-agent conversation features (added in v0.6.0) include synthetic data generation. It is unknown whether the generated synthetic data produces models that actually perform better as agents, or whether the format matches what agent frameworks (LangChain, CrewAI, AutoGen) expect.

- **Does the data prep pipeline work with real datasets?**
  Data loading and formatting was tested structurally but not with real-world datasets (Alpaca, ShareGPT, UltraChat, LMSYS). Real datasets have encoding issues, missing fields, and format inconsistencies that synthetic test data does not capture.

- **What is the actual test pass rate?**
  Tests could not be run locally during this audit because PyTorch was not available in the environment. The CI/CD pipeline status and actual test pass rate are unknown. Some of the bugs found (ORPO crash, device mismatch) should cause test failures if tests exist for those paths.

- **Are there production users?**
  No telemetry, GitHub issues from external users, or community engagement was found that would indicate production usage. The library's actual reliability in the wild is unknown.
