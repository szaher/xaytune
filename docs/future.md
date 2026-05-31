# Future Features

Potential features in rough order of user impact.

**Delivery standard:** Every feature must ship with all of the following before it is considered complete:

- **Unit tests** — comprehensive coverage of every public function, edge cases, and error paths
- **Integration tests** — end-to-end tests that exercise the feature through the Python API, CLI, and config/recipe integration
- **Example notebook** — a new notebook in `examples/` that walks through the feature with real (or realistic mock) data, runnable on a fresh install
- **CLI verification** — if the feature adds CLI commands, subprocess-based tests that invoke them and check output
- **Documentation** — update the docs site and README if the feature changes the public API or adds new commands

---

## 1. Quantized Inference / vLLM Export

Close the train-to-deploy loop. Add GGUF export or vLLM-ready packaging so users can go from fine-tuning to serving without leaving xaytune.

**Testing requirements:**
- Unit tests for each export format (GGUF quantization levels, vLLM-compatible output)
- Integration test: finetune a tiny model → export → verify the exported artifact loads correctly
- Example notebook: `examples/08_export_and_serving.ipynb` — full train → export → serve workflow
- CLI tests for `xaytune export vllm` and any new export subcommands

## ~~2. Dataset Preparation Toolkit~~ ✅ Implemented (v0.7.0)

Shipped: dedup, quality filtering, format conversion, synthetic data generation, pipeline chaining, CLI, recipe integration.

## 3. Hyperparameter Sweep / Auto-Tuning

Beyond the existing LR finder: grid, random, and Bayesian search over key parameters (LoRA rank, alpha, epochs, batch size, learning rate). Integrate with W&B Sweeps or Optuna.

**Testing requirements:**
- Unit tests for each search strategy (grid, random, Bayesian) with mock training runs
- Integration test: run a sweep over 2-3 configs on a tiny model and verify the best config is selected
- Example notebook: `examples/09_hyperparameter_sweep.ipynb` — sweep over LoRA rank and learning rate, visualize results
- CLI tests for `xaytune sweep` command

## 4. Multi-Modal Fine-Tuning

Vision-language model support (LLaVA-style). Either a new recipe or an extension of the finetune recipe to handle image+text inputs.

**Testing requirements:**
- Unit tests for image+text data loading, tokenization, and collation
- Integration test: finetune a tiny VLM on a small image-caption dataset end-to-end
- Example notebook: `examples/10_multimodal_finetuning.ipynb` — fine-tune a vision-language model on image-caption pairs
- Tests for new data formats (image paths, base64-encoded images, HuggingFace image datasets)

## 5. Continued Pre-Training Improvements

Domain adaptation workflows with curriculum learning, data mixing ratios, and phased training schedules.

**Testing requirements:**
- Unit tests for curriculum scheduler, data mixing logic, and phase transitions
- Integration test: run a 2-phase pre-training on a tiny model with different data mixtures per phase
- Example notebook: `examples/11_domain_adaptation.ipynb` — continued pre-training with curriculum learning
- Config validation tests for the new curriculum/mixing YAML schema

## ~~6. Model Merging~~ ✅ Implemented (v0.7.0)

Shipped: Linear, SLERP, TIES, DARE merge algorithms. Python API, CLI (`xaytune export model-merge`), example notebook.

## 7. Agent Fine-Tuning

Train models to be better agents — tool use, multi-step reasoning, and task completion. This is the first opinionated toolkit for agent fine-tuning with a clean CLI-first API.

### 7a. Tool-Use Data Formats

New data formats for agent training data:

- **function_calling** — OpenAI-style function calling conversations with tool definitions, tool calls, and tool results
- **react** — ReAct-style traces (Thought → Action → Observation loops)
- **trajectory** — full agent trajectories with multi-step tool use, branching, and error recovery

Each format maps to the existing `register_format` system and tokenizes tool schemas, calls, and results with appropriate special tokens.

### 7b. Agent SFT (Supervised Fine-Tuning on Trajectories)

Extend the `finetune` recipe to handle agent data:

- Fine-tune on successful tool-use trajectories (teach the model which tools to call and when)
- Loss masking on tool results (model learns to generate actions, not predict tool output)
- Support for multi-turn conversations with interleaved tool calls
- Compatible with LoRA/QLoRA for efficient training

### 7c. Agent Alignment (RLHF/GRPO with Tool Execution)

Extend the `align` recipe with agent-specific reward functions:

- **Task completion reward** — did the agent achieve the goal?
- **Tool-use quality reward** — were the right tools called with correct parameters?
- **Efficiency reward** — fewer steps = better (penalize unnecessary tool calls)
- **Execution feedback** — integrate actual tool execution results into the reward signal
- Online RL with tool execution in the loop (builds on existing GRPO/PPO + online generation pipeline)

### 7d. Agent Evaluation

Agent-specific eval metrics and benchmarks:

- Tool-use accuracy (correct tool + correct parameters)
- Task success rate (end-to-end completion)
- Step efficiency (steps taken vs minimum required)
- Error recovery rate (handles tool failures gracefully)
- Integration with existing `xaytune.eval` and lm-eval-harness

**Testing requirements:**
- Unit tests for each data format (function_calling, react, trajectory) with tokenization verification
- Unit tests for loss masking on tool results
- Unit tests for each reward function with known trajectories
- Integration test: fine-tune a tiny model on tool-use data → evaluate tool-use accuracy
- Integration test: run GRPO alignment with a mock tool execution environment
- Example notebook: `examples/11_agent_finetuning.ipynb` — full agent training pipeline: data prep → SFT on trajectories → alignment with tool-use rewards → evaluation
- CLI tests for any new subcommands or format options

### 7e. Synthetic Agent Data Generation (future)

Extend `xaytune.data.prep.generate` with agent-aware generation modes:

- **trajectory_gen** — give an LLM a task + tool definitions, have it produce a multi-turn tool-use conversation as training data
- **trajectory_verify** — execute generated trajectories against real tools to validate correctness before training
- Builds on the existing augment/distill/evolve pipeline but produces structured agent formats instead of flat instruction/output pairs

Deferred until 7a data formats are proven. The formats ship first with support for loading and training on existing agent data.

### 7f. Multi-Agent Conversations (future)

Training on multi-agent collaboration data where multiple agents interact. Extends the `trajectory` format with a `name` field per turn to distinguish agents. Covers CrewAI, AutoGen, and custom multi-agent orchestration patterns. Deferred until single-agent training is proven.
