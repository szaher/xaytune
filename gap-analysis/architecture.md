# Architecture Summary

Last updated: 2026-06-03 15:00

## 1.1 Technology Stack

| Aspect | Detail |
|---|---|
| Language | Python 3.10+ |
| ML Framework | PyTorch >= 2.0 |
| Model Loading | HuggingFace Transformers >= 4.40 |
| Parameter-Efficient FT | PEFT >= 0.10 |
| Quantization | bitsandbytes >= 0.43 |
| Data Handling | HuggingFace Datasets >= 2.18 |
| Config Parsing | PyYAML >= 6.0 |
| Config Validation | Pydantic >= 2.0 |
| Console Output | Rich >= 13.0 |
| Build System | Hatchling |
| Package Name | `xaytune` (PyPI) |
| CLI Entry Point | `xaytune.cli:main` |
| License | Apache-2.0 |

### Optional Dependencies

| Extra Group | Packages |
|---|---|
| `deepspeed` | deepspeed >= 0.14 |
| `eval` | lm-eval >= 0.4 |
| `wandb` | wandb >= 0.16 |
| `mlflow` | mlflow >= 2.10 |
| `tensorboard` | tensorboard >= 2.14 |
| `studio` | gradio >= 5.0, plotly >= 5.0 |
| `data-prep` | datasketch >= 1.6, langdetect >= 1.0 |
| `synth` | openai >= 1.0 |
| `data-all` | data-prep + synth |
| `docs` | mkdocs-material >= 9.5, mkdocstrings[python] >= 0.25 |
| `all` | deepspeed + eval + wandb + mlflow + tensorboard + studio + data-prep + synth |

## 1.2 C4-Style Architecture Diagram

### Context Diagram

```
+-------------------+
|      User         |
| (ML Engineer)     |
+--------+----------+
         |
         | CLI / Python API / Studio UI
         v
+-------------------+        +--------------------+
|     xaytune       |------->| HuggingFace Hub    |
|   library         |        | (model download,   |
|   v0.6.0          |        |  dataset fetch,    |
+--------+----------+        |  model push)       |
         |                   +--------------------+
         |
         v
+-------------------+        +--------------------+
| GPU Hardware      |        | Experiment Tracking|
| (CUDA / MPS /     |        | (W&B, MLflow,      |
|  CPU fallback)    |        |  TensorBoard)      |
+-------------------+        +--------------------+
```

### Container Diagram

```
xaytune package
+----------------------------------------------------------------+
|                                                                |
|  +-----------+   +------------+   +-------------------------+  |
|  |   CLI     |   | Python API |   | Training Studio         |  |
|  | xaytune   |   | finetune() |   | (Gradio Web UI)         |  |
|  | train     |   | pretrain() |   | xaytune studio          |  |
|  | eval      |   | align()    |   | --port 7860             |  |
|  | export    |   | evaluate() |   +-------------------------+  |
|  | data      |   | lr_find()  |                                |
|  | lr-find   |   +------------+                                |
|  | launch    |                                                 |
|  | compare   |                                                 |
|  | list      |                                                 |
|  +-----------+                                                 |
|                                                                |
+----------------------------------------------------------------+
```

### Component Diagram (by module)

```
xaytune/
|
+-- config/
|   +-- schema.py        Pydantic models: TrainConfig, ModelConfig, DataConfig,
|   |                    TrainerConfig, EvalConfig, LoggingConfig, OutputConfig,
|   |                    LoraConfig, FSDPConfig, DeepSpeedConfig, OnlineRLConfig,
|   |                    GenerationConfig
|   +-- parser.py        YAML loading, config inheritance (base field),
|   |                    dot-notation overrides, deep merge
|   +-- validation.py    Cross-field validation (validate_config),
|                        environment preflight checks (preflight_check)
|
+-- data/
|   +-- loader.py        Dataset loading (local JSONL, HuggingFace Hub, streaming)
|   +-- formats.py       Format registry: alpaca, sharegpt, chat, text
|   +-- agent_formats.py Agent format registry: function_calling, react,
|   |                    trajectory, multi_agent
|   +-- tokenizer.py     Tokenization, StreamingTokenizedDataset,
|   |                    collate functions (tokenized, preference, prompt)
|   +-- agent_tokenizer.py  Agent-specific tokenization with trainable masks
|   +-- packing.py       Sequence packing to reduce padding waste
|   +-- preferences.py   Preference data handling (chosen/rejected pairs)
|   +-- validation.py    Dataset sample validation before training
|   +-- registry.py      FormatRegistry (format_registry instance)
|   +-- prep/            Data preparation pipeline
|       +-- pipeline.py  Multi-step YAML-driven prep pipeline
|       +-- dedup.py     Deduplication (exact hash, MinHash)
|       +-- filters.py   Filtering (length, language, regex)
|       +-- convert.py   Format conversion between data formats
|       +-- generate.py  Synthetic data generation (augment, distill, evolve)
|       +-- report.py    Prep step reporting
|
+-- models/
|   +-- loader.py        HuggingFace AutoModelForCausalLM loading, quantization,
|   |                    dtype selection; returns ModelResult(model, tokenizer, name)
|   +-- peft.py          LoRA/QLoRA adapter application via PEFT library
|   +-- registry.py      Model registry (model_registry instance)
|
+-- recipes/
|   +-- base.py          setup_training() — assembles TrainingComponents:
|   |                    model + tokenizer + dataloaders + trainer + callbacks
|   +-- finetune.py      finetune() — SFT recipe (full/lora/qlora)
|   +-- pretrain.py      pretrain() — continued pre-training recipe
|   +-- align/
|       +-- align.py     align() — alignment recipe entry point
|       +-- loss_dispatch.py  Factory: create_alignment_loss_fn() for all methods
|       +-- dpo.py       DPO loss computation
|       +-- grpo.py      GRPO loss computation
|       +-- orpo.py      ORPO loss (SFT + odds-ratio preference)
|       +-- simpo.py     SimPO loss (length-normalized, reference-free)
|       +-- ppo.py       PPO clip loss + REINFORCE loss
|       +-- logprobs.py  Shared: get_sequence_logps() utility
|       +-- generation.py  Online generation for RL methods
|       +-- online_step.py  OnlineRLStep: generate -> score -> train step
|       +-- online_eval.py  Periodic online evaluation during RL training
|       +-- reward_scoring.py  Reward function dispatch
|
+-- trainer/
|   +-- loop.py          Trainer class: training loop with AMP autocast,
|   |                    gradient accumulation, gradient clipping, optimizer step
|   +-- callbacks.py     TrainState dataclass, CallbackManager (event-driven hooks)
|   |                    Events: train_start, train_end, epoch_start, epoch_end,
|   |                    step_start, step_end, eval_start, eval_end,
|   |                    checkpoint_saved, error
|   +-- checkpointing.py save_checkpoint(), load_checkpoint(), find_latest_checkpoint()
|   +-- checkpoint_callback.py  register_checkpoint_callbacks() — auto-save on step/end
|   +-- async_checkpoint.py     AsyncCheckpointSaver — background thread writes
|   +-- distributed.py   DistributedContext, init_distributed(), wrap_model_distributed()
|   |                    Strategies: DDP, FSDP (full config), DeepSpeed (ZeRO 0-3)
|   +-- scheduler.py     LR schedulers: cosine, linear, constant, constant_with_warmup
|   +-- lr_finder.py     lr_find() — LR range test with exponential sweep
|   +-- eval_callback.py register_eval_callbacks() — periodic evaluation during training
|   +-- early_stopping.py register_early_stopping_callbacks() — patience-based stopping
|   +-- progress.py      register_progress_callbacks() — Rich progress bar
|   +-- device.py        Device detection (CUDA/MPS/CPU), seed_all(), AMP support checks
|
+-- eval/
|   +-- evaluate.py      evaluate() — run model on dataset, compute metrics
|   +-- metrics.py       Metric registry: loss, perplexity, token_accuracy
|   +-- benchmarks.py    benchmark_evaluate() — lm-eval-harness integration
|   +-- agent_metrics.py Agent-specific evaluation metrics
|
+-- export/
|   +-- merge.py         merge() — LoRA adapter merging into base model
|   +-- gguf.py          to_gguf() — GGUF format conversion
|   +-- hub.py           push_to_hub() — HuggingFace Hub upload
|   +-- model_merge.py   model_merge() — weight merging: linear, slerp, ties, dare
|
+-- logging/
|   +-- base.py          LoggingManager, setup_logging() — backend multiplexer
|   +-- console.py       Rich console logging backend
|   +-- tensorboard.py   TensorBoard logging backend
|   +-- wandb.py         Weights & Biases logging backend
|   +-- mlflow.py        MLflow logging backend
|
+-- studio/
|   +-- app.py           Main Gradio application layout
|   +-- server.py        launch() — Gradio server startup
|   +-- jobs.py          JobManager — training job lifecycle management
|   +-- code_runner.py   In-browser Python code execution
|   +-- codegen.py       Config-to-code generation for training scripts
|   +-- data_preview.py  Dataset preview widget
|   +-- dataset_browser.py  Dataset browsing and exploration UI
|   +-- hub_browser.py   HuggingFace Hub model browser UI
|   +-- gpu_metrics.py   GPU utilization monitoring
|   +-- events.py        Studio event system (SSE for job updates)
|   +-- examples.py      Example configurations for the UI
|
+-- utils/
|   +-- registry.py      Registry pattern: register/get/list/has
|
+-- plugins.py           Entry-point plugin discovery for four groups:
|                        xaytune.recipes, xaytune.models, xaytune.formats,
|                        xaytune.metrics
+-- cli.py               Argument parsing and command dispatch
+-- __init__.py          Public API exports: finetune, pretrain, align, evaluate,
                         lr_find, JobManager, discover_plugins
```

## 1.3 Data Flow

### Standard Training (SFT / Pretrain)

```
YAML Config
    |
    v
load_config() -----> TrainConfig (Pydantic validated)
    |
    v
setup_training()
    |-- load_model()        --> ModelResult(model, tokenizer, name)
    |-- apply_lora()        --> LoRA-wrapped model (if method=lora/qlora)
    |-- load_dataset()      --> raw samples (list[dict] or IterableDataset)
    |-- format_fn()         --> {"text": "..."} per sample
    |-- tokenize_dataset()  --> [{input_ids, attention_mask, labels}, ...]
    |-- pack_sequences()    --> packed sequences (optional, reduces padding)
    |-- DataLoader          --> batched tensors
    |-- Trainer()           --> configured trainer with callbacks
    |-- register_*_callbacks() --> eval, checkpoint, early stopping, progress, logging
    |
    v
TrainingComponents(model, tokenizer, train_dataloader, eval_dataloader, trainer)
    |
    v
trainer.train()
    |-- epoch loop
    |   |-- step loop
    |   |   |-- training_step() --> forward, loss, backward, optimizer step
    |   |   |-- fire("step_end") --> callbacks (logging, eval, checkpoint)
    |   |-- fire("epoch_end")
    |-- fire("train_end")
    |
    v
TrainState(step, epoch, global_step, metrics)
```

### Alignment Training (DPO / GRPO / ORPO / SimPO / PPO / REINFORCE)

```
TrainConfig (method="dpo"|"grpo"|...)
    |
    v
setup_training()  --> same as above, but with preference tokenization
    |
    v
align()
    |-- create_alignment_loss_fn(method, ref_model, **params)
    |       --> callable: (model, batch, outputs) -> loss
    |-- [if online RL] OnlineRLStep(ref_model, tokenizer, generation_config, ...)
    |       --> generate completions -> score with reward_fn -> compute advantages
    |
    v
trainer.train(loss_fn=alignment_loss_fn)
    |-- training_step() uses loss_fn instead of model.loss
    |
    v
TrainState
```

### Data Preparation Pipeline

```
CLI: xaytune data pipeline --config prep.yaml
    |
    v
pipeline(config="prep.yaml")
    |-- Step 1: filter (length, language, regex)
    |-- Step 2: deduplicate (exact, minhash, both)
    |-- Step 3: convert (format transformation)
    |-- Each step: input JSONL -> process -> output JSONL + PrepReport
    |
    v
Prepped dataset file (JSONL)
```

## 1.4 External Integrations

| Integration | Purpose | Module | Optional Dep |
|---|---|---|---|
| HuggingFace Hub | Model download, dataset fetch, model push | `models/loader.py`, `data/loader.py`, `export/hub.py` | No (core) |
| HuggingFace PEFT | LoRA/QLoRA adapter creation and merging | `models/peft.py`, `export/merge.py` | No (core) |
| bitsandbytes | 4-bit / 8-bit quantization | `models/loader.py` | No (core) |
| Weights & Biases | Experiment tracking and logging | `logging/wandb.py` | Yes (`wandb`) |
| MLflow | Experiment tracking and logging | `logging/mlflow.py` | Yes (`mlflow`) |
| TensorBoard | Metric visualization and logging | `logging/tensorboard.py` | Yes (`tensorboard`) |
| lm-eval-harness | Benchmark evaluation (MMLU, GSM8K, etc.) | `eval/benchmarks.py` | Yes (`eval`) |
| OpenAI API | Synthetic data generation (augment, distill, evolve) | `data/prep/generate.py` | Yes (`synth`) |
| DeepSpeed | ZeRO optimization stages 0-3, CPU offload | `trainer/distributed.py` | Yes (`deepspeed`) |
| Gradio | Training Studio web UI | `studio/` | Yes (`studio`) |
| Plotly | GPU metrics visualization in Studio | `studio/gpu_metrics.py` | Yes (`studio`) |
| datasketch | MinHash-based deduplication | `data/prep/dedup.py` | Yes (`data-prep`) |
| langdetect | Language-based filtering | `data/prep/filters.py` | Yes (`data-prep`) |

## 1.5 Statelessness

xaytune is a pure library with no persistent data stores, message queues, or caches.

- **No databases**: All state is in-memory during training or serialized to files.
- **No queues**: No async job queue; the Studio `JobManager` runs jobs in-process.
- **No caches**: No HTTP cache, no model cache (relies on HuggingFace's `~/.cache/huggingface`).
- **File I/O only**: Checkpoints (`.pt` + `metadata.json`), training logs, JSONL datasets, exported models, GGUF files.
- **Stateless between runs**: Each `train`/`eval`/`export` invocation is self-contained. Resume is achieved by loading checkpoint files from disk via `find_latest_checkpoint()`.
