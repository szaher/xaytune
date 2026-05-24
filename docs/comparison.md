# How xaytune Compares

The LLM fine-tuning ecosystem has several strong tools. This page explains where xaytune fits and when you might prefer one of the alternatives.

---

## At a Glance

| | xaytune | TRL | Axolotl | LLaMA-Factory | Unsloth | torchtune |
|---|---|---|---|---|---|---|
| **Primary interface** | Python API + YAML + CLI | Python API | YAML config | Web UI + YAML | Python API | Config + CLI |
| **SFT** | :material-check: full, LoRA, QLoRA | :material-check: | :material-check: | :material-check: | :material-check: | :material-check: |
| **Pre-training** | :material-check: | :material-close: | :material-close: | :material-check: | :material-close: | :material-close: |
| **Alignment** | DPO, GRPO, PPO, ORPO, SimPO, REINFORCE | DPO, PPO, GRPO, KTO, ORPO | DPO, ORPO, KTO | DPO, PPO, GRPO, ORPO, SimPO, KTO | DPO, GRPO, ORPO | DPO, PPO |
| **One-liner API** | :material-check: `xaytune.finetune(...)` | :material-close: | :material-close: | :material-close: | :material-close: | :material-close: |
| **Config-driven** | :material-check: Pydantic + YAML | :material-close: | :material-check: YAML | :material-check: YAML | :material-close: | :material-check: YAML |
| **Web UI** | :material-check: Training Studio | :material-close: | :material-close: | :material-check: LlamaBoard | :material-close: | :material-close: |
| **Built-in eval** | :material-check: metrics + lm-eval | :material-close: | :material-close: | :material-check: | :material-close: | :material-check: limited |
| **Export (merge/GGUF/Hub)** | :material-check: | :material-close: | :material-close: | :material-check: | :material-check: | :material-close: |
| **Distributed** | DDP, FSDP, DeepSpeed | DDP, FSDP, DeepSpeed | DDP, FSDP, DeepSpeed | DDP, FSDP, DeepSpeed | :material-close: single-GPU | DDP, FSDP |
| **Extensibility** | Registry decorators | Subclass trainers | YAML plugins | :material-close: limited | :material-close: | Recipe classes |

---

## xaytune's Design Philosophy

xaytune is built around a principle that most tools only partially address: **the same library should serve both the quick experiment and the production pipeline, with the same config format and the same code path.**

```python
# 30 seconds to first result
xaytune.finetune(model="meta-llama/Llama-3.1-8B", dataset="data.jsonl", method="lora")
```

```yaml
# Same run, reproducible and version-controlled
recipe: finetune
method: lora
model:
  name: meta-llama/Llama-3.1-8B
data:
  path: data.jsonl
  format: alpaca
```

Both paths go through the same validated `TrainConfig`, the same `setup_training()`, and the same `Trainer`. There's no "quick mode" with different behavior.

---

## Detailed Comparisons

### vs. TRL (Hugging Face)

[TRL](https://github.com/huggingface/trl) is the Hugging Face alignment library and the standard for RLHF research.

**Choose TRL when:**

- You need bleeding-edge alignment algorithms as soon as they're published
- Your team already uses the HF Trainer API and wants to stay in that ecosystem
- You need tight integration with `transformers.Trainer` callbacks and logging

**Choose xaytune when:**

- You want a single tool that covers pre-training through alignment and export
- You prefer one-liner APIs over subclassing `SFTTrainer` / `DPOTrainer`
- You want config validation that catches mistakes before training starts
- You need built-in evaluation and GGUF export without additional tools

**Key difference:** TRL gives you a trainer class per algorithm (`SFTTrainer`, `DPOTrainer`, `GRPOTrainer`). xaytune gives you one function per recipe (`finetune()`, `align()`) and selects the algorithm via `method=`. This means less API surface to learn, but also less per-algorithm customization.

---

### vs. Axolotl

[Axolotl](https://github.com/axolotl-ai-cloud/axolotl) is a config-driven fine-tuning framework popular in production settings.

**Choose Axolotl when:**

- You need sequence parallelism for very long contexts (Ring FlashAttention)
- You want the most battle-tested config-driven workflow
- Your team is already standardized on Axolotl configs

**Choose xaytune when:**

- You want a Python API alongside the config system, not just YAML
- You need pre-training support (Axolotl is SFT/alignment only)
- You want type-safe configs with Pydantic validation and actionable error messages
- You want to extend the system with decorators (`@register_format`, `@register_metric`) rather than YAML plugins

**Key difference:** Axolotl is YAML-first — you configure everything in YAML and run via CLI. xaytune treats Python and YAML as equal citizens: the same `TrainConfig` object drives both, and you can mix and match (load YAML, override in Python, pass to `trainer.train()`).

---

### vs. LLaMA-Factory

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) has the largest community (68K+ stars) and the lowest barrier to entry thanks to its web UI.

**Choose LLaMA-Factory when:**

- You want a point-and-click training experience with no code
- You need broad model support out of the box (100+ model families)
- You want Unsloth speed optimization with a web UI on top

**Choose xaytune when:**

- You need to embed training in a Python application or pipeline
- You want to customize loss functions, add callbacks, or write custom data formats
- You want type-safe, validated configuration rather than a form-based UI
- You need programmatic access to training state and metrics

**Key difference:** LLaMA-Factory optimizes for the first-run experience — you can launch training from a browser in minutes. xaytune optimizes for the hundredth run — when you need reproducibility, extensibility, and integration with your own code.

---

### vs. Unsloth

[Unsloth](https://github.com/unslothai/unsloth) is the speed champion, achieving 2-5x faster training through custom Triton kernels.

**Choose Unsloth when:**

- Training speed and VRAM efficiency are your top priority
- You're working with a single GPU and need to maximize what it can do
- You want the fastest possible LoRA/QLoRA fine-tuning

**Choose xaytune when:**

- You need multi-GPU distributed training (DDP, FSDP, DeepSpeed)
- You want a unified tool that also handles pre-training, evaluation, and export
- You want a config system and callback hooks for production workflows
- You need extensibility through registries rather than a fixed pipeline

**Key difference:** Unsloth is a speed optimizer — it makes training faster on a single GPU. xaytune is a workflow framework — it manages the full lifecycle from data to deployment. They solve different problems, and in fact Unsloth's optimizations can be used as a backend by other tools (LLaMA-Factory does this).

---

### vs. torchtune

[torchtune](https://github.com/pytorch/torchtune) is PyTorch's official fine-tuning library.

**Choose torchtune when:**

- You want to stay as close to raw PyTorch as possible
- You prefer reading and modifying recipe files over using a framework
- You primarily work with Meta's model family (Llama)

**Choose xaytune when:**

- You want higher-level APIs without giving up control
- You need broader model support beyond Meta's family
- You want built-in eval, export, and logging integrations
- You prefer a registry/decorator pattern over subclassing

**Key difference:** torchtune gives you recipe scripts that you fork and modify. xaytune gives you a framework with extension points. torchtune says "here's a good starting point, make it yours." xaytune says "here's a system, extend it through the defined interfaces."

---

## Summary

| If you need... | Consider |
|---|---|
| Fastest single-GPU training | Unsloth |
| Point-and-click web UI | LLaMA-Factory |
| Bleeding-edge alignment research | TRL |
| Battle-tested YAML-driven production | Axolotl |
| Raw PyTorch control | torchtune |
| **One tool from pre-training to deployment** | **xaytune** |
| **Python API + YAML + CLI + Web UI** | **xaytune** |
| **Extensible registries and callbacks** | **xaytune** |

xaytune's niche is the **full-lifecycle training library**: a single tool with a layered API (one-liners → config → full control) that covers pre-training, fine-tuning, alignment, evaluation, and export. If you only need one piece of that pipeline, a specialized tool might serve you better. If you want one cohesive system for all of it, that's what xaytune is for.
