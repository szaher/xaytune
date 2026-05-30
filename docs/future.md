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

## 6. Model Merging

TIES, DARE, and SLERP merge methods for combining fine-tuned adapters or full models. Useful for ensembling specialized models without retraining.

**Testing requirements:**
- Unit tests for each merge algorithm (TIES, DARE, SLERP) with small random weight tensors
- Integration test: train 2 tiny LoRA adapters → merge them → verify the merged model produces valid outputs
- Example notebook: `examples/12_model_merging.ipynb` — merge two fine-tuned adapters and compare with individual models
- CLI tests for `xaytune export merge --method ties/dare/slerp`
