# Future Features

Potential features in rough order of user impact.

## 1. Quantized Inference / vLLM Export

Close the train-to-deploy loop. Add GGUF export or vLLM-ready packaging so users can go from fine-tuning to serving without leaving xaytune.

## 2. Dataset Preparation Toolkit

Synthetic data generation, deduplication, quality filtering, and format conversion. Data prep is often harder than training — a built-in toolkit would lower the barrier significantly.

## 3. Hyperparameter Sweep / Auto-Tuning

Beyond the existing LR finder: grid, random, and Bayesian search over key parameters (LoRA rank, alpha, epochs, batch size, learning rate). Integrate with W&B Sweeps or Optuna.

## 4. Multi-Modal Fine-Tuning

Vision-language model support (LLaVA-style). Either a new recipe or an extension of the finetune recipe to handle image+text inputs.

## 5. Continued Pre-Training Improvements

Domain adaptation workflows with curriculum learning, data mixing ratios, and phased training schedules.

## 6. Model Merging

TIES, DARE, and SLERP merge methods for combining fine-tuned adapters or full models. Useful for ensembling specialized models without retraining.
