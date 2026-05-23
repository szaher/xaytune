#!/usr/bin/env python3
"""
trainlib — End-to-End Demo
==========================

A self-contained script that exercises every major trainlib feature using a tiny
GPT-2 model (~500 KB). No GPU required — runs on CPU in under a minute.

Features demonstrated:
  1. Data pipeline     — load_dataset, format_alpaca, tokenize_dataset, collate
  2. Fine-tuning       — finetune() one-liner API
  3. Pre-training      — pretrain() one-liner API
  4. Alignment (DPO)   — align() one-liner with method_params
  5. Config-driven     — TrainConfig + setup_training() + trainer.train()
  6. Evaluation        — evaluate() with loss & perplexity
  7. LR finder         — lr_find() to suggest optimal learning rate
  8. Callbacks         — custom step_end callback
  9. Checkpointing     — save & resume from checkpoint
 10. Export            — save model + tokenizer to disk

Usage:
    cd trainlib/
    uv run python examples/end_to_end.py
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import trainlib
from trainlib.config.schema import (
    DataConfig,
    EvalConfig,
    ModelConfig,
    OutputConfig,
    TrainConfig,
    TrainerConfig,
)
from trainlib.data import load_dataset
from trainlib.data.tokenizer import collate_tokenized, tokenize_dataset
from trainlib.eval import evaluate
from trainlib.export import save
from trainlib.recipes.align.dpo import dpo_loss
from trainlib.recipes.align.logprobs import get_sequence_logps
from trainlib.recipes.base import setup_training
from trainlib.trainer import CallbackManager
from trainlib.trainer.callbacks import TrainState
from trainlib.trainer.checkpointing import load_checkpoint, save_checkpoint
from trainlib.trainer.lr_finder import lr_find

# ---------------------------------------------------------------------------
# 0. Setup — tiny model + sample data
# ---------------------------------------------------------------------------
TINY_MODEL = "sshleifer/tiny-gpt2"
print("=" * 60)
print("trainlib end-to-end demo")
print("=" * 60)

# Create temporary workspace
work_dir = Path(tempfile.mkdtemp(prefix="trainlib_demo_"))
data_dir = work_dir / "data"
output_dir = work_dir / "output"
data_dir.mkdir()
output_dir.mkdir()

# Write alpaca-format training data
alpaca_path = data_dir / "train.jsonl"
alpaca_samples = [
    {"instruction": "Translate to Spanish", "input": "hello", "output": "hola"},
    {"instruction": "What is 2+2?", "input": "", "output": "4"},
    {"instruction": "Reverse the string", "input": "abc", "output": "cba"},
    {"instruction": "Capitalize", "input": "world", "output": "WORLD"},
    {"instruction": "Say goodbye", "input": "", "output": "Goodbye!"},
    {"instruction": "Count to 3", "input": "", "output": "1, 2, 3"},
    {"instruction": "Greet the user", "input": "Alice", "output": "Hello, Alice!"},
    {"instruction": "What color is the sky?", "input": "", "output": "Blue"},
]
with open(alpaca_path, "w") as f:
    for s in alpaca_samples:
        f.write(json.dumps(s) + "\n")

# Write raw text data for pre-training
text_path = data_dir / "corpus.jsonl"
text_samples = [
    {"text": "The quick brown fox jumps over the lazy dog."},
    {"text": "Machine learning is a subset of artificial intelligence."},
    {"text": "Python is a popular programming language for data science."},
    {"text": "Large language models can generate human-like text."},
    {"text": "Fine-tuning adapts a pre-trained model to a specific task."},
    {"text": "Reinforcement learning from human feedback improves model alignment."},
]
with open(text_path, "w") as f:
    for s in text_samples:
        f.write(json.dumps(s) + "\n")

print(f"\nWork directory: {work_dir}")
print(f"Training data:  {len(alpaca_samples)} alpaca samples")
print(f"Text corpus:    {len(text_samples)} samples")

# ---------------------------------------------------------------------------
# 1. Data Pipeline — format, tokenize, collate
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("1. DATA PIPELINE")
print("-" * 60)

# Load and format
raw_data = load_dataset(str(alpaca_path), format="alpaca")
print(f"   Loaded {len(raw_data)} samples, format: alpaca")
print(f"   Sample keys: {list(raw_data[0].keys())}")
print(f"   Sample text: {raw_data[0]['text'][:80]}...")

# Tokenize
tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenized = tokenize_dataset(raw_data, tokenizer, max_seq_length=128)
print(f"   Tokenized: {len(tokenized)} samples")
print(f"   Token keys: {list(tokenized[0].keys())}")
print(f"   Seq length: {len(tokenized[0]['input_ids'])} tokens")

# Collate into a batch
batch = collate_tokenized(tokenized[:4], pad_token_id=tokenizer.pad_token_id)
print(f"   Batch shape: input_ids={list(batch['input_ids'].shape)}")
print("   OK")

# ---------------------------------------------------------------------------
# 2. Fine-Tuning — one-liner API
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("2. FINE-TUNING (finetune one-liner)")
print("-" * 60)

ft_output = str(output_dir / "finetuned")
state = trainlib.finetune(
    model=TINY_MODEL,
    dataset=str(alpaca_path),
    method="full",
    format="alpaca",
    num_epochs=1,
    learning_rate=5e-4,
    batch_size=2,
    max_steps=5,
    mixed_precision="fp32",
)
print(f"   Final loss:   {state.metrics.get('loss', 'N/A'):.4f}")
print(f"   Global steps: {state.global_step}")
print("   OK")

# ---------------------------------------------------------------------------
# 3. Pre-Training — one-liner API
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("3. PRE-TRAINING (pretrain one-liner)")
print("-" * 60)

state = trainlib.pretrain(
    model=TINY_MODEL,
    dataset=str(text_path),
    format="text",
    num_epochs=1,
    learning_rate=3e-4,
    batch_size=2,
    max_steps=5,
    mixed_precision="fp32",
)
print(f"   Final loss:   {state.metrics.get('loss', 'N/A'):.4f}")
print(f"   Global steps: {state.global_step}")
print("   OK")

# ---------------------------------------------------------------------------
# 4. Alignment — DPO loss function demo
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("4. ALIGNMENT — DPO loss function")
print("-" * 60)

# DPO operates on tokenized preference pairs (chosen_input_ids / rejected_input_ids).
# Here we demonstrate the DPO loss computation directly with the tiny model.
policy_model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
ref_model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
ref_model.eval()

# Tokenize a chosen/rejected pair
chosen_text = "AI is artificial intelligence, a branch of computer science."
rejected_text = "AI is magic."
chosen_enc = tokenizer(chosen_text, return_tensors="pt", padding=True)
rejected_enc = tokenizer(rejected_text, return_tensors="pt", padding=True)

with torch.no_grad():
    ref_chosen_out = ref_model(**chosen_enc)
    ref_rejected_out = ref_model(**rejected_enc)
    ref_chosen_logps = get_sequence_logps(
        ref_chosen_out.logits, chosen_enc["input_ids"], chosen_enc["attention_mask"],
    )
    ref_rejected_logps = get_sequence_logps(
        ref_rejected_out.logits, rejected_enc["input_ids"], rejected_enc["attention_mask"],
    )

policy_chosen_out = policy_model(**chosen_enc)
policy_rejected_out = policy_model(**rejected_enc)
policy_chosen_logps = get_sequence_logps(
    policy_chosen_out.logits, chosen_enc["input_ids"], chosen_enc["attention_mask"],
)
policy_rejected_logps = get_sequence_logps(
    policy_rejected_out.logits, rejected_enc["input_ids"], rejected_enc["attention_mask"],
)

loss = dpo_loss(
    policy_chosen_logps=policy_chosen_logps,
    policy_rejected_logps=policy_rejected_logps,
    ref_chosen_logps=ref_chosen_logps,
    ref_rejected_logps=ref_rejected_logps,
    beta=0.2,
)
print("   Method:       DPO (beta=0.2)")
print(f"   DPO loss:     {loss.item():.4f}")
assert math.isfinite(loss.item())
assert loss.requires_grad
print(f"   Differentiable: {loss.requires_grad}")
print("   OK")

# ---------------------------------------------------------------------------
# 5. Config-Driven Training — TrainConfig + setup_training
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("5. CONFIG-DRIVEN TRAINING")
print("-" * 60)

config = TrainConfig(
    recipe="finetune",
    method="full",
    model=ModelConfig(name=TINY_MODEL),
    data=DataConfig(
        path=str(alpaca_path),
        format="alpaca",
        eval_split=0.2,
        max_seq_length=128,
    ),
    trainer=TrainerConfig(
        batch_size=2,
        learning_rate=5e-4,
        num_epochs=1,
        max_steps=6,
        mixed_precision="fp32",
        scheduler="cosine",
        warmup_steps=2,
    ),
    eval=EvalConfig(
        every_n_steps=3,
        metrics=["loss", "perplexity"],
    ),
    output=OutputConfig(dir=str(output_dir / "config-run")),
)

components = setup_training(config)
print(f"   Model:       {TINY_MODEL}")
print(f"   Train data:  {len(components.train_dataloader.dataset)} samples")
if components.eval_dataloader:
    print(f"   Eval data:   {len(components.eval_dataloader.dataset)} samples")
print("   Scheduler:   cosine with 2 warmup steps")

state = components.trainer.train(
    model=components.model,
    train_dataloader=components.train_dataloader,
)
print(f"   Final loss:   {state.metrics.get('loss', 'N/A'):.4f}")
print(f"   Global steps: {state.global_step}")
if "eval_loss" in state.metrics:
    print(f"   Eval loss:    {state.metrics['eval_loss']:.4f}")
print("   OK")

# ---------------------------------------------------------------------------
# 6. Custom Callbacks
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("6. CUSTOM CALLBACKS")
print("-" * 60)

loss_history: list[float] = []
cb = CallbackManager()


@cb.on("step_end")
def track_loss(train_state):
    loss_history.append(train_state.metrics.get("loss", 0.0))


# Quick training run with callback
components = setup_training(
    TrainConfig(
        recipe="finetune",
        method="full",
        model=ModelConfig(name=TINY_MODEL),
        data=DataConfig(path=str(alpaca_path), format="alpaca", max_seq_length=128),
        trainer=TrainerConfig(
            batch_size=2,
            learning_rate=1e-3,
            num_epochs=1,
            max_steps=8,
            mixed_precision="fp32",
        ),
        output=OutputConfig(dir=str(output_dir / "callback-run")),
    ),
    callback_manager=cb,
)
state = components.trainer.train(
    model=components.model,
    train_dataloader=components.train_dataloader,
)
print(f"   Tracked {len(loss_history)} loss values via step_end callback")
print(f"   Loss trend: {loss_history[0]:.4f} → {loss_history[-1]:.4f}")
decreased = loss_history[-1] < loss_history[0]
print(f"   Loss decreased: {decreased}")
print("   OK")

# ---------------------------------------------------------------------------
# 7. LR Finder
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("7. LR FINDER")
print("-" * 60)

model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
tokenized_data = tokenize_dataset(
    load_dataset(str(alpaca_path), format="alpaca"),
    tokenizer,
    max_seq_length=128,
)

pad_id = tokenizer.pad_token_id or 0


def collate_fn(batch, pid=pad_id):
    return collate_tokenized(batch, pad_token_id=pid)


lr_loader = DataLoader(tokenized_data, batch_size=2, collate_fn=collate_fn)

result = lr_find(
    model,
    lr_loader,
    start_lr=1e-6,
    end_lr=0.1,
    num_iterations=20,
)
print(f"   Suggested LR: {result.suggested_lr:.2e}")
print(f"   Tested range: {result.lrs[0]:.2e} → {result.lrs[-1]:.2e}")
print(f"   Loss points:  {len(result.losses)}")
print("   OK")

# ---------------------------------------------------------------------------
# 8. Evaluation
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("8. EVALUATION")
print("-" * 60)

eval_model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
eval_batches = [
    collate_tokenized(tokenized_data[i : i + 2], pad_token_id=pad_id)
    for i in range(0, min(4, len(tokenized_data)), 2)
]

results = evaluate(model=eval_model, dataset=eval_batches, metrics=["loss", "perplexity"])
print(f"   Loss:       {results['loss']:.4f}")
print(f"   Perplexity: {results['perplexity']:.4f}")
assert math.isfinite(results["loss"])
assert math.isfinite(results["perplexity"])
print("   OK")

# ---------------------------------------------------------------------------
# 9. Checkpointing — save & resume
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("9. CHECKPOINT SAVE & RESUME")
print("-" * 60)

ckpt_dir = output_dir / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)

train_model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
optimizer = torch.optim.AdamW(train_model.parameters(), lr=1e-4)

# Train a few steps
train_model.train()
last_loss = 0.0
for i, batch in enumerate(lr_loader):
    optimizer.zero_grad()
    outputs = train_model(**batch)
    outputs.loss.backward()
    optimizer.step()
    last_loss = outputs.loss.item()
    if i >= 2:
        break

# Save checkpoint
ckpt_state = TrainState(step=2, epoch=0, global_step=3, num_epochs=1)
ckpt_state.metrics["loss"] = last_loss
save_checkpoint(
    model=train_model,
    optimizer=optimizer,
    state=ckpt_state,
    output_dir=str(ckpt_dir / "step-3"),
)
print("   Saved checkpoint at step 3")

# Resume from checkpoint
resumed = load_checkpoint(
    checkpoint_dir=str(ckpt_dir / "step-3"),
    model=train_model,
    optimizer=optimizer,
)
print(f"   Resumed from step {resumed.step}, epoch {resumed.epoch}")
print(f"   Resumed metrics: loss={resumed.metrics.get('loss', 'N/A'):.4f}")
print("   OK")

# ---------------------------------------------------------------------------
# 10. Export — save model + tokenizer
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("10. EXPORT — save model & tokenizer")
print("-" * 60)

export_dir = str(output_dir / "exported-model")
save(
    model=train_model,
    tokenizer=tokenizer,
    output_dir=export_dir,
    metadata={
        "base_model": TINY_MODEL,
        "training_steps": 3,
        "framework": "trainlib",
    },
)

exported_files = list(Path(export_dir).iterdir())
print(f"   Exported to: {export_dir}")
print(f"   Files: {[f.name for f in exported_files]}")
assert Path(export_dir, "trainlib_metadata.json").exists()
print("   OK")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ALL 10 SECTIONS PASSED")
print("=" * 60)
print(f"\nWork directory: {work_dir}")
print("To clean up:   rm -rf", work_dir)

# Cleanup
shutil.rmtree(work_dir, ignore_errors=True)
print("\nCleaned up temporary files.")
