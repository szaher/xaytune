from __future__ import annotations

import json
import math

import pytest
import torch

from trainlib.config.schema import TrainerConfig
from trainlib.data.formats import format_alpaca
from trainlib.data.tokenizer import (
    collate_preference,
    collate_tokenized,
    tokenize_dataset,
    tokenize_preference_dataset,
)
from trainlib.trainer.callbacks import CallbackManager

pytestmark = pytest.mark.slow

TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="session")
def tiny_model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    return model, tokenizer


@pytest.fixture
def alpaca_samples():
    return [
        {"instruction": "Say hello", "input": "", "output": "Hello!"},
        {"instruction": "Add numbers", "input": "2+2", "output": "4"},
        {"instruction": "Translate", "input": "hello", "output": "hola"},
        {"instruction": "Reverse", "input": "abc", "output": "cba"},
        {"instruction": "Capitalize", "input": "hello", "output": "HELLO"},
    ]


@pytest.fixture
def alpaca_jsonl(tmp_path, alpaca_samples):
    path = tmp_path / "train.jsonl"
    with open(path, "w") as f:
        for sample in alpaca_samples:
            f.write(json.dumps(sample) + "\n")
    return str(path)


class TestRealModelPipeline:
    def test_tokenize_and_collate(self, tiny_model_and_tokenizer, alpaca_samples):
        _, tokenizer = tiny_model_and_tokenizer
        formatted = [format_alpaca(s) for s in alpaca_samples]
        tokenized = tokenize_dataset(formatted, tokenizer, max_seq_length=64)

        assert len(tokenized) == len(alpaca_samples)
        assert all("input_ids" in s for s in tokenized)
        assert all("labels" in s for s in tokenized)
        assert all("attention_mask" in s for s in tokenized)
        assert all(len(s["input_ids"]) <= 64 for s in tokenized)

        batch = collate_tokenized(tokenized[:2], pad_token_id=tokenizer.pad_token_id)
        assert batch["input_ids"].shape[0] == 2
        assert batch["input_ids"].dtype == torch.long

    def test_forward_pass(self, tiny_model_and_tokenizer, alpaca_samples):
        model, tokenizer = tiny_model_and_tokenizer
        formatted = [format_alpaca(s) for s in alpaca_samples[:2]]
        tokenized = tokenize_dataset(formatted, tokenizer, max_seq_length=64)
        batch = collate_tokenized(tokenized, pad_token_id=tokenizer.pad_token_id)

        model.train()
        outputs = model(**batch)
        assert hasattr(outputs, "loss")
        assert outputs.loss.requires_grad
        assert math.isfinite(outputs.loss.item())

    def test_training_step_with_backward(self, tiny_model_and_tokenizer, alpaca_samples):
        model, tokenizer = tiny_model_and_tokenizer
        formatted = [format_alpaca(s) for s in alpaca_samples[:2]]
        tokenized = tokenize_dataset(formatted, tokenizer, max_seq_length=64)
        batch = collate_tokenized(tokenized, pad_token_id=tokenizer.pad_token_id)

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, "No gradients computed — backward pass failed"

        optimizer.step()

    def test_loss_decreases_over_steps(self, tiny_model_and_tokenizer, alpaca_samples):
        model, tokenizer = tiny_model_and_tokenizer
        formatted = [format_alpaca(s) for s in alpaca_samples[:2]]
        tokenized = tokenize_dataset(formatted, tokenizer, max_seq_length=64)
        batch = collate_tokenized(tokenized, pad_token_id=tokenizer.pad_token_id)

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


class TestFullPipelineIntegration:
    def test_load_format_tokenize_train(self, tiny_model_and_tokenizer, alpaca_jsonl):
        model, tokenizer = tiny_model_and_tokenizer
        from trainlib.data import load_dataset

        data = load_dataset(alpaca_jsonl, format="alpaca")
        assert all("text" in s for s in data)

        tokenized = tokenize_dataset(data, tokenizer, max_seq_length=64)
        assert all("input_ids" in s for s in tokenized)

        from torch.utils.data import DataLoader

        pad_id = tokenizer.pad_token_id or 0
        loader = DataLoader(
            tokenized,
            batch_size=2,
            collate_fn=lambda b, pid=pad_id: collate_tokenized(b, pad_token_id=pid),
        )

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for batch in loader:
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()

    def test_trainer_loop_integration(self, tiny_model_and_tokenizer, alpaca_jsonl):
        model, tokenizer = tiny_model_and_tokenizer
        from trainlib.data import load_dataset
        from trainlib.trainer import Trainer

        data = load_dataset(alpaca_jsonl, format="alpaca")
        tokenized = tokenize_dataset(data, tokenizer, max_seq_length=64)

        from torch.utils.data import DataLoader

        pad_id = tokenizer.pad_token_id or 0
        loader = DataLoader(
            tokenized,
            batch_size=2,
            collate_fn=lambda b, pid=pad_id: collate_tokenized(b, pad_token_id=pid),
        )

        trainer_config = TrainerConfig(
            batch_size=2,
            num_epochs=1,
            learning_rate=1e-4,
            max_steps=3,
        )
        cb = CallbackManager()
        trainer = Trainer(config=trainer_config, callback_manager=cb)

        model.train()
        state = trainer.train(model=model, train_dataloader=loader)

        assert state.global_step == 3
        assert "loss" in state.metrics
        assert math.isfinite(state.metrics["loss"])

    def test_eval_during_training(self, tiny_model_and_tokenizer, alpaca_jsonl):
        model, tokenizer = tiny_model_and_tokenizer
        from trainlib.data import load_dataset
        from trainlib.trainer import Trainer
        from trainlib.trainer.eval_callback import register_eval_callbacks

        data = load_dataset(alpaca_jsonl, format="alpaca")
        tokenized = tokenize_dataset(data, tokenizer, max_seq_length=64)

        split_idx = 3
        train_data = tokenized[:split_idx]
        eval_data = tokenized[split_idx:]

        from torch.utils.data import DataLoader

        pad_id = tokenizer.pad_token_id or 0

        def collate_fn(b, pid=pad_id):
            return collate_tokenized(b, pad_token_id=pid)

        train_loader = DataLoader(train_data, batch_size=2, collate_fn=collate_fn)
        eval_loader = DataLoader(eval_data, batch_size=2, collate_fn=collate_fn)

        trainer_config = TrainerConfig(
            batch_size=2,
            num_epochs=2,
            learning_rate=1e-4,
        )
        cb = CallbackManager()
        trainer = Trainer(config=trainer_config, callback_manager=cb)

        register_eval_callbacks(
            callback_manager=cb,
            model=model,
            eval_dataloader=eval_loader,
            every_n_steps=1,
            metrics=["loss"],
            is_main_process=True,
        )

        model.train()
        state = trainer.train(model=model, train_dataloader=train_loader)

        assert "eval_loss" in state.metrics
        assert math.isfinite(state.metrics["eval_loss"])


class TestPreferenceDataPipeline:
    @pytest.fixture
    def preference_samples(self):
        return [
            {"prompt": "What is AI? ", "chosen": "Artificial intelligence.", "rejected": "Magic."},
            {"prompt": "Say hi. ", "chosen": "Hello!", "rejected": "No."},
            {"prompt": "2+2? ", "chosen": "4", "rejected": "5"},
            {"prompt": "Color of sky? ", "chosen": "Blue", "rejected": "Green"},
        ]

    @pytest.fixture
    def preference_jsonl(self, tmp_path, preference_samples):
        path = tmp_path / "prefs.jsonl"
        with open(path, "w") as f:
            for sample in preference_samples:
                f.write(json.dumps(sample) + "\n")
        return str(path)

    def test_tokenize_preference_and_collate(
        self, tiny_model_and_tokenizer, preference_samples,
    ):
        _, tokenizer = tiny_model_and_tokenizer
        tokenized = tokenize_preference_dataset(
            preference_samples, tokenizer, max_seq_length=64,
        )
        assert len(tokenized) == len(preference_samples)
        assert all("chosen_input_ids" in s for s in tokenized)
        assert all("rejected_input_ids" in s for s in tokenized)

        batch = collate_preference(tokenized[:2], pad_token_id=tokenizer.pad_token_id)
        assert batch["chosen_input_ids"].shape[0] == 2
        assert batch["rejected_input_ids"].shape[0] == 2
        assert batch["chosen_input_ids"].dtype == torch.long

    def test_dpo_loss_end_to_end(
        self, tiny_model_and_tokenizer, preference_samples,
    ):
        import copy

        from trainlib.recipes.align.loss_dispatch import create_alignment_loss_fn

        model, tokenizer = tiny_model_and_tokenizer
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        tokenized = tokenize_preference_dataset(
            preference_samples, tokenizer, max_seq_length=64,
        )
        batch = collate_preference(tokenized[:2], pad_token_id=tokenizer.pad_token_id)

        loss_fn = create_alignment_loss_fn(method="dpo", ref_model=ref_model, beta=0.1)
        model.train()
        loss = loss_fn(model, batch, None)

        assert loss.requires_grad
        assert math.isfinite(loss.item())

        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        assert has_grad, "No gradients from DPO loss"

    def test_dpo_training_loop(
        self, tiny_model_and_tokenizer, preference_samples,
    ):
        import copy

        from trainlib.recipes.align.loss_dispatch import create_alignment_loss_fn
        from trainlib.trainer import Trainer

        model, tokenizer = tiny_model_and_tokenizer
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        tokenized = tokenize_preference_dataset(
            preference_samples, tokenizer, max_seq_length=64,
        )

        from torch.utils.data import DataLoader

        pad_id = tokenizer.pad_token_id or 0
        loader = DataLoader(
            tokenized,
            batch_size=2,
            collate_fn=lambda b, pid=pad_id: collate_preference(b, pad_token_id=pid),
        )

        loss_fn = create_alignment_loss_fn(method="dpo", ref_model=ref_model, beta=0.1)
        trainer_config = TrainerConfig(
            batch_size=2,
            num_epochs=1,
            learning_rate=5e-5,
            max_steps=3,
            mixed_precision="fp32",
        )
        trainer = Trainer(config=trainer_config)

        model.train()
        state = trainer.train(
            model=model, train_dataloader=loader, loss_fn=loss_fn,
        )

        assert state.global_step >= 1
        assert "loss" in state.metrics
        assert math.isfinite(state.metrics["loss"])
