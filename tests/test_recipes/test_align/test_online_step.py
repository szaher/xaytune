from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from xaytune.config.schema import GenerationConfig
from xaytune.recipes.align import online_step as online_step_mod
from xaytune.recipes.align.generation import GenerationResult
from xaytune.recipes.align.online_step import OnlineRLStep


def _make_mock_model():
    model = MagicMock()
    model.training = True
    logits = torch.randn(2, 5, 100)
    output = MagicMock()
    output.logits = logits
    model.return_value = output
    model.generate.return_value = torch.randint(1, 100, (2, 5))
    return model


def _make_mock_tokenizer():
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    tok.decode.return_value = "mock text"
    return tok


def _make_prompt_batch(batch_size: int = 2, seq_len: int = 3) -> dict[str, torch.Tensor]:
    return {
        "prompt_input_ids": torch.randint(1, 100, (batch_size, seq_len)),
        "prompt_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }


def _fake_gen_result(batch_size: int = 2) -> GenerationResult:
    return GenerationResult(
        response_ids=torch.randint(1, 100, (batch_size, 5)),
        response_texts=[f"resp{i}" for i in range(batch_size)],
        prompt_ids=torch.randint(1, 100, (batch_size, 3)),
        prompt_lengths=torch.tensor([3] * batch_size),
        attention_mask=torch.ones(batch_size, 5, dtype=torch.long),
    )


class TestOnlineRLStepFallback:
    def test_falls_back_when_advantages_present(self):
        model = _make_mock_model()
        ref_model = _make_mock_model()
        tok = _make_mock_tokenizer()
        gen_config = GenerationConfig(max_new_tokens=2, group_size=1)

        step = OnlineRLStep(
            ref_model=ref_model,
            tokenizer=tok,
            method="grpo",
            generation_config=gen_config,
        )

        batch = {
            "input_ids": torch.randint(1, 100, (2, 5)),
            "attention_mask": torch.ones(2, 5, dtype=torch.long),
            "advantages": torch.tensor([1.0, -1.0]),
        }

        loss = step(model, batch, MagicMock())
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0


class TestOnlineRLStepGRPO:
    def test_grpo_returns_scalar_loss(self):
        with (
            patch.object(online_step_mod, "generate_completions", return_value=_fake_gen_result()),
            patch.object(
                online_step_mod, "score_completions", return_value=torch.tensor([1.0, 0.5])
            ),
        ):
            model = _make_mock_model()
            ref_model = _make_mock_model()
            tok = _make_mock_tokenizer()
            gen_config = GenerationConfig(max_new_tokens=2, group_size=1)

            step = OnlineRLStep(
                ref_model=ref_model,
                tokenizer=tok,
                method="grpo",
                generation_config=gen_config,
                kl_coeff=0.04,
            )

            batch = _make_prompt_batch()
            loss = step(model, batch, MagicMock())

            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0


class TestOnlineRLStepPPO:
    def test_ppo_returns_scalar_loss(self):
        with (
            patch.object(online_step_mod, "generate_completions", return_value=_fake_gen_result()),
            patch.object(
                online_step_mod, "score_completions", return_value=torch.tensor([1.0, 0.5])
            ),
        ):
            model = _make_mock_model()
            ref_model = _make_mock_model()
            tok = _make_mock_tokenizer()
            gen_config = GenerationConfig(max_new_tokens=2, group_size=1)

            step = OnlineRLStep(
                ref_model=ref_model,
                tokenizer=tok,
                method="ppo",
                generation_config=gen_config,
                clip_eps=0.2,
            )

            batch = _make_prompt_batch()
            loss = step(model, batch, MagicMock())

            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0


class TestOnlineRLStepREINFORCE:
    def test_reinforce_returns_scalar_loss(self):
        with (
            patch.object(online_step_mod, "generate_completions", return_value=_fake_gen_result()),
            patch.object(
                online_step_mod, "score_completions", return_value=torch.tensor([1.0, 0.5])
            ),
        ):
            model = _make_mock_model()
            ref_model = _make_mock_model()
            tok = _make_mock_tokenizer()
            gen_config = GenerationConfig(max_new_tokens=2, group_size=1)

            step = OnlineRLStep(
                ref_model=ref_model,
                tokenizer=tok,
                method="reinforce",
                generation_config=gen_config,
            )

            batch = _make_prompt_batch()
            loss = step(model, batch, MagicMock())

            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0


class TestOnlineRLStepInvalidMethod:
    def test_unsupported_method_raises(self):
        with (
            patch.object(online_step_mod, "generate_completions", return_value=_fake_gen_result()),
            patch.object(
                online_step_mod, "score_completions", return_value=torch.tensor([1.0, 0.5])
            ),
        ):
            model = _make_mock_model()
            ref_model = _make_mock_model()
            tok = _make_mock_tokenizer()
            gen_config = GenerationConfig(max_new_tokens=2, group_size=1)

            step = OnlineRLStep(
                ref_model=ref_model,
                tokenizer=tok,
                method="dpo",
                generation_config=gen_config,
            )

            batch = _make_prompt_batch()
            with pytest.raises(ValueError, match="not supported"):
                step(model, batch, MagicMock())
