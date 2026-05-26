from unittest.mock import MagicMock, patch

import pytest
import torch

from xaytune.recipes.align.rewards import (
    LLMJudgeWrapper,
    _llm_judge_cache,
    _reward_model_cache,
    reward_registry,
)


@pytest.fixture(autouse=True)
def _clear_caches():
    _reward_model_cache.clear()
    _llm_judge_cache.clear()
    yield
    _reward_model_cache.clear()
    _llm_judge_cache.clear()


class TestRewardModelReward:
    def test_registered(self):
        assert reward_registry.has("reward_model")

    def test_requires_model_name(self):
        fn = reward_registry.get("reward_model")
        with pytest.raises(ValueError, match="model_name"):
            fn("prompt", "response")

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    def test_scores_with_mock(self, mock_model_cls, mock_tok_cls):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tok_cls.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.85]])
        mock_model.return_value = mock_output
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model_cls.return_value = mock_model

        fn = reward_registry.get("reward_model")
        score = fn("hello", "world", model_name="test-rm", device="cpu")

        assert isinstance(score, float)
        assert abs(score - 0.85) < 0.01

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    def test_caches_model(self, mock_model_cls, mock_tok_cls):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_tok_cls.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.5]])
        mock_model.return_value = mock_output
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model_cls.return_value = mock_model

        fn = reward_registry.get("reward_model")
        fn("a", "b", model_name="cached-rm", device="cpu")
        fn("c", "d", model_name="cached-rm", device="cpu")

        assert mock_model_cls.call_count == 1


class TestLLMJudgeReward:
    def test_registered(self):
        assert reward_registry.has("llm_judge")

    def test_requires_model_name(self):
        fn = reward_registry.get("llm_judge")
        with pytest.raises(ValueError, match="model_name"):
            fn("prompt", "response")

    @patch("transformers.pipeline")
    def test_parses_score(self, mock_pipeline_fn):
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"generated_text": "4"}]
        mock_pipeline_fn.return_value = mock_pipe

        fn = reward_registry.get("llm_judge")
        score = fn("hello", "world", model_name="test-judge", device="cpu")

        assert score == 0.75

    def test_judge_wrapper_parses_digit(self):
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"generated_text": "4"}]

        wrapper = LLMJudgeWrapper.__new__(LLMJudgeWrapper)
        wrapper._pipe = mock_pipe
        wrapper._template = LLMJudgeWrapper.DEFAULT_TEMPLATE

        score = wrapper.judge("prompt", "response")
        assert score == 0.75

    def test_judge_wrapper_fallback_on_garbage(self):
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"generated_text": "I cannot rate this"}]

        wrapper = LLMJudgeWrapper.__new__(LLMJudgeWrapper)
        wrapper._pipe = mock_pipe
        wrapper._template = LLMJudgeWrapper.DEFAULT_TEMPLATE

        score = wrapper.judge("prompt", "response")
        assert score == 0.0

    def test_judge_wrapper_parses_boundary_scores(self):
        wrapper = LLMJudgeWrapper.__new__(LLMJudgeWrapper)
        wrapper._template = LLMJudgeWrapper.DEFAULT_TEMPLATE

        for digit, expected in [("1", 0.0), ("2", 0.25), ("3", 0.5), ("5", 1.0)]:
            mock_pipe = MagicMock()
            mock_pipe.return_value = [{"generated_text": digit}]
            wrapper._pipe = mock_pipe
            assert wrapper.judge("p", "r") == expected
