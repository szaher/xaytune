from unittest.mock import MagicMock

import torch

from xaytune.config.schema import GenerationConfig
from xaytune.recipes.align.generation import GenerationResult, generate_completions


class TestGenerateCompletions:
    def _make_mock_model(self, output_ids: torch.Tensor):
        model = MagicMock()
        model.training = True
        model.generate.return_value = output_ids
        return model

    def _make_mock_tokenizer(self, pad_id: int = 0):
        tok = MagicMock()
        tok.pad_token_id = pad_id
        tok.eos_token_id = 1
        tok.decode.side_effect = lambda ids, **kw: f"text_{len(ids)}"
        return tok

    def test_returns_generation_result(self):
        prompt_ids = torch.tensor([[10, 20, 30]])
        prompt_mask = torch.ones(1, 3, dtype=torch.long)
        output = torch.tensor([[10, 20, 30, 40, 50]])
        model = self._make_mock_model(output)
        tok = self._make_mock_tokenizer()
        config = GenerationConfig(max_new_tokens=2, group_size=1)

        result = generate_completions(model, tok, prompt_ids, prompt_mask, config)

        assert isinstance(result, GenerationResult)
        assert result.response_ids.shape == (1, 5)
        assert len(result.response_texts) == 1
        assert result.prompt_lengths.shape == (1,)

    def test_group_sampling_expands_batch(self):
        prompt_ids = torch.tensor([[10, 20]])
        prompt_mask = torch.ones(1, 2, dtype=torch.long)
        output = torch.tensor(
            [
                [10, 20, 40, 50],
                [10, 20, 60, 70],
                [10, 20, 80, 90],
            ]
        )
        model = self._make_mock_model(output)
        tok = self._make_mock_tokenizer()
        config = GenerationConfig(max_new_tokens=2, group_size=3)

        result = generate_completions(model, tok, prompt_ids, prompt_mask, config)

        assert result.response_ids.shape[0] == 3
        assert result.prompt_ids.shape[0] == 3
        assert result.prompt_lengths.shape == (3,)
        assert len(result.response_texts) == 3

    def test_model_restored_to_training(self):
        prompt_ids = torch.tensor([[10, 20]])
        prompt_mask = torch.ones(1, 2, dtype=torch.long)
        output = torch.tensor([[10, 20, 40]])
        model = self._make_mock_model(output)
        model.training = True
        tok = self._make_mock_tokenizer()
        config = GenerationConfig(max_new_tokens=1, group_size=1)

        generate_completions(model, tok, prompt_ids, prompt_mask, config)

        model.eval.assert_called_once()
        model.train.assert_called_once()

    def test_eval_model_stays_eval(self):
        prompt_ids = torch.tensor([[10, 20]])
        prompt_mask = torch.ones(1, 2, dtype=torch.long)
        output = torch.tensor([[10, 20, 40]])
        model = self._make_mock_model(output)
        model.training = False
        tok = self._make_mock_tokenizer()
        config = GenerationConfig(max_new_tokens=1, group_size=1)

        generate_completions(model, tok, prompt_ids, prompt_mask, config)

        model.eval.assert_called_once()
        model.train.assert_not_called()

    def test_attention_mask_shape_matches_output(self):
        prompt_ids = torch.tensor([[10, 20], [30, 40]])
        prompt_mask = torch.ones(2, 2, dtype=torch.long)
        output = torch.tensor([[10, 20, 50, 60], [30, 40, 70, 80]])
        model = self._make_mock_model(output)
        tok = self._make_mock_tokenizer()
        config = GenerationConfig(max_new_tokens=2, group_size=1)

        result = generate_completions(model, tok, prompt_ids, prompt_mask, config)

        assert result.attention_mask.shape == output.shape

    def test_generate_called_with_config_params(self):
        prompt_ids = torch.tensor([[10, 20]])
        prompt_mask = torch.ones(1, 2, dtype=torch.long)
        output = torch.tensor([[10, 20, 40]])
        model = self._make_mock_model(output)
        tok = self._make_mock_tokenizer(pad_id=0)
        config = GenerationConfig(
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            group_size=2,
        )

        generate_completions(model, tok, prompt_ids, prompt_mask, config)

        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 64
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["top_k"] == 50
        assert call_kwargs["do_sample"] is True
        assert call_kwargs["num_return_sequences"] == 2

    def test_greedy_ignores_temperature(self):
        prompt_ids = torch.tensor([[10, 20]])
        prompt_mask = torch.ones(1, 2, dtype=torch.long)
        output = torch.tensor([[10, 20, 40]])
        model = self._make_mock_model(output)
        tok = self._make_mock_tokenizer()
        config = GenerationConfig(
            max_new_tokens=1,
            temperature=0.5,
            do_sample=False,
            group_size=1,
        )

        generate_completions(model, tok, prompt_ids, prompt_mask, config)

        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["temperature"] == 1.0
        assert call_kwargs["do_sample"] is False
