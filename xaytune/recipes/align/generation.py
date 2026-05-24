from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from xaytune.config.schema import GenerationConfig


@dataclass
class GenerationResult:
    response_ids: torch.Tensor
    response_texts: list[str]
    prompt_ids: torch.Tensor
    prompt_lengths: torch.Tensor
    attention_mask: torch.Tensor


def generate_completions(
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    config: GenerationConfig,
) -> GenerationResult:
    """Generate completions from prompt token IDs.

    Args:
        model: HuggingFace model with a ``generate()`` method.
        tokenizer: Tokenizer for decoding outputs.
        prompt_ids: Prompt token IDs, shape ``(batch, seq_len)``.
        prompt_mask: Attention mask for prompts, shape ``(batch, seq_len)``.
        config: Generation parameters.

    Returns:
        GenerationResult with generated token IDs, decoded texts, and metadata.
    """
    prompt_lens = prompt_mask.sum(dim=-1)

    was_training = model.training
    model.eval()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature if config.do_sample else 1.0,
            top_p=config.top_p if config.do_sample else 1.0,
            top_k=config.top_k if config.do_sample else 0,
            do_sample=config.do_sample,
            num_return_sequences=config.group_size,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    if was_training:
        model.train()

    # output_ids shape: (batch * group_size, prompt_len + new_tokens)
    # Expand prompt_ids and prompt_lens for group sampling
    if config.group_size > 1:
        expanded_prompt_ids = prompt_ids.repeat_interleave(config.group_size, dim=0)
        expanded_prompt_lens = prompt_lens.repeat_interleave(config.group_size)
    else:
        expanded_prompt_ids = prompt_ids
        expanded_prompt_lens = prompt_lens

    out_mask = torch.ones_like(output_ids)
    for i in range(output_ids.shape[0]):
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        if pad_id is not None:
            out_mask[i] = (output_ids[i] != pad_id).long()

    response_texts = []
    for i in range(output_ids.shape[0]):
        p_len = int(expanded_prompt_lens[i].item())
        resp_ids = output_ids[i, p_len:]
        response_texts.append(tokenizer.decode(resp_ids, skip_special_tokens=True))

    return GenerationResult(
        response_ids=output_ids,
        response_texts=response_texts,
        prompt_ids=expanded_prompt_ids,
        prompt_lengths=expanded_prompt_lens,
        attention_mask=out_mask,
    )
