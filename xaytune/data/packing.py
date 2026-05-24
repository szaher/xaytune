from __future__ import annotations

IGNORE_INDEX = -100


def pack_sequences(
    sequences: list[dict[str, list[int]]],
    *,
    max_seq_length: int,
    pad_token_id: int,
) -> list[dict[str, list[int]]]:
    """Pack multiple short sequences into fixed-length blocks to reduce padding.

    Concatenates tokenized samples end-to-end and splits at
    *max_seq_length* boundaries.  Remaining space is padded with
    *pad_token_id* (labels use ``-100``).

    Args:
        sequences: Tokenized samples, each with ``"input_ids"`` keys.
        max_seq_length: Target sequence length for packed blocks.
        pad_token_id: Token id used for input padding.

    Returns:
        Packed samples with ``input_ids``, ``attention_mask``, and ``labels``.
    """
    if not sequences:
        return []

    packed: list[dict[str, list[int]]] = []
    current_ids: list[int] = []
    current_labels: list[int] = []

    for seq in sequences:
        ids = seq["input_ids"][:max_seq_length]

        if len(current_ids) + len(ids) > max_seq_length:
            if current_ids:
                packed.append(
                    _pad_and_finalize(current_ids, current_labels, max_seq_length, pad_token_id)
                )
            current_ids = ids[:]
            current_labels = ids[:]
        else:
            current_ids.extend(ids)
            current_labels.extend(ids)

    if current_ids:
        packed.append(_pad_and_finalize(current_ids, current_labels, max_seq_length, pad_token_id))

    return packed


def _pad_and_finalize(
    input_ids: list[int],
    labels: list[int],
    max_seq_length: int,
    pad_token_id: int,
) -> dict[str, list[int]]:
    seq_len = len(input_ids)
    pad_len = max_seq_length - seq_len

    attention_mask = [1] * seq_len + [0] * pad_len
    padded_ids = input_ids + [pad_token_id] * pad_len
    padded_labels = labels + [IGNORE_INDEX] * pad_len

    return {
        "input_ids": padded_ids,
        "attention_mask": attention_mask,
        "labels": padded_labels,
    }
