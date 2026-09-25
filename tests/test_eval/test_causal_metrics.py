"""Issue #36: legacy evaluation scores the next token, and weights loss by targets.

``xaytune.eval.evaluate()`` and the in-training evaluation callback used to
compare the logits at *i* with the label at *i* -- which rewards a model for
echoing its input -- and to average batch losses, so a batch of 3 targets
weighed as much as one of 300. These tests pin the corrected definitions:

```text
token_accuracy   logits[..., :-1].argmax(-1) against labels[..., 1:], where != -100
loss             Σ(batch_loss × next-token targets) / Σ(next-token targets)
perplexity       exp(loss)
```
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from tests.training_fixtures import tiny_model
from xaytune.eval.evaluate import evaluate
from xaytune.eval.metrics import compute_loss, compute_perplexity
from xaytune.trainer.callbacks import CallbackManager, TrainState
from xaytune.trainer.eval_callback import register_eval_callbacks

VOCAB = 6


class _FixedLogits(torch.nn.Module):
    """A model whose logits put all their weight on chosen tokens, per position."""

    def __init__(self, choose: Any) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.choose = choose

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, **_: Any) -> Any:
        chosen = self.choose(input_ids)
        logits = torch.full((*input_ids.shape, VOCAB), -30.0)
        logits.scatter_(-1, chosen.unsqueeze(-1), 30.0)
        output = MagicMock()
        output.logits = logits
        output.loss = None
        return output


def echoes_current(ids: torch.Tensor) -> torch.Tensor:
    return ids


def predicts_next(ids: torch.Tensor) -> torch.Tensor:
    return torch.roll(ids, shifts=-1, dims=-1)


def _batch(ids: list[int], labels: list[int] | None = None) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([ids]),
        "labels": torch.tensor([labels if labels is not None else ids]),
    }


def _callback_metrics(model: Any, batches: list[dict], metrics: list[str]) -> dict:
    callbacks = CallbackManager()
    register_eval_callbacks(
        callback_manager=callbacks,
        model=model,
        eval_dataloader=batches,
        every_n_steps=1,
        metrics=metrics,
    )
    state = TrainState(global_step=1)
    callbacks.fire("step_end", state)
    return {name.removeprefix("eval_"): value for name, value in state.metrics.items()}


def _both(model: Any, batches: list[dict], metrics: list[str]) -> list[dict]:
    """The same measurement through evaluate() and through the training callback."""
    return [evaluate(model=model, dataset=batches, metrics=metrics)] + [
        _callback_metrics(model, batches, metrics)
    ]


# ---- token accuracy scores the next token ----------------------------------------


def test_echoing_the_current_token_is_not_perfect_accuracy() -> None:
    for measured in _both(_FixedLogits(echoes_current), [_batch([1, 2, 3, 4])], ["token_accuracy"]):
        assert measured["token_accuracy"] == 0.0


def test_predicting_the_next_token_is() -> None:
    for measured in _both(_FixedLogits(predicts_next), [_batch([1, 2, 3, 4])], ["token_accuracy"]):
        assert measured["token_accuracy"] == 1.0


def test_masked_labels_count_in_neither_numerator_nor_denominator() -> None:
    """Three targets; one masked. Right on one of the remaining two: 0.5, not 1/3 or 2/3."""

    def right_once(ids: torch.Tensor) -> torch.Tensor:
        # Position 0 predicts label[1] = 2 correctly; position 2 would be right
        # about label[3], which is masked; position 1 is wrong about label[2].
        return torch.tensor([[2, 5, 4, 0]])

    batch = _batch([1, 2, 3, 4], labels=[1, 2, 3, -100])
    for measured in _both(_FixedLogits(right_once), [batch], ["token_accuracy"]):
        assert measured["token_accuracy"] == 0.5


# ---- loss is a mean over targets, whatever the batching ---------------------------


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory) -> Any:
    from transformers import AutoModelForCausalLM

    directory = tiny_model(tmp_path_factory.mktemp("model") / "tiny")
    return AutoModelForCausalLM.from_pretrained(Path(directory), dtype=torch.float32)


SEQUENCES = [[5, 6, 8], [7, 5, 6, 8, 5, 6, 8, 1], [6, 8]]


def _padded(sequences: list[list[int]]) -> dict[str, torch.Tensor]:
    """One right-padded batch, the padding masked out of the labels."""
    width = max(len(s) for s in sequences)
    ids = [s + [0] * (width - len(s)) for s in sequences]
    labels = [s + [-100] * (width - len(s)) for s in sequences]
    mask = [[1] * len(s) + [0] * (width - len(s)) for s in sequences]
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(mask),
    }


def test_the_same_examples_batched_differently_give_the_same_loss(model: Any) -> None:
    one_each = [_padded([s]) for s in SEQUENCES]
    together = [_padded(SEQUENCES)]
    two_then_one = [_padded(SEQUENCES[:2]), _padded(SEQUENCES[2:])]

    metrics = ["loss", "perplexity"]
    for measure in (
        lambda b: evaluate(model=model, dataset=b, metrics=metrics),
        lambda b: _callback_metrics(model, b, metrics),
    ):
        reference = measure(one_each)
        for partition in (together, two_then_one):
            measured = measure(partition)
            assert measured["loss"] == pytest.approx(reference["loss"], rel=1e-5)
            assert measured["perplexity"] == pytest.approx(reference["perplexity"], rel=1e-5)
            assert measured["perplexity"] == pytest.approx(math.exp(measured["loss"]), rel=1e-6)


def test_the_loss_is_the_mean_over_every_target(model: Any) -> None:
    """Checked against the definition: summed token losses over the token count."""
    with torch.no_grad():
        summed, targets = 0.0, 0
        for sequence in SEQUENCES:
            ids = torch.tensor([sequence])
            logits = model(input_ids=ids).logits
            summed += torch.nn.functional.cross_entropy(
                logits[0, :-1], ids[0, 1:], reduction="sum"
            ).item()
            targets += len(sequence) - 1
    measured = evaluate(model=model, dataset=[_padded(SEQUENCES)], metrics=["loss"])
    assert measured["loss"] == pytest.approx(summed / targets, rel=1e-5)


def test_a_batch_with_no_target_contributes_nothing(model: Any) -> None:
    """A one-token sequence has no next token: its loss -- NaN from the model -- is dropped.

    Neither a loss nor a weight is recorded for it, in either evaluation path.
    """
    with torch.no_grad():
        unscorable = model(**_padded([[5]]), return_dict=True).loss
    assert unscorable is None or not math.isfinite(unscorable.item()), "the premise"

    for measure in (
        lambda b: evaluate(model=model, dataset=b, metrics=["loss", "perplexity"]),
        lambda b: _callback_metrics(model, b, ["loss", "perplexity"]),
    ):
        alone = measure([_padded(SEQUENCES)])
        with_empty = measure([_padded(SEQUENCES), _padded([[5]])])
        assert math.isfinite(with_empty["loss"]) and math.isfinite(with_empty["perplexity"])
        assert with_empty["loss"] == pytest.approx(alone["loss"], rel=1e-6)


# ---- the metric functions ---------------------------------------------------------


def test_losses_are_weighted_by_their_target_counts() -> None:
    assert compute_loss([1.0, 4.0], weights=[3, 1]) == pytest.approx(7 / 4)
    assert compute_perplexity([1.0, 4.0], weights=[3, 1]) == pytest.approx(math.exp(7 / 4))


def test_without_weights_the_losses_are_averaged_exactly_as_before() -> None:
    """For losses that arrive with no labels to count targets from."""
    assert compute_loss([1.0, 4.0]) == 2.5
    assert compute_perplexity([1.0, 4.0]) == math.exp(2.5)
    assert compute_loss([]) == 0.0
    assert compute_perplexity([]) == 0.0


@pytest.mark.parametrize(
    ("losses", "weights"),
    [([1.0, 4.0], [3]), ([1.0], [3, 1]), ([], [3])],
    ids=["fewer-weights", "more-weights", "weights-without-losses"],
)
def test_weights_must_pair_one_to_one_with_losses(losses: list, weights: list) -> None:
    """Never truncated to fit: a mismatch is a caller's bug."""
    for metric in (compute_loss, compute_perplexity):
        with pytest.raises(ValueError, match="weights"):
            metric(losses, weights=weights)


def test_a_negative_weight_is_refused() -> None:
    for metric in (compute_loss, compute_perplexity):
        with pytest.raises(ValueError, match="negative"):
            metric([1.0, 4.0], weights=[3, -1])


def test_weights_that_count_no_target_report_zero_as_no_losses_always_have() -> None:
    for metric in (compute_loss, compute_perplexity):
        assert metric([1.0, 4.0], weights=[0, 0]) == 0.0
