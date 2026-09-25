"""The native evaluation worker measures what its metrics say, on the data it names.

The metrics are next-token and token-weighted (see
``xaytune.evaluation.native``). These tests hold them to that definition
directly -- on logits built by hand, where the right answer is known -- and
on the tiny model, against what transformers itself computes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tests.training_fixtures import tiny_model
from xaytune.workers.eval_native import (
    DatasetChangedError,
    NextTokenTally,
    NothingToMeasureError,
    measure,
)
from xaytune.workers.eval_native_schema import (
    NATIVE_EVALUATION_API_VERSION,
    NativeEvaluationWorkerConfig,
    content_digest,
)

VOCAB = 5


def _one_hot(tokens: list[int]) -> torch.Tensor:
    """Logits that put all their weight on *tokens*, one position each."""
    logits = torch.full((1, len(tokens), VOCAB), -30.0)
    for position, token in enumerate(tokens):
        logits[0, position, token] = 30.0
    return logits


# ---- the definitions, on logits whose answer is known -------------------------------


def test_position_i_is_scored_against_the_token_at_i_plus_one() -> None:
    labels = torch.tensor([[1, 2, 3, 4]])

    predicts_the_next = NextTokenTally()
    predicts_the_next.add(_one_hot([2, 3, 4, 0]), labels)
    assert predicts_the_next.metrics(["token_accuracy"]) == {"token_accuracy": 1.0}

    # What an unshifted comparison would call perfect is wrong every time.
    echoes_the_current = NextTokenTally()
    echoes_the_current.add(_one_hot([1, 2, 3, 4]), labels)
    assert echoes_the_current.metrics(["token_accuracy"]) == {"token_accuracy": 0.0}


def test_the_last_position_predicts_nothing_and_ignored_labels_are_not_scored() -> None:
    tally = NextTokenTally()
    tally.add(_one_hot([2, 0, 0, 0]), torch.tensor([[1, 2, -100, -100]]))
    assert tally.tokens == 1  # only 1 -> 2 is a prediction of a real token


def test_loss_is_mean_cross_entropy_per_token_and_perplexity_its_exponent() -> None:
    logits = torch.randn(2, 4, VOCAB, generator=torch.Generator().manual_seed(0))
    labels = torch.tensor([[1, 2, 3, 4], [0, 1, -100, -100]])
    tally = NextTokenTally()
    tally.add(logits, labels)

    predicted, target = logits[:, :-1].reshape(-1, VOCAB), labels[:, 1:].reshape(-1)
    expected = torch.nn.functional.cross_entropy(predicted, target, ignore_index=-100).item()
    values = tally.metrics(["loss", "perplexity"])
    assert values["loss"] == pytest.approx(expected, rel=1e-6)
    assert values["perplexity"] == pytest.approx(math.exp(expected), rel=1e-6)


def test_tokens_not_batches_are_what_is_averaged() -> None:
    """A short sequence and a long one: batch means would weight them equally."""
    generator = torch.Generator().manual_seed(1)
    short = torch.randn(1, 2, VOCAB, generator=generator)
    long = torch.randn(1, 6, VOCAB, generator=generator)
    short_labels, long_labels = torch.tensor([[1, 2]]), torch.tensor([[1, 2, 3, 4, 0, 1]])

    tally = NextTokenTally()
    tally.add(short, short_labels)
    tally.add(long, long_labels)

    alone = [NextTokenTally(), NextTokenTally()]
    alone[0].add(short, short_labels)
    alone[1].add(long, long_labels)
    by_token = (alone[0].loss_sum + alone[1].loss_sum) / (alone[0].tokens + alone[1].tokens)
    assert tally.metrics(["loss"])["loss"] == pytest.approx(by_token)
    by_batch = (alone[0].metrics(["loss"])["loss"] + alone[1].metrics(["loss"])["loss"]) / 2
    assert tally.metrics(["loss"])["loss"] != pytest.approx(by_batch)


def test_a_metric_over_nothing_is_refused() -> None:
    tally = NextTokenTally()
    tally.add(_one_hot([1]), torch.tensor([[1]]))
    with pytest.raises(NothingToMeasureError):
        tally.metrics(["loss"])


# ---- measure(), on the tiny model -------------------------------------------------


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny_model(tmp_path_factory.mktemp("model") / "tiny")


def _data(path: Path, texts: list[str]) -> Path:
    path.write_text("\n".join(json.dumps({"text": t}) for t in texts) + "\n", encoding="utf-8")
    return path


def _config(
    model: Path, data: Path, tmp_path: Path, **measure: object
) -> NativeEvaluationWorkerConfig:
    return NativeEvaluationWorkerConfig.model_validate(
        {
            "api_version": NATIVE_EVALUATION_API_VERSION,
            "model_uri": str(model),
            "data": {
                "path": str(data),
                "content_digest": content_digest(data),
                "format": "text",
                "max_seq_length": 32,
            },
            "measure": {
                "metrics": ["loss", "perplexity", "token_accuracy"],
                "batch_size": 2,
                "precision": "fp32",
                **measure,
            },
            "realization": {
                "evaluator_name": "native",
                "evaluator_version": "0.1.0",
                "seed": 7,
                "output_dir": str(tmp_path / "out"),
            },
        }
    )


TEXTS = ["hello world .", "train hello world . hello world .", "world ."]


def test_one_sequence_scores_what_transformers_computes(model_dir: Path, tmp_path: Path) -> None:
    """For one unpadded sequence, a causal LM's own loss is the same definition."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data = _data(tmp_path / "one.jsonl", [TEXTS[1]])
    values, record = measure(_config(model_dir, data, tmp_path, batch_size=1))

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    ids = AutoTokenizer.from_pretrained(model_dir)(TEXTS[1], return_tensors="pt")["input_ids"]
    with torch.no_grad():
        expected = model(input_ids=ids, labels=ids).loss.item()
    assert values["loss"] == pytest.approx(expected, rel=1e-5)
    assert record["tokens_scored"] == ids.shape[1] - 1


def test_the_batch_size_groups_work_and_does_not_change_the_answer(
    model_dir: Path, tmp_path: Path
) -> None:
    data = _data(tmp_path / "held-out.jsonl", TEXTS)
    one, _ = measure(_config(model_dir, data, tmp_path, batch_size=1))
    three, _ = measure(_config(model_dir, data, tmp_path, batch_size=3))
    for name in one:
        assert three[name] == pytest.approx(one[name], rel=1e-5), name


def test_what_is_measured_is_accounted_for(model_dir: Path, tmp_path: Path) -> None:
    data = _data(tmp_path / "held-out.jsonl", [*TEXTS, ""])
    values, record = measure(_config(model_dir, data, tmp_path, metrics=["perplexity"]))
    assert set(values) == {"perplexity"}
    assert (record["records"], record["sequences"], record["skipped_empty"]) == (4, 3, 1)
    assert set(record["environment"]) == {"device", "python", "torch", "transformers", "xaytune"}


def test_data_that_changed_since_it_was_named_is_not_measured(
    model_dir: Path, tmp_path: Path
) -> None:
    data = _data(tmp_path / "held-out.jsonl", TEXTS)
    config = _config(model_dir, data, tmp_path)
    _data(data, [*TEXTS, "hello ."])
    with pytest.raises(DatasetChangedError, match="would measure different data"):
        measure(config)


def test_a_model_without_its_tokenizer_is_not_measured(model_dir: Path, tmp_path: Path) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    for name in ("config.json", "model.safetensors"):
        (bare / name).write_bytes((model_dir / name).read_bytes())
    data = _data(tmp_path / "held-out.jsonl", TEXTS)
    with pytest.raises(FileNotFoundError, match="tokenizer_config.json"):
        measure(_config(bare, data, tmp_path))


def test_data_with_nothing_to_score_fails_rather_than_reporting_zero(
    model_dir: Path, tmp_path: Path
) -> None:
    data = _data(tmp_path / "empty.jsonl", ["", ""])
    with pytest.raises(NothingToMeasureError):
        measure(_config(model_dir, data, tmp_path))
