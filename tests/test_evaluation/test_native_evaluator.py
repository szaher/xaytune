"""NativeEvaluator: exact within a narrow surface, refused outside it.

Torch-free, as the evaluator is: it prepares in the controller. What the
worker measures is ``tests/test_workers/test_eval_native_worker.py``; the
evaluation end to end is ``tests/test_experiment/test_native_evaluation.py``.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from tests.evaluation_fixtures import NATIVE_CONFIG
from xaytune.core.domain.evaluation import EvaluationSpec, EvaluatorDeterminism, EvaluatorSpec
from xaytune.core.ids import ArtifactId
from xaytune.core.refs import ArtifactRef, DatasetRef
from xaytune.evaluation import EvaluationContext, Evaluator, UnsupportedEvaluationError
from xaytune.evaluation.native import NativeEvaluator, local_dataset
from xaytune.workers.eval_native_schema import NativeEvaluationWorkerConfig

DIGEST = "sha256:" + "a" * 64


def _spec(config: dict | None = None, **dataset: object) -> EvaluationSpec:
    fields: dict[str, object] = {"uri": "/data/held-out.jsonl", "content_digest": DIGEST}
    fields.update(dataset)
    return EvaluationSpec(
        evaluator=EvaluatorSpec(name="native", config=NATIVE_CONFIG if config is None else config),
        dataset=DatasetRef(**fields),  # type: ignore[arg-type]
    )


def _subject(**fields: object) -> ArtifactRef:
    values: dict[str, object] = {"id": ArtifactId.generate(), "kind": "model", "uri": "/m/run_1"}
    values.update(fields)
    return ArtifactRef(**values)  # type: ignore[arg-type]


def _context(**fields: object) -> EvaluationContext:
    values: dict[str, object] = {
        "experiment_id": "exp_1",
        "node_id": "node_1",
        "evaluation_run_id": "evalrun_1",
        "seed": 7,
        "replicate": 1,
        "output_uri": "/out/evalrun_1",
    }
    values.update(fields)
    return EvaluationContext(**values)  # type: ignore[arg-type]


def test_it_is_an_evaluator_that_declares_itself_seeded() -> None:
    """Never DETERMINISTIC: reuse would take that as the same number anywhere, forever."""
    evaluator = NativeEvaluator()
    assert isinstance(evaluator, Evaluator)
    assert evaluator.determinism is EvaluatorDeterminism.SEEDED


def test_a_fully_declared_evaluation_is_supported() -> None:
    assert NativeEvaluator().supports(_spec())


# ---- what it refuses, at submission ------------------------------------------------


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"format": "alpaca"}, "evaluator.config.format"),
        ({"precision": "bf16"}, "evaluator.config.precision"),
        ({"metrics": []}, "evaluator.config.metrics"),
        ({"metrics": ["loss", "loss"]}, "names one more than once"),
        ({"metrics": ["bleu"]}, "evaluator.config.metrics"),
        ({"batch_size": 0}, "evaluator.config.batch_size"),
        ({"max_seq_length": 0}, "evaluator.config.max_seq_length"),
        ({"shuffle": True}, "evaluator.config.shuffle"),
    ],
)
def test_a_config_it_cannot_honour_exactly_is_refused(change: dict, reason: str) -> None:
    reasons = NativeEvaluator().supports(_spec({**NATIVE_CONFIG, **change})).reasons
    assert reasons and any(reason in r for r in reasons), reasons


@pytest.mark.parametrize("missing", sorted(NATIVE_CONFIG))
def test_nothing_that_changes_the_numbers_is_defaulted(missing: str) -> None:
    config = {k: v for k, v in NATIVE_CONFIG.items() if k != missing}
    [refusal] = NativeEvaluator().supports(_spec(config)).reasons
    assert f"evaluator.config.{missing}" in refusal


@pytest.mark.parametrize(
    ("dataset", "reason"),
    [
        ({"uri": "data/held-out.jsonl"}, "not an absolute local path"),
        ({"content_digest": None}, "content_digest is undeclared"),
        ({"content_digest": "md5:abc"}, "is not 'sha256:'"),
        ({"revision": "main"}, "revision is declared"),
        ({"split": "test"}, "split is declared"),
        ({"tokenizer_fingerprint": "sha256:t"}, "tokenizer_fingerprint is declared"),
        ({"template_fingerprint": "sha256:t"}, "template_fingerprint is declared"),
        ({"transform_fingerprint": "sha256:t"}, "transform_fingerprint is declared"),
    ],
)
def test_data_it_cannot_pin_is_refused(dataset: dict, reason: str) -> None:
    [refusal] = NativeEvaluator().supports(_spec(**dataset)).reasons
    assert reason in refusal


def test_no_dataset_and_slices_are_refused() -> None:
    spec = EvaluationSpec(
        evaluator=EvaluatorSpec(name="native", config=NATIVE_CONFIG), slices=("hard",)
    )
    reasons = NativeEvaluator().supports(spec).reasons
    assert any("slices" in r for r in reasons)
    assert any("dataset is undeclared" in r for r in reasons)


def test_every_reason_is_given_at_once() -> None:
    reasons = (
        NativeEvaluator()
        .supports(_spec({**NATIVE_CONFIG, "precision": "bf16"}, content_digest=None, split="test"))
        .reasons
    )
    assert len(reasons) == 3


# ---- what prepare() adds: the subject and the run ---------------------------------


@pytest.mark.parametrize(
    ("subject", "context", "reason"),
    [
        ({"kind": "evaluation_report"}, {}, "not a model"),
        ({"uri": "s3://bucket/model"}, {}, "not an absolute local path"),
        ({}, {"seed": None}, "has no seed"),
        ({}, {"output_uri": None}, "output_uri"),
        ({}, {"output_uri": "out/relative"}, "output_uri"),
    ],
)
def test_prepare_refuses_a_subject_or_run_it_cannot_measure(
    subject: dict, context: dict, reason: str
) -> None:
    with pytest.raises(UnsupportedEvaluationError) as refused:
        NativeEvaluator().prepare(_subject(**subject), _spec(), _context(**context))
    assert refused.value.evaluator == "native"
    assert any(reason in r for r in refused.value.reasons)


def test_prepare_also_refuses_what_supports_refuses() -> None:
    with pytest.raises(UnsupportedEvaluationError, match="content_digest"):
        NativeEvaluator().prepare(_subject(), _spec(content_digest=None), _context())


# ---- what it prepares --------------------------------------------------------------


def test_the_plan_carries_everything_the_worker_needs_explicitly() -> None:
    subject = _subject()
    spec = NativeEvaluator().prepare(subject, _spec(), _context())

    assert spec.subject == subject
    assert spec.evaluation_fingerprint == _spec().evaluation_fingerprint()
    assert (spec.evaluator.name, spec.evaluator.version) == ("native", "0.1.0")
    assert spec.entrypoint.module == "xaytune.workers.eval_native"  # type: ignore[union-attr]

    config = NativeEvaluationWorkerConfig.model_validate(dict(spec.config))
    assert config.model_uri == "/m/run_1"
    assert (config.data.path, config.data.content_digest) == ("/data/held-out.jsonl", DIGEST)
    assert (config.data.format, config.data.max_seq_length) == ("text", 32)
    assert config.measure.metrics == ("loss", "perplexity", "token_accuracy")
    assert (config.measure.batch_size, config.measure.precision) == (2, "fp32")
    realization = config.realization
    assert (realization.evaluator_name, realization.evaluator_version) == ("native", "0.1.0")
    assert (realization.seed, realization.output_dir) == (7, "/out/evalrun_1")


def test_preparing_is_deterministic() -> None:
    """A re-issued evaluation is rebuilt from the record; its digest must not move."""
    subject = _subject()
    first = NativeEvaluator().prepare(subject, _spec(), _context())
    assert NativeEvaluator().prepare(subject, _spec(), _context()) == first


def test_preparing_reads_no_file() -> None:
    """The dataset path need not exist: the worker checks it, where it runs."""
    spec = _spec(uri="/nowhere/at/all.jsonl")
    assert NativeEvaluator().prepare(_subject(), spec, _context()).config


@pytest.mark.parametrize(
    "change",
    [
        {"max_seq_length": 16},
        {"batch_size": 4},
        {"metrics": ["loss"]},
    ],
    ids=["truncation", "batching", "metrics"],
)
def test_what_changes_the_numbers_changes_the_fingerprint(change: dict) -> None:
    changed = _spec({**NATIVE_CONFIG, **change}).evaluation_fingerprint()
    assert changed != _spec().evaluation_fingerprint()


def test_different_bytes_are_a_different_evaluation() -> None:
    other = _spec(content_digest="sha256:" + "b" * 64).evaluation_fingerprint()
    assert other != _spec().evaluation_fingerprint()


# ---- pinning a local file ----------------------------------------------------------


def test_local_dataset_pins_the_bytes(tmp_path: Path) -> None:
    path = tmp_path / "held-out.jsonl"
    path.write_bytes(b'{"text": "hello"}\n')
    pinned = local_dataset(path)
    assert pinned.uri == str(path.resolve())
    assert pinned.content_digest == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    path.write_bytes(b'{"text": "hello!"}\n')
    assert local_dataset(path).content_digest != pinned.content_digest


def test_the_evaluator_needs_no_training_stack() -> None:
    """It prepares in the controller (ADR-010)."""
    code = (
        "import sys, xaytune.evaluation.native\n"
        "heavy = [m for m in ('torch', 'transformers', 'trl', 'datasets') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
